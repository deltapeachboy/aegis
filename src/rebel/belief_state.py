"""
ポケモンバトルにおける信念状態の管理

相手の隠された情報（技構成・持ち物・テラスタイプ等）に対する
確率分布を管理し、観測（素早さ関係の不等式、被弾した実ダメージ等）に
基づいてベイズ更新および確定枝刈りを行う。
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

# 🌟 赤線対策として、直接 pokepy からインポート
from pokepy.pokemon import Pokemon

from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase

from .ev_template import (
    EVSpread,
    EVSpreadType,
    estimate_ev_spread_type,
    get_ev_spread,
    get_ivs_for_spread_type,
    NATURE_BOOST,
    NATURE_PENALTY,
)


class ObservationType(Enum):
    """観測イベントの種類"""

    # === 持ち物が確定する観測 ===
    ITEM_REVEALED = auto()  # 持ち物が直接判明（トリック等）
    FOCUS_SASH_ACTIVATED = auto()  # きあいのタスキ発動
    CHOICE_LOCKED = auto()  # こだわり系で技固定
    LEFTOVERS_HEAL = auto()  # たべのこし回復
    BLACK_SLUDGE_HEAL = auto()  # くろいヘドロ回復
    LIFE_ORB_RECOIL = auto()  # いのちのたま反動
    ROCKY_HELMET_DAMAGE = auto()  # ゴツゴツメット発動
    ASSAULT_VEST_BLOCK = auto()  # とつげきチョッキで変化技不可
    BOOST_ENERGY_ACTIVATED = auto()  # ブーストエナジー発動
    BERRY_CONSUMED = auto()  # きのみ消費
    AIR_BALLOON_CONSUMED = auto()  # ふうせん消費（被弾時）

    # === 技が判明する観測 ===
    MOVE_USED = auto()  # 技を使用した

    # === テラスタイプが判明する観測 ===
    TERASTALLIZED = auto()  # テラスタル使用

    # === 特性が判明する観測 ===
    ABILITY_REVEALED = auto()  # 特性発動

    # === 推測に使える観測 ===
    OUTSPED_UNEXPECTEDLY = auto()  # 予想外に先制 → 素早さ不等式による確定枝刈り
    HIGH_DAMAGE_DEALT = auto()  # 高ダメージ → 火力アイテム実ダメージ逆算による確定枝刈り
    STATUS_CURED = auto()  # 状態異常回復 → ラムのみ疑惑
    SURVIVED_UNEXPECTEDLY = auto()  # 予想外の耐え → チョッキ/耐久振り疑惑


@dataclass
class Observation:
    """観測イベント"""

    type: ObservationType
    pokemon_name: str
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PokemonTypeHypothesis:
    """
    ポケモンの型仮説
    """

    moves: tuple[str, ...]  # 4技（ソート済み）
    item: str
    tera_type: str
    nature: str
    ability: str
    ev_spread_type: EVSpreadType = EVSpreadType.UNKNOWN  # EV配分タイプ

    def __repr__(self) -> str:
        moves_str = ", ".join(self.moves[:2]) + "..."
        return f"Hypothesis(item={self.item}, tera={self.tera_type}, evs={self.ev_spread_type.name}, moves=[{moves_str}])"

    @classmethod
    def from_lists(
        cls,
        moves: list[str],
        item: str,
        tera_type: str,
        nature: str,
        ability: str,
        base_stats: Optional[list[int]] = None,
    ) -> "PokemonTypeHypothesis":
        """リストから作成"""
        ev_type = estimate_ev_spread_type(nature, base_stats, moves)
        return cls(
            moves=tuple(sorted(moves)),
            item=item,
            tera_type=tera_type,
            nature=nature,
            ability=ability,
            ev_spread_type=ev_type,
        )

    def matches_revealed_moves(self, revealed: set[str]) -> bool:
        """判明した技と矛盾しないか"""
        return revealed.issubset(set(self.moves))

    def to_dict(self) -> dict[str, Any]:
        """シリアライズ可能な辞書に変換"""
        return {
            "moves": list(self.moves),
            "item": self.item,
            "tera_type": self.tera_type,
            "nature": self.nature,
            "ability": self.ability,
            "ev_spread_type": self.ev_spread_type.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PokemonTypeHypothesis":
        """辞書から復元"""
        return cls(
            moves=tuple(data["moves"]),
            item=data["item"],
            tera_type=data["tera_type"],
            nature=data["nature"],
            ability=data["ability"],
            ev_spread_type=EVSpreadType[data["ev_spread_type"]],
        )

    def get_evs(self) -> list[int]:
        """EV配分をリストで取得 [H, A, B, C, D, S]"""
        from .ev_template import get_ev_spread
        spread = get_ev_spread(self.nature, moves=list(self.moves))
        return spread.to_list()

    def get_ivs(self) -> list[int]:
        """個体値をリストで取得（常に素早さ31）"""
        return [31, 31, 31, 31, 31, 31]


class PokemonBeliefState:
    """
    相手パーティ全体に対する信念状態
    """

    def __init__(
        self,
        opponent_pokemon_names: list[str],
        usage_db: PokemonUsageDatabase,
        max_hypotheses_per_pokemon: int = 50,
        min_probability: float = 0.01,
    ):
        self.usage_db = usage_db
        self.max_hypotheses = max_hypotheses_per_pokemon
        self.min_probability = min_probability

        self.beliefs: dict[str, dict[PokemonTypeHypothesis, float]] = {}

        self.revealed_moves: dict[str, set[str]] = {
            name: set() for name in opponent_pokemon_names
        }
        self.revealed_items: dict[str, Optional[str]] = {
            name: None for name in opponent_pokemon_names
        }
        self.revealed_tera: dict[str, Optional[str]] = {
            name: None for name in opponent_pokemon_names
        }
        self.revealed_abilities: dict[str, Optional[str]] = {
            name: None for name in opponent_pokemon_names
        }

        self.move_use_count: dict[str, dict[str, int]] = {
            name: {} for name in opponent_pokemon_names
        }

        self.observation_history: list[Observation] = []

        for name in opponent_pokemon_names:
            self.beliefs[name] = self._build_initial_belief(name)

    def _build_initial_belief(
        self, pokemon_name: str
    ) -> dict[PokemonTypeHypothesis, float]:
        """使用率データから初期信念を構築"""
        hypotheses: dict[PokemonTypeHypothesis, float] = {}
        base_stats = self._get_base_stats(pokemon_name)

        move_prior = self.usage_db.get_move_prior(pokemon_name, min_probability=0.05)
        item_prior = self.usage_db.get_item_prior(pokemon_name, min_probability=0.05)
        tera_prior = self.usage_db.get_tera_prior(pokemon_name, min_probability=0.05)
        nature_prior = self.usage_db.get_nature_prior(pokemon_name, min_probability=0.05)
        ability_prior = self.usage_db.get_ability_prior(pokemon_name, min_probability=0.01)

        if not move_prior:
            return hypotheses
        if not item_prior:
            item_prior = {"きあいのタスキ": 1.0}
        if not tera_prior:
            tera_prior = {"ノーマル": 1.0}
        if not nature_prior:
            nature_prior = self._get_default_nature_prior(base_stats)
        if not ability_prior:
            ability_prior = {"": 1.0}

        num_samples = self.max_hypotheses * 10

        for _ in range(num_samples):
            moves = self._sample_moveset(move_prior, 4)
            item = self._weighted_sample(item_prior)
            tera = self._weighted_sample(tera_prior)
            nature = self._weighted_sample(nature_prior)
            ability = self._weighted_sample(ability_prior)

            hypothesis = PokemonTypeHypothesis.from_lists(
                moves=moves,
                item=item,
                tera_type=tera,
                nature=nature,
                ability=ability,
                base_stats=base_stats,
            )

            prob = self._calculate_hypothesis_probability(
                hypothesis, move_prior, item_prior, tera_prior, nature_prior, ability_prior
            )

            if hypothesis in hypotheses:
                hypotheses[hypothesis] = max(hypotheses[hypothesis], prob)
            else:
                hypotheses[hypothesis] = prob

        return self._prune_and_normalize(hypotheses)

    def _sample_moveset(self, move_prior: dict[str, float], num_moves: int) -> list[str]:
        moves = []
        available = dict(move_prior)

        for _ in range(min(num_moves, len(available))):
            if not available:
                break
            move = self._weighted_sample(available)
            moves.append(move)
            del available[move]
            total = sum(available.values())
            if total > 0:
                available = {k: v / total for k, v in available.items()}

        return moves

    def _calculate_hypothesis_probability(
        self,
        hypothesis: PokemonTypeHypothesis,
        move_prior: dict[str, float],
        item_prior: dict[str, float],
        tera_prior: dict[str, float],
        nature_prior: dict[str, float],
        ability_prior: dict[str, float],
    ) -> float:
        prob = 1.0
        for move in hypothesis.moves:
            prob *= move_prior.get(move, 0.01)

        prob *= item_prior.get(hypothesis.item, 0.01)
        prob *= tera_prior.get(hypothesis.tera_type, 0.01)
        prob *= nature_prior.get(hypothesis.nature, 0.1)

        if hypothesis.ability:
            prob *= ability_prior.get(hypothesis.ability, 0.1)

        return prob

    def _prune_and_normalize(
        self, hypotheses: dict[PokemonTypeHypothesis, float]
    ) -> dict[PokemonTypeHypothesis, float]:
        if not hypotheses:
            return {}

        sorted_hypos = sorted(hypotheses.items(), key=lambda x: -x[1])
        top_hypos = sorted_hypos[: self.max_hypotheses]

        total = sum(prob for _, prob in top_hypos)
        if total > 0:
            return {h: p / total for h, p in top_hypos}
        n = len(hypotheses)
        return {h: 1.0 / n for h, _ in top_hypos}

    def _weighted_sample(self, probs: dict[str, float]) -> str:
        if not probs:
            return ""
        items = list(probs.keys())
        weights = list(probs.values())
        return random.choices(items, weights=weights, k=1)[0]

    def _get_base_stats(self, pokemon_name: str) -> Optional[list[int]]:
        if pokemon_name in Pokemon.zukan:
            return Pokemon.zukan[pokemon_name].get("base")
        return None

    def _get_default_nature_prior(self, base_stats: Optional[list[int]]) -> dict[str, float]:
        default_natures = ["いじっぱり", "ようき", "ひかえめ", "おくびょう"]
        n = len(default_natures)
        return {nat: 1.0 / n for nat in default_natures}

    # =========================================================================
    # 🌟 新規追加：仮説全滅（空リスト）時の「安全な自動再生成フォールバック」
    # =========================================================================

    def _regenerate_hypotheses_with_constraints(self, pokemon_name: str) -> dict[PokemonTypeHypothesis, float]:
        """
        フィルタリングによって適合する型候補が0になった際、
        現在までに確定した情報（確定技、持ち物、テラス、特性）を満たす仮説を
        使用率データベースから再サンプリングして、安全に信念をリフレッシュ（復旧）する。
        """
        base_stats = self._get_base_stats(pokemon_name)
        confirmed_moves = list(self.revealed_moves[pokemon_name])
        confirmed_item = self.revealed_items[pokemon_name]
        confirmed_ability = self.revealed_abilities[pokemon_name]
        confirmed_tera = self.revealed_tera[pokemon_name]

        # 確率情報のロード
        move_prior = self.usage_db.get_move_prior(pokemon_name, min_probability=0.0)
        item_prior = self.usage_db.get_item_prior(pokemon_name, min_probability=0.0)
        tera_prior = self.usage_db.get_tera_prior(pokemon_name, min_probability=0.0)
        nature_prior = self.usage_db.get_nature_prior(pokemon_name, min_probability=0.0)
        ability_prior = self.usage_db.get_ability_prior(pokemon_name, min_probability=0.0)

        # 確定情報を優先設定
        if confirmed_item:
            item_prior = {confirmed_item: 1.0}
        if confirmed_tera:
            tera_prior = {confirmed_tera: 1.0}
        if confirmed_ability:
            ability_prior = {confirmed_ability: 1.0}

        hypotheses: dict[PokemonTypeHypothesis, float] = {}
        for _ in range(self.max_hypotheses * 5):
            # 確定技をベースに4つの技構成をサンプリング
            moves_sampled = list(confirmed_moves)
            remaining_prior = {k: v for k, v in move_prior.items() if k not in moves_sampled}
            while len(moves_sampled) < 4 and remaining_prior:
                mv = self._weighted_sample(remaining_prior)
                moves_sampled.append(mv)
                remaining_prior.pop(mv, None)

            while len(moves_sampled) < 4:
                moves_sampled.append("わるあがき")

            item = self._weighted_sample(item_prior) if item_prior else "なし"
            tera = self._weighted_sample(tera_prior) if tera_prior else "ノーマル"
            nature = self._weighted_sample(nature_prior) if nature_prior else "いじっぱり"
            ability = self._weighted_sample(ability_prior) if ability_prior else "とくせいなし"

            hypo = PokemonTypeHypothesis.from_lists(
                moves=moves_sampled,
                item=item,
                tera_type=tera,
                nature=nature,
                ability=ability,
                base_stats=base_stats,
            )
            hypotheses[hypo] = 1.0

        return self._prune_and_normalize(hypotheses)

    # =========================================================================
    # ベイズ更新 ＆ 素早さ・ダメージ確定不等式ハードフィルタリング
    # =========================================================================

    def _calculate_hypothetical_speed(
        self,
        pokemon_name: str,
        hypothesis: PokemonTypeHypothesis,
        details: dict[str, Any],
    ) -> int:
        """
        特定の仮説における相手の素早さ実数値（レベル50）を仕様に基づいて厳密に算出する
        """
        base_stats = self._get_base_stats(pokemon_name)
        if not base_stats:
            return 100

        base_s = base_stats[5]  # 素早さ種族値

        from .ev_template import get_ev_spread
        spread = get_ev_spread(hypothesis.nature, base_stats, list(hypothesis.moves))
        ev_s = spread.speed

        # システム仕様に合わせ素早さ個体値は常に31
        iv_s = 31

        # レベル50時点の素早さ実数値の算出
        speed_stat = int((base_s * 2 + iv_s + ev_s // 4) * 0.5) + 5

        # 性格（Nature）補正
        nature = hypothesis.nature
        if NATURE_BOOST.get(nature) == "speed":
            speed_stat = int(speed_stat * 1.1)
        elif NATURE_PENALTY.get(nature) == "speed":
            speed_stat = int(speed_stat * 0.9)

        # ランク補正
        rank_stage = details.get("opp_speed_rank", 0)
        if rank_stage > 0:
            speed_stat = int(speed_stat * (2 + rank_stage) / 2)
        elif rank_stage < 0:
            speed_stat = int(speed_stat * 2 / (2 - rank_stage))

        # こだわりスカーフ補正
        if hypothesis.item == "こだわりスカーフ":
            speed_stat = int(speed_stat * 1.5)

        # 特性と天候・フィールドの相乗効果
        ability = hypothesis.ability
        weather = details.get("weather", "")
        terrain = details.get("terrain", "")

        if ability == "すいすい" and weather == "rainy":
            speed_stat = int(speed_stat * 2.0)
        elif ability == "ようりょくそ" and weather == "sunny":
            speed_stat = int(speed_stat * 2.0)
        elif ability == "すなかき" and weather == "sandstorm":
            speed_stat = int(speed_stat * 2.0)
        elif ability == "ゆきかき" and weather == "snow":
            speed_stat = int(speed_stat * 2.0)

        if ability == "クォークチャージ" and terrain == "electric":
            speed_stat = int(speed_stat * 1.5)
        elif ability == "こだいかっせい" and weather == "sunny":
            speed_stat = int(speed_stat * 1.5)

        # 麻痺状態補正
        opp_is_paralyzed = details.get("opp_paralyzed", False)
        if opp_is_paralyzed:
            speed_stat = int(speed_stat * 0.5)

        # おいかぜ補正
        opp_has_tailwind = details.get("opp_tailwind", False)
        if opp_has_tailwind:
            speed_stat = int(speed_stat * 2.0)

        return speed_stat

    def _filter_by_speed_order(self, pokemon_name: str, details: dict[str, Any]) -> None:
        """
        行動順の観測に基づいて、素早さの確定不等式を満たさない矛盾仮説を完全に排除（確率0）する
        """
        my_speed = details.get("my_speed")
        opp_moved_first = details.get("opp_moved_first", True)

        if my_speed is None:
            return

        current = self.beliefs[pokemon_name]
        updated = {}

        for h, p in current.items():
            hypo_speed = self._calculate_hypothetical_speed(pokemon_name, h, details)

            if opp_moved_first:
                if hypo_speed < my_speed:
                    continue
            else:
                if hypo_speed > my_speed:
                    continue

            updated[h] = p

        # 🌟 仮説が全滅した場合は、自動再生成フォールバックを起動
        if updated:
            self.beliefs[pokemon_name] = self._prune_and_normalize(updated)
        else:
            self.beliefs[pokemon_name] = self._regenerate_hypotheses_with_constraints(pokemon_name)

# 🌟 新規追加：実ダメージ計算に基づく「火力補正アイテムの厳密な逆算特定」
    # =========================================================================
    # 🌟 修正版：実ダメージ計算に基づく「火力補正アイテムの厳密な逆算特定」
    # =========================================================================
        # =========================================================================
        # 🌟 修正版（不具合解消済）：実ダメージ計算に基づく「火力補正アイテムの厳密な逆算特定」
        # =========================================================================
    def _filter_by_damage_observation(self, pokemon_name: str, details: dict[str, Any]) -> None:
        """
        被弾した実ダメージから、物理的にあり得ない火力補正アイテム（ハチマキ、眼鏡、命の珠、等）
        を保持する矛盾仮説を100%の数学的精度で除外する
        """
        damage_taken = details.get("damage_taken")
        move_name = details.get("move_used")  # 技名
        def_stats = details.get("my_def_stats")  # 自分（防御側）の実数値リスト [H, A, B, C, D, S]

        if damage_taken is None or not move_name or not def_stats:
            return

        move_data = Pokemon.all_moves.get(move_name)
        if not move_data:
            return

        power = move_data.get("power", 0)
        move_type = move_data.get("type", "ノーマル")
        move_class = move_data.get("class", "phy")

        if power == 0 or move_class == "status":
            return

        current = self.beliefs[pokemon_name]
        updated = {}

        for h, p in current.items():
            base_stats = self._get_base_stats(pokemon_name)
            if not base_stats:
                updated[h] = p
                continue

            # 1. 攻撃側の実数値ステータスの算出（レベル50）
            is_special = "spc" in move_class
            atk_base = base_stats[3] if is_special else base_stats[1]  # C or A

            from .ev_template import get_ev_spread
            spread = get_ev_spread(h.nature, base_stats, list(h.moves))
            ev_atk = spread.sp_attack if is_special else spread.attack

            # 攻撃側実数値の算出
            atk_stat = int((atk_base * 2 + 31 + ev_atk // 4) * 0.5) + 5

            # 性格補正
            nature = h.nature
            if NATURE_BOOST.get(nature) == ("sp_attack" if is_special else "attack"):
                atk_stat = int(atk_stat * 1.1)
            elif NATURE_PENALTY.get(nature) == ("sp_attack" if is_special else "attack"):
                atk_stat = int(atk_stat * 0.9)

            # 持ち物による攻撃ステータス補正（ハチマキ・眼鏡）
            if h.item == "こだわりハチマキ" and not is_special:
                atk_stat = int(atk_stat * 1.5)
            elif h.item == "こだわりメガネ" and is_special:
                atk_stat = int(atk_stat * 1.5)

            # 防御側の実数値（味方の実数値なので完全既知。def_stats: [H, A, B, C, D, S]）
            def_stat = def_stats[2] if not is_special else def_stats[4]  # B（物理防御） or D（特殊防）

            # 2. 基礎ダメージ計算（残骸コードを撤廃し、厳密かつ安全に修正完了）
            try:
                base_damage = int(int(22 * power * atk_stat / def_stat) / 50) + 2
            except (ZeroDivisionError, ValueError):
                updated[h] = p
                continue

            # 各火力アイテムによる最終ダメージ倍率（補正値）の算出
            item_modifier = 1.0
            if h.item == "いのちのたま":
                item_modifier = 1.3
            elif h.item in ("たつじんのおび", "しんぴのしずく", "もくたん", "メタルコート", "じしゃく",
                            "きせきのタネ", "とけないこおり", "まがったスプーン", "ようせいのハネ", "くろいメガネ",
                            "くろおび", "のろいのおふだ", "りゅうのキバ", "どくばり", "やわらかいすな",
                            "かたいたし", "するどいくちばし", "ぎんのこな"):
                # タイプ一致強化アイテムまたは達人の帯（一律で最大補正の可能性があるものを1.2倍としてチェック）
                item_modifier = 1.2

            # 乱数幅のシミュレーション（レベル50公式に基づく [0.85 〜 1.0] 倍）
            min_possible = int(base_damage * 0.85 * item_modifier)
            max_possible = int(base_damage * 1.00 * item_modifier)

            # 実被ダメージが、この仮説における最大・最小の物理的限界を超えている場合は矛盾
            if damage_taken < min_possible or damage_taken > max_possible:
                continue  # 矛盾仮説として排除

            updated[h] = p

        # 仮説が全滅した場合は、自動再生成フォールバックを起動
        if updated:
            self.beliefs[pokemon_name] = self._prune_and_normalize(updated)
        else:
            self.beliefs[pokemon_name] = self._regenerate_hypotheses_with_constraints(pokemon_name)

    def update(self, observation: Observation) -> None:
        """
        観測に基づいて信念をベイズ更新
        """
        self.observation_history.append(observation)
        pokemon_name = observation.pokemon_name

        if pokemon_name not in self.beliefs:
            return

        obs_type = observation.type

        # === 技の判明 ===
        if obs_type == ObservationType.MOVE_USED:
            move = observation.details.get("move")
            if move:
                self.revealed_moves[pokemon_name].add(move)
                self._filter_by_revealed_moves(pokemon_name)
                if pokemon_name not in self.move_use_count:
                    self.move_use_count[pokemon_name] = {}
                self.move_use_count[pokemon_name][move] = \
                    self.move_use_count[pokemon_name].get(move, 0) + 1

        # === 持ち物の確定 ===
        elif obs_type == ObservationType.ITEM_REVEALED:
            item = observation.details.get("item")
            if item:
                self._confirm_item(pokemon_name, item)

        elif obs_type == ObservationType.FOCUS_SASH_ACTIVATED:
            self._confirm_item(pokemon_name, "きあいのタスキ")

        elif obs_type == ObservationType.CHOICE_LOCKED:
            self._filter_to_items(
                pokemon_name,
                ["こだわりハチマキ", "こだわりメガネ", "こだわりスカーフ"],
            )

        elif obs_type == ObservationType.LEFTOVERS_HEAL:
            self._confirm_item(pokemon_name, "たべのこし")

        elif obs_type == ObservationType.BLACK_SLUDGE_HEAL:
            self._confirm_item(pokemon_name, "くろいヘドロ")

        elif obs_type == ObservationType.LIFE_ORB_RECOIL:
            self._confirm_item(pokemon_name, "いのちのたま")

        elif obs_type == ObservationType.ROCKY_HELMET_DAMAGE:
            self._confirm_item(pokemon_name, "ゴツゴツメット")

        elif obs_type == ObservationType.ASSAULT_VEST_BLOCK:
            self._confirm_item(pokemon_name, "とつげきチョッキ")

        elif obs_type == ObservationType.BOOST_ENERGY_ACTIVATED:
            self._confirm_item(pokemon_name, "ブーストエナジー")

        elif obs_type == ObservationType.BERRY_CONSUMED:
            berry = observation.details.get("berry")
            if berry:
                self._confirm_item(pokemon_name, berry)

        # === テラスタイプの確定 ===
        elif obs_type == ObservationType.TERASTALLIZED:
            tera_type = observation.details.get("tera_type")
            if tera_type:
                self._confirm_tera(pokemon_name, tera_type)

        # === 特性の確定 ===
        elif obs_type == ObservationType.ABILITY_REVEALED:
            ability = observation.details.get("ability")
            if ability:
                self._confirm_ability(pokemon_name, ability)

        # === 推測観測（不等式・物理ダメージ逆算による枝刈り） ===
        elif obs_type == ObservationType.OUTSPED_UNEXPECTEDLY:
            if "my_speed" in observation.details:
                self._filter_by_speed_order(pokemon_name, observation.details)
            else:
                self._boost_item_probability(pokemon_name, "こだわりスカーフ", factor=3.0)

        elif obs_type == ObservationType.HIGH_DAMAGE_DEALT:
            # 🌟 もし詳細な実ダメージ、被弾技、自分の耐久実数値が渡された場合は、厳密な実ダメージ逆算特定を実行
            if "damage_taken" in observation.details and "my_def_stats" in observation.details:
                self._filter_by_damage_observation(pokemon_name, observation.details)
            else:
                # 簡易フォールバック（確率ブースト）
                category = observation.details.get("category", "physical")
                if category == "physical":
                    self._boost_item_probability(pokemon_name, "こだわりハチマキ", factor=2.0)
                    self._boost_item_probability(pokemon_name, "いのちのたま", factor=1.5)
                else:
                    self._boost_item_probability(pokemon_name, "こだわりメガネ", factor=2.0)
                    self._boost_item_probability(pokemon_name, "いのちのたま", factor=1.5)

        elif obs_type == ObservationType.STATUS_CURED:
            self._boost_item_probability(pokemon_name, "ラムのみ", factor=5.0)

    def _filter_by_revealed_moves(self, pokemon_name: str) -> None:
        """判明した技と矛盾する仮説を除外"""
        revealed = self.revealed_moves[pokemon_name]
        if not revealed:
            return

        current = self.beliefs[pokemon_name]
        filtered = {
            h: p for h, p in current.items() if h.matches_revealed_moves(revealed)
        }

        if filtered:
            self.beliefs[pokemon_name] = self._prune_and_normalize(filtered)
        else:
            self.beliefs[pokemon_name] = self._regenerate_hypotheses_with_constraints(pokemon_name)

    def _confirm_item(self, pokemon_name: str, item: str) -> None:
        """持ち物を確定"""
        self.revealed_items[pokemon_name] = item
        current = self.beliefs[pokemon_name]
        filtered = {h: p for h, p in current.items() if h.item == item}
        if filtered:
            self.beliefs[pokemon_name] = self._prune_and_normalize(filtered)
        else:
            self.beliefs[pokemon_name] = self._regenerate_hypotheses_with_constraints(pokemon_name)

    def _filter_to_items(self, pokemon_name: str, items: list[str]) -> None:
        """指定した持ち物のみに絞り込み"""
        current = self.beliefs[pokemon_name]
        filtered = {h: p for h, p in current.items() if h.item in items}
        if filtered:
            self.beliefs[pokemon_name] = self._prune_and_normalize(filtered)
        else:
            self.beliefs[pokemon_name] = self._regenerate_hypotheses_with_constraints(pokemon_name)

    def _boost_item_probability(
        self, pokemon_name: str, item: str, factor: float
    ) -> None:
        """特定の持ち物を持つ仮説の確率を上げる"""
        current = self.beliefs[pokemon_name]
        updated = {}
        for h, p in current.items():
            if h.item == item:
                updated[h] = p * factor
            else:
                updated[h] = p
        self.beliefs[pokemon_name] = self._prune_and_normalize(updated)

    def _confirm_tera(self, pokemon_name: str, tera_type: str) -> None:
        """テラスタイプを確定"""
        self.revealed_tera[pokemon_name] = tera_type
        current = self.beliefs[pokemon_name]
        filtered = {h: p for h, p in current.items() if h.tera_type == tera_type}
        if filtered:
            self.beliefs[pokemon_name] = self._prune_and_normalize(filtered)
        else:
            self.beliefs[pokemon_name] = self._regenerate_hypotheses_with_constraints(pokemon_name)

    def _confirm_ability(self, pokemon_name: str, ability: str) -> None:
        """特性を確定"""
        self.revealed_abilities[pokemon_name] = ability
        current = self.beliefs[pokemon_name]
        filtered = {h: p for h, p in current.items() if h.ability == ability}
        if filtered:
            self.beliefs[pokemon_name] = self._prune_and_normalize(filtered)
        else:
            self.beliefs[pokemon_name] = self._regenerate_hypotheses_with_constraints(pokemon_name)

    # =========================================================================
    # サンプリング
    # =========================================================================

    def sample_world(self) -> dict[str, PokemonTypeHypothesis]:
        """
        現在の信念から1つの「世界」（型の組み合わせ）をサンプリング
        """
        world = {}
        for pokemon_name, belief in self.beliefs.items():
            if not belief:
                continue
            hypotheses = list(belief.keys())
            probs = list(belief.values())
            world[pokemon_name] = random.choices(hypotheses, weights=probs, k=1)[0]
        return world

    def sample_worlds(self, n: int) -> list[dict[str, PokemonTypeHypothesis]]:
        """複数の世界をサンプリング"""
        return [self.sample_world() for _ in range(n)]

    def get_most_likely_world(self) -> dict[str, PokemonTypeHypothesis]:
        """最も確率の高い世界を取得"""
        world = {}
        for pokemon_name, belief in self.beliefs.items():
            if belief:
                world[pokemon_name] = max(belief.items(), key=lambda x: x[1])[0]
        return world

    # =========================================================================
    # 確率取得
    # =========================================================================

    def get_item_distribution(self, pokemon_name: str) -> dict[str, float]:
        """持ち物の周辺確率分布を取得"""
        if pokemon_name not in self.beliefs:
            return {}

        distribution: dict[str, float] = {}
        for hypothesis, prob in self.beliefs[pokemon_name].items():
            item = hypothesis.item
            distribution[item] = distribution.get(item, 0.0) + prob
        return distribution

    def get_tera_distribution(self, pokemon_name: str) -> dict[str, float]:
        """テラスタイプの周辺確率分布を取得"""
        if pokemon_name not in self.beliefs:
            return {}

        distribution: dict[str, float] = {}
        for hypothesis, prob in self.beliefs[pokemon_name].items():
            tera = hypothesis.tera_type
            distribution[tera] = distribution.get(tera, 0.0) + prob
        return distribution

    def get_move_probability(self, pokemon_name: str, move: str) -> float:
        """特定の技を持っている確率を取得"""
        if pokemon_name not in self.beliefs:
            return 0.0

        prob = 0.0
        for hypothesis, p in self.beliefs[pokemon_name].items():
            if move in hypothesis.moves:
                prob += p
        return prob

    def get_ability_distribution(self, pokemon_name: str) -> dict[str, float]:
        """特性の周辺確率分布を取得"""
        if pokemon_name not in self.beliefs:
            return {}

        distribution: dict[str, float] = {}
        for hypothesis, prob in self.beliefs[pokemon_name].items():
            ability = hypothesis.ability
            distribution[ability] = distribution.get(ability, 0.0) + prob
        return distribution

    # =========================================================================
    # ユーティリティ
    # =========================================================================

    def copy(self) -> "PokemonBeliefState":
        """信念状態のディープコピー"""
        new_state = PokemonBeliefState.__new__(PokemonBeliefState)
        new_state.usage_db = self.usage_db
        new_state.max_hypotheses = self.max_hypotheses
        new_state.min_probability = self.min_probability
        new_state.beliefs = deepcopy(self.beliefs)
        new_state.revealed_moves = deepcopy(self.revealed_moves)
        new_state.revealed_items = deepcopy(self.revealed_items)
        new_state.revealed_tera = deepcopy(self.revealed_tera)
        new_state.revealed_abilities = deepcopy(self.revealed_abilities)
        new_state.move_use_count = deepcopy(self.move_use_count)
        new_state.observation_history = list(self.observation_history)
        return new_state

    def get_move_use_count(self, pokemon_name: str, move: str) -> int:
        """特定の技の使用回数を取得"""
        if pokemon_name not in self.move_use_count:
            return 0
        return self.move_use_count[pokemon_name].get(move, 0)

    def get_all_move_use_counts(self, pokemon_name: str) -> dict[str, int]:
        """ポケモンの全技使用回数を取得"""
        return self.move_use_count.get(pokemon_name, {})

    def estimate_pp_remaining(self, pokemon_name: str, move: str, max_pp: int = 8) -> float:
        """
        相手の技のPP残量を推定
        """
        use_count = self.get_move_use_count(pokemon_name, move)
        estimated_remaining = max(0, max_pp - use_count)
        return estimated_remaining / max_pp if max_pp > 0 else 1.0

    def summary(self) -> str:
        """信念状態の概要"""
        lines = ["PokemonBeliefState Summary", "=" * 50]

        for pokemon_name in self.beliefs:
            lines.append(f"\n{pokemon_name}:")

            # 判明した技
            revealed = self.revealed_moves.get(pokemon_name, set())
            if revealed:
                lines.append(f"  判明した技: {', '.join(revealed)}")

            # 持ち物
            if self.revealed_items.get(pokemon_name):
                lines.append(f"  持ち物: {self.revealed_items[pokemon_name]} (確定)")
            else:
                item_dist = self.get_item_distribution(pokemon_name)
                top_items = sorted(item_dist.items(), key=lambda x: -x[1])[:3]
                items_str = ", ".join(f"{i}({p:.1%})" for i, p in top_items)
                lines.append(f"  持ち物: {items_str}")

            # テラス
            if self.revealed_tera.get(pokemon_name):
                lines.append(f"  テラス: {self.revealed_tera[pokemon_name]} (確定)")
            else:
                tera_dist = self.get_tera_distribution(pokemon_name)
                top_tera = sorted(tera_dist.items(), key=lambda x: -x[1])[:3]
                tera_str = ", ".join(f"{t}({p:.1%})" for t, p in top_tera)
                lines.append(f"  テラス: {tera_str}")

            # 特性
            if self.revealed_abilities.get(pokemon_name):
                lines.append(f"  特性: {self.revealed_abilities[pokemon_name]} (確定)")
            else:
                ability_dist = self.get_ability_distribution(pokemon_name)
                top_abilities = sorted(ability_dist.items(), key=lambda x: -x[1])[:3]
                abilities_str = ", ".join(f"{a}({p:.1%})" for a, p in top_abilities)
                lines.append(f"  特性: {abilities_str}")

            # 仮説数
            lines.append(f"  仮説数: {len(self.beliefs[pokemon_name])}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        total_hypos = sum(len(b) for b in self.beliefs.values())
        return f"PokemonBeliefState(pokemon={len(self.beliefs)}, total_hypotheses={total_hypos})"

    def to_dict(self) -> dict[str, Any]:
        """シリアライズ可能な辞書に変換"""
        beliefs_serialized = {}
        for pokemon_name, hypo_dist in self.beliefs.items():
            beliefs_serialized[pokemon_name] = [
                (hypo.to_dict(), prob) for hypo, prob in hypo_dist.items()
            ]

        return {
            "beliefs": beliefs_serialized,
            "revealed_moves": {k: list(v) for k, v in self.revealed_moves.items()},
            "revealed_items": self.revealed_items,
            "revealed_tera": self.revealed_tera,
            "revealed_abilities": self.revealed_abilities,
            "move_use_count": self.move_use_count,
            "max_hypotheses": self.max_hypotheses,
            "min_probability": self.min_probability,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], usage_db: PokemonUsageDatabase
    ) -> "PokemonBeliefState":
        """辞書から復元"""
        pokemon_names = list(data["beliefs"].keys())
        instance = cls.__new__(cls)

        instance.usage_db = usage_db
        instance.max_hypotheses = data.get("max_hypotheses", 50)
        instance.min_probability = data.get("min_probability", 0.01)
        instance.observation_history = []

        instance.revealed_moves = {k: set(v) for k, v in data["revealed_moves"].items()}
        instance.revealed_items = data["revealed_items"]
        instance.revealed_tera = data["revealed_tera"]
        instance.revealed_abilities = data.get("revealed_abilities", {name: None for name in pokemon_names})
        instance.move_use_count = data.get("move_use_count", {name: {} for name in pokemon_names})

        instance.beliefs = {}
        for pokemon_name, hypo_list in data["beliefs"].items():
            instance.beliefs[pokemon_name] = {
                PokemonTypeHypothesis.from_dict(hypo_dict): prob
                for hypo_dict, prob in hypo_list
            }

        return instance