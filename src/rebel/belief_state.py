"""
ポケモンバトルにおける信念状態の管理

相手の隠された情報（技構成・持ち物・テラスタイプ等）に対する
確率分布を管理し、観測に基づいてベイズ更新を行う。
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase

from .ev_template import (
    EVSpread,
    EVSpreadType,
    estimate_ev_spread_type,
    get_ev_spread,
    get_ivs_for_spread_type,
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
    OUTSPED_UNEXPECTEDLY = auto()  # 予想外に先制 → スカーフ疑惑
    HIGH_DAMAGE_DEALT = auto()  # 高ダメージ → 火力アイテム疑惑
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

    1体のポケモンがどのような構成かの仮説。
    frozen=True でハッシュ可能にし、辞書のキーとして使用可能。
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
        """リストから作成（技はソートして正規化、EVは性格と種族値から推定）"""
        ev_type = estimate_ev_spread_type(nature, base_stats)
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
        from .ev_template import EV_TEMPLATES
        spread = EV_TEMPLATES.get(self.ev_spread_type, EV_TEMPLATES[EVSpreadType.UNKNOWN])
        return spread.to_list()

    def get_ivs(self) -> list[int]:
        """個体値をリストで取得 [H, A, B, C, D, S]"""
        return get_ivs_for_spread_type(self.ev_spread_type)


class PokemonBeliefState:
    """
    相手パーティ全体に対する信念状態

    各ポケモンについて、型仮説の確率分布を保持し、
    観測に基づいてベイズ更新を行う。

    計算量削減のため、確率の低い仮説は枝刈りする。
    """

    def __init__(
        self,
        opponent_pokemon_names: list[str],
        usage_db: PokemonUsageDatabase,
        max_hypotheses_per_pokemon: int = 50,
        min_probability: float = 0.01,
    ):
        """
        Args:
            opponent_pokemon_names: 相手の見せ合いで見えたポケモン名
            usage_db: 使用率データベース
            max_hypotheses_per_pokemon: ポケモンあたりの最大仮説数
            min_probability: 仮説の最小確率（これ以下は枝刈り）
        """
        self.usage_db = usage_db
        self.max_hypotheses = max_hypotheses_per_pokemon
        self.min_probability = min_probability

        # 各ポケモンの信念: {pokemon_name: {hypothesis: probability}}
        self.beliefs: dict[str, dict[PokemonTypeHypothesis, float]] = {}

        # 観測済み情報（確定情報）
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

        # 技使用回数の追跡（PP枯渇の推測に使用）
        # {pokemon_name: {move_name: use_count}}
        self.move_use_count: dict[str, dict[str, int]] = {
            name: {} for name in opponent_pokemon_names
        }

        # 観測履歴
        self.observation_history: list[Observation] = []

        # 初期信念を構築
        for name in opponent_pokemon_names:
            self.beliefs[name] = self._build_initial_belief(name)

    def _build_initial_belief(
        self, pokemon_name: str
    ) -> dict[PokemonTypeHypothesis, float]:
        """
        使用率データから初期信念を構築

        技・持ち物・テラス・性格・特性の組み合わせを
        独立と仮定してサンプリングし、上位N個を保持。
        EVは性格と種族値から自動推定する。
        """
        hypotheses: dict[PokemonTypeHypothesis, float] = {}

        # 種族値を取得（EV推定に使用）
        base_stats = self._get_base_stats(pokemon_name)

        # 使用率データを取得
        move_prior = self.usage_db.get_move_prior(pokemon_name, min_probability=0.05)
        item_prior = self.usage_db.get_item_prior(pokemon_name, min_probability=0.05)
        tera_prior = self.usage_db.get_tera_prior(pokemon_name, min_probability=0.05)
        nature_prior = self.usage_db.get_nature_prior(
            pokemon_name, min_probability=0.05
        )
        ability_prior = self.usage_db.get_ability_prior(
            pokemon_name, min_probability=0.01
        )

        # データがない場合はデフォルト
        if not move_prior:
            return hypotheses
        if not item_prior:
            item_prior = {"きあいのタスキ": 1.0}
        if not tera_prior:
            tera_prior = {"ノーマル": 1.0}
        if not nature_prior:
            # 種族値から物理/特殊を判断してデフォルト性格を決定
            nature_prior = self._get_default_nature_prior(base_stats)
        if not ability_prior:
            ability_prior = {"": 1.0}

        # 組み合わせをサンプリング（確率上位の組み合わせを生成）
        # 完全な直積は計算量が爆発するため、確率的にサンプリング
        num_samples = self.max_hypotheses * 10  # 多めにサンプリングして上位を取る

        for _ in range(num_samples):
            # 各要素を確率に従ってサンプリング
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
                base_stats=base_stats,  # 種族値を渡してEV推定
            )

            # 確率を計算（独立仮定）
            prob = self._calculate_hypothesis_probability(
                hypothesis, move_prior, item_prior, tera_prior, nature_prior, ability_prior
            )

            if hypothesis in hypotheses:
                hypotheses[hypothesis] = max(hypotheses[hypothesis], prob)
            else:
                hypotheses[hypothesis] = prob

        # 上位N個に絞り込み、正規化
        return self._prune_and_normalize(hypotheses)

    def _sample_moveset(self, move_prior: dict[str, float], num_moves: int) -> list[str]:
        """技構成をサンプリング"""
        moves = []
        available = dict(move_prior)

        for _ in range(min(num_moves, len(available))):
            if not available:
                break
            move = self._weighted_sample(available)
            moves.append(move)
            del available[move]
            # 再正規化
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
        """仮説の確率を計算（独立仮定）"""
        prob = 1.0

        # 技の確率（4技の積）
        for move in hypothesis.moves:
            prob *= move_prior.get(move, 0.01)

        # 持ち物
        prob *= item_prior.get(hypothesis.item, 0.01)

        # テラス
        prob *= tera_prior.get(hypothesis.tera_type, 0.01)

        # 性格
        prob *= nature_prior.get(hypothesis.nature, 0.1)

        # 特性
        if hypothesis.ability:
            prob *= ability_prior.get(hypothesis.ability, 0.1)

        return prob

    def _prune_and_normalize(
        self, hypotheses: dict[PokemonTypeHypothesis, float]
    ) -> dict[PokemonTypeHypothesis, float]:
        """仮説を枝刈りして正規化"""
        if not hypotheses:
            return {}

        # 確率順にソート
        sorted_hypos = sorted(hypotheses.items(), key=lambda x: -x[1])

        # 上位N個を取得
        top_hypos = sorted_hypos[: self.max_hypotheses]

        # 正規化
        total = sum(prob for _, prob in top_hypos)
        if total > 0:
            return {h: p / total for h, p in top_hypos}
        return {h: 1.0 / len(top_hypos) for h, _ in top_hypos}

    def _weighted_sample(self, probs: dict[str, float]) -> str:
        """重み付きサンプリング"""
        if not probs:
            return ""
        items = list(probs.keys())
        weights = list(probs.values())
        return random.choices(items, weights=weights, k=1)[0]

    def _get_base_stats(self, pokemon_name: str) -> Optional[list[int]]:
        """
        ポケモン名から種族値を取得

        Args:
            pokemon_name: ポケモン名

        Returns:
            種族値 [H, A, B, C, D, S]、見つからない場合は None
        """
        try:
            from src.pokemon_battle_sim.pokemon import Pokemon

            if pokemon_name in Pokemon.zukan:
                return Pokemon.zukan[pokemon_name].get("base")
        except (ImportError, AttributeError):
            pass
        return None

    def _get_default_nature_prior(
        self, base_stats: Optional[list[int]]
    ) -> dict[str, float]:
        """
        種族値からデフォルトの性格確率分布を取得

        Args:
            base_stats: 種族値 [H, A, B, C, D, S]

        Returns:
            性格の確率分布
        """
        if base_stats is None:
            # 種族値不明の場合は汎用的なデフォルト
            return {"いじっぱり": 0.3, "ようき": 0.3, "ひかえめ": 0.2, "おくびょう": 0.2}

        attack = base_stats[1]  # A
        sp_attack = base_stats[3]  # C
        speed = base_stats[5]  # S

        # 物理 vs 特殊を判断
        is_physical = attack >= sp_attack

        # 素早さが高いか（90以上をアタッカー向け素早さと判断）
        is_fast = speed >= 90

        if is_physical:
            if is_fast:
                # 物理アタッカー（素早さ重視）: ようき > いじっぱり
                return {"ようき": 0.6, "いじっぱり": 0.4}
            else:
                # 物理アタッカー（火力重視）: いじっぱり > ゆうかん
                return {"いじっぱり": 0.7, "ゆうかん": 0.2, "ようき": 0.1}
        else:
            if is_fast:
                # 特殊アタッカー（素早さ重視）: おくびょう > ひかえめ
                return {"おくびょう": 0.6, "ひかえめ": 0.4}
            else:
                # 特殊アタッカー（火力重視）: ひかえめ > れいせい
                return {"ひかえめ": 0.7, "れいせい": 0.2, "おくびょう": 0.1}

    # =========================================================================
    # 観測による更新
    # =========================================================================

    def update(self, observation: Observation) -> None:
        """
        観測に基づいて信念をベイズ更新

        Args:
            observation: 観測イベント
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
                # 技使用回数を更新
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

        # === 推測観測（確率更新） ===
        elif obs_type == ObservationType.OUTSPED_UNEXPECTEDLY:
            self._boost_item_probability(pokemon_name, "こだわりスカーフ", factor=3.0)

        elif obs_type == ObservationType.HIGH_DAMAGE_DEALT:
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

    def _confirm_item(self, pokemon_name: str, item: str) -> None:
        """持ち物を確定"""
        self.revealed_items[pokemon_name] = item
        current = self.beliefs[pokemon_name]
        filtered = {h: p for h, p in current.items() if h.item == item}
        if filtered:
            self.beliefs[pokemon_name] = self._prune_and_normalize(filtered)

    def _filter_to_items(self, pokemon_name: str, items: list[str]) -> None:
        """指定した持ち物のみに絞り込み"""
        current = self.beliefs[pokemon_name]
        filtered = {h: p for h, p in current.items() if h.item in items}
        if filtered:
            self.beliefs[pokemon_name] = self._prune_and_normalize(filtered)

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

    def _confirm_ability(self, pokemon_name: str, ability: str) -> None:
        """特性を確定"""
        self.revealed_abilities[pokemon_name] = ability
        current = self.beliefs[pokemon_name]
        filtered = {h: p for h, p in current.items() if h.ability == ability}
        if filtered:
            self.beliefs[pokemon_name] = self._prune_and_normalize(filtered)

    # =========================================================================
    # サンプリング
    # =========================================================================

    def sample_world(self) -> dict[str, PokemonTypeHypothesis]:
        """
        現在の信念から1つの「世界」（型の組み合わせ）をサンプリング

        Returns:
            {pokemon_name: hypothesis} の辞書
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

        Args:
            pokemon_name: ポケモン名
            move: 技名
            max_pp: 技の最大PP（デフォルト8、実際はポイントアップで増加可能）

        Returns:
            推定PP残量（0.0-1.0の比率）
        """
        use_count = self.get_move_use_count(pokemon_name, move)
        # PPを使い切った可能性を考慮
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
        # beliefs の変換: {pokemon_name: [(hypothesis_dict, probability), ...]}
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
        # まず空のポケモン名リストでインスタンスを作成
        pokemon_names = list(data["beliefs"].keys())
        instance = cls.__new__(cls)

        # 基本属性を設定
        instance.usage_db = usage_db
        instance.max_hypotheses = data.get("max_hypotheses", 50)
        instance.min_probability = data.get("min_probability", 0.01)
        instance.observation_history = []

        # revealed情報を復元
        instance.revealed_moves = {k: set(v) for k, v in data["revealed_moves"].items()}
        instance.revealed_items = data["revealed_items"]
        instance.revealed_tera = data["revealed_tera"]
        instance.revealed_abilities = data.get("revealed_abilities", {name: None for name in pokemon_names})
        instance.move_use_count = data.get("move_use_count", {name: {} for name in pokemon_names})

        # beliefs を復元
        instance.beliefs = {}
        for pokemon_name, hypo_list in data["beliefs"].items():
            instance.beliefs[pokemon_name] = {
                PokemonTypeHypothesis.from_dict(hypo_dict): prob
                for hypo_dict, prob in hypo_list
            }

        return instance
