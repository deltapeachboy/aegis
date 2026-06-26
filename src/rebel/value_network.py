"""
ReBeL Value Network (Version 5.3 - 特性・みがわり ＋ 天候交差シナジー特徴量統合版)

Public Belief State (PBS) から両プレイヤーの期待値を予測する
ニューラルネットワーク。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .belief_state import PokemonBeliefState, PokemonTypeHypothesis
from .move_effectiveness import MoveEffectivenessCalculator
from .public_state import PublicBeliefState, PublicGameState, PublicPokemonState


class ResidualBlock(nn.Module):
    """残差ブロック"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class PBSEncoder(nn.Module):
    """
    Public Belief State をテンソルにエンコードするモジュール

    PBS の各コンポーネントを適切にエンコードし、
    Value Network の入力となる固定長ベクトルを生成する。
    """

    def __init__(
        self,
        pokemon_name_dim: int = 32,
        move_dim: int = 16,
        item_dim: int = 16,
        tera_dim: int = 8,
        max_pokemon_vocab: int = 500,
        max_move_vocab: int = 500,
        max_item_vocab: int = 300,
        num_types: int = 19,  # 18タイプ + ステラ
        num_belief_samples: int = 10,  # 信念状態で考慮するサンプル数
        pokemon_to_id: Optional[dict[str, int]] = None,
        move_to_id: Optional[dict[str, int]] = None,
        item_to_id: Optional[dict[str, int]] = None,
        use_move_effectiveness: bool = True,  # チェックポイント互換性用
    ):
        super().__init__()

        self.pokemon_name_dim = pokemon_name_dim
        self.move_dim = move_dim
        self.item_dim = item_dim
        self.tera_dim = tera_dim
        self.num_belief_samples = num_belief_samples
        self.use_move_effectiveness = use_move_effectiveness

        # 埋め込み層
        self.pokemon_embed = nn.Embedding(
            max_pokemon_vocab, pokemon_name_dim, padding_idx=0
        )
        self.move_embed = nn.Embedding(max_move_vocab, move_dim, padding_idx=0)
        self.item_embed = nn.Embedding(max_item_vocab, item_dim, padding_idx=0)
        self.type_embed = nn.Embedding(num_types + 1, tera_dim, padding_idx=0)

        # ID 変換辞書（動的に構築、または事前設定）
        self.pokemon_to_id: dict[str, int] = pokemon_to_id or {}
        self.move_to_id: dict[str, int] = move_to_id or {}
        self.item_to_id: dict[str, int] = item_to_id or {}
        self.type_to_id: dict[str, int] = self._default_type_to_id()

        # 次のIDは既存辞書の最大値+1から開始
        self._next_pokemon_id = max(self.pokemon_to_id.values(), default=0) + 1
        self._next_move_id = max(self.move_to_id.values(), default=0) + 1
        self._next_item_id = max(self.item_to_id.values(), default=0) + 1

        # 技有効性計算用
        self.move_effectiveness_calculator = (
            MoveEffectivenessCalculator() if use_move_effectiveness else None
        )

        # 出力次元を計算
        self._calc_output_dim()

    def _default_type_to_id(self) -> dict[str, int]:
        types = [
            "ノーマル", "ほのお", "みず", "でんき", "くさ", "こおり", "かくとう", "どく", "じめん", "ひこう",
            "エスパー", "むし", "いわ", "ゴースト", "ドラゴン", "あく", "はがね", "フェアリー", "ステラ",
        ]
        return {t: i + 1 for i, t in enumerate(types)}

    def _get_pokemon_id(self, name: str) -> int:
        if not name:
            return 0
        if name not in self.pokemon_to_id:
            self.pokemon_to_id[name] = self._next_pokemon_id
            self._next_pokemon_id += 1
        return self.pokemon_to_id[name]

    def _get_move_id(self, name: str) -> int:
        if not name:
            return 0
        if name not in self.move_to_id:
            self.move_to_id[name] = self._next_move_id
            self._next_move_id += 1
        return self.move_to_id[name]

    def _get_item_id(self, name: str) -> int:
        if not name:
            return 0
        if name not in self.item_to_id:
            self.item_to_id[name] = self._next_item_id
            self._next_item_id += 1
        return self.item_to_id[name]

    def _get_type_id(self, name: str) -> int:
        return self.type_to_id.get(name, 0)

    def _calc_output_dim(self) -> None:
        """出力次元を計算"""
        # 拡張要素: ばけのかわ稼働フラグ (1次元) + みがわりHP比率 (1次元) = 計 2次元
        extended_status_dim = 2

        # 自分の場のポケモン
        my_active_dim = (
            self.pokemon_name_dim  # name
            + 1  # hp_ratio
            + 7  # ailment (one-hot)
            + 2  # ailment details (bad_poison_counter, sleep_counter)
            + 8  # rank
            + self.tera_dim * 2  # types
            + self.item_dim  # item
            + self.move_dim * 4  # moves
            + 4  # pp ratios for 4 moves
            + 1  # terastallized
            + self.tera_dim  # tera_type
            + extended_status_dim  # [ばけのかわ残存, みがわりHP割合]
        )

        # 自分の控え (2体)
        my_bench_dim = my_active_dim * 2

        # 相手の場のポケモン（持ち物・技は信念で表現）
        opp_active_dim = (
            self.pokemon_name_dim  # name
            + 1  # hp_ratio
            + 7  # ailment
            + 2  # ailment details (bad_poison_counter, sleep_counter)
            + 8  # rank
            + self.tera_dim * 2  # types
            + 1  # terastallized
            + self.tera_dim  # tera_type (if used)
            + extended_status_dim  # [ばけのかわ残存, みがわりHP割合]
        )

        # 相手の控え (2体、簡略化)
        opp_bench_dim = (self.pokemon_name_dim + 1) * 2  # name + hp_ratio

        # 場の状態 (天候、フィールド、壁、おいかぜ、設置技)
        field_dim = 28

        # 信念状態（相手3体分）
        belief_dim = 3 * (
            10  # 上位10持ち物の確率
            + 10  # 上位10テラスタイプの確率
            + 10  # 上位10技の確率
        )

        # 戦略エンコーディング
        strategy_dim = 20  # 上位行動の確率

        # テラスタル可否
        tera_flags_dim = 2

        # 技有効性情報
        move_effectiveness_dim = (4 + 4 + 1 + 1) if self.use_move_effectiveness else 0

        # 🌟 拡張特徴量: 天候シナジー（交差特徴量） (自分側4天候 + 相手側4天候 = 計 8次元)
        weather_synergy_dim = 8

        self.output_dim = (
            my_active_dim
            + my_bench_dim
            + opp_active_dim
            + opp_bench_dim
            + field_dim
            + belief_dim
            + strategy_dim
            + tera_flags_dim
            + move_effectiveness_dim
            + weather_synergy_dim  # 🌟 拡張
        )

    def get_output_dim(self) -> int:
        return self.output_dim

    def encode_my_pokemon(
        self, pokemon: "PokemonState", device: torch.device
    ) -> torch.Tensor:
        """自分のポケモンをエンコード（完全情報 ＋ 状態変化拡張）"""
        features = []

        pokemon_id = torch.tensor([self._get_pokemon_id(pokemon.name)], device=device)
        features.append(self.pokemon_embed(pokemon_id).squeeze(0))
        features.append(torch.tensor([pokemon.hp_ratio], device=device))

        ailment_map = {"": 0, "どく": 1, "もうどく": 2, "やけど": 3, "まひ": 4, "ねむり": 5, "こおり": 6}
        ailment_idx = ailment_map.get(pokemon.ailment, 0)
        ailment = F.one_hot(torch.tensor(ailment_idx, device=device), num_classes=7)
        features.append(ailment.float())

        bad_poison_counter = getattr(pokemon, "bad_poison_counter", 0) / 16.0
        sleep_counter = getattr(pokemon, "sleep_counter", 0) / 3.0
        features.append(torch.tensor([bad_poison_counter, sleep_counter], device=device))

        rank = torch.tensor(pokemon.rank[:8], device=device, dtype=torch.float) / 6.0
        features.append(rank)

        type_ids = [self._get_type_id(t) for t in pokemon.types[:2]]
        while len(type_ids) < 2:
            type_ids.append(0)
        for tid in type_ids:
            features.append(self.type_embed(torch.tensor([tid], device=device)).squeeze(0))

        item_id = torch.tensor([self._get_item_id(pokemon.item)], device=device)
        features.append(self.item_embed(item_id).squeeze(0))

        for i in range(4):
            move = pokemon.moves[i] if i < len(pokemon.moves) else ""
            move_id = torch.tensor([self._get_move_id(move)], device=device)
            features.append(self.move_embed(move_id).squeeze(0))

        pp_list = getattr(pokemon, "pp", []) or []
        pp_ratios = []
        for i in range(4):
            if i < len(pp_list) and i < len(pokemon.moves):
                pp_ratios.append(min(pp_list[i] / 32.0, 1.0))
            else:
                pp_ratios.append(1.0)
        features.append(torch.tensor(pp_ratios, device=device, dtype=torch.float))

        features.append(torch.tensor([1.0 if pokemon.terastallized else 0.0], device=device))

        tera_id = torch.tensor([self._get_type_id(pokemon.tera_type)], device=device)
        features.append(self.type_embed(tera_id).squeeze(0))

        # 拡張特徴量: ばけのかわ（Disguise）稼働フラグ
        ability_name = getattr(pokemon, "ability", "")
        is_disguise_active = 1.0 if ability_name == "ばけのかわ+" else 0.0

        # 拡張特徴量: みがわり（Substitute）の残りHP比率
        sub_hp = getattr(pokemon, "sub_hp", 0)
        max_hp = 100.0
        if hasattr(pokemon, "status") and pokemon.status and len(pokemon.status) > 0:
            max_hp = pokemon.status[0] if pokemon.status[0] > 0 else 100.0
        sub_hp_ratio = min(max(0.0, sub_hp / max_hp), 1.0)

        features.append(torch.tensor([is_disguise_active, sub_hp_ratio], device=device, dtype=torch.float))

        return torch.cat(features)

    def encode_opp_pokemon(
        self, pokemon: "PublicPokemonState", device: torch.device
    ) -> torch.Tensor:
        """相手のポケモンをエンコード（公開情報 ＋ 状態変化拡張）"""
        features = []

        pokemon_id = torch.tensor([self._get_pokemon_id(pokemon.name)], device=device)
        features.append(self.pokemon_embed(pokemon_id).squeeze(0))
        features.append(torch.tensor([pokemon.hp_ratio], device=device))

        ailment_map = {"": 0, "どく": 1, "もうどく": 2, "やけど": 3, "まひ": 4, "ねむり": 5, "こおり": 6}
        ailment_idx = ailment_map.get(pokemon.ailment, 0)
        ailment = F.one_hot(torch.tensor(ailment_idx, device=device), num_classes=7)
        features.append(ailment.float())

        bad_poison_counter = getattr(pokemon, "bad_poison_counter", 0) / 16.0
        sleep_counter = getattr(pokemon, "sleep_counter", 0) / 3.0
        features.append(torch.tensor([bad_poison_counter, sleep_counter], device=device))

        rank = torch.tensor(pokemon.rank[:8], device=device, dtype=torch.float) / 6.0
        features.append(rank)

        type_ids = [self._get_type_id(t) for t in pokemon.types[:2]]
        while len(type_ids) < 2:
            type_ids.append(0)
        for tid in type_ids:
            features.append(self.type_embed(torch.tensor([tid], device=device)).squeeze(0))

        features.append(torch.tensor([1.0 if pokemon.terastallized else 0.0], device=device))

        tera_id = torch.tensor([self._get_type_id(pokemon.tera_type)], device=device)
        features.append(self.type_embed(tera_id).squeeze(0))

        # 拡張特徴量: 相手のばけのかわ（Disguise）稼働フラグ
        opp_ability = getattr(pokemon, "revealed_ability", "")
        if not opp_ability:
            opp_ability = getattr(pokemon, "ability", "") or ""
        is_disguise_active = 1.0 if opp_ability == "ばけのかわ+" else 0.0

        # 拡張特徴量: 相手のみがわり（Substitute）の残りHP比率
        sub_hp = getattr(pokemon, "sub_hp", 0)
        max_hp = 100.0
        if hasattr(pokemon, "status") and pokemon.status and len(pokemon.status) > 0:
            max_hp = pokemon.status[0] if pokemon.status[0] > 0 else 100.0
        sub_hp_ratio = min(max(0.0, sub_hp / max_hp), 1.0)

        features.append(torch.tensor([is_disguise_active, sub_hp_ratio], device=device, dtype=torch.float))

        return torch.cat(features)

    def encode_field(
        self, field: "FieldCondition", device: torch.device
    ) -> torch.Tensor:
        """場の状態をエンコード（28次元）"""
        features = [
            # 天候 (4)
            field.sunny / 5.0,
            field.rainy / 5.0,
            field.snow / 5.0,
            field.sandstorm / 5.0,
            # フィールド (4)
            field.electric_field / 5.0,
            field.grass_field / 5.0,
            field.psychic_field / 5.0,
            field.mist_field / 5.0,
            # その他場の効果 (2)
            field.gravity / 5.0,
            field.trick_room / 5.0,
            # 壁 (4)
            field.reflector[0] / 5.0,
            field.light_screen[0] / 5.0,
            field.reflector[1] / 5.0,
            field.light_screen[1] / 5.0,
            # おいかぜ (2)
            field.tailwind[0] / 4.0,
            field.tailwind[1] / 4.0,
            # しんぴのまもり・しろいきり (4)
            field.safeguard[0] / 5.0 if hasattr(field, "safeguard") else 0.0,
            field.safeguard[1] / 5.0 if hasattr(field, "safeguard") else 0.0,
            field.mist[0] / 5.0 if hasattr(field, "mist") else 0.0,
            field.mist[1] / 5.0 if hasattr(field, "mist") else 0.0,
            # 設置技 (8)
            field.spikes[0] / 3.0,
            field.toxic_spikes[0] / 2.0,
            float(field.stealth_rock[0]),
            float(field.sticky_web[0]),
            field.spikes[1] / 3.0,
            field.toxic_spikes[1] / 2.0,
            float(field.stealth_rock[1]),
            float(field.sticky_web[1]),
        ]
        return torch.tensor(features, device=device, dtype=torch.float)

    def encode_belief(
        self, belief: PokemonBeliefState, device: torch.device
    ) -> torch.Tensor:
        """信念状態をエンコード"""
        features = []

        for pokemon_name in list(belief.beliefs.keys())[:3]:
            item_dist = belief.get_item_distribution(pokemon_name)
            top_items = sorted(item_dist.items(), key=lambda x: -x[1])[:10]
            for i in range(10):
                if i < len(top_items):
                    features.append(top_items[i][1])
                else:
                    features.append(0.0)

            tera_dist = belief.get_tera_distribution(pokemon_name)
            top_tera = sorted(tera_dist.items(), key=lambda x: -x[1])[:10]
            for i in range(10):
                if i < len(top_tera):
                    features.append(top_tera[i][1])
                else:
                    features.append(0.0)

            hypo_belief = belief.beliefs.get(pokemon_name, {})
            move_probs: dict[str, float] = {}
            for hypo, prob in hypo_belief.items():
                for move in hypo.moves:
                    move_probs[move] = move_probs.get(move, 0.0) + prob
            top_moves = sorted(move_probs.items(), key=lambda x: -x[1])[:10]
            for i in range(10):
                if i < len(top_moves):
                    features.append(top_moves[i][1])
                else:
                    features.append(0.0)

        while len(features) < 3 * 30:
            features.append(0.0)

        return torch.tensor(features, device=device, dtype=torch.float)

    def encode_strategy(
        self, strategy: dict[int, float], device: torch.device
    ) -> torch.Tensor:
        """戦略をエンコード"""
        sorted_actions = sorted(strategy.items())[:20]
        features = []
        for i in range(20):
            if i < len(sorted_actions):
                features.append(sorted_actions[i][1])
            else:
                features.append(0.0)
        return torch.tensor(features, device=device, dtype=torch.float)

    def encode_move_effectiveness(
        self,
        my_pokemon: "PokemonState",
        opp_pokemon: "PublicPokemonState",
        gravity: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """
        技の有効性をエンコード
        """
        features = []

        my_moves = my_pokemon.moves if my_pokemon.moves else []
        opp_types = opp_pokemon.types if opp_pokemon.types else []
        opp_ability = opp_pokemon.revealed_ability
        opp_item = opp_pokemon.revealed_item

        my_effective_flags = []
        my_effectiveness_values = []
        my_has_effective = False

        for i in range(4):
            if i < len(my_moves) and my_moves[i]:
                result = self.move_effectiveness_calculator.check_move_effectiveness(
                    my_moves[i],
                    opp_types,
                    opp_ability,
                    opp_item,
                    gravity,
                )
                my_effective_flags.append(1.0 if result.is_effective else 0.0)
                normalized = min(result.effectiveness / 4.0, 1.0)
                my_effectiveness_values.append(normalized)
                if result.is_effective:
                    my_has_effective = True
            else:
                my_effective_flags.append(0.0)
                my_effectiveness_values.append(0.0)

        features.extend(my_effective_flags)
        features.extend(my_effectiveness_values)
        features.append(1.0 if my_has_effective else 0.0)

        my_types = my_pokemon.types if my_pokemon.types else []
        my_ability = my_pokemon.ability if hasattr(my_pokemon, "ability") else None
        my_item = my_pokemon.item if hasattr(my_pokemon, "item") else None

        opp_can_hit = 1.0
        if my_item == "ふうせん" and not gravity:
            if opp_types and opp_types[0] == "じめん":
                opp_can_hit = 0.5

        features.append(opp_can_hit)

        return torch.tensor(features, device=device, dtype=torch.float)

    # =========================================================================
    # 🌟 【アプローチ2】天候 ✕ 特性・属性の交差特徴量（天候シナジー）エンコーダ (8次元)
    # =========================================================================
    def encode_weather_synergy(
        self,
        my_poke: "PokemonState",
        opp_poke: "PublicPokemonState",
        field: "FieldCondition",
        device: torch.device,
    ) -> torch.Tensor:
        """
        現在フィールドに出ている双方のポケモンについて、雨・晴れ・砂・雪天候とのシナジー（交差）を明示的に判定します。

        返却テンソル（8次元）:
        [my_rain, my_sun, my_sand, my_snow, opp_rain, opp_sun, opp_sand, opp_snow]
        """
        synergy_features = []

        for pokemon in [my_poke, opp_poke]:
            # 安全ガード（瀕死または不在の場合はシナジーなし）
            hp_ratio = getattr(pokemon, "hp_ratio", 0.0)
            if pokemon is None or hp_ratio <= 1e-5:
                synergy_features.extend([0.0, 0.0, 0.0, 0.0])
                continue

            types = getattr(pokemon, "types", []) or []
            # 公開特性の安全取得
            ability = getattr(pokemon, "revealed_ability", "")
            if not ability:
                ability = getattr(pokemon, "ability", "") or ""

            # 1. 雨シナジー（すいすい / みずタイプ）
            is_rain_active = 1.0 if field.rainy > 0 else 0.0
            has_rain_compatibility = 1.0 if ("みず" in types or "すいすい" in ability) else 0.0
            synergy_features.append(is_rain_active * has_rain_compatibility)

            # 2. 晴れシナジー（こだいかっせい / ようりょくそ / ほのおタイプ）
            is_sun_active = 1.0 if field.sunny > 0 else 0.0
            has_sun_compatibility = 1.0 if ("ほのお" in types or ability in ["ようりょくそ", "こだいかっせい"]) else 0.0
            synergy_features.append(is_sun_active * has_sun_compatibility)

            # 3. 砂嵐シナジー（すなかき / すなのちから / すなのよろい / 岩・地面・鋼タイプ）
            is_sand_active = 1.0 if field.sandstorm > 0 else 0.0
            has_sand_compatibility = 1.0 if (
                any(t in types for t in ["いわ", "じめん", "はがね"]) or
                any(a in ability for a in ["すなかき", "すなのちから", "すながくれ"])
            ) else 0.0
            synergy_features.append(is_sand_active * has_sand_compatibility)

            # 4. 雪シナジー（ゆきかき / こおりタイプ）
            is_snow_active = 1.0 if field.snow > 0 else 0.0
            has_snow_compatibility = 1.0 if ("こおり" in types or "ゆきかき" in ability) else 0.0
            synergy_features.append(is_snow_active * has_snow_compatibility)

        return torch.tensor(synergy_features, device=device, dtype=torch.float)

    def forward(self, pbs: PublicBeliefState) -> torch.Tensor:
        """PBS を固定長ベクトルにエンコード"""
        device = next(self.parameters()).device
        features = []

        ps = pbs.public_state

        # 自分の場のポケモン
        features.append(self.encode_my_pokemon(ps.my_pokemon, device))

        # 自分の控え（2体、パディング）
        for i in range(2):
            if i < len(ps.my_bench):
                features.append(self.encode_my_pokemon(ps.my_bench[i], device))
            else:
                features.append(torch.zeros(features[0].shape[0], device=device))

        # 相手の場のポケモン
        features.append(self.encode_opp_pokemon(ps.opp_pokemon, device))

        # 相手の控え（簡略化: 名前 + HP比率のみ）
        for i in range(2):
            if i < len(ps.opp_bench):
                bench = ps.opp_bench[i]
                pokemon_id = torch.tensor(
                    [self._get_pokemon_id(bench.name)], device=device
                )
                features.append(self.pokemon_embed(pokemon_id).squeeze(0))
                features.append(torch.tensor([bench.hp_ratio], device=device))
            else:
                features.append(torch.zeros(self.pokemon_name_dim, device=device))
                features.append(torch.zeros(1, device=device))

        # 場の状態
        features.append(self.encode_field(ps.field, device))

        # 信念状態
        features.append(self.encode_belief(pbs.belief, device))

        # 戦略
        features.append(self.encode_strategy(pbs.my_strategy, device))

        # テラスタル可否
        features.append(
            torch.tensor(
                [float(ps.my_tera_available), float(ps.opp_tera_available)],
                device=device,
            )
        )

        # 技有効性情報
        if self.use_move_effectiveness:
            gravity = ps.field.gravity > 0 if hasattr(ps.field, "gravity") else False
            features.append(
                self.encode_move_effectiveness(
                    ps.my_pokemon, ps.opp_pokemon, gravity, device
                )
            )

        # 🌟 【アプローチ2】天候交差（シナジー）特徴量の結合 (8次元)
        features.append(
            self.encode_weather_synergy(
                ps.my_pokemon, ps.opp_pokemon, ps.field, device
            )
        )

        return torch.cat(features)


class ReBeLValueNetwork(nn.Module):
    """
    ReBeL Value Network

    PBS を入力として、両プレイヤーの期待値を予測する。
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        dropout: float = 0.1,
        encoder_config: Optional[dict] = None,
        use_move_effectiveness: bool = True,
    ):
        super().__init__()

        # PBS エンコーダー
        encoder_config = encoder_config or {}
        encoder_config["use_move_effectiveness"] = use_move_effectiveness
        self.encoder = PBSEncoder(**encoder_config)
        input_dim = self.encoder.get_output_dim()

        self.hidden_dim = hidden_dim

        # 入力層
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 残差ブロック
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        # Value Head（両プレイヤーの期待値を出力）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # [my_value, opp_value]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    def forward(self, pbs: PublicBeliefState) -> tuple[float, float]:
        """
        PBS から両プレイヤーの期待値を予測
        """
        self.eval()
        with torch.no_grad():
            x = self.encoder(pbs).unsqueeze(0)  # [1, input_dim]
            h = self.input_layer(x)
            for block in self.res_blocks:
                h = block(h)
            values = self.value_head(h).squeeze(0)  # [2]
            values = torch.sigmoid(values)
        return values[0].item(), values[1].item()

    def forward_batch(
        self, pbs_list: list[PublicBeliefState]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        バッチ処理
        """
        encoded = torch.stack([self.encoder(pbs) for pbs in pbs_list])
        h = self.input_layer(encoded)
        for block in self.res_blocks:
            h = block(h)
        values = self.value_head(h)  # [batch, 2]
        values = torch.sigmoid(values)
        return values[:, 0], values[:, 1]

    def predict(self, pbs: PublicBeliefState) -> tuple[float, float]:
        """推論用（勾配なし）"""
        self.eval()
        with torch.no_grad():
            return self.forward(pbs)


class ReBeLPolicyValueNetwork(nn.Module):
    """
    ReBeL Policy-Value Network
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        num_actions: int = 10,
        dropout: float = 0.1,
        encoder_config: Optional[dict] = None,
    ):
        super().__init__()

        # PBS エンコーダー
        encoder_config = encoder_config or {}
        self.encoder = PBSEncoder(**encoder_config)
        input_dim = self.encoder.get_output_dim()

        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # 入力層
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 残差ブロック
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    def forward(
        self,
        pbs: PublicBeliefState,
        action_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        PBS から Policy と Value を予測
        """
        x = self.encoder(pbs).unsqueeze(0)
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)

        # Policy Head
        policy_logits = self.policy_head(h).squeeze(0)
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, float("-inf"))
        policy = F.softmax(policy_logits, dim=-1)

        # Value Head
        values = torch.sigmoid(self.value_head(h).squeeze(0))

        return policy, values