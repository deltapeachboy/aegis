"""
技の有効性計算モジュール

観測情報および信念確率から各技が相手に有効かどうかを判定する。
これにより、Value Network が「詰み」状態を正しく認識できるようになる。

判定する無効化パターン:
1. タイプ相性による無効（ノーマル→ゴースト、じめん→ひこう等）
2. 特性による無効化（おうごんのからだ、ふゆう、ちくでん等）
3. 持ち物による無効化（ふうせん→じめん技無効）
4. 場の状態による無効化（重力解除で浮いている等）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

# 🌟 赤線警告対策として、仮想空間ではなく直接 pokepy パッケージからインポートします
from pokepy.pokemon import Pokemon


@dataclass
class MoveEffectivenessResult:
    """技の有効性判定結果"""

    move_name: str
    is_effective: bool  # 技が有効かどうか
    effectiveness: float  # タイプ相性倍率（0.0 = 無効, 0.5 = 半減, 1.0 = 等倍, 2.0 = 抜群）
    reason: str  # 無効の理由（デバッグ用）


class MoveEffectivenessCalculator:
    """
    技の有効性を計算するクラス

    観測済み情報および信念状態の確率から、各技が相手に有効かを論理的・確率的に判定する。
    """

    # おうごんのからだで無効化される変化技を持つ特性
    GOLDEN_BODY_ABILITY = "おうごんのからだ"

    # 地面技を無効化する特性
    GROUND_IMMUNE_ABILITIES = {"ふゆう"}

    # 電気技を無効化する特性
    ELECTRIC_IMMUNE_ABILITIES = {"ちくでん", "ひらいしん", "でんきエンジン"}

    # 炎技を無効化する特性
    FIRE_IMMUNE_ABILITIES = {"もらいび"}

    # 草技を無効化する特性
    GRASS_IMMUNE_ABILITIES = {"そうしょく"}

    # 水技を無効化する特性
    WATER_IMMUNE_ABILITIES = {"よびみず", "ちょすい", "かんそうはだ"}

    # 地面技を無効化する持ち物
    GROUND_IMMUNE_ITEMS = {"ふうせん"}

    # タイプIDマッピング
    TYPE_IDS = {
        "ノーマル": 0,
        "ほのお": 1,
        "みず": 2,
        "でんき": 3,
        "くさ": 4,
        "こおり": 5,
        "かくとう": 6,
        "どく": 7,
        "じめん": 8,
        "ひこう": 9,
        "エスパー": 10,
        "むし": 11,
        "いわ": 12,
        "ゴースト": 13,
        "ドラゴン": 14,
        "あく": 15,
        "はがね": 16,
        "フェアリー": 17,
        "ステラ": 18,
    }

    def __init__(self):
        # Pokemon.init() が呼ばれていることを前提
        pass

    def calculate_type_effectiveness(
        self,
        move_type: str,
        defender_types: list[str],
    ) -> float:
        """
        タイプ相性を計算

        Args:
            move_type: 技のタイプ
            defender_types: 防御側のタイプリスト

        Returns:
            タイプ相性倍率
        """
        if not move_type or move_type not in self.TYPE_IDS:
            return 1.0

        atk_type_id = self.TYPE_IDS[move_type]
        effectiveness = 1.0

        for def_type in defender_types:
            if def_type not in self.TYPE_IDS:
                continue
            def_type_id = self.TYPE_IDS[def_type]
            effectiveness *= Pokemon.type_corrections[atk_type_id][def_type_id]

        return effectiveness

    def is_status_move(self, move_name: str) -> bool:
        """技が変化技かどうか判定"""
        if move_name not in Pokemon.all_moves:
            return False
        move_data = Pokemon.all_moves[move_name]
        return "sta" in move_data.get("class", "")

    def get_move_type(self, move_name: str) -> Optional[str]:
        """技のタイプを取得"""
        if move_name not in Pokemon.all_moves:
            return None
        return Pokemon.all_moves[move_name].get("type")

    def check_move_effectiveness(
        self,
        move_name: str,
        defender_types: list[str],
        defender_ability: Optional[str] = None,
        defender_item: Optional[str] = None,
        gravity: bool = False,
        # 🌟 確率補完用の追加引数
        belief_state: Optional[Any] = None,
        pokemon_name: Optional[str] = None,
    ) -> MoveEffectivenessResult:
        """
        技の有効性を判定（未確定情報は信念状態から確率的に補完）

        Args:
            move_name: 技名
            defender_types: 防御側のタイプリスト
            defender_ability: 防御側の特性（観測済みの場合）
            defender_item: 防御側の持ち物（観測済みの場合）
            gravity: 重力状態かどうか
            belief_state: 現在の信念状態
            pokemon_name: 相手のポケモン名

        Returns:
            MoveEffectivenessResult
        """
        # 技データを取得
        if move_name not in Pokemon.all_moves:
            return MoveEffectivenessResult(
                move_name=move_name,
                is_effective=True,  # 不明な技は有効と仮定
                effectiveness=1.0,
                reason="unknown_move",
            )

        move_data = Pokemon.all_moves[move_name]
        move_type = move_data.get("type")
        move_class = move_data.get("class", "")

        # 🌟 未観測の特性を信念状態の周辺確率から補完
        active_ability = defender_ability
        if active_ability is None and belief_state is not None and pokemon_name:
            ability_dist = belief_state.get_ability_distribution(pokemon_name)
            if ability_dist:
                top_ability, prob = max(ability_dist.items(), key=lambda x: x[1])
                if prob >= 0.70:  # 確信度が70%以上なら適用
                    active_ability = top_ability

        # 🌟 未観測の持ち物を信念状態の周辺確率から補完
        active_item = defender_item
        if active_item is None and belief_state is not None and pokemon_name:
            item_dist = belief_state.get_item_distribution(pokemon_name)
            if item_dist:
                top_item, prob = max(item_dist.items(), key=lambda x: x[1])
                if prob >= 0.70:  # 確信度が70%以上なら適用
                    active_item = top_item

        # 1. おうごんのからだによる変化技無効
        if active_ability == self.GOLDEN_BODY_ABILITY and "sta" in move_class:
            return MoveEffectivenessResult(
                move_name=move_name,
                is_effective=False,
                effectiveness=0.0,
                reason="golden_body_blocks_status",
            )

        # 2. タイプ相性を計算
        effectiveness = self.calculate_type_effectiveness(move_type, defender_types)

        # タイプ相性で無効（0倍）の場合
        if effectiveness == 0.0:
            return MoveEffectivenessResult(
                move_name=move_name,
                is_effective=False,
                effectiveness=0.0,
                reason="type_immunity",
            )

        # 3. 地面技に対する無効化チェック
        if move_type == "じめん":
            # ふうせん（重力で無効化される）
            if active_item in self.GROUND_IMMUNE_ITEMS and not gravity:
                return MoveEffectivenessResult(
                    move_name=move_name,
                    is_effective=False,
                    effectiveness=0.0,
                    reason="balloon_blocks_ground",
                )

            # ふゆう特性（重力で無効化される）
            if active_ability in self.GROUND_IMMUNE_ABILITIES and not gravity:
                return MoveEffectivenessResult(
                    move_name=move_name,
                    is_effective=False,
                    effectiveness=0.0,
                    reason="levitate_blocks_ground",
                )

            # ひこうタイプ（重力で無効化される）
            if "ひこう" in defender_types and not gravity:
                return MoveEffectivenessResult(
                    move_name=move_name,
                    is_effective=False,
                    effectiveness=0.0,
                    reason="flying_type_blocks_ground",
                )

        # 4. 電気技に対する無効化チェック
        if move_type == "でんき" and active_ability in self.ELECTRIC_IMMUNE_ABILITIES:
            return MoveEffectivenessResult(
                move_name=move_name,
                is_effective=False,
                effectiveness=0.0,
                reason="ability_blocks_electric",
            )

        # 5. 炎技に対する無効化チェック
        if move_type == "ほのお" and active_ability in self.FIRE_IMMUNE_ABILITIES:
            return MoveEffectivenessResult(
                move_name=move_name,
                is_effective=False,
                effectiveness=0.0,
                reason="ability_blocks_fire",
            )

        # 6. 草技に対する無効化チェック
        if move_type == "くさ" and active_ability in self.GRASS_IMMUNE_ABILITIES:
            return MoveEffectivenessResult(
                move_name=move_name,
                is_effective=False,
                effectiveness=0.0,
                reason="ability_blocks_grass",
            )

        # 7. 水技に対する無効化チェック
        if move_type == "みず" and active_ability in self.WATER_IMMUNE_ABILITIES:
            return MoveEffectivenessResult(
                move_name=move_name,
                is_effective=False,
                effectiveness=0.0,
                reason="ability_blocks_water",
            )

        # 有効
        return MoveEffectivenessResult(
            move_name=move_name,
            is_effective=True,
            effectiveness=effectiveness,
            reason="effective",
        )

    def check_all_moves_effectiveness(
        self,
        moves: list[str],
        defender_types: list[str],
        defender_ability: Optional[str] = None,
        defender_item: Optional[str] = None,
        gravity: bool = False,
        belief_state: Optional[Any] = None,
        pokemon_name: Optional[str] = None,
    ) -> list[MoveEffectivenessResult]:
        """
        複数の技の有効性を一括判定
        """
        return [
            self.check_move_effectiveness(
                move, defender_types, defender_ability, defender_item, gravity, belief_state, pokemon_name
            )
            for move in moves
        ]

    def has_effective_move(
        self,
        moves: list[str],
        defender_types: list[str],
        defender_ability: Optional[str] = None,
        defender_item: Optional[str] = None,
        gravity: bool = False,
        belief_state: Optional[Any] = None,
        pokemon_name: Optional[str] = None,
    ) -> bool:
        """
        有効な技が1つでもあるかどうか判定
        """
        results = self.check_all_moves_effectiveness(
            moves, defender_types, defender_ability, defender_item, gravity, belief_state, pokemon_name
        )
        return any(r.is_effective for r in results)

    def count_effective_moves(
        self,
        moves: list[str],
        defender_types: list[str],
        defender_ability: Optional[str] = None,
        defender_item: Optional[str] = None,
        gravity: bool = False,
        belief_state: Optional[Any] = None,
        pokemon_name: Optional[str] = None,
    ) -> int:
        """
        有効な技の数をカウント
        """
        results = self.check_all_moves_effectiveness(
            moves, defender_types, defender_ability, defender_item, gravity, belief_state, pokemon_name
        )
        return sum(1 for r in results if r.is_effective)


def should_consider_surrender(
    attacker_moves: list[str],
    defender_types: list[str],
    defender_ability: str | None = None,
    defender_item: str | None = None,
    gravity: bool = False,
    can_switch: bool = False,
    belief_state: Optional[Any] = None,
    pokemon_name: Optional[str] = None,
) -> bool:
    """
    降参を検討すべき状況かどうかを判定
    """
    # 交代可能なら降参は不要
    if can_switch:
        return False

    calculator = MoveEffectivenessCalculator()
    return not calculator.has_effective_move(
        attacker_moves,
        defender_types,
        defender_ability,
        defender_item,
        gravity,
        belief_state,
        pokemon_name,
    )


def encode_move_effectiveness_flags(
    attacker_moves: list[str],
    defender_types: list[str],
    defender_ability: Optional[str] = None,
    defender_item: Optional[str] = None,
    gravity: bool = False,
    belief_state: Optional[Any] = None,
    pokemon_name: Optional[str] = None,
) -> list[float]:
    """
    技の有効性をエンコード用のフラグリストに変換
    """
    calculator = MoveEffectivenessCalculator()

    flags = []
    effectiveness_values = []
    has_any = False

    for i in range(4):
        if i < len(attacker_moves) and attacker_moves[i]:
            result = calculator.check_move_effectiveness(
                attacker_moves[i],
                defender_types,
                defender_ability,
                defender_item,
                gravity,
                belief_state,
                pokemon_name,
            )
            flags.append(1.0 if result.is_effective else 0.0)
            # 有効性を0-1に正規化（4倍=1.0, 2倍=0.75, 等倍=0.5, 半減=0.25, 無効=0.0）
            normalized = min(result.effectiveness / 4.0, 1.0)
            effectiveness_values.append(normalized)
            if result.is_effective:
                has_any = True
        else:
            flags.append(0.0)
            effectiveness_values.append(0.0)

    # [有効フラグx4, 有効性x4, 有効技があるか]
    return flags + effectiveness_values + [1.0 if has_any else 0.0]