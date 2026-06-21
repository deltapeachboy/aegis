"""
統合ダメージ計算エンジン

各計算モジュールを統合し、完全なダメージ計算機能を提供
既存のBattle.oneshot_damages() の全機能を独立実装
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from src.damage_calculator_api.calculators.stat_calculator import StatCalculator
from src.damage_calculator_api.calculators.type_calculator import TypeCalculator
from src.damage_calculator_api.models.pokemon_models import (
    BattleConditions,
    DamageResult,
    MoveInput,
    PokemonState,
    StatusAilment,
    TerrainCondition,
    WeatherCondition,
)
from src.damage_calculator_api.utils.data_loader import get_data_loader

logger = logging.getLogger(__name__)


class DamageCalculator:
    """
    統合ダメージ計算エンジン

    既存のBattle.oneshot_damages() の全機能を統合し、
    完全独立したダメージ計算機能を提供
    """

    def __init__(self):
        self.data_loader = get_data_loader()
        self.type_calculator = TypeCalculator()
        self.stat_calculator = StatCalculator()

    def calculate_damage(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        conditions: Optional[BattleConditions] = None,
    ) -> DamageResult:
        """
        メインダメージ計算関数

        Args:
            attacker: 攻撃側ポケモン
            defender: 防御側ポケモン
            move: 使用技
            conditions: バトル状況

        Returns:
            ダメージ計算結果

        Raises:
            ValueError: 無効な技名など
        """
        if conditions is None:
            conditions = BattleConditions()

        # 技データを取得
        move_data = self.data_loader.get_move_data(move.name)
        if not move_data:
            raise ValueError(f"Unknown move: {move.name}")

        # 変化技の場合はダメージなし
        if move_data.is_status:
            return DamageResult(
                damage_range=[0],
                damage_percentage=[0.0],
                ko_probability=0.0,
                guaranteed_ko_hits=999,
                calculation_details={"note": "Status move deals no damage"},
            )

        # 技威力を計算
        power = self.stat_calculator.calculate_move_power(
            attacker, defender, move, move_data, conditions
        )
        if power <= 0:
            return DamageResult(
                damage_range=[0],
                damage_percentage=[0.0],
                ko_probability=0.0,
                guaranteed_ko_hits=999,
                calculation_details={"note": "Move has no power"},
            )

        # 攻撃・防御実数値を計算
        attack_stat = self.stat_calculator.calculate_attack_stat(
            attacker, move_data, conditions, defender
        )
        defense_stat = self.stat_calculator.calculate_defense_stat(
            defender, move_data, conditions, attacker
        )

        # タイプ相性・STAB補正を計算
        type_effectiveness = self.type_calculator.calculate_type_effectiveness(
            attacker, defender, move, move_data, conditions
        )
        stab_modifier = self.type_calculator.calculate_stab_modifier(
            attacker, move, move_data, conditions
        )

        # タイプ相性で無効の場合
        if type_effectiveness == 0.0:
            return DamageResult(
                damage_range=[0],
                damage_percentage=[0.0],
                ko_probability=0.0,
                guaranteed_ko_hits=999,
                calculation_details={
                    "note": "Move has no effect due to type immunity",
                    "type_effectiveness": type_effectiveness,
                },
            )

        # 基本ダメージ計算
        base_damage = self._calculate_base_damage(
            level=attacker.level, power=power, attack=attack_stat, defense=defense_stat
        )

        # 最終補正
        final_modifier = self.stat_calculator.calculate_final_damage_modifier(
            attacker, defender, move, move_data, conditions
        )

        # 急所補正
        critical_modifier = 1.5 if move.is_critical else 1.0

        # やけど補正（物理技のみ、特性考慮）
        burn_modifier = self._calculate_burn_modifier(attacker, move_data)

        # 全ての補正を適用
        total_modifier = (
            type_effectiveness
            * stab_modifier
            * final_modifier
            * critical_modifier
            * burn_modifier
        )

        # 16段階のダメージロールを計算
        damage_range = self._calculate_damage_rolls(base_damage, total_modifier)

        # HP割合とKO情報を計算
        defender_max_hp = defender.stats.get("hp", 155)
        current_hp = int(defender_max_hp * defender.hp_ratio)

        damage_percentage = [dmg / defender_max_hp for dmg in damage_range]
        ko_probability = self._calculate_ko_probability(damage_range, current_hp)
        guaranteed_ko_hits = self._calculate_guaranteed_ko_hits(
            damage_range, current_hp
        )

        # 計算詳細を記録
        calculation_details = {
            "base_damage": base_damage,
            "level": attacker.level,
            "power": power,
            "attack_stat": attack_stat,
            "defense_stat": defense_stat,
            "type_effectiveness": type_effectiveness,
            "stab_modifier": stab_modifier,
            "final_modifier": final_modifier,
            "critical_modifier": critical_modifier,
            "burn_modifier": burn_modifier,
            "total_modifier": total_modifier,
            "defender_max_hp": defender_max_hp,
            "defender_current_hp": current_hp,
            "move_category": move_data.category,
            "attacker_ability": attacker.ability,
            "defender_ability": defender.ability,
            "attacker_item": attacker.item,
            "defender_item": defender.item,
            "weather": conditions.weather.value,
            "terrain": conditions.terrain.value,
        }

        return DamageResult(
            damage_range=damage_range,
            damage_percentage=damage_percentage,
            ko_probability=ko_probability,
            guaranteed_ko_hits=guaranteed_ko_hits,
            calculation_details=calculation_details,
        )

    def calculate_damage_range_analysis(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        conditions: Optional[BattleConditions] = None,
    ) -> Dict[str, Any]:
        """
        より詳細なダメージ範囲分析

        Returns:
            詳細分析結果
        """
        result = self.calculate_damage(attacker, defender, move, conditions)

        if not result.damage_range or max(result.damage_range) == 0:
            return {
                "no_damage": True,
                "reason": result.calculation_details.get("note", "Unknown"),
            }

        defender_max_hp = defender.stats.get("hp", 155)
        current_hp = int(defender_max_hp * defender.hp_ratio)

        # ダメージ分布分析
        damage_counts = {}
        for damage in result.damage_range:
            damage_counts[damage] = damage_counts.get(damage, 0) + 1

        # KO確率詳細分析
        ko_damages = [dmg for dmg in result.damage_range if dmg >= current_hp]
        ko_probability = len(ko_damages) / len(result.damage_range)

        # 確定数分析
        max_damage = max(result.damage_range)
        min_damage = min(result.damage_range)

        # 各確定数でのKO確率
        ko_analysis = {}
        for hits in range(1, 6):  # 1~5回攻撃
            remaining_hp = current_hp
            total_scenarios = 0
            ko_scenarios = 0

            # 簡易的な確率計算（実際は組み合わせ論が必要）
            if hits == 1:
                ko_scenarios = len(
                    [d for d in result.damage_range if d >= remaining_hp]
                )
                total_scenarios = len(result.damage_range)
            else:
                # 複数回攻撃の場合の簡易計算
                avg_damage = sum(result.damage_range) / len(result.damage_range)
                if avg_damage * hits >= remaining_hp:
                    ko_scenarios = total_scenarios = 1

            if total_scenarios > 0:
                ko_analysis[f"{hits}_hit_ko_probability"] = (
                    ko_scenarios / total_scenarios
                )

        return {
            "damage_distribution": damage_counts,
            "min_damage": min_damage,
            "max_damage": max_damage,
            "average_damage": sum(result.damage_range) / len(result.damage_range),
            "min_percentage": min_damage / defender_max_hp * 100,
            "max_percentage": max_damage / defender_max_hp * 100,
            "average_percentage": result.average_damage / defender_max_hp * 100,
            "ko_probability": ko_probability,
            "guaranteed_ko_hits": result.guaranteed_ko_hits,
            "ko_analysis": ko_analysis,
            "calculation_details": result.calculation_details,
        }

    def compare_moves(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        moves: List[MoveInput],
        conditions: Optional[BattleConditions] = None,
    ) -> List[Dict[str, Any]]:
        """
        複数技のダメージ比較

        Args:
            moves: 比較する技のリスト

        Returns:
            各技の分析結果リスト
        """
        results = []

        for move in moves:
            try:
                analysis = self.calculate_damage_range_analysis(
                    attacker, defender, move, conditions
                )
                analysis["move_name"] = move.name
                results.append(analysis)
            except Exception as e:
                logger.error(f"Failed to calculate damage for {move.name}: {e}")
                results.append(
                    {"move_name": move.name, "error": str(e), "no_damage": True}
                )

        # 平均ダメージでソート（降順）
        valid_results = [r for r in results if not r.get("no_damage", False)]
        invalid_results = [r for r in results if r.get("no_damage", False)]

        valid_results.sort(key=lambda x: x.get("average_damage", 0), reverse=True)

        return valid_results + invalid_results

    def _calculate_base_damage(
        self, level: int, power: int, attack: int, defense: int
    ) -> int:
        """
        基本ダメージ計算式

        ポケモンの基本ダメージ計算式：
        ((level * 0.4 + 2) * power * attack / defense) / 50 + 2
        """
        damage = ((level * 0.4 + 2) * power * attack / defense) / 50 + 2
        return int(damage)

    def _calculate_damage_rolls(self, base_damage: int, modifier: float) -> List[int]:
        """
        16段階のダメージロールを計算

        ダメージの乱数は85%〜100%の16段階
        """
        damage_rolls = []

        for i in range(16):
            # 85% + i * 1% = 85% ~ 100%
            random_factor = 0.85 + (i * 0.01)
            final_damage = int(base_damage * modifier * random_factor)
            damage_rolls.append(max(1, final_damage))  # 最低1ダメージ

        return sorted(damage_rolls)

    def _calculate_burn_modifier(self, attacker: PokemonState, move_data) -> float:
        """やけど状態による物理技威力半減"""
        if (
            attacker.status_ailment == StatusAilment.BURN
            and move_data.is_physical
            and attacker.ability not in ["こんじょう", "からかい"]
        ):  # 無視する特性
            return 0.5
        return 1.0

    def _calculate_ko_probability(
        self, damage_range: List[int], current_hp: int
    ) -> float:
        """KO確率を計算"""
        if not damage_range:
            return 0.0
        ko_count = sum(1 for damage in damage_range if damage >= current_hp)
        return ko_count / len(damage_range)

    def _calculate_guaranteed_ko_hits(
        self, damage_range: List[int], current_hp: int
    ) -> int:
        """確定KOまでの攻撃回数を計算"""
        if not damage_range:
            return 999

        max_damage = max(damage_range)

        if max_damage >= current_hp:
            return 1
        elif max_damage <= 0:
            return 999
        else:
            return math.ceil(current_hp / max_damage)

    def get_supported_moves(self) -> List[str]:
        """サポートされている技名のリストを取得"""
        return list(self.data_loader.move_data.keys())

    def get_supported_pokemon(self) -> List[str]:
        """サポートされているポケモン名のリストを取得"""
        return list(self.data_loader.pokemon_data.keys())

    def validate_pokemon_state(self, pokemon: PokemonState) -> bool:
        """ポケモンの状態が有効かチェック"""
        try:
            # ポケモン種族データの存在確認
            species_data = self.data_loader.get_pokemon_data(pokemon.species)
            if not species_data:
                return False

            # 能力値の妥当性チェック
            for stat_name, value in pokemon.stats.items():
                if not isinstance(value, int) or value < 1 or value > 999:
                    return False

            # ランク補正の範囲チェック
            for rank_name, value in pokemon.stat_boosts.items():
                if not isinstance(value, int) or value < -6 or value > 6:
                    return False

            # HP割合の妥当性チェック
            if not (0.0 <= pokemon.hp_ratio <= 1.0):
                return False

            return True

        except Exception as e:
            logger.error(f"Pokemon validation error: {e}")
            return False

    def validate_move_input(self, move: MoveInput) -> bool:
        """技入力が有効かチェック"""
        try:
            # 技データの存在確認
            move_data = self.data_loader.get_move_data(move.name)
            if not move_data:
                return False

            # 威力補正の妥当性チェック
            if not (0.1 <= move.power_modifier <= 10.0):
                return False

            return True

        except Exception as e:
            logger.error(f"Move validation error: {e}")
            return False


# 便利関数
def create_simple_pokemon(
    species: str,
    level: int = 50,
    nature: str = "まじめ",
    ability: str = "",
    item: Optional[str] = None,
    evs: Optional[Dict[str, int]] = None,
    ivs: Optional[Dict[str, int]] = None,
    paradox_boost_stat: Optional[str] = None,
    gender: Optional[str] = None,
    fainted_teammates: int = 0,
    moves_last: bool = False,
    flash_fire_active: bool = False,
) -> PokemonState:
    """
    簡単なポケモン作成ヘルパー関数

    Args:
        species: ポケモン名
        level: レベル
        nature: 性格
        ability: 特性
        item: 道具
        evs: 努力値 {"hp": 252, "attack": 252, ...}
        ivs: 個体値 {"hp": 31, "attack": 31, ...}
        paradox_boost_stat: クォークチャージ・古代活性で上昇する能力値 ("attack", "defense", "sp_attack", "sp_defense")
        gender: 性別 ("male", "female", "genderless") - とうそうしん用
        fainted_teammates: 倒れた味方の数 - そうだいしょう用
        moves_last: 後攻かどうか - アナライズ用
        flash_fire_active: もらい火発動中かどうか - もらい火用
    """
    data_loader = get_data_loader()
    species_data = data_loader.get_pokemon_data(species)

    if not species_data:
        raise ValueError(f"Unknown Pokemon: {species}")

    # デフォルト個体値・努力値
    default_ivs = {
        "hp": 31,
        "attack": 31,
        "defense": 31,
        "sp_attack": 31,
        "sp_defense": 31,
        "speed": 31,
    }
    default_evs = {
        "hp": 0,
        "attack": 0,
        "defense": 0,
        "sp_attack": 0,
        "sp_defense": 0,
        "speed": 0,
    }

    if ivs is None:
        ivs = default_ivs
    else:
        # 不足しているキーをデフォルト値で補完
        for key in default_ivs:
            if key not in ivs:
                ivs[key] = default_ivs[key]

    if evs is None:
        evs = default_evs
    else:
        # 不足しているキーをデフォルト値で補完
        for key in default_evs:
            if key not in evs:
                evs[key] = default_evs[key]

    # 性格補正取得
    nature_corrections = data_loader.get_nature_correction(nature)

    # 実数値計算
    def calc_hp(base, iv, ev, level):
        return int(((base * 2 + iv + ev // 4) * level // 100) + level + 10)

    def calc_stat(base, iv, ev, level, nature_mod):
        return int((((base * 2 + iv + ev // 4) * level // 100) + 5) * nature_mod)

    stats = {
        "hp": calc_hp(species_data.hp, ivs["hp"], evs["hp"], level),
        "attack": calc_stat(
            species_data.attack,
            ivs["attack"],
            evs["attack"],
            level,
            nature_corrections[1],
        ),
        "defense": calc_stat(
            species_data.defense,
            ivs["defense"],
            evs["defense"],
            level,
            nature_corrections[2],
        ),
        "sp_attack": calc_stat(
            species_data.sp_attack,
            ivs["sp_attack"],
            evs["sp_attack"],
            level,
            nature_corrections[3],
        ),
        "sp_defense": calc_stat(
            species_data.sp_defense,
            ivs["sp_defense"],
            evs["sp_defense"],
            level,
            nature_corrections[4],
        ),
        "speed": calc_stat(
            species_data.speed, ivs["speed"], evs["speed"], level, nature_corrections[5]
        ),
    }

    # デフォルト特性（abilityがNoneの場合のみ）
    if ability is None and species_data.abilities:
        ability = species_data.abilities[0]

    return PokemonState(
        species=species,
        level=level,
        stats=stats,
        nature=nature,
        ability=ability,
        item=item,
        paradox_boost_stat=paradox_boost_stat,
        gender=gender,
        fainted_teammates=fainted_teammates,
        moves_last=moves_last,
        flash_fire_active=flash_fire_active,
    )
