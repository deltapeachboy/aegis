"""
ダメージ計算関連のAPIエンドポイント
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from src.damage_calculator_api.calculators.damage_calculator import DamageCalculator
from src.damage_calculator_api.models.pokemon_models import (
    BattleConditions,
    MoveInput,
    PokemonState,
)
from src.damage_calculator_api.schemas.requests import (
    DamageCalculationRequest,
    MoveComparisonRequest,
    RangeAnalysisRequest,
)
from src.damage_calculator_api.schemas.responses import (
    CalculationDetails,
    DamageResult,
    MoveComparisonResponse,
    RangeAnalysisResult,
)
from src.damage_calculator_api.utils.data_loader import get_data_loader

router = APIRouter()


def get_damage_calculator() -> DamageCalculator:
    """ダメージ計算機の依存性注入"""
    return DamageCalculator()


def convert_pokemon_request(pokemon_req) -> PokemonState:
    """リクエストからPokemonStateに変換"""
    from src.damage_calculator_api.utils.stat_calculator import calculate_actual_stats
    
    # 実数値を計算または取得
    if pokemon_req.evs:
        # 努力値が指定されている場合は実数値を計算
        stats = calculate_actual_stats(
            species=pokemon_req.species,
            level=pokemon_req.level,
            evs=pokemon_req.evs,
            ivs=pokemon_req.ivs,  # 省略時は31がデフォルト
            nature=pokemon_req.nature
        )
    else:
        # 努力値が指定されていない場合は直接指定された実数値を使用
        stats = pokemon_req.stats or {}
    
    return PokemonState(
        species=pokemon_req.species,
        level=pokemon_req.level,
        stats=stats,
        nature=pokemon_req.nature,
        ability=pokemon_req.ability,
        item=pokemon_req.item,
        terastal_type=pokemon_req.terastal_type,
        is_terastalized=pokemon_req.is_terastalized,
        status_ailment=pokemon_req.status_ailment,
        hp_ratio=pokemon_req.hp_ratio,
        stat_boosts=pokemon_req.stat_boosts or {},
        paradox_boost_stat=pokemon_req.paradox_boost_stat,
        gender=pokemon_req.gender,
        fainted_teammates=pokemon_req.fainted_teammates,
        moves_last=pokemon_req.moves_last,
        flash_fire_active=pokemon_req.flash_fire_active
    )


def convert_move_request(move_req) -> MoveInput:
    """リクエストからMoveInputに変換"""
    return MoveInput(
        name=move_req.name,
        move_type=move_req.move_type,
        is_critical=move_req.is_critical,
        power_modifier=move_req.power_modifier,
    )


def convert_battle_conditions(conditions_req) -> BattleConditions:
    """リクエストからBattleConditionsに変換"""
    if not conditions_req:
        return BattleConditions()

    return BattleConditions(
        weather=conditions_req.weather,
        terrain=conditions_req.terrain,
        trick_room=conditions_req.trick_room,
        gravity=conditions_req.gravity,
        magic_room=conditions_req.magic_room,
        wonder_room=conditions_req.wonder_room,
        reflect=conditions_req.reflect,
        light_screen=conditions_req.light_screen,
        aurora_veil=conditions_req.aurora_veil,
        tailwind=conditions_req.tailwind,
    )


@router.post("/calculate", response_model=DamageResult)
async def calculate_damage(
    request: DamageCalculationRequest,
    calculator: DamageCalculator = Depends(get_damage_calculator),
):
    """
    ダメージ計算を実行

    完全な16段階乱数でのダメージ計算を行い、
    各種補正・確定数・KO確率なども含めた結果を返す
    """
    try:
        # リクエストデータをモデルに変換
        attacker = convert_pokemon_request(request.attacker)
        defender = convert_pokemon_request(request.defender)
        move = convert_move_request(request.move)
        conditions = convert_battle_conditions(request.conditions)

        # ダメージ計算実行
        result = calculator.calculate_damage(attacker, defender, move, conditions)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"計算中にエラーが発生しました: {str(e)}"
        )


@router.post("/compare", response_model=MoveComparisonResponse)
async def compare_moves(
    request: MoveComparisonRequest,
    calculator: DamageCalculator = Depends(get_damage_calculator),
):
    """
    複数の技のダメージを比較

    同じポケモン・条件で複数の技を比較し、
    最適な技の推奨も含めた結果を返す
    """
    try:
        # リクエストデータをモデルに変換
        attacker = convert_pokemon_request(request.attacker)
        defender = convert_pokemon_request(request.defender)
        conditions = convert_battle_conditions(request.conditions)

        results = []

        # 各技のダメージを計算
        for move_req in request.moves:
            move = convert_move_request(move_req)
            damage_result = calculator.calculate_damage(
                attacker, defender, move, conditions
            )

            # 技比較結果を作成
            move_result = {
                "move_name": move.name,
                "min_damage": damage_result.min_damage,
                "max_damage": damage_result.max_damage,
                "average_damage": damage_result.average_damage,
                "ko_probability": damage_result.ko_probability,
                "guaranteed_ko_hits": damage_result.guaranteed_ko_hits,
                "damage_percentage_range": damage_result.damage_percentage,
            }
            results.append(move_result)

        # 最適技を決定（平均ダメージで比較）
        best_move = max(results, key=lambda x: x["average_damage"])
        recommendation = best_move["move_name"]

        # 分析サマリーを作成
        analysis_summary = {
            "total_moves": len(results),
            "best_move": recommendation,
            "damage_range": {
                "min": min(r["min_damage"] for r in results),
                "max": max(r["max_damage"] for r in results),
            },
            "average_damage_range": {
                "min": min(r["average_damage"] for r in results),
                "max": max(r["average_damage"] for r in results),
            },
        }

        return MoveComparisonResponse(
            results=results,
            recommendation=recommendation,
            analysis_summary=analysis_summary,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"比較中にエラーが発生しました: {str(e)}"
        )


@router.post("/analyze", response_model=RangeAnalysisResult)
async def analyze_damage_range(
    request: RangeAnalysisRequest,
    calculator: DamageCalculator = Depends(get_damage_calculator),
):
    """
    ダメージ範囲の詳細分析

    16段階乱数の詳細な分布や、
    各確定数でのKO確率など詳細な分析結果を返す
    """
    try:
        # リクエストデータをモデルに変換
        attacker = convert_pokemon_request(request.attacker)
        defender = convert_pokemon_request(request.defender)
        move = convert_move_request(request.move)
        conditions = convert_battle_conditions(request.conditions)

        # ダメージ計算実行
        damage_result = calculator.calculate_damage(
            attacker, defender, move, conditions
        )

        # ダメージ分布を計算
        damage_distribution = {}
        for damage in damage_result.damage_range:
            damage_distribution[damage] = damage_distribution.get(damage, 0) + 1

        # 確定数分析
        defender_hp = damage_result.calculation_details.defender_max_hp
        ko_analysis = {}

        for hits in range(1, 6):  # 1~5確定を分析
            ko_count = sum(
                1
                for damage in damage_result.damage_range
                if damage * hits >= defender_hp
            )
            ko_analysis[f"{hits}確定"] = ko_count / len(damage_result.damage_range)

        return RangeAnalysisResult(
            damage_distribution=damage_distribution,
            min_damage=damage_result.min_damage,
            max_damage=damage_result.max_damage,
            average_damage=damage_result.average_damage,
            min_percentage=min(damage_result.damage_percentage),
            max_percentage=max(damage_result.damage_percentage),
            average_percentage=sum(damage_result.damage_percentage)
            / len(damage_result.damage_percentage),
            ko_probability=damage_result.ko_probability,
            guaranteed_ko_hits=damage_result.guaranteed_ko_hits,
            ko_analysis=ko_analysis,
            calculation_details=damage_result.calculation_details,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"分析中にエラーが発生しました: {str(e)}"
        )
