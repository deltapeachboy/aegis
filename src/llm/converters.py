from __future__ import annotations

"""
pokemon_battle_sim と damage_calculator_api の橋渡しユーティリティ

- Pokemon -> PokemonState 変換
- Battle.condition -> BattleConditions 変換
"""

from typing import Dict

from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon
from src.damage_calculator_api.models.pokemon_models import (
    BattleConditions,
    PokemonState,
    StatusAilment,
    TerrainCondition,
    WeatherCondition,
)


def pokemon_to_pokemon_state(p: Pokemon) -> PokemonState:
    """バトルシミュレータ側の Pokemon からダメージ計算用 PokemonState を生成する。"""

    # 実数値ステータス [H, A, B, C, D, S]
    status = p.status
    stats: Dict[str, int] = {
        "hp": int(status[0]),
        "attack": int(status[1]),
        "defense": int(status[2]),
        "sp_attack": int(status[3]),
        "sp_defense": int(status[4]),
        "speed": int(status[5]),
    }

    # 能力ランク [-6, +6]
    rank = p.rank
    stat_boosts = {
        "attack": int(rank[1]),
        "defense": int(rank[2]),
        "sp_attack": int(rank[3]),
        "sp_defense": int(rank[4]),
        "speed": int(rank[5]),
        "accuracy": int(rank[6]),
        "evasion": int(rank[7]),
    }

    # 状態異常
    if not p.ailment:
        status_ailment = StatusAilment.NONE
    else:
        try:
            status_ailment = StatusAilment(p.ailment)
        except ValueError:
            # 未知の表現は一旦「異常なし」とみなす
            status_ailment = StatusAilment.NONE

    # HP 割合
    hp_ratio = float(p.hp_ratio)

    return PokemonState(
        species=p.name,
        level=p.level,
        stats=stats,
        nature=p.nature,
        ability=p.ability,
        item=p.item,
        terastal_type=p.Ttype if p.terastal else None,
        is_terastalized=bool(p.terastal),
        status_ailment=status_ailment,
        stat_boosts=stat_boosts,
        hp_ratio=hp_ratio,
    )


def battle_to_battle_conditions(battle: Battle, attacker: int) -> BattleConditions:
    """
    Battle.condition などからダメージ計算用 BattleConditions を構築する。

    attacker:
        天候・フィールドの一部は「ばんのうがさ」など攻撃側の状態に依存するため、
        攻撃側プレイヤー番号を受け取る。
    """
    # 天候
    weather_str = battle.weather(attacker)
    if not weather_str:
        weather = WeatherCondition.NONE
    else:
        try:
            weather = WeatherCondition(weather_str)
        except ValueError:
            weather = WeatherCondition.NONE

    # フィールド
    field_str = battle.field()
    if not field_str:
        terrain = TerrainCondition.NONE
    else:
        try:
            terrain = TerrainCondition(field_str)
        except ValueError:
            terrain = TerrainCondition.NONE

    cond = battle.condition

    return BattleConditions(
        weather=weather,
        terrain=terrain,
        trick_room=bool(cond.get("trickroom", 0)),
        gravity=bool(cond.get("gravity", 0)),
        magic_room=False,
        wonder_room=False,
        reflect=bool(cond.get("reflector", [0, 0])[attacker]),
        light_screen=bool(cond.get("lightwall", [0, 0])[attacker]),
        aurora_veil=False,
        tailwind=bool(cond.get("oikaze", [0, 0])[attacker]),
    )


