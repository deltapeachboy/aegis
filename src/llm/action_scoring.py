from __future__ import annotations

"""
ダメージ計算 API を用いた行動スコアリング

- Battle / Pokemon から PokemonState / BattleConditions へ変換
- 各合法手（技 / 交代）の「行動スコア」を計算
"""

from typing import Dict, List

from src.damage_calculator_api.calculators.damage_calculator import DamageCalculator
from src.damage_calculator_api.models.pokemon_models import MoveInput, PokemonState
from src.llm.converters import battle_to_battle_conditions, pokemon_to_pokemon_state
from src.llm.state_representation import LLMAction
from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon


def _build_move_inputs_from_actions(
    me: Pokemon, actions: List[LLMAction]
) -> Dict[str, MoveInput]:
    """LLMAction から MoveInput を構築する（技アクションのみ）。"""
    move_inputs: Dict[str, MoveInput] = {}
    for a in actions:
        if a.type != "move" or not a.move_name:
            continue
        if a.id in move_inputs:
            continue
        move_inputs[a.id] = MoveInput(name=a.move_name)
    return move_inputs


def score_actions_with_damage(
    battle: Battle, player: int, actions: List[LLMAction]
) -> Dict[str, float]:
    """
    ダメージ計算 API に基づいて各行動のスコアを計算する。

    スコア設計（暫定）:
        - 技: 期待ダメージ + KO 確率 * ボーナス
        - 交代: 相手から受ける最大平均ダメージのマイナス（= 受け性能）
    """
    calc = DamageCalculator()

    me: Pokemon = battle.pokemon[player]
    opp: Pokemon = battle.pokemon[1 - player]

    attacker_state: PokemonState = pokemon_to_pokemon_state(me)
    defender_state: PokemonState = pokemon_to_pokemon_state(opp)
    conditions = battle_to_battle_conditions(battle, attacker=player)

    scores: Dict[str, float] = {}

    # --- 技アクションのスコアリング ---
    move_inputs = _build_move_inputs_from_actions(me, actions)
    if move_inputs:
        # DamageCalculator.compare_moves で一括評価
        move_list = list(move_inputs.values())
        analyses = calc.compare_moves(
            attacker=attacker_state,
            defender=defender_state,
            moves=move_list,
            conditions=conditions,
        )

        # move_name -> (avg_damage, ko_prob) の辞書を作る
        tmp: Dict[str, Dict[str, float]] = {}
        for ana in analyses:
            if ana.get("no_damage"):
                avg = 0.0
                ko_prob = 0.0
            else:
                avg = float(ana.get("average_damage", 0.0))
                ko_prob = float(ana.get("ko_probability", 0.0))
            tmp[str(ana["move_name"])] = {
                "avg": avg,
                "ko_prob": ko_prob,
            }

        for a in actions:
            if a.type != "move" or not a.move_name:
                continue
            key = a.move_name
            info = tmp.get(key, {"avg": 0.0, "ko_prob": 0.0})
            # 暫定スコア: 平均ダメージ + KO 確率 * ボーナス
            scores[a.id] = info["avg"] + info["ko_prob"] * 50.0

    # --- 交代アクションのスコアリング ---
    # 「相手から見たときのダメージ」を最小化するような単純な防御スコア
    opp_attacker_state: PokemonState = pokemon_to_pokemon_state(opp)
    opp_conditions = battle_to_battle_conditions(battle, attacker=1 - player)

    # 相手の技一覧
    opp_move_actions: List[LLMAction] = []
    for i, move_name in enumerate(opp.moves):
        if not move_name:
            continue
        opp_move_actions.append(
            LLMAction(id=f"OPP_MOVE_{i}", type="move", move_name=move_name)
        )
    opp_move_inputs = _build_move_inputs_from_actions(opp, opp_move_actions)
    opp_move_list = list(opp_move_inputs.values())

    for a in actions:
        if a.type != "switch":
            continue
        # 交代先ポケモンを特定
        try:
            index = int(a.id.split("_", 1)[1])
        except (IndexError, ValueError):
            # 想定外の形式はスコア 0 とする
            scores.setdefault(a.id, 0.0)
            continue

        party: List[Pokemon] = battle.selected[player]
        if index < 0 or index >= len(party):
            scores.setdefault(a.id, 0.0)
            continue
        target_poke = party[index]
        if target_poke.hp <= 0:
            scores.setdefault(a.id, 0.0)
            continue

        switchin_defender_state = pokemon_to_pokemon_state(target_poke)

        # 相手の技でこの交代先にどれだけ入るかを計算
        if opp_move_list:
            analyses = calc.compare_moves(
                attacker=opp_attacker_state,
                defender=switchin_defender_state,
                moves=opp_move_list,
                conditions=opp_conditions,
            )
            # 相手から見た最大平均ダメージ
            max_avg = 0.0
            for ana in analyses:
                if ana.get("no_damage"):
                    continue
                avg = float(ana.get("average_damage", 0.0))
                if avg > max_avg:
                    max_avg = avg
            # 防御スコア: マイナス最大平均ダメージ
            scores[a.id] = -max_avg
        else:
            scores[a.id] = 0.0

    return scores


