from __future__ import annotations

"""
LLM 方策の評価用ユーティリティ

- 自己対戦における勝率・平均ターン数・最終HP差などの指標を計算
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from src.llm.policy import LLMPolicy
from src.llm.selfplay_rl import action_to_command
from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon


def _total_hp_ratio(party: List[Pokemon]) -> float:
    return float(sum(p.hp_ratio for p in party))


@dataclass
class EvalStats:
    num_games: int
    win_rate: float
    average_turns: float
    average_hp_diff: float  # (self_HP_sum - opp_HP_sum) の平均


def simulate_match(
    battle: Battle,
    policy_a: LLMPolicy,
    policy_b: LLMPolicy,
    max_turns: int = 200,
) -> Tuple[int, int, float]:
    """
    policy_a vs policy_b を 1 試合だけ行う。

    返り値:
        (winner, turns, final_hp_diff_for_player0)
    """
    import copy

    turns = 0
    while battle.winner() is None and battle.turn < max_turns:
        commands = [None, None]
        for pl, policy in ((0, policy_a), (1, policy_b)):
            po = policy.select_action(battle, pl)
            cmd = action_to_command(battle, pl, po.action)
            commands[pl] = cmd

        battle.command = commands
        battle.proceed()
        turns += 1

        if battle.winner() is not None or battle.turn >= max_turns:
            break

    winner = battle.winner()
    # 最終 HP 差（player0 視点）
    hp0 = _total_hp_ratio(battle.selected[0])
    hp1 = _total_hp_ratio(battle.selected[1])
    hp_diff = hp0 - hp1
    return winner if winner is not None else -1, turns, hp_diff


def evaluate_policy_selfplay(
    battle_factory: Callable[[], Battle],
    policy: LLMPolicy,
    num_games: int = 50,
) -> EvalStats:
    """
    1 つの LLMPolicy を 2 プレイヤーで自己対戦させ、性能を評価する。

    battle_factory:
        新しい初期状態の Battle を返す関数。
        例: トレーナー構築から 3vs3 をサンプリングして Battle を構築する関数。
    """
    wins = 0
    total_turns = 0
    total_hp_diff = 0.0

    for _ in range(num_games):
        battle = battle_factory()
        winner, turns, hp_diff = simulate_match(battle, policy, policy)
        if winner == 0:
            wins += 1
        total_turns += turns
        total_hp_diff += hp_diff

    if num_games == 0:
        return EvalStats(num_games=0, win_rate=0.0, average_turns=0.0, average_hp_diff=0.0)

    return EvalStats(
        num_games=num_games,
        win_rate=wins / num_games,
        average_turns=total_turns / num_games,
        average_hp_diff=total_hp_diff / num_games,
    )


