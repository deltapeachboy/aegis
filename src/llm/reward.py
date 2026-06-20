from __future__ import annotations

"""
自己対戦用の報酬設計ユーティリティ

- 最終報酬（勝ち/負け/引き分け）
- 中間報酬（与ダメージ−被ダメージ、残りHP差 など）
"""

from typing import Tuple

from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon


def _total_hp_ratio(party: list[Pokemon]) -> float:
    """パーティ全体の HP 割合合計を計算する。"""
    return float(sum(p.hp_ratio for p in party))


def final_reward(battle: Battle, player: int) -> float:
    """
    対戦終了時の最終報酬。

    - 勝ち: +1
    - 負け: -1
    - 引き分け / TOD 引き分け: 0
    """
    winner = battle.winner()
    if winner is None:
        return 0.0
    if winner == player:
        return 1.0
    if winner == (1 - player):
        return -1.0
    return 0.0


def hp_diff_reward(
    before: Battle, after: Battle, player: int, scale: float = 0.5
) -> float:
    """
    ターン間での「残り HP 差」の変化に基づく中間報酬。

    定義:
        r_t = scale * ( (ΔHP_self - ΔHP_opp) )
    ここで ΔHP は味方/相手パーティの HP 割合合計の変化量。
    """
    self_before = _total_hp_ratio(before.selected[player])
    opp_before = _total_hp_ratio(before.selected[1 - player])

    self_after = _total_hp_ratio(after.selected[player])
    opp_after = _total_hp_ratio(after.selected[1 - player])

    # HP は減少するので、(before - after) が与ダメージに対応
    delta_self = self_before - self_after
    delta_opp = opp_before - opp_after

    return scale * (delta_opp - delta_self)


def composite_reward(
    before: Battle,
    after: Battle,
    player: int,
    done: bool,
    final_weight: float = 1.0,
    intermediate_weight: float = 1.0,
) -> Tuple[float, float, float]:
    """
    中間報酬 + 最終報酬を合成した報酬を返す。

    返り値:
        (total_reward, intermediate_part, final_part)
    """
    r_inter = hp_diff_reward(before, after, player)
    r_final = final_reward(after, player) if done else 0.0
    total = intermediate_weight * r_inter + final_weight * r_final
    return total, r_inter, r_final


