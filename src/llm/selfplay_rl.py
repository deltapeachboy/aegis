from __future__ import annotations

"""
LLM 自己対戦ループ用ユーティリティ

- LLMPolicy を用いた 2 体自己対戦
- (状態, 行動, 報酬, 終了フラグ) のログ生成

実際の PPO / RLHF 学習は `trl` などを利用して別スクリプトで行う想定。
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from src.llm.policy import LLMPolicy
from src.llm.reward import composite_reward
from src.llm.state_representation import LLMAction, battle_to_llm_state
from src.pokemon_battle_sim.battle import Battle


@dataclass
class Transition:
    state_text: str
    action_text: str
    reward: float
    reward_intermediate: float
    reward_final: float
    done: bool
    info: Dict[str, Any]


def action_to_command(battle: Battle, player: int, action: LLMAction) -> int:
    """
    LLMAction を Battle のコマンド整数に変換する。

    - MOVE_i   -> i
    - SWITCH_i -> 20 + i
    非法行動の場合は available_commands の先頭にフォールバック。
    """
    commands = battle.available_commands(player, phase="battle")

    if action.type == "move" and action.id.startswith("MOVE_"):
        try:
            idx = int(action.id.split("_", 1)[1])
            if idx in commands:
                return idx
        except ValueError:
            pass

    if action.type == "switch" and action.id.startswith("SWITCH_"):
        try:
            idx = int(action.id.split("_", 1)[1])
            cmd = 20 + idx
            if cmd in commands:
                return cmd
        except ValueError:
            pass

    # いずれもマッチしない場合はフォールバック
    return commands[0]


def run_selfplay_episode(
    battle: Battle,
    policy: LLMPolicy,
    max_turns: int = 200,
) -> Dict[int, List[Transition]]:
    """
    2 プレイヤーとも同じ LLMPolicy を用いて 1 エピソード分の自己対戦を行い、
    各プレイヤー視点の遷移列を返す。

    返り値:
        {player: [Transition, ...]}
    """
    trajectories: Dict[int, List[Transition]] = {0: [], 1: []}

    # 初期状態のコピーを作る（報酬計算用）
    import copy

    prev_battle = copy.deepcopy(battle)

    while battle.winner() is None and battle.turn < max_turns:
        # 両プレイヤーのコマンドを決定
        commands: List[Optional[int]] = [None, None]
        for pl in (0, 1):
            llm_state = battle_to_llm_state(battle, pl)
            po = policy.select_action(battle, pl)
            cmd = action_to_command(battle, pl, po.action)
            commands[pl] = cmd

            # 状態テキストはプレイヤーごとに別になるので、個別に保存
            # 報酬はターン後にまとめて計算
            trajectories[pl].append(
                Transition(
                    state_text=llm_state.state_text,
                    action_text=po.action.to_command_string(),
                    reward=0.0,  # 後で更新
                    reward_intermediate=0.0,
                    reward_final=0.0,
                    done=False,
                    info={"raw_text": po.raw_text, "match": po.info.get("match")},
                )
            )

        # コマンドをセットして 1 ターン進める
        battle.command = commands
        battle.proceed()

        # 報酬計算
        done = battle.winner() is not None or battle.turn >= max_turns
        for pl in (0, 1):
            total_r, r_inter, r_final = composite_reward(prev_battle, battle, pl, done=done)
            # 直近の Transition に報酬を書き戻す
            traj = trajectories[pl]
            traj[-1].reward = total_r
            traj[-1].reward_intermediate = r_inter
            traj[-1].reward_final = r_final
            traj[-1].done = done

        prev_battle = copy.deepcopy(battle)

        if done:
            break

    return trajectories


def trajectories_to_jsonl_rows(
    trajectories: Dict[int, List[Transition]]
) -> List[str]:
    """自己対戦ログを JSONL 形式の1行文字列リストに変換する。"""
    import json

    rows: List[str] = []
    for player, traj in trajectories.items():
        for t in traj:
            data = asdict(t)
            data["player"] = player
            rows.append(json.dumps(data, ensure_ascii=False))
    return rows


