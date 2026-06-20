from __future__ import annotations

"""
ダメージ計算ベースの静的教師データセット生成

出力フォーマット（1 サンプル）:
    {
        "state_text": str,             # LLM 入力となる盤面テキスト
        "actions": [                   # 合法手一覧
            {"id": str, "text": str}
        ],
        "label_action_id": str,        # 最良手（argmax）
        "policy_dist": {id: float},    # 正規化された行動分布
    }
"""

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List

from src.llm.action_scoring import score_actions_with_damage
from src.llm.state_representation import LLMAction, LLMState, battle_to_llm_state
from src.pokemon_battle_sim.battle import Battle


@dataclass
class StaticDatasetExample:
    state_text: str
    actions: List[Dict[str, str]]
    label_action_id: str
    policy_dist: Dict[str, float]


def _softmax(scores: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """簡易 softmax（オーバーフロー対策として最大値を引く）。"""
    if not scores:
        return {}
    import math

    max_score = max(scores.values())
    exps = {k: math.exp((v - max_score) / max(temperature, 1e-6)) for k, v in scores.items()}
    total = sum(exps.values())
    if total <= 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores.keys()}
    return {k: v / total for k, v in exps.items()}


def build_example_from_battle(battle: Battle, player: int) -> StaticDatasetExample:
    """
    Battle / player から 1 サンプル分の静的データを構築する。
    """
    llm_state: LLMState = battle_to_llm_state(battle, player)
    actions: List[LLMAction] = llm_state.actions

    scores = score_actions_with_damage(battle, player, actions)
    if not scores:
        # フォールバックとして一様分布
        scores = {a.id: 0.0 for a in actions}

    # argmax を教師ラベルにする
    label_action_id = max(scores.items(), key=lambda kv: kv[1])[0]

    # 方策分布
    policy_dist = _softmax(scores)

    actions_for_dump = [
        {"id": a.id, "text": a.to_command_string()} for a in actions
    ]

    return StaticDatasetExample(
        state_text=llm_state.state_text,
        actions=actions_for_dump,
        label_action_id=label_action_id,
        policy_dist=policy_dist,
    )


def iter_examples_from_battles(
    battles: Iterable[Battle], for_players: Iterable[int] = (0, 1)
) -> Iterable[StaticDatasetExample]:
    """
    複数の Battle から連続してサンプルを生成する簡易イテレータ。
    """
    for battle in battles:
        for pl in for_players:
            yield build_example_from_battle(battle, pl)


def example_to_json(example: StaticDatasetExample) -> str:
    """JSON Lines 形式で保存しやすいように 1 行 JSON 文字列を返す。"""
    import json

    return json.dumps(asdict(example), ensure_ascii=False)


