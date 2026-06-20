from __future__ import annotations

"""
LLM-jp 3.1 系向け SFT 用フォーマット変換

`llm-jp/llm-jp-3.1-1.8b-instruct4` などのチャット形式モデルに対して、
静的データセット (`StaticDatasetExample`) を

    messages = [
        {"role": "system", "content": ...},
        {"role": "user", "content": ...},
        {"role": "assistant", "content": ...},
    ]

の形に変換する。

モデル利用例はモデルカードを参照:
`llm-jp/llm-jp-3.1-1.8b-instruct4`
https://huggingface.co/llm-jp/llm-jp-3.1-1.8b-instruct4
"""

from dataclasses import dataclass, asdict
from typing import Dict, List

from src.llm.static_dataset import StaticDatasetExample


SYSTEM_PROMPT = (
    "あなたはポケモン対戦のエージェントです。"
    "ユーザーは現在の盤面情報と選択可能な行動リストを与えます。"
    "あなたのタスクは、勝率が高くなるような次の1手を1つだけ選ぶことです。\n\n"
    "出力ルール:\n"
    "- 出力は1行のみとし、日本語の説明文は一切書かないこと。\n"
    "- 行動は次のいずれかの形式で出力すること:\n"
    "  - 「MOVE: 技名」\n"
    "  - 「SWITCH: ポケモン名」\n"
    "- 「技名」「ポケモン名」は、与えられた legal_actions に含まれるものだけを使うこと。\n"
)


@dataclass
class ChatExample:
    """LLM-jp チャットテンプレート用の 1 サンプル。"""

    messages: List[Dict[str, str]]
    label_action_id: str
    policy_dist: Dict[str, float]


def static_example_to_chat(
    example: StaticDatasetExample, system_prompt: str = SYSTEM_PROMPT
) -> ChatExample:
    """
    StaticDatasetExample -> ChatExample への変換。

    - user: 盤面テキスト + 合法手の一覧
    - assistant: ラベル行動のコマンド文字列（MOVE/SWITCH）
    """
    # ラベル行動を特定
    label_id = example.label_action_id
    label_text = None
    for a in example.actions:
        if a["id"] == label_id:
            label_text = a["text"]
            break
    if label_text is None:
        # フォールバック: 先頭の行動を選ぶ
        first = example.actions[0]
        label_id = first["id"]
        label_text = first["text"]

    # user メッセージには、盤面テキストをそのまま埋め込む
    user_content_lines: List[str] = []
    user_content_lines.append("以下は現在の盤面情報です。")
    user_content_lines.append("")
    user_content_lines.append(example.state_text)
    user_content_lines.append("")
    user_content_lines.append(
        "上記の `legal_actions` に含まれる行動から、次の1手として最も良いものを1つ選んでください。"
    )
    user_content_lines.append(
        "出力は必ず「MOVE: 技名」または「SWITCH: ポケモン名」の1行のみとし、他の文章は書かないでください。"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_content_lines)},
        {"role": "assistant", "content": label_text},
    ]

    return ChatExample(
        messages=messages,
        label_action_id=label_id,
        policy_dist=example.policy_dist,
    )


def chat_example_to_json(chat_ex: ChatExample) -> str:
    """JSON Lines 形式で保存するためのユーティリティ。"""
    import json

    return json.dumps(asdict(chat_ex), ensure_ascii=False)


