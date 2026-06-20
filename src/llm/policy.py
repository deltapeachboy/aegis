from __future__ import annotations

"""
LLM ベース方策クラス

Hugging Face 上の `llm-jp/llm-jp-3.1-1.8b-instruct4` などのチャット形式モデルを用いて、
Battle から 1 手をサンプリングするためのラッパー。

モデルカード:
https://huggingface.co/llm-jp/llm-jp-3.1-1.8b-instruct4
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm.state_representation import (
    LLMAction,
    LLMState,
    battle_to_llm_state,
    parse_llm_action_output,
)
from src.pokemon_battle_sim.battle import Battle


@dataclass
class PolicyOutput:
    action: LLMAction
    raw_text: str
    info: Dict[str, Any]


class LLMPolicy:
    """
    LLM-jp 系モデルを用いた方策。

    ベースモデルとして `llm-jp/llm-jp-3.1-1.8b-instruct4` を想定するが、
    任意のチャットテンプレート対応モデルも利用可能。
    """

    def __init__(
        self,
        model_name: str = "llm-jp/llm-jp-3.1-1.8b-instruct4",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch_dtype,
        )
        self.model.to(self.device)

        # チャットテンプレートが存在する前提（llm-jp は apply_chat_template 対応）

    def build_messages(self, llm_state: LLMState) -> list[dict[str, str]]:
        """1 ターン分の状態からチャットプロンプトを構築する。"""
        user_lines = [
            "以下は現在の盤面情報です。",
            "",
            llm_state.state_text,
            "",
            "上記の `legal_actions` から次の1手を1つ選んでください。",
            "出力は必ず「MOVE: 技名」または「SWITCH: ポケモン名」の1行のみとし、"
            "日本語の説明文は書かないでください。",
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "あなたはポケモン対戦のエージェントです。"
                    "ユーザーが与える盤面情報をもとに、勝率が高くなる次の1手を選びます。"
                    "出力は1行のみのコマンドとし、追加の説明は出力してはいけません。"
                ),
            },
            {"role": "user", "content": "\n".join(user_lines)},
        ]
        return messages

    @torch.no_grad()
    def select_action(
        self,
        battle: Battle,
        player: int,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
    ) -> PolicyOutput:
        """
        Battle / player から 1 手をサンプリングし、LLMAction へ変換して返す。
        """
        llm_state = battle_to_llm_state(battle, player)
        messages = self.build_messages(llm_state)

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(
            tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )[0]

        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # プロンプト部分を粗く削除（最後の改行以降を出力とみなす簡易実装）
        # 必要に応じてより厳密なパースに差し替え可能
        if "\n" in text:
            last_line = text.split("\n")[-1].strip()
        else:
            last_line = text.strip()

        action, match_info = parse_llm_action_output(last_line, llm_state.actions)

        return PolicyOutput(
            action=action,
            raw_text=text,
            info={
                "last_line": last_line,
                "match": match_info,
            },
        )


