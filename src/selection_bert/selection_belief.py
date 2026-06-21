"""
選出信念 (Selection Belief) - ReBeL統合用

Pokemon BERTの選出予測をReBeL信念に統合する。
相手の選出を予測し、それに基づいて自分の選出を決定する。
"""

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .dataset import PokemonVocab
from .model import PokemonBertConfig, PokemonBertForTokenClassification


@dataclass
class SelectionPrediction:
    """選出予測結果"""

    # 各ポケモンの選出確率 (6次元)
    selection_probs: list[float]
    # 先発確率 (6次元、選出されたポケモン中での確率)
    lead_probs: list[float]

    def top_k_selections(self, k: int = 5) -> list[tuple[tuple[int, ...], float]]:
        """
        上位k個の選出パターンと確率を返す

        Returns:
            [(選出インデックスのタプル, 確率), ...]
        """
        # 3体選出の全組み合わせを列挙
        all_combinations = list(combinations(range(6), 3))

        # 各組み合わせの確率を計算（選出確率の積）
        combo_probs = []
        for combo in all_combinations:
            prob = 1.0
            for i in range(6):
                if i in combo:
                    prob *= self.selection_probs[i]
                else:
                    prob *= 1 - self.selection_probs[i]
            combo_probs.append((combo, prob))

        # 確率でソート
        combo_probs.sort(key=lambda x: x[1], reverse=True)
        return combo_probs[:k]

    def sample_selection(self, temperature: float = 1.0) -> tuple[list[int], int]:
        """
        確率に基づいて選出をサンプリング

        Returns:
            (選出3体のインデックス, 先発インデックス)
        """
        # 選出確率を温度でスケール
        probs = torch.tensor(self.selection_probs)
        if temperature != 1.0:
            probs = F.softmax(torch.log(probs + 1e-8) / temperature, dim=0)

        # 3体を選出（重複なし）
        selected = []
        remaining_probs = probs.clone()
        for _ in range(3):
            # 正規化
            remaining_probs = remaining_probs / remaining_probs.sum()
            idx = torch.multinomial(remaining_probs, 1).item()
            selected.append(idx)
            remaining_probs[idx] = 0  # 選んだものは除外

        # 先発を決定（選出された3体の中から）
        lead_probs = torch.tensor([self.lead_probs[i] for i in selected])
        lead_probs = lead_probs / lead_probs.sum()
        lead_local_idx = torch.multinomial(lead_probs, 1).item()
        lead_idx = selected[lead_local_idx]

        # 先発を最初に持ってくる
        selected.remove(lead_idx)
        selected = [lead_idx] + selected

        return selected, lead_idx


class SelectionBeliefPredictor:
    """
    選出信念予測器

    Pokemon BERTを使って相手の選出を予測し、
    それに基づいて自分の最適な選出を決定する。
    """

    def __init__(
        self,
        model: PokemonBertForTokenClassification,
        vocab: PokemonVocab,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.device = device

    @classmethod
    def load(cls, model_dir: Path, device: str = "cpu") -> "SelectionBeliefPredictor":
        """学習済みモデルを読み込み"""
        vocab = PokemonVocab.load(model_dir / "vocab.json")

        with open(model_dir / "config.json") as f:
            config_dict = json.load(f)
        config = PokemonBertConfig(**config_dict)

        model = PokemonBertForTokenClassification(config)
        model.load_state_dict(
            torch.load(model_dir / "best_model.pt", map_location=device)
        )

        return cls(model, vocab, device)

    def predict(
        self, my_team: list[str], opp_team: list[str]
    ) -> tuple[SelectionPrediction, SelectionPrediction]:
        """
        自分と相手の選出を予測

        Args:
            my_team: 自分の6体のポケモン名
            opp_team: 相手の6体のポケモン名

        Returns:
            (自分の選出予測, 相手の選出予測)
        """
        # 入力作成: [CLS] my1-6 [SEP] opp1-6 [SEP]
        input_ids = (
            [self.vocab.cls_id]
            + [self.vocab.encode(p) for p in my_team]
            + [self.vocab.sep_id]
            + [self.vocab.encode(p) for p in opp_team]
            + [self.vocab.sep_id]
        )

        token_type_ids = [0] * (1 + 6 + 1) + [1] * (6 + 1)

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        token_type_tensor = torch.tensor(
            [token_type_ids], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            output = self.model.predict_selection(input_tensor, token_type_ids=token_type_tensor)

        # 結果を取り出し
        my_probs = output["my_selection_probs"][0].cpu()  # [6, 3]
        opp_probs = output["opp_selection_probs"][0].cpu()  # [6, 3]
        my_lead = output["my_lead_probs"][0].cpu()  # [6]
        opp_lead = output["opp_lead_probs"][0].cpu()  # [6]

        # SELECTED + LEAD の確率を選出確率とする
        my_selection_probs = (my_probs[:, 1] + my_probs[:, 2]).tolist()
        opp_selection_probs = (opp_probs[:, 1] + opp_probs[:, 2]).tolist()

        my_pred = SelectionPrediction(
            selection_probs=my_selection_probs,
            lead_probs=my_lead.tolist(),
        )
        opp_pred = SelectionPrediction(
            selection_probs=opp_selection_probs,
            lead_probs=opp_lead.tolist(),
        )

        return my_pred, opp_pred

    def get_opponent_belief(
        self, my_team: list[str], opp_team: list[str], top_k: int = 10
    ) -> list[tuple[tuple[int, ...], float]]:
        """
        相手の選出信念を取得

        Returns:
            [(選出パターン, 確率), ...]
            選出パターンは (idx1, idx2, idx3) のタプル、idx1が先発
        """
        _, opp_pred = self.predict(my_team, opp_team)
        return opp_pred.top_k_selections(top_k)

    def select_team(
        self,
        my_team: list[str],
        opp_team: list[str],
        temperature: float = 1.0,
        deterministic: bool = True,
    ) -> tuple[list[int], int]:
        """
        最適な選出を決定

        Args:
            my_team: 自分の6体
            opp_team: 相手の6体
            temperature: サンプリング温度（deterministic=Falseの場合）
            deterministic: Trueなら確率最大の選出、Falseならサンプリング

        Returns:
            (選出3体のインデックス, 先発インデックス)
        """
        my_pred, _ = self.predict(my_team, opp_team)

        if deterministic:
            # 最も確率の高い3体を選出
            top_selections = my_pred.top_k_selections(1)
            if top_selections:
                selected = list(top_selections[0][0])
                # 先発は選出された中で最も先発確率が高いもの
                lead_probs = [(i, my_pred.lead_probs[i]) for i in selected]
                lead_probs.sort(key=lambda x: x[1], reverse=True)
                lead_idx = lead_probs[0][0]
                selected.remove(lead_idx)
                return [lead_idx] + selected, lead_idx
            else:
                # フォールバック
                return [0, 1, 2], 0
        else:
            return my_pred.sample_selection(temperature)
