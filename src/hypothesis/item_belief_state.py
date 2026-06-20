"""
対戦中の相手持ち物に対する信念状態を管理

ベイズ更新により、観測情報から持ち物の確率分布を更新する。
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .item_prior_database import ItemPriorDatabase


class ObservationType(Enum):
    """観測イベントの種類"""

    # 持ち物が確定する観測
    ITEM_REVEALED = auto()  # 持ち物が直接判明（例: トリック）
    FOCUS_SASH_ACTIVATED = auto()  # きあいのタスキ発動
    CHOICE_LOCKED = auto()  # こだわり系で技固定
    LEFTOVERS_HEAL = auto()  # たべのこし回復
    BLACK_SLUDGE_HEAL = auto()  # くろいヘドロ回復
    LIFE_ORB_RECOIL = auto()  # いのちのたま反動
    ROCKY_HELMET_DAMAGE = auto()  # ゴツゴツメット発動
    ASSAULT_VEST_BLOCK = auto()  # とつげきチョッキで変化技不可
    BOOST_ENERGY_ACTIVATED = auto()  # ブーストエナジー発動

    # 持ち物を推測できる観測
    OUTSPED_UNEXPECTEDLY = auto()  # 予想外に先制された → スカーフ疑惑
    HIGH_DAMAGE_DEALT = auto()  # ダメージが高い → ハチマキ/メガネ疑惑
    SURVIVED_WITH_BERRY = auto()  # きのみで耐えた
    STATUS_CURED = auto()  # 状態異常が治った → ラムのみ疑惑


@dataclass
class Observation:
    """観測イベント"""

    type: ObservationType
    pokemon_name: str
    details: dict = field(default_factory=dict)


class ItemBeliefState:
    """
    対戦中の相手持ち物に対する信念状態

    各相手ポケモンについて、持ち物の確率分布を保持し、
    観測に基づいてベイズ更新を行う。
    """

    def __init__(
        self,
        opponent_pokemon_names: list[str],
        prior_db: ItemPriorDatabase,
        min_probability: float = 0.05,
    ):
        """
        Args:
            opponent_pokemon_names: 相手の見せ合いで見えたポケモン名リスト
            prior_db: 持ち物事前確率データベース
            min_probability: 事前確率のカットオフ
        """
        self.prior_db = prior_db
        self.min_probability = min_probability

        # 各ポケモンの持ち物信念 {pokemon_name: {item: probability}}
        self.beliefs: dict[str, dict[str, float]] = {}
        for name in opponent_pokemon_names:
            self.beliefs[name] = prior_db.get_item_prior(name, min_probability)

        # 確定した持ち物 {pokemon_name: item_name}
        self.confirmed_items: dict[str, str] = {}

        # 観測履歴
        self.observation_history: list[Observation] = []

    def update(self, observation: Observation) -> None:
        """
        観測に基づいて信念をベイズ更新

        Args:
            observation: 観測イベント
        """
        self.observation_history.append(observation)
        pokemon_name = observation.pokemon_name

        if pokemon_name not in self.beliefs:
            return

        # 既に確定している場合はスキップ
        if pokemon_name in self.confirmed_items:
            return

        obs_type = observation.type

        # 持ち物が確定する観測
        if obs_type == ObservationType.ITEM_REVEALED:
            item = observation.details.get("item")
            if item:
                self._confirm_item(pokemon_name, item)

        elif obs_type == ObservationType.FOCUS_SASH_ACTIVATED:
            self._confirm_item(pokemon_name, "きあいのタスキ")

        elif obs_type == ObservationType.CHOICE_LOCKED:
            # こだわり系のどれかに絞り込み
            self._filter_to_items(
                pokemon_name,
                ["こだわりハチマキ", "こだわりメガネ", "こだわりスカーフ"],
            )

        elif obs_type == ObservationType.LEFTOVERS_HEAL:
            self._confirm_item(pokemon_name, "たべのこし")

        elif obs_type == ObservationType.BLACK_SLUDGE_HEAL:
            self._confirm_item(pokemon_name, "くろいヘドロ")

        elif obs_type == ObservationType.LIFE_ORB_RECOIL:
            self._confirm_item(pokemon_name, "いのちのたま")

        elif obs_type == ObservationType.ROCKY_HELMET_DAMAGE:
            self._confirm_item(pokemon_name, "ゴツゴツメット")

        elif obs_type == ObservationType.ASSAULT_VEST_BLOCK:
            self._confirm_item(pokemon_name, "とつげきチョッキ")

        elif obs_type == ObservationType.BOOST_ENERGY_ACTIVATED:
            self._confirm_item(pokemon_name, "ブーストエナジー")

        # 持ち物を推測できる観測（確率更新）
        elif obs_type == ObservationType.OUTSPED_UNEXPECTEDLY:
            self._boost_item_probability(pokemon_name, "こだわりスカーフ", factor=3.0)

        elif obs_type == ObservationType.HIGH_DAMAGE_DEALT:
            move_category = observation.details.get("category", "physical")
            if move_category == "physical":
                self._boost_item_probability(
                    pokemon_name, "こだわりハチマキ", factor=2.0
                )
                self._boost_item_probability(pokemon_name, "いのちのたま", factor=1.5)
            else:
                self._boost_item_probability(pokemon_name, "こだわりメガネ", factor=2.0)
                self._boost_item_probability(pokemon_name, "いのちのたま", factor=1.5)

        elif obs_type == ObservationType.STATUS_CURED:
            self._boost_item_probability(pokemon_name, "ラムのみ", factor=5.0)

    def _confirm_item(self, pokemon_name: str, item: str) -> None:
        """持ち物を確定"""
        self.confirmed_items[pokemon_name] = item
        self.beliefs[pokemon_name] = {item: 1.0}

    def _filter_to_items(self, pokemon_name: str, items: list[str]) -> None:
        """指定した持ち物のみに絞り込み"""
        current = self.beliefs[pokemon_name]
        filtered = {item: prob for item, prob in current.items() if item in items}

        if not filtered:
            # 候補がない場合は均等分布
            filtered = {item: 1.0 / len(items) for item in items}

        # 正規化
        total = sum(filtered.values())
        self.beliefs[pokemon_name] = {
            item: prob / total for item, prob in filtered.items()
        }

    def _boost_item_probability(
        self, pokemon_name: str, item: str, factor: float
    ) -> None:
        """特定の持ち物の確率を上げる（ベイズ更新の簡易版）"""
        current = self.beliefs[pokemon_name]

        if item in current:
            current[item] *= factor

        # 正規化
        total = sum(current.values())
        if total > 0:
            self.beliefs[pokemon_name] = {
                item: prob / total for item, prob in current.items()
            }

    def get_belief(self, pokemon_name: str) -> dict[str, float]:
        """指定ポケモンの持ち物確率分布を取得"""
        return self.beliefs.get(pokemon_name, {})

    def is_confirmed(self, pokemon_name: str) -> bool:
        """持ち物が確定しているか"""
        return pokemon_name in self.confirmed_items

    def get_confirmed_item(self, pokemon_name: str) -> Optional[str]:
        """確定した持ち物を取得（未確定ならNone）"""
        return self.confirmed_items.get(pokemon_name)

    def sample_hypothesis(self) -> dict[str, str]:
        """
        現在の信念から持ち物の組み合わせを1つサンプリング

        Returns:
            {pokemon_name: item_name} の辞書
        """
        hypothesis = {}

        for pokemon_name, belief in self.beliefs.items():
            if pokemon_name in self.confirmed_items:
                hypothesis[pokemon_name] = self.confirmed_items[pokemon_name]
            else:
                # 確率に基づいてサンプリング
                items = list(belief.keys())
                probs = list(belief.values())
                hypothesis[pokemon_name] = random.choices(items, weights=probs, k=1)[0]

        return hypothesis

    def sample_hypotheses(self, n: int) -> list[dict[str, str]]:
        """
        複数の仮説をサンプリング

        Args:
            n: サンプル数

        Returns:
            [{pokemon_name: item_name}, ...] のリスト
        """
        return [self.sample_hypothesis() for _ in range(n)]

    def get_most_likely_hypothesis(self) -> dict[str, str]:
        """最も確率の高い持ち物の組み合わせを取得"""
        hypothesis = {}

        for pokemon_name, belief in self.beliefs.items():
            if pokemon_name in self.confirmed_items:
                hypothesis[pokemon_name] = self.confirmed_items[pokemon_name]
            else:
                # 最大確率の持ち物を選択
                hypothesis[pokemon_name] = max(belief.items(), key=lambda x: x[1])[0]

        return hypothesis

    def copy(self) -> ItemBeliefState:
        """信念状態のディープコピーを作成"""
        new_state = ItemBeliefState.__new__(ItemBeliefState)
        new_state.prior_db = self.prior_db
        new_state.min_probability = self.min_probability
        new_state.beliefs = deepcopy(self.beliefs)
        new_state.confirmed_items = deepcopy(self.confirmed_items)
        new_state.observation_history = list(self.observation_history)
        return new_state

    def summary(self) -> str:
        """信念状態の概要を文字列で返す"""
        lines = ["ItemBeliefState Summary", "=" * 40]

        for pokemon_name, belief in self.beliefs.items():
            if pokemon_name in self.confirmed_items:
                lines.append(f"\n{pokemon_name}: {self.confirmed_items[pokemon_name]} (確定)")
            else:
                lines.append(f"\n{pokemon_name}:")
                sorted_items = sorted(belief.items(), key=lambda x: -x[1])
                for item, prob in sorted_items[:5]:
                    lines.append(f"  - {item}: {prob:.1%}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        confirmed = len(self.confirmed_items)
        total = len(self.beliefs)
        return f"ItemBeliefState(confirmed={confirmed}/{total})"
