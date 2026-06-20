"""
ポケモンごとの持ち物事前確率データベース

トップランカーのデータから、各ポケモンがどの持ち物を持っている確率が高いかを算出する。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


class ItemPriorDatabase:
    """ポケモンごとの持ち物事前確率を管理するデータベース"""

    def __init__(self):
        # {pokemon_name: {item_name: probability}}
        self._priors: dict[str, dict[str, float]] = {}
        # {pokemon_name: {item_name: count}}
        self._counts: dict[str, Counter] = defaultdict(Counter)
        # 未知のポケモンに対するデフォルト持ち物リスト
        self._default_items: list[str] = [
            "きあいのタスキ",
            "こだわりハチマキ",
            "こだわりメガネ",
            "こだわりスカーフ",
            "たべのこし",
            "いのちのたま",
            "とつげきチョッキ",
            "ゴツゴツメット",
            "オボンのみ",
            "ラムのみ",
        ]

    @classmethod
    def from_trainer_json(cls, json_path: str | Path) -> ItemPriorDatabase:
        """トレーナーJSONファイルから事前確率データベースを構築"""
        db = cls()
        db.load_trainer_json(json_path)
        return db

    def load_trainer_json(self, json_path: str | Path) -> None:
        """トレーナーJSONファイルを読み込んでカウントを更新"""
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            trainers = json.load(f)

        for trainer in trainers:
            for pokemon in trainer.get("pokemons", []):
                name = pokemon.get("name")
                item = pokemon.get("item")
                if name and item:
                    # 表記揺れ対応（例: "こだわりはちまき" → "こだわりハチマキ"）
                    item = self._normalize_item_name(item)
                    self._counts[name][item] += 1

        # カウントから確率を計算
        self._compute_priors()

    def _normalize_item_name(self, item: str) -> str:
        """持ち物名の表記揺れを正規化"""
        # 既知の表記揺れを修正
        normalizations = {
            "こだわりはちまき": "こだわりハチマキ",
            "こだわりめがね": "こだわりメガネ",
            "こだわりすかーふ": "こだわりスカーフ",
        }
        return normalizations.get(item, item)

    def _compute_priors(self) -> None:
        """カウントから確率を計算"""
        self._priors = {}
        for pokemon_name, item_counts in self._counts.items():
            total = sum(item_counts.values())
            if total > 0:
                self._priors[pokemon_name] = {
                    item: count / total for item, count in item_counts.items()
                }

    def get_item_prior(
        self, pokemon_name: str, min_probability: float = 0.05
    ) -> dict[str, float]:
        """
        指定ポケモンの持ち物事前確率を取得

        Args:
            pokemon_name: ポケモン名
            min_probability: この確率以下の持ち物は除外（デフォルト5%）

        Returns:
            {item_name: probability} の辞書
        """
        if pokemon_name in self._priors:
            priors = self._priors[pokemon_name]
            # 低確率の持ち物を除外して正規化
            filtered = {
                item: prob for item, prob in priors.items() if prob >= min_probability
            }
            if filtered:
                total = sum(filtered.values())
                return {item: prob / total for item, prob in filtered.items()}
            return priors
        else:
            # 未知のポケモンは均等分布
            n = len(self._default_items)
            return {item: 1.0 / n for item in self._default_items}

    def get_top_items(
        self, pokemon_name: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        指定ポケモンの持ち物を確率順にtop_k個取得

        Args:
            pokemon_name: ポケモン名
            top_k: 取得する持ち物の数

        Returns:
            [(item_name, probability), ...] のリスト（確率降順）
        """
        priors = self.get_item_prior(pokemon_name, min_probability=0.0)
        sorted_items = sorted(priors.items(), key=lambda x: -x[1])
        return sorted_items[:top_k]

    def get_all_pokemon_names(self) -> list[str]:
        """データベースに登録されている全ポケモン名を取得"""
        return list(self._priors.keys())

    def get_item_count(self, pokemon_name: str, item_name: str) -> int:
        """指定ポケモンの指定持ち物のカウント数を取得"""
        return self._counts.get(pokemon_name, Counter()).get(item_name, 0)

    def get_total_count(self, pokemon_name: str) -> int:
        """指定ポケモンの総カウント数を取得"""
        return sum(self._counts.get(pokemon_name, Counter()).values())

    def __contains__(self, pokemon_name: str) -> bool:
        """ポケモンがデータベースに存在するか"""
        return pokemon_name in self._priors

    def __repr__(self) -> str:
        return f"ItemPriorDatabase(pokemon_count={len(self._priors)})"

    def summary(self, top_n_pokemon: int = 10, top_k_items: int = 3) -> str:
        """データベースの概要を文字列で返す"""
        lines = [f"ItemPriorDatabase Summary", f"=" * 40]
        lines.append(f"登録ポケモン数: {len(self._priors)}")
        lines.append("")

        # 出現頻度順にソート
        sorted_pokemon = sorted(
            self._priors.keys(), key=lambda p: self.get_total_count(p), reverse=True
        )

        lines.append(f"出現頻度TOP{top_n_pokemon}のポケモンと持ち物:")
        for pokemon_name in sorted_pokemon[:top_n_pokemon]:
            count = self.get_total_count(pokemon_name)
            lines.append(f"\n  {pokemon_name} (n={count})")
            for item, prob in self.get_top_items(pokemon_name, top_k_items):
                lines.append(f"    - {item}: {prob:.1%}")

        return "\n".join(lines)
