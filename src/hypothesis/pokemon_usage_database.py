"""
ポケモン使用率データベース

pokedb.tokyoから取得したデータを元に、各ポケモンの
技・持ち物・テラスタイプ・性格・特性の事前確率を管理する。

ReBeL実装での信念状態の基盤として使用。
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PokemonHypothesis:
    """ポケモンの型仮説（サンプリング結果）"""

    pokemon_name: str
    moves: list[str]  # 4技
    item: str
    tera_type: str
    nature: str
    ability: str

    def __repr__(self) -> str:
        moves_str = ", ".join(self.moves[:2]) + "..."
        return (
            f"PokemonHypothesis({self.pokemon_name}, "
            f"item={self.item}, tera={self.tera_type}, moves=[{moves_str}])"
        )


@dataclass
class PokemonUsageEntry:
    """単一ポケモンの使用率データ"""

    pokemon_name: str
    pokemon_id: str = ""
    season: int = 0

    # 各種採用率 {名前: 確率(0-1)}
    moves: dict[str, float] = field(default_factory=dict)
    items: dict[str, float] = field(default_factory=dict)
    abilities: dict[str, float] = field(default_factory=dict)
    tera_types: dict[str, float] = field(default_factory=dict)
    natures: dict[str, float] = field(default_factory=dict)


class PokemonUsageDatabase:
    """
    ポケモン使用率データベース

    pokedb.tokyoのデータを読み込み、各ポケモンの型情報の事前確率を提供する。
    """

    def __init__(self):
        self._entries: dict[str, PokemonUsageEntry] = {}

        # デフォルト値（データがない場合に使用）
        self._default_items = [
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
        self._default_tera_types = [
            "ノーマル", "ほのお", "みず", "でんき", "くさ",
            "こおり", "かくとう", "どく", "じめん", "ひこう",
            "エスパー", "むし", "いわ", "ゴースト", "ドラゴン",
            "あく", "はがね", "フェアリー",
        ]
        self._default_natures = [
            "いじっぱり", "ようき", "ひかえめ", "おくびょう",
            "わんぱく", "しんちょう", "ずぶとい", "おだやか",
        ]

    @classmethod
    def from_json(cls, json_path: str | Path) -> PokemonUsageDatabase:
        """JSONファイルからデータベースを構築"""
        db = cls()
        db.load_json(json_path)
        return db

    def load_json(self, json_path: str | Path) -> None:
        """JSONファイルを読み込み"""
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            data_list = json.load(f)

        for data in data_list:
            entry = PokemonUsageEntry(
                pokemon_name=data.get("pokemon_name", ""),
                pokemon_id=data.get("pokemon_id", ""),
                season=data.get("season", 0),
                moves=data.get("moves", {}),
                items=data.get("items", {}),
                abilities=data.get("abilities", {}),
                tera_types=data.get("tera_types", {}),
                natures=data.get("natures", {}),
            )
            self._entries[entry.pokemon_name] = entry

    def get_entry(self, pokemon_name: str) -> Optional[PokemonUsageEntry]:
        """ポケモンの使用率データを取得"""
        return self._entries.get(pokemon_name)

    def __contains__(self, pokemon_name: str) -> bool:
        return pokemon_name in self._entries

    def get_all_pokemon_names(self) -> list[str]:
        """登録されている全ポケモン名を取得"""
        return list(self._entries.keys())

    # =========================================================================
    # 事前確率取得メソッド
    # =========================================================================

    def get_move_prior(
        self, pokemon_name: str, min_probability: float = 0.01
    ) -> dict[str, float]:
        """技の事前確率を取得"""
        entry = self._entries.get(pokemon_name)
        if entry and entry.moves:
            return self._filter_and_normalize(entry.moves, min_probability)
        return {}

    def get_item_prior(
        self, pokemon_name: str, min_probability: float = 0.05
    ) -> dict[str, float]:
        """持ち物の事前確率を取得"""
        entry = self._entries.get(pokemon_name)
        if entry and entry.items:
            return self._filter_and_normalize(entry.items, min_probability)
        # デフォルト均等分布
        n = len(self._default_items)
        return {item: 1.0 / n for item in self._default_items}

    def get_tera_prior(
        self, pokemon_name: str, min_probability: float = 0.05
    ) -> dict[str, float]:
        """テラスタイプの事前確率を取得"""
        entry = self._entries.get(pokemon_name)
        if entry and entry.tera_types:
            return self._filter_and_normalize(entry.tera_types, min_probability)
        # デフォルト均等分布
        n = len(self._default_tera_types)
        return {t: 1.0 / n for t in self._default_tera_types}

    def get_nature_prior(
        self, pokemon_name: str, min_probability: float = 0.05
    ) -> dict[str, float]:
        """性格の事前確率を取得"""
        entry = self._entries.get(pokemon_name)
        if entry and entry.natures:
            return self._filter_and_normalize(entry.natures, min_probability)
        # デフォルト均等分布
        n = len(self._default_natures)
        return {nature: 1.0 / n for nature in self._default_natures}

    def get_ability_prior(
        self, pokemon_name: str, min_probability: float = 0.01
    ) -> dict[str, float]:
        """特性の事前確率を取得"""
        entry = self._entries.get(pokemon_name)
        if entry and entry.abilities:
            return self._filter_and_normalize(entry.abilities, min_probability)
        return {}

    def _filter_and_normalize(
        self, probs: dict[str, float], min_prob: float
    ) -> dict[str, float]:
        """低確率を除外して正規化"""
        filtered = {k: v for k, v in probs.items() if v >= min_prob}
        if not filtered:
            return probs  # フィルタ後空なら元データを返す
        total = sum(filtered.values())
        if total > 0:
            return {k: v / total for k, v in filtered.items()}
        return filtered

    # =========================================================================
    # サンプリングメソッド
    # =========================================================================

    def sample_item(self, pokemon_name: str) -> str:
        """持ち物を確率的にサンプリング"""
        prior = self.get_item_prior(pokemon_name)
        return self._weighted_sample(prior)

    def sample_tera_type(self, pokemon_name: str) -> str:
        """テラスタイプを確率的にサンプリング"""
        prior = self.get_tera_prior(pokemon_name)
        return self._weighted_sample(prior)

    def sample_nature(self, pokemon_name: str) -> str:
        """性格を確率的にサンプリング"""
        prior = self.get_nature_prior(pokemon_name)
        return self._weighted_sample(prior)

    def sample_moveset(
        self, pokemon_name: str, num_moves: int = 4
    ) -> list[str]:
        """
        技構成を確率的にサンプリング

        採用率の高い技から順に重み付きサンプリングで選択。
        """
        prior = self.get_move_prior(pokemon_name, min_probability=0.01)
        if not prior:
            return []

        moves = []
        available = dict(prior)

        for _ in range(min(num_moves, len(available))):
            if not available:
                break
            move = self._weighted_sample(available)
            moves.append(move)
            del available[move]
            # 残りを再正規化
            total = sum(available.values())
            if total > 0:
                available = {k: v / total for k, v in available.items()}

        return moves

    def sample_hypothesis(self, pokemon_name: str) -> PokemonHypothesis:
        """ポケモンの型全体をサンプリング"""
        entry = self._entries.get(pokemon_name)

        # 技
        moves = self.sample_moveset(pokemon_name, 4)

        # 持ち物
        item = self.sample_item(pokemon_name)

        # テラスタイプ
        tera_type = self.sample_tera_type(pokemon_name)

        # 性格
        nature = self.sample_nature(pokemon_name)

        # 特性
        ability_prior = self.get_ability_prior(pokemon_name)
        ability = self._weighted_sample(ability_prior) if ability_prior else ""

        return PokemonHypothesis(
            pokemon_name=pokemon_name,
            moves=moves,
            item=item,
            tera_type=tera_type,
            nature=nature,
            ability=ability,
        )

    def _weighted_sample(self, probs: dict[str, float]) -> str:
        """重み付きサンプリング"""
        if not probs:
            return ""
        items = list(probs.keys())
        weights = list(probs.values())
        return random.choices(items, weights=weights, k=1)[0]

    # =========================================================================
    # トップN取得
    # =========================================================================

    def get_top_moves(
        self, pokemon_name: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """技を採用率順にtop_k個取得"""
        prior = self.get_move_prior(pokemon_name, min_probability=0.0)
        return sorted(prior.items(), key=lambda x: -x[1])[:top_k]

    def get_top_items(
        self, pokemon_name: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """持ち物を採用率順にtop_k個取得"""
        prior = self.get_item_prior(pokemon_name, min_probability=0.0)
        return sorted(prior.items(), key=lambda x: -x[1])[:top_k]

    def get_top_tera_types(
        self, pokemon_name: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """テラスタイプを採用率順にtop_k個取得"""
        prior = self.get_tera_prior(pokemon_name, min_probability=0.0)
        return sorted(prior.items(), key=lambda x: -x[1])[:top_k]

    # =========================================================================
    # ユーティリティ
    # =========================================================================

    def __repr__(self) -> str:
        return f"PokemonUsageDatabase(pokemon_count={len(self._entries)})"

    def summary(self, top_n_pokemon: int = 5) -> str:
        """データベースの概要"""
        lines = ["PokemonUsageDatabase Summary", "=" * 50]
        lines.append(f"登録ポケモン数: {len(self._entries)}")

        for pokemon_name in list(self._entries.keys())[:top_n_pokemon]:
            entry = self._entries[pokemon_name]
            lines.append(f"\n{pokemon_name} (Season {entry.season}):")

            if entry.moves:
                top_moves = sorted(entry.moves.items(), key=lambda x: -x[1])[:3]
                moves_str = ", ".join(f"{m}({v:.1%})" for m, v in top_moves)
                lines.append(f"  技: {moves_str}")

            if entry.items:
                top_items = sorted(entry.items.items(), key=lambda x: -x[1])[:3]
                items_str = ", ".join(f"{i}({v:.1%})" for i, v in top_items)
                lines.append(f"  持ち物: {items_str}")

            if entry.tera_types:
                top_tera = sorted(entry.tera_types.items(), key=lambda x: -x[1])[:3]
                tera_str = ", ".join(f"{t}({v:.1%})" for t, v in top_tera)
                lines.append(f"  テラス: {tera_str}")

        if len(self._entries) > top_n_pokemon:
            lines.append(f"\n... and {len(self._entries) - top_n_pokemon} more")

        return "\n".join(lines)


# =============================================================================
# ItemPriorDatabase との互換レイヤー
# =============================================================================


class ItemPriorDatabaseAdapter:
    """
    PokemonUsageDatabaseをItemPriorDatabaseのインターフェースで使用するアダプター

    既存のHypothesisMCTSとの互換性を保つ。
    """

    def __init__(self, usage_db: PokemonUsageDatabase):
        self._usage_db = usage_db

    def get_item_prior(
        self, pokemon_name: str, min_probability: float = 0.05
    ) -> dict[str, float]:
        return self._usage_db.get_item_prior(pokemon_name, min_probability)

    def get_top_items(
        self, pokemon_name: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        return self._usage_db.get_top_items(pokemon_name, top_k)

    def __contains__(self, pokemon_name: str) -> bool:
        return pokemon_name in self._usage_db

    def get_all_pokemon_names(self) -> list[str]:
        return self._usage_db.get_all_pokemon_names()
