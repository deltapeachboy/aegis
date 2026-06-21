"""
データ読み込みユーティリティ

既存のPokemon.init()の機能を独立させたデータローダー
src/pokemon_battle_sim/pokemon.py の Pokemon.init() メソッドを参考に実装
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.damage_calculator_api.models.pokemon_models import (
    ItemData,
    MoveData,
    PokemonSpeciesData,
)

logger = logging.getLogger(__name__)


class PokemonDataLoader:
    """
    ポケモンの静的データを読み込むクラス

    既存のPokemon.init()の機能を独立させて実装
    data/ディレクトリから各種データファイルを読み込み
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Args:
            data_dir: データディレクトリのパス。Noneの場合は自動検出
        """
        if data_dir is None:
            # プロジェクトルートのdataディレクトリを自動検出
            current_dir = Path(__file__).parent
            # src/damage_calculator_api/utils -> src/damage_calculator_api -> src -> project_root -> data
            self.data_dir = current_dir.parent.parent.parent / "data"
        else:
            self.data_dir = data_dir

        # データ格納用辞書
        self.pokemon_data: Dict[str, PokemonSpeciesData] = {}
        self.move_data: Dict[str, MoveData] = {}
        self.item_data: Dict[str, ItemData] = {}
        self.type_chart: List[List[float]] = []
        self.nature_corrections: Dict[str, List[float]] = {}

        # 既にロード済みかのフラグ
        self._loaded = False

    def load_all_data(self) -> None:
        """全てのデータを読み込み"""
        if self._loaded:
            return

        logger.info(f"Loading Pokemon data from {self.data_dir}")

        try:
            self._load_pokemon_species()
            self._load_moves()
            self._load_items()
            self._load_type_chart()
            self._load_nature_corrections()

            self._loaded = True
            logger.info("Successfully loaded all Pokemon data")

        except Exception as e:
            logger.error(f"Failed to load Pokemon data: {e}")
            raise

    def _load_pokemon_species(self) -> None:
        """ポケモン種族データを読み込み (zukan.txt)"""
        zukan_path = self.data_dir / "zukan.txt"

        if not zukan_path.exists():
            raise FileNotFoundError(f"zukan.txt not found at {zukan_path}")

        with open(zukan_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # ヘッダー行をスキップ
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 12:  # 最低限必要な列数
                continue

            try:
                # 図鑑番号（ハイフン付きフォルム対応）
                number_str = parts[0]
                if "-" in number_str:
                    # フォルム違いの場合（例: "898-2"）
                    base_number = int(number_str.split("-")[0])
                    number = base_number
                else:
                    number = int(number_str)
                
                name = parts[1]

                # タイプ情報
                types = [parts[2]]
                if parts[3] != "-":
                    types.append(parts[3])

                # 特性情報
                abilities = []
                for i in range(4, 8):  # Ability1-4
                    if i < len(parts) and parts[i] != "-":
                        abilities.append(parts[i])

                # 種族値 [H, A, B, C, D, S]
                base_stats = [int(parts[i]) for i in range(8, 14)]

                species_data = PokemonSpeciesData(
                    number=number,
                    name=name,
                    types=types,
                    abilities=abilities,
                    base_stats=base_stats,
                )

                self.pokemon_data[name] = species_data

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse Pokemon data line: {line} - {e}")
                continue

    def _load_moves(self) -> None:
        """技データを読み込み (move.txt)"""
        move_path = self.data_dir / "move.txt"

        if not move_path.exists():
            raise FileNotFoundError(f"move.txt not found at {move_path}")

        with open(move_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # ヘッダー行をスキップ
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 6:
                continue

            try:
                name = parts[0]
                move_type = parts[1]
                category = parts[2]
                power = int(parts[3]) if parts[3] != "0" else 0
                accuracy = int(parts[4]) if parts[4] != "10000" else 100
                pp = int(parts[5])

                # カテゴリの正規化
                if category.startswith("物理"):
                    category = "phy"
                elif category.startswith("特殊"):
                    category = "spe"
                else:
                    category = "sta"

                move_data = MoveData(
                    name=name,
                    move_type=move_type,
                    category=category,
                    power=power,
                    accuracy=accuracy,
                    pp=pp,
                )

                self.move_data[name] = move_data

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse move data line: {line} - {e}")
                continue

    def _load_items(self) -> None:
        """道具データを読み込み (item.txt)"""
        item_path = self.data_dir / "item.txt"

        if not item_path.exists():
            raise FileNotFoundError(f"item.txt not found at {item_path}")

        with open(item_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # ヘッダー行をスキップ
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 6:
                continue

            try:
                name = parts[0]
                fling_power = int(parts[1]) if parts[1] != "-" else 0
                boost_type = parts[2] if parts[2] != "-" else None
                resist_type = parts[3] if parts[3] != "-" else None
                power_modifier = float(parts[4]) if parts[4] != "1" else 1.0
                is_consumable = bool(int(parts[5])) if parts[5] in ["0", "1"] else False

                item_data = ItemData(
                    name=name,
                    fling_power=fling_power,
                    boost_type=boost_type,
                    resist_type=resist_type,
                    power_modifier=power_modifier,
                    is_consumable=is_consumable,
                )

                self.item_data[name] = item_data

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse item data line: {line} - {e}")
                continue

    def _load_type_chart(self) -> None:
        """タイプ相性表を読み込み"""
        # 既存のPokemon.type_correctionsと同等の19x19マトリックスを構築
        # ここでは簡略化して基本的なタイプ相性のみ実装

        type_names = [
            "ノーマル",
            "ほのお",
            "みず",
            "でんき",
            "くさ",
            "こおり",
            "かくとう",
            "どく",
            "じめん",
            "ひこう",
            "エスパー",
            "むし",
            "いわ",
            "ゴースト",
            "ドラゴン",
            "あく",
            "はがね",
            "フェアリー",
            "ステラ",
        ]

        # 19x19のタイプ相性表を初期化（デフォルト1.0倍）
        self.type_chart = [[1.0 for _ in range(19)] for _ in range(19)]

        # 基本的なタイプ相性を設定（例）
        # ほのお vs くさ = 2.0倍
        self.type_chart[1][4] = 2.0
        # みず vs ほのお = 2.0倍
        self.type_chart[2][1] = 2.0
        # でんき vs みず = 2.0倍
        self.type_chart[3][2] = 2.0
        # くさ vs みず = 2.0倍
        self.type_chart[4][2] = 2.0

        # TODO: 完全なタイプ相性表を実装
        # 既存のPokemon.type_correctionsの内容を移植する必要がある

    def _load_nature_corrections(self) -> None:
        """性格補正を読み込み (nature.txt)"""
        nature_path = self.data_dir / "nature.txt"

        if not nature_path.exists():
            # デフォルトの性格補正を設定
            self._set_default_natures()
            return

        try:
            with open(nature_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines[1:]:  # ヘッダー行をスキップ
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 7:
                    continue

                nature_name = parts[0]
                # [H, A, B, C, D, S] の補正値
                corrections = [float(parts[i]) for i in range(1, 7)]

                self.nature_corrections[nature_name] = corrections

        except Exception as e:
            logger.warning(f"Failed to load nature data: {e}")
            self._set_default_natures()

    def _set_default_natures(self) -> None:
        """デフォルトの性格補正を設定"""
        # 基本的な性格のみ実装
        self.nature_corrections = {
            "まじめ": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 無補正
            "いじっぱり": [1.0, 1.1, 1.0, 0.9, 1.0, 1.0],  # A↑C↓
            "ひかえめ": [1.0, 0.9, 1.0, 1.1, 1.0, 1.0],  # C↑A↓
            "ようき": [1.0, 1.0, 1.0, 0.9, 1.0, 1.1],  # S↑C↓
            "おくびょう": [1.0, 0.9, 1.0, 1.0, 1.0, 1.1],  # S↑A↓
            "わんぱく": [1.0, 1.0, 1.1, 0.9, 1.0, 1.0],  # B↑C↓
            "しんちょう": [1.0, 1.0, 1.0, 0.9, 1.1, 1.0],  # D↑C↓
        }

    def get_pokemon_data(self, name: str) -> Optional[PokemonSpeciesData]:
        """ポケモン種族データを取得"""
        if not self._loaded:
            self.load_all_data()
        return self.pokemon_data.get(name)

    def get_move_data(self, name: str) -> Optional[MoveData]:
        """技データを取得"""
        if not self._loaded:
            self.load_all_data()
        return self.move_data.get(name)

    def get_item_data(self, name: str) -> Optional[ItemData]:
        """道具データを取得"""
        if not self._loaded:
            self.load_all_data()
        return self.item_data.get(name)

    def get_type_effectiveness(
        self, attacking_type: str, defending_types: List[str]
    ) -> float:
        """タイプ相性を取得"""
        if not self._loaded:
            self.load_all_data()

        # TODO: タイプ名からIDへの変換を実装
        # 現在は簡略化
        return 1.0

    def get_nature_correction(self, nature: str) -> List[float]:
        """性格補正を取得"""
        if not self._loaded:
            self.load_all_data()
        return self.nature_corrections.get(nature, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


# シングルトンインスタンス
_data_loader: Optional[PokemonDataLoader] = None


def get_data_loader() -> PokemonDataLoader:
    """データローダーのシングルトンインスタンスを取得"""
    global _data_loader
    if _data_loader is None:
        _data_loader = PokemonDataLoader()
        _data_loader.load_all_data()
    return _data_loader
