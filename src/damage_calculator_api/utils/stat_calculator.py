"""
実数値計算ユーティリティ

努力値・個体値・性格から実数値を計算する機能を提供
"""

from typing import Dict, Optional

from src.damage_calculator_api.utils.data_loader import get_data_loader


def calculate_actual_stats(
    species: str,
    level: int = 50,
    evs: Optional[Dict[str, int]] = None,
    ivs: Optional[Dict[str, int]] = None,
    nature: str = "まじめ",
) -> Dict[str, int]:
    """
    努力値・個体値・性格から実数値を計算

    Args:
        species: ポケモン名
        level: レベル
        evs: 努力値 {"hp": 252, "attack": 0, ...}
        ivs: 個体値 {"hp": 31, "attack": 31, ...}
        nature: 性格

    Returns:
        実数値辞書 {"hp": 155, "attack": 100, ...}
    """
    # デフォルト値設定
    if evs is None:
        evs = {}
    if ivs is None:
        ivs = {}

    # 個体値のデフォルトは31
    default_ivs = {
        "hp": 31,
        "attack": 31,
        "defense": 31,
        "sp_attack": 31,
        "sp_defense": 31,
        "speed": 31,
    }
    default_ivs.update(ivs)
    ivs = default_ivs

    # 努力値のデフォルトは0
    default_evs = {
        "hp": 0,
        "attack": 0,
        "defense": 0,
        "sp_attack": 0,
        "sp_defense": 0,
        "speed": 0,
    }
    default_evs.update(evs)
    evs = default_evs

    # データローダーからポケモンデータを取得
    data_loader = get_data_loader()
    if species not in data_loader.pokemon_data:
        raise ValueError(f"ポケモン '{species}' が見つかりません")

    pokemon_data = data_loader.pokemon_data[species]

    # 種族値を取得
    base_stats = {
        "hp": pokemon_data.hp,
        "attack": pokemon_data.attack,
        "defense": pokemon_data.defense,
        "sp_attack": pokemon_data.sp_attack,
        "sp_defense": pokemon_data.sp_defense,
        "speed": pokemon_data.speed,
    }

    # 性格補正を取得
    nature_modifiers = get_nature_modifiers(nature)

    # 実数値計算
    actual_stats = {}

    # HP計算（性格補正なし）
    actual_stats["hp"] = (
        int((base_stats["hp"] * 2 + ivs["hp"] + int(evs["hp"] / 4)) * level / 100)
        + level
        + 10
    )

    # その他のステータス計算（性格補正あり）
    for stat in ["attack", "defense", "sp_attack", "sp_defense", "speed"]:
        base_value = (
            int((base_stats[stat] * 2 + ivs[stat] + int(evs[stat] / 4)) * level / 100)
            + 5
        )

        # 性格補正適用
        modifier = nature_modifiers.get(stat, 1.0)
        actual_stats[stat] = int(base_value * modifier)

    return actual_stats


def get_nature_modifiers(nature: str) -> Dict[str, float]:
    """
    性格による能力補正を取得

    Args:
        nature: 性格名

    Returns:
        補正値辞書 {"attack": 1.1, "defense": 0.9, ...}
    """
    # 既存のnature.txtファイルから性格補正を読み込み
    from pathlib import Path

    # データファイルのパス
    current_dir = Path(__file__).parent
    nature_file_path = current_dir.parent.parent.parent / "data" / "nature.txt"

    nature_corrections = {}

    if nature_file_path.exists():
        with open(nature_file_path, encoding="utf-8") as fin:
            for line in fin:
                data = line.strip().split()
                if len(data) >= 7:
                    nature_name = data[0]
                    corrections = list(map(float, data[1:7]))
                    # [HP, Attack, Defense, Sp.Attack, Sp.Defense, Speed]
                    nature_corrections[nature_name] = {
                        "hp": corrections[0],
                        "attack": corrections[1],
                        "defense": corrections[2],
                        "sp_attack": corrections[3],
                        "sp_defense": corrections[4],
                        "speed": corrections[5],
                    }

    return nature_corrections.get(
        nature,
        {
            "hp": 1.0,
            "attack": 1.0,
            "defense": 1.0,
            "sp_attack": 1.0,
            "sp_defense": 1.0,
            "speed": 1.0,
        },
    )


def validate_evs(evs: Dict[str, int]) -> Dict[str, int]:
    """
    努力値の妥当性チェック

    Args:
        evs: 努力値辞書

    Returns:
        検証済み努力値辞書

    Raises:
        ValueError: 努力値が無効な場合
    """
    if not evs:
        return {}

    # 各努力値の範囲チェック
    for stat, value in evs.items():
        if not 0 <= value <= 252:
            raise ValueError(f"努力値 {stat} は0-252の範囲で指定してください: {value}")

    # 合計努力値チェック
    total = sum(evs.values())
    if total > 510:
        raise ValueError(f"努力値の合計は510以下である必要があります: {total}")

    return evs


def validate_ivs(ivs: Dict[str, int]) -> Dict[str, int]:
    """
    個体値の妥当性チェック

    Args:
        ivs: 個体値辞書

    Returns:
        検証済み個体値辞書

    Raises:
        ValueError: 個体値が無効な場合
    """
    if not ivs:
        return {}

    # 各個体値の範囲チェック
    for stat, value in ivs.items():
        if not 0 <= value <= 31:
            raise ValueError(f"個体値 {stat} は0-31の範囲で指定してください: {value}")

    return ivs
