"""
EV (努力値) テンプレート

性格と種族値から適切なEV配分を推定する。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class EVSpreadType(Enum):
    """EV配分のタイプ"""

    # アタッカー型
    PHYSICAL_SPEED = auto()  # AS252 (攻撃・素早さ)
    SPECIAL_SPEED = auto()  # CS252 (特攻・素早さ)
    PHYSICAL_HP = auto()  # HA252 (HP・攻撃)
    SPECIAL_HP = auto()  # HC252 (HP・特攻)

    # 耐久型
    PHYSICAL_BULK = auto()  # HB252 (HP・防御)
    SPECIAL_BULK = auto()  # HD252 (HP・特防)
    BALANCED_BULK = auto()  # H252 B128 D128

    # トリックルーム型
    PHYSICAL_TRICK_ROOM = auto()  # HA252 S0 (攻撃・HP、素早さ最遅)
    SPECIAL_TRICK_ROOM = auto()  # HC252 S0 (特攻・HP、素早さ最遅)

    # その他
    UNKNOWN = auto()  # 不明（デフォルト）


@dataclass
class EVSpread:
    """EV配分"""

    hp: int = 0
    attack: int = 0
    defense: int = 0
    sp_attack: int = 0
    sp_defense: int = 0
    speed: int = 0

    def to_list(self) -> list[int]:
        """リスト形式に変換 [H, A, B, C, D, S]"""
        return [
            self.hp,
            self.attack,
            self.defense,
            self.sp_attack,
            self.sp_defense,
            self.speed,
        ]

    @classmethod
    def from_list(cls, evs: list[int]) -> "EVSpread":
        """リストから作成"""
        return cls(
            hp=evs[0] if len(evs) > 0 else 0,
            attack=evs[1] if len(evs) > 1 else 0,
            defense=evs[2] if len(evs) > 2 else 0,
            sp_attack=evs[3] if len(evs) > 3 else 0,
            sp_defense=evs[4] if len(evs) > 4 else 0,
            speed=evs[5] if len(evs) > 5 else 0,
        )

    def total(self) -> int:
        """合計EV"""
        return sum(self.to_list())

    def __repr__(self) -> str:
        parts = []
        labels = ["H", "A", "B", "C", "D", "S"]
        for label, val in zip(labels, self.to_list()):
            if val > 0:
                parts.append(f"{label}{val}")
        return " ".join(parts) if parts else "H0"


# ============================================================
# 定番のEV配分テンプレート
# ============================================================

EV_TEMPLATES: dict[EVSpreadType, EVSpread] = {
    # アタッカー型
    EVSpreadType.PHYSICAL_SPEED: EVSpread(hp=4, attack=252, speed=252),
    EVSpreadType.SPECIAL_SPEED: EVSpread(hp=4, sp_attack=252, speed=252),
    EVSpreadType.PHYSICAL_HP: EVSpread(hp=252, attack=252, speed=4),
    EVSpreadType.SPECIAL_HP: EVSpread(hp=252, sp_attack=252, speed=4),
    # 耐久型
    EVSpreadType.PHYSICAL_BULK: EVSpread(hp=252, defense=252, sp_defense=4),
    EVSpreadType.SPECIAL_BULK: EVSpread(hp=252, sp_defense=252, defense=4),
    EVSpreadType.BALANCED_BULK: EVSpread(hp=252, defense=128, sp_defense=128),
    # トリックルーム型
    EVSpreadType.PHYSICAL_TRICK_ROOM: EVSpread(hp=252, attack=252, defense=4),
    EVSpreadType.SPECIAL_TRICK_ROOM: EVSpread(hp=252, sp_attack=252, defense=4),
    # 不明
    EVSpreadType.UNKNOWN: EVSpread(hp=84, attack=84, defense=84, sp_attack=84, sp_defense=84, speed=84),
}


# ============================================================
# 性格からEV配分を推定
# ============================================================

# 性格 → 上昇ステータス
NATURE_BOOST: dict[str, Optional[str]] = {
    # 攻撃上昇
    "いじっぱり": "attack",
    "さみしがり": "attack",
    "やんちゃ": "attack",
    "ゆうかん": "attack",
    # 防御上昇
    "ずぶとい": "defense",
    "わんぱく": "defense",
    "のうてんき": "defense",
    "のんき": "defense",
    # 特攻上昇
    "ひかえめ": "sp_attack",
    "おっとり": "sp_attack",
    "うっかりや": "sp_attack",
    "れいせい": "sp_attack",
    # 特防上昇
    "おだやか": "sp_defense",
    "おとなしい": "sp_defense",
    "しんちょう": "sp_defense",
    "なまいき": "sp_defense",
    # 素早さ上昇
    "おくびょう": "speed",
    "せっかち": "speed",
    "ようき": "speed",
    "むじゃき": "speed",
    # 無補正
    "がんばりや": None,
    "きまぐれ": None,
    "すなお": None,
    "てれや": None,
    "まじめ": None,
}

# 性格 → 下降ステータス
NATURE_PENALTY: dict[str, Optional[str]] = {
    # 攻撃下降
    "ずぶとい": "attack",
    "ひかえめ": "attack",
    "おだやか": "attack",
    "おくびょう": "attack",
    # 防御下降
    "さみしがり": "defense",
    "おっとり": "defense",
    "おとなしい": "defense",
    "せっかち": "defense",
    # 特攻下降
    "いじっぱり": "sp_attack",
    "わんぱく": "sp_attack",
    "しんちょう": "sp_attack",
    "ようき": "sp_attack",
    # 特防下降
    "やんちゃ": "sp_defense",
    "のうてんき": "sp_defense",
    "うっかりや": "sp_defense",
    "むじゃき": "sp_defense",
    # 素早さ下降
    "ゆうかん": "speed",
    "のんき": "speed",
    "れいせい": "speed",
    "なまいき": "speed",
    # 無補正
    "がんばりや": None,
    "きまぐれ": None,
    "すなお": None,
    "てれや": None,
    "まじめ": None,
}


def estimate_ev_spread_type(
    nature: str,
    base_stats: Optional[list[int]] = None,
) -> EVSpreadType:
    """
    性格と種族値からEV配分タイプを推定

    Args:
        nature: 性格名
        base_stats: 種族値 [H, A, B, C, D, S]（Noneの場合は性格のみで判断）

    Returns:
        推定されるEV配分タイプ
    """
    boost = NATURE_BOOST.get(nature)
    penalty = NATURE_PENALTY.get(nature)

    # 素早さ下降 → トリックルーム型の可能性
    if penalty == "speed":
        if boost == "attack":
            return EVSpreadType.PHYSICAL_TRICK_ROOM
        elif boost == "sp_attack":
            return EVSpreadType.SPECIAL_TRICK_ROOM

    # 攻撃上昇系
    if boost == "attack":
        if penalty == "sp_attack":
            # いじっぱり: AS or HA
            return EVSpreadType.PHYSICAL_SPEED
        else:
            return EVSpreadType.PHYSICAL_SPEED

    # 特攻上昇系
    if boost == "sp_attack":
        if penalty == "attack":
            # ひかえめ: CS or HC
            return EVSpreadType.SPECIAL_SPEED
        else:
            return EVSpreadType.SPECIAL_SPEED

    # 素早さ上昇系
    if boost == "speed":
        if penalty == "attack":
            # おくびょう: CS
            return EVSpreadType.SPECIAL_SPEED
        elif penalty == "sp_attack":
            # ようき: AS
            return EVSpreadType.PHYSICAL_SPEED
        elif base_stats:
            # 種族値で判断
            if base_stats[1] >= base_stats[3]:
                return EVSpreadType.PHYSICAL_SPEED
            else:
                return EVSpreadType.SPECIAL_SPEED

    # 防御上昇系
    if boost == "defense":
        return EVSpreadType.PHYSICAL_BULK

    # 特防上昇系
    if boost == "sp_defense":
        return EVSpreadType.SPECIAL_BULK

    # 無補正性格の場合、種族値で判断
    if boost is None and base_stats:
        if base_stats[1] >= base_stats[3]:
            return EVSpreadType.PHYSICAL_SPEED
        else:
            return EVSpreadType.SPECIAL_SPEED

    return EVSpreadType.UNKNOWN


def get_ev_spread(
    nature: str,
    base_stats: Optional[list[int]] = None,
) -> EVSpread:
    """
    性格と種族値から推定されるEV配分を取得

    Args:
        nature: 性格名
        base_stats: 種族値 [H, A, B, C, D, S]

    Returns:
        推定されるEV配分
    """
    spread_type = estimate_ev_spread_type(nature, base_stats)
    return EV_TEMPLATES[spread_type]


def get_ev_spread_from_pokemon_name(
    pokemon_name: str,
    nature: str,
) -> EVSpread:
    """
    ポケモン名と性格からEV配分を推定

    種族値を自動で取得して推定する。

    Args:
        pokemon_name: ポケモン名
        nature: 性格名

    Returns:
        推定されるEV配分
    """
    from src.pokemon_battle_sim.pokemon import Pokemon

    # 種族値を取得
    base_stats = None
    if pokemon_name in Pokemon.zukan:
        base_stats = Pokemon.zukan[pokemon_name].get("base")

    return get_ev_spread(nature, base_stats)


# ============================================================
# IVs (個体値) の推定
# ============================================================

def get_default_ivs() -> list[int]:
    """デフォルトの個体値 (6V)"""
    return [31, 31, 31, 31, 31, 31]


def get_trick_room_ivs() -> list[int]:
    """トリックルーム用の個体値 (素早さ0)"""
    return [31, 31, 31, 31, 31, 0]


def get_ivs_for_spread_type(spread_type: EVSpreadType) -> list[int]:
    """EV配分タイプに応じた個体値を取得"""
    if spread_type in (
        EVSpreadType.PHYSICAL_TRICK_ROOM,
        EVSpreadType.SPECIAL_TRICK_ROOM,
    ):
        return get_trick_room_ivs()
    return get_default_ivs()
