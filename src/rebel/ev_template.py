"""
EV (努力値) テンプレート

性格と種族値、および技の属性（物理・特殊）から適切なEV配分を高度に推定・抽選する。
個体値はシステム仕様に合わせ、素早さを含めて常に31にクランプする。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class EVSpreadType(Enum):
    """EV配分のタイプ"""
    PHYSICAL_SPEED = auto()  # AS極振り
    SPECIAL_SPEED = auto()   # CS極振り
    PHYSICAL_HP = auto()     # HA極振り
    SPECIAL_HP = auto()      # HC極振り
    PHYSICAL_BULK = auto()   # HB極振り
    SPECIAL_BULK = auto()    # HD極振り
    BALANCED_BULK = auto()   # HBD複合調整
    MIXED_ATTACKER = auto()  # AC両刀型
    HYBRID_ADJUSTED = auto() # 各種複合調整型
    UNKNOWN = auto()         # デフォルト


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


# 性格 → 上昇ステータス
NATURE_BOOST: dict[str, Optional[str]] = {
    "いじっぱり": "attack", "さみしがり": "attack", "やんちゃ": "attack", "ゆうかん": "attack",
    "ずぶとい": "defense", "わんぱく": "defense", "のうてんき": "defense", "のんき": "defense",
    "ひかえめ": "sp_attack", "おっとり": "sp_attack", "うっかりや": "sp_attack", "れいせい": "sp_attack",
    "おだやか": "sp_defense", "おとなしい": "sp_defense", "しんちょう": "sp_defense", "なまいき": "sp_defense",
    "おくびょう": "speed", "せっかち": "speed", "ようき": "speed", "むじゃき": "speed",
    "がんばりや": None, "きまぐれ": None, "すなお": None, "てれや": None, "まじめ": None,
}

# 性格 → 下降ステータス
NATURE_PENALTY: dict[str, Optional[str]] = {
    "ずぶとい": "attack", "ひかえめ": "attack", "おだやか": "attack", "おくびょう": "attack",
    "さみしがり": "defense", "おっとり": "defense", "おとなしい": "defense", "せっかち": "defense",
    "いじっぱり": "sp_attack", "わんぱく": "sp_attack", "しんちょう": "sp_attack", "ようき": "sp_attack",
    "やんちゃ": "sp_defense", "のうてんき": "sp_defense", "うっかりや": "sp_defense", "むじゃき": "sp_defense",
    "ゆうかん": "speed", "のんき": "speed", "れいせい": "speed", "なまいき": "speed",
    "がんばりや": None, "きまぐれ": None, "すなお": None, "てれや": None, "まじめ": None,
}


def allocate_stat_points_randomly(indices: list[int], total_points: int = 66, max_single: int = 32) -> list[int]:
    """指定されたインデックスに対して努力値ポイント（1点=8EV）を確率配分する"""
    points = [0] * 6
    if not indices:
        return points
    actual_total = min(total_points, len(indices) * max_single)
    allocated = 0
    while allocated < actual_total:
        idx = random.choice(indices)
        if points[idx] < max_single:
            points[idx] += 1
            allocated += 1
    return points


def estimate_ev_spread_type(
    nature: str,
    base_stats: Optional[list[int]] = None,
    moves: Optional[list[str]] = None,
) -> EVSpreadType:
    """性格、種族値、および技構成から大まかなタイプを推定（下位互換性維持用）"""
    boost = NATURE_BOOST.get(nature)
    if boost == "attack":
        return EVSpreadType.PHYSICAL_SPEED
    elif boost == "sp_attack":
        return EVSpreadType.SPECIAL_SPEED
    elif boost == "speed":
        if base_stats and base_stats[1] >= base_stats[3]:
            return EVSpreadType.PHYSICAL_SPEED
        return EVSpreadType.SPECIAL_SPEED
    elif boost in ("defense", "sp_defense"):
        return EVSpreadType.PHYSICAL_BULK
    return EVSpreadType.UNKNOWN


def get_ev_spread(
    nature: str,
    base_stats: Optional[list[int]] = None,
    moves: Optional[list[str]] = None,
) -> EVSpread:
    """
    【高度化された努力値推定ロジック】
    チームビルドで確立された「物理/特殊アタッカー判定 ➔ 両刀/調整/極振りの確率抽選」を
    完全に踏襲し、実戦的で整合性の取れた努力値実数値を動的に生成する。
    """
    from pokepy.pokemon import Pokemon

    # 1. 技から攻撃型属性（物理/特殊）を判定
    has_physical_attack = False
    has_special_attack = False

    if moves:
        for mv in moves:
            mv_data = Pokemon.all_moves.get(mv)
            if mv_data:
                mv_class = mv_data.get("class", "")
                if "phy" in mv_class:
                    has_physical_attack = True
                elif "spc" in mv_class:
                    has_special_attack = True

        if any(m in moves for m in ["つるぎのまい", "りゅうのまい", "ビルドアップ"]):
            has_physical_attack = True
        if any(m in moves for m in ["わるだくみ", "めいそう"]):
            has_special_attack = True

    # 技がない場合、性格から物理/特殊を仮想定
    if not has_physical_attack and not has_special_attack:
        boost = NATURE_BOOST.get(nature)
        if boost == "attack":
            has_physical_attack = True
        elif boost == "sp_attack":
            has_special_attack = True
        else:
            has_physical_attack = True  # デフォルト

    # 2. 確率的な努力値振り分けの実施（両刀4%、複合調整46%、極振り50%）
    rand_ev = random.random()
    stat_points = [0] * 6

    if rand_ev < 0.04:
        # A. 【両刀型】 (4%)
        if random.random() < 0.5:
            stat_points = allocate_stat_points_randomly([1, 3, 5], total_points=66, max_single=32)
        else:
            stat_points = allocate_stat_points_randomly([0, 1, 3], total_points=66, max_single=32)
    elif rand_ev < 0.50:
        # B. 【複合調整型】 (46%)
        hybrid_patterns = [
            ("HBD", [0, 2, 4]), ("HBDS", [0, 2, 4, 5]), ("HABS", [0, 1, 2, 5]),
            ("HBCS", [0, 2, 3, 5]), ("HAB", [0, 1, 2]), ("HBC", [0, 2, 3]),
            ("HAD", [0, 1, 4]), ("HCD", [0, 3, 4]), ("HAS", [0, 1, 5]),
            ("HCS", [0, 3, 5]), ("HBS", [0, 2, 5]), ("HDS", [0, 4, 5])
        ]
        valid_patterns = []
        for name_pat, idx_list in hybrid_patterns:
            if "A" in name_pat and has_special_attack and not has_physical_attack:
                continue
            if "C" in name_pat and has_physical_attack and not has_special_attack:
                continue
            valid_patterns.append(idx_list)

        if not valid_patterns:
            valid_patterns = [p[1] for p in hybrid_patterns]

        target_indices = random.choice(valid_patterns)
        stat_points = allocate_stat_points_randomly(target_indices, total_points=66, max_single=32)
    else:
        # C. 【極振り(ブッパ)型】 (50%)
        max_out_candidates = ["HB", "HD", "HS"]
        if has_physical_attack or not has_special_attack:
            max_out_candidates += ["HA", "AS"]
        if has_special_attack or not has_physical_attack:
            max_out_candidates += ["HC", "CS"]

        chosen_max_type = random.choice(max_out_candidates)
        if chosen_max_type == "HA":
            stat_points[0], stat_points[1], stat_points[5] = 32, 32, 2
        elif chosen_max_type == "HB":
            stat_points[0], stat_points[2], stat_points[4] = 32, 32, 2
        elif chosen_max_type == "HC":
            stat_points[0], stat_points[3], stat_points[5] = 32, 32, 2
        elif chosen_max_type == "HD":
            stat_points[0], stat_points[4], stat_points[2] = 32, 32, 2
        elif chosen_max_type == "HS":
            stat_points[0], stat_points[5], stat_points[4] = 32, 32, 2
        elif chosen_max_type == "AS":
            stat_points[0], stat_points[1], stat_points[5] = 2, 32, 32
        elif chosen_max_type == "CS":
            stat_points[0], stat_points[3], stat_points[5] = 2, 32, 32

    effort = [min(252, sp * 8) for sp in stat_points]
    return EVSpread.from_list(effort)


def get_ev_spread_from_pokemon_name(
    pokemon_name: str,
    nature: str,
) -> EVSpread:
    """ポケモン名と性格からEV配分を推定"""
    from pokepy.pokemon import Pokemon
    base_stats = None
    if pokemon_name in Pokemon.zukan:
        base_stats = Pokemon.zukan[pokemon_name].get("base")
    return get_ev_spread(nature, base_stats)


# ============================================================
# IVs (個体値) の推定 (システム仕様に適合)
# ============================================================

def get_default_ivs() -> list[int]:
    """デフォルトの個体値 (S31固定仕様適合)"""
    return [31, 31, 31, 31, 31, 31]


def get_trick_room_ivs() -> list[int]:
    """トリックルーム用の個体値 (S31固定仕様適合に修正)"""
    return [31, 31, 31, 31, 31, 31]


def get_ivs_for_spread_type(spread_type: EVSpreadType) -> list[int]:
    """EV配分タイプに応じた個体値を取得（常に素早さ31）"""
    return get_default_ivs()