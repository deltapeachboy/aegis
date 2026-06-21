"""
ポケモン関連のデータモデル定義

既存のPokemonクラスの機能を最小限に抽出した軽量版
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from enum import Enum


class StatusAilment(str, Enum):
    """状態異常の種類"""
    NONE = "normal"
    POISON = "PSN"
    PARALYSIS = "PAR"
    BURN = "BRN"
    SLEEP = "SLP"
    FREEZE = "FLZ"


class WeatherCondition(str, Enum):
    """天気の種類"""
    NONE = "normal"
    SUN = "sunny"
    RAIN = "rainy"
    SANDSTORM = "sandstorm"
    SNOW = "snow"


class TerrainCondition(str, Enum):
    """フィールドの種類"""
    NONE = "normal"
    ELECTRIC = "elecfield"
    PSYCHIC = "psycofield"
    GRASSY = "glassfield"
    MISTY = "mistfield"


@dataclass
class PokemonState:
    """
    ダメージ計算に必要なポケモンの状態
    
    既存のPokemonクラスから必要最小限の情報を抽出
    """
    species: str  # ポケモン名（例: "ピカチュウ", "オーガポン(かまど)"）
    level: int = 50
    
    # 実数値ステータス [H, A, B, C, D, S]
    stats: Dict[str, int] = field(default_factory=dict)
    
    # 個体情報
    nature: str = "まじめ"
    ability: str = ""
    item: Optional[str] = None
    
    # テラスタル
    terastal_type: Optional[str] = None
    is_terastalized: bool = False
    
    # 状態
    status_ailment: StatusAilment = StatusAilment.NONE
    
    # 能力ランク補正 [-6 ~ +6]
    stat_boosts: Dict[str, int] = field(default_factory=lambda: {
        "attack": 0,
        "defense": 0,
        "sp_attack": 0,
        "sp_defense": 0,
        "speed": 0,
        "accuracy": 0,
        "evasion": 0
    })
    
    # バトル中の状態
    hp_ratio: float = 1.0  # HP割合 (0.0 ~ 1.0)
    
    # クォークチャージ・古代活性用
    paradox_boost_stat: Optional[str] = None  # "attack", "defense", "sp_attack", "sp_defense" のいずれか
    
    # 特殊特性用パラメータ
    gender: Optional[str] = None  # "male", "female", "genderless" - とうそうしん用
    fainted_teammates: int = 0  # そうだいしょう用（倒れた味方の数）
    moves_last: bool = False  # アナライズ用（後攻かどうか）
    flash_fire_active: bool = False  # もらい火発動中かどうか（攻撃力1.5倍）
    
    def __post_init__(self):
        """初期化後の処理"""
        # デフォルト能力値設定（レベル50、無補正）
        if not self.stats:
            self.stats = {
                "hp": 155,        # H: 100
                "attack": 105,    # A: 80  
                "defense": 105,   # B: 80
                "sp_attack": 105, # C: 80
                "sp_defense": 105,# D: 80
                "speed": 105      # S: 80
            }


@dataclass 
class MoveInput:
    """
    技の入力情報
    """
    name: str  # 技名
    move_type: Optional[str] = None  # タイプ変更特性用
    is_critical: bool = False
    power_modifier: float = 1.0  # 威力補正（例：いのちのたま 1.3倍）
    
    # 技の基本情報（データから自動設定される）
    base_power: Optional[int] = None
    move_category: Optional[str] = None  # "phy", "spe", "sta"
    accuracy: Optional[int] = None


@dataclass
class BattleConditions:
    """
    バトルフィールドの状態
    """
    weather: WeatherCondition = WeatherCondition.NONE
    terrain: TerrainCondition = TerrainCondition.NONE
    
    # フィールド効果
    trick_room: bool = False
    gravity: bool = False
    magic_room: bool = False
    wonder_room: bool = False
    
    # 壁系
    reflect: bool = False
    light_screen: bool = False
    aurora_veil: bool = False
    
    # その他
    tailwind: bool = False


@dataclass
class DamageResult:
    """
    ダメージ計算結果
    """
    # 基本ダメージ情報
    damage_range: List[int]  # [最小, 最大] 16段階のダメージロール
    damage_percentage: List[float]  # HP割合でのダメージ
    
    # KO情報
    ko_probability: float  # 一撃KO確率
    guaranteed_ko_hits: int  # 確定KOまでの攻撃回数
    
    # 計算詳細
    calculation_details: Dict[str, Union[float, int, str]] = field(default_factory=dict)
    
    @property
    def min_damage(self) -> int:
        """最小ダメージ"""
        return min(self.damage_range) if self.damage_range else 0
    
    @property  
    def max_damage(self) -> int:
        """最大ダメージ"""
        return max(self.damage_range) if self.damage_range else 0
    
    @property
    def average_damage(self) -> float:
        """平均ダメージ"""
        return sum(self.damage_range) / len(self.damage_range) if self.damage_range else 0.0


@dataclass
class PokemonSpeciesData:
    """
    ポケモン種族データ（zukan.txtから読み込み）
    """
    number: int
    name: str
    types: List[str]
    abilities: List[str]
    base_stats: List[int]  # [H, A, B, C, D, S]
    weight: float = 0.0
    
    @property
    def hp(self) -> int:
        return self.base_stats[0]
    
    @property 
    def attack(self) -> int:
        return self.base_stats[1]
        
    @property
    def defense(self) -> int:
        return self.base_stats[2]
        
    @property
    def sp_attack(self) -> int:
        return self.base_stats[3]
        
    @property
    def sp_defense(self) -> int:
        return self.base_stats[4]
        
    @property
    def speed(self) -> int:
        return self.base_stats[5]


@dataclass
class MoveData:
    """
    技データ（move.txtから読み込み）
    """
    name: str
    move_type: str
    category: str  # "phy", "spe", "sta"
    power: int
    accuracy: int
    pp: int
    
    # 特殊効果フラグ
    is_contact: bool = False
    is_sound: bool = False
    is_wind: bool = False
    has_recoil: bool = False
    
    @property
    def is_physical(self) -> bool:
        return self.category == "phy"
    
    @property
    def is_special(self) -> bool:
        return self.category == "spe"
    
    @property 
    def is_status(self) -> bool:
        return self.category == "sta"


@dataclass
class ItemData:
    """
    道具データ（item.txtから読み込み）
    """
    name: str
    fling_power: int = 0
    boost_type: Optional[str] = None  # 強化タイプ
    resist_type: Optional[str] = None  # 半減タイプ
    power_modifier: float = 1.0  # 威力補正
    is_consumable: bool = False  # 消耗品かどうか