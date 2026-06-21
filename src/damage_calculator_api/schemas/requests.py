"""
APIリクエストスキーマ定義

Pydanticモデルを使用してリクエスト形式を定義
"""

from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class StatusAilmentEnum(str, Enum):
    """状態異常の種類"""

    NONE = "normal"
    POISON = "PSN"
    PARALYSIS = "PAR"
    BURN = "BRN"
    SLEEP = "SLP"
    FREEZE = "FLZ"


class WeatherEnum(str, Enum):
    """天気の種類"""

    NONE = "normal"
    SUN = "sunny"
    RAIN = "rainy"
    SANDSTORM = "sandstorm"
    SNOW = "snow"


class TerrainEnum(str, Enum):
    """フィールドの種類"""

    NONE = "normal"
    ELECTRIC = "elecfield"
    PSYCHIC = "psycofield"
    GRASSY = "glassfield"
    MISTY = "mistfield"


class PokemonRequest(BaseModel):
    """ポケモンのリクエストデータ"""

    species: str = Field(..., description="ポケモン名", example="ピカチュウ")
    level: int = Field(50, ge=1, le=100, description="レベル")

    # 個体情報
    nature: str = Field("まじめ", description="性格", example="ひかえめ")
    ability: str = Field("", description="特性", example="せいでんき")
    item: Optional[str] = Field(None, description="道具", example="いのちのたま")

    # 努力値・個体値（実数値計算用）
    evs: Optional[Dict[str, int]] = Field(
        None, description="努力値", example={"sp_attack": 252, "speed": 252, "hp": 4}
    )
    ivs: Optional[Dict[str, int]] = Field(
        None,
        description="個体値（省略時は31）",
        example={
            "hp": 31,
            "attack": 31,
            "defense": 31,
            "sp_attack": 31,
            "sp_defense": 31,
            "speed": 31,
        },
    )

    # 実数値ステータス（上級者向け：直接指定する場合）
    stats: Optional[Dict[str, int]] = Field(
        None,
        description="実数値ステータス（努力値が指定されている場合は無視されます）",
        example={
            "hp": 145,
            "attack": 86,
            "defense": 80,
            "sp_attack": 137,
            "sp_defense": 90,
            "speed": 156,
        },
    )

    # テラスタル
    terastal_type: Optional[str] = Field(
        None, description="テラスタルタイプ", example="でんき"
    )
    is_terastalized: bool = Field(False, description="テラスタル状態か")

    # 状態
    status_ailment: StatusAilmentEnum = Field(
        StatusAilmentEnum.NONE, description="状態異常"
    )
    hp_ratio: float = Field(1.0, ge=0.0, le=1.0, description="HP割合")

    # 能力ランク補正
    stat_boosts: Optional[Dict[str, int]] = Field(
        None, description="能力ランク補正", example={"attack": 2, "speed": 1}
    )

    # 特殊特性用パラメータ（テスト準拠）
    paradox_boost_stat: Optional[str] = Field(
        None, 
        description="クォークチャージ・古代活性で上昇する能力値",
        example="attack"
    )
    gender: Optional[str] = Field(
        None, 
        description="性別（とうそうしん用）",
        example="male"
    )
    fainted_teammates: int = Field(
        0, 
        ge=0, 
        le=5, 
        description="倒れた味方の数（そうだいしょう用）"
    )
    moves_last: bool = Field(
        False, 
        description="後攻かどうか（アナライズ用）"
    )
    flash_fire_active: bool = Field(
        False, 
        description="もらい火発動中かどうか"
    )


    @validator("stat_boosts")
    def validate_stat_boosts(cls, v):
        if v is None:
            return {}
        for stat, boost in v.items():
            if not -6 <= boost <= 6:
                raise ValueError(
                    f"能力ランク補正は-6~+6の範囲である必要があります: {stat}={boost}"
                )
        return v

    @validator("evs")
    def validate_evs(cls, v):
        if v is None:
            return {}
        for stat, ev in v.items():
            if not 0 <= ev <= 252:
                raise ValueError(
                    f"努力値は0~252の範囲である必要があります: {stat}={ev}"
                )
        total_evs = sum(v.values())
        if total_evs > 510:
            raise ValueError(f"努力値の合計は510以下である必要があります: {total_evs}")
        return v

    @validator("ivs")
    def validate_ivs(cls, v):
        if v is None:
            return {}
        for stat, iv in v.items():
            if not 0 <= iv <= 31:
                raise ValueError(f"個体値は0~31の範囲である必要があります: {stat}={iv}")
        return v


class MoveRequest(BaseModel):
    """技のリクエストデータ"""

    name: str = Field(..., description="技名", example="10まんボルト")
    move_type: Optional[str] = Field(
        None, description="技タイプ（特性による変更時）", example="でんき"
    )
    is_critical: bool = Field(False, description="急所か")
    power_modifier: float = Field(
        1.0, ge=0.1, le=10.0, description="威力補正", example=1.3
    )


class BattleConditionsRequest(BaseModel):
    """バトル状況のリクエストデータ"""

    weather: WeatherEnum = Field(WeatherEnum.NONE, description="天気")
    terrain: TerrainEnum = Field(TerrainEnum.NONE, description="フィールド")

    # フィールド効果
    trick_room: bool = Field(False, description="トリックルーム")
    gravity: bool = Field(False, description="じゅうりょく")
    magic_room: bool = Field(False, description="マジックルーム")
    wonder_room: bool = Field(False, description="ワンダールーム")

    # 壁系
    reflect: bool = Field(False, description="リフレクター")
    light_screen: bool = Field(False, description="ひかりのかべ")
    aurora_veil: bool = Field(False, description="オーロラベール")

    # その他
    tailwind: bool = Field(False, description="おいかぜ")


class DamageCalculationRequest(BaseModel):
    """ダメージ計算リクエスト"""

    attacker: PokemonRequest = Field(..., description="攻撃側ポケモン")
    defender: PokemonRequest = Field(..., description="防御側ポケモン")
    move: MoveRequest = Field(..., description="使用技")
    conditions: Optional[BattleConditionsRequest] = Field(
        None, description="バトル状況"
    )


class MoveComparisonRequest(BaseModel):
    """技比較リクエスト"""

    attacker: PokemonRequest = Field(..., description="攻撃側ポケモン")
    defender: PokemonRequest = Field(..., description="防御側ポケモン")
    moves: List[MoveRequest] = Field(
        ..., min_items=1, max_items=10, description="比較する技のリスト"
    )
    conditions: Optional[BattleConditionsRequest] = Field(
        None, description="バトル状況"
    )


class PokemonBuildRequest(BaseModel):
    """ポケモン構築リクエスト（簡易作成用）"""

    species: str = Field(..., description="ポケモン名")
    level: int = Field(50, ge=1, le=100, description="レベル")
    nature: str = Field("まじめ", description="性格")
    ability: str = Field("", description="特性")
    item: Optional[str] = Field(None, description="道具")
    evs: Optional[Dict[str, int]] = Field(None, description="努力値")
    ivs: Optional[Dict[str, int]] = Field(None, description="個体値")


class RangeAnalysisRequest(BaseModel):
    """ダメージ範囲分析リクエスト"""

    attacker: PokemonRequest = Field(..., description="攻撃側ポケモン")
    defender: PokemonRequest = Field(..., description="防御側ポケモン")
    move: MoveRequest = Field(..., description="使用技")
    conditions: Optional[BattleConditionsRequest] = Field(
        None, description="バトル状況"
    )
