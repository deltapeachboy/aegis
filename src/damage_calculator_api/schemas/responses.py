"""
APIレスポンススキーマ定義

Pydanticモデルを使用してレスポンス形式を定義
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class CalculationDetails(BaseModel):
    """計算詳細情報"""
    
    base_damage: int = Field(..., description="基本ダメージ")
    level: int = Field(..., description="レベル")
    power: int = Field(..., description="技威力")
    attack_stat: int = Field(..., description="攻撃実数値")
    defense_stat: int = Field(..., description="防御実数値")
    type_effectiveness: float = Field(..., description="タイプ相性")
    stab_modifier: float = Field(..., description="タイプ一致補正")
    final_modifier: float = Field(..., description="最終補正")
    critical_modifier: float = Field(..., description="急所補正")
    burn_modifier: float = Field(..., description="やけど補正")
    total_modifier: float = Field(..., description="総合補正")
    defender_max_hp: int = Field(..., description="防御側最大HP")
    defender_current_hp: int = Field(..., description="防御側現在HP")
    move_category: str = Field(..., description="技分類")
    attacker_ability: str = Field(..., description="攻撃側特性")
    defender_ability: str = Field(..., description="防御側特性")
    attacker_item: Optional[str] = Field(None, description="攻撃側道具")
    defender_item: Optional[str] = Field(None, description="防御側道具")
    weather: str = Field(..., description="天気")
    terrain: str = Field(..., description="フィールド")


class DamageResult(BaseModel):
    """ダメージ計算結果"""
    
    damage_range: List[int] = Field(..., description="ダメージ範囲（16段階）")
    damage_percentage: List[float] = Field(..., description="HP割合でのダメージ")
    min_damage: int = Field(..., description="最小ダメージ")
    max_damage: int = Field(..., description="最大ダメージ")
    average_damage: float = Field(..., description="平均ダメージ")
    ko_probability: float = Field(..., description="一撃KO確率")
    guaranteed_ko_hits: int = Field(..., description="確定KOまでの攻撃回数")
    calculation_details: CalculationDetails = Field(..., description="計算詳細")


class RangeAnalysisResult(BaseModel):
    """ダメージ範囲分析結果"""
    
    damage_distribution: Dict[int, int] = Field(..., description="ダメージ分布")
    min_damage: int = Field(..., description="最小ダメージ")
    max_damage: int = Field(..., description="最大ダメージ")
    average_damage: float = Field(..., description="平均ダメージ")
    min_percentage: float = Field(..., description="最小ダメージHP割合")
    max_percentage: float = Field(..., description="最大ダメージHP割合")
    average_percentage: float = Field(..., description="平均ダメージHP割合")
    ko_probability: float = Field(..., description="KO確率")
    guaranteed_ko_hits: int = Field(..., description="確定数")
    ko_analysis: Dict[str, float] = Field(..., description="各確定数でのKO確率")
    calculation_details: CalculationDetails = Field(..., description="計算詳細")


class MoveComparisonResult(BaseModel):
    """技比較結果"""
    
    move_name: str = Field(..., description="技名")
    min_damage: int = Field(..., description="最小ダメージ")
    max_damage: int = Field(..., description="最大ダメージ")
    average_damage: float = Field(..., description="平均ダメージ")
    ko_probability: float = Field(..., description="KO確率")
    guaranteed_ko_hits: int = Field(..., description="確定数")
    damage_percentage_range: List[float] = Field(..., description="ダメージHP割合範囲")


class MoveComparisonResponse(BaseModel):
    """技比較レスポンス"""
    
    results: List[MoveComparisonResult] = Field(..., description="技比較結果リスト")
    recommendation: Optional[str] = Field(None, description="推奨技")
    analysis_summary: Dict[str, Any] = Field(..., description="分析サマリー")


class PokemonInfo(BaseModel):
    """ポケモン情報"""
    
    number: int = Field(..., description="図鑑番号")
    name: str = Field(..., description="ポケモン名")
    types: List[str] = Field(..., description="タイプ")
    abilities: List[str] = Field(..., description="特性")
    base_stats: Dict[str, int] = Field(..., description="種族値")
    weight: float = Field(..., description="重さ")


class MoveInfo(BaseModel):
    """技情報"""
    
    name: str = Field(..., description="技名")
    move_type: str = Field(..., description="技タイプ")
    category: str = Field(..., description="分類")
    power: int = Field(..., description="威力")
    accuracy: int = Field(..., description="命中率")
    pp: int = Field(..., description="PP")


class ItemInfo(BaseModel):
    """道具情報"""
    
    name: str = Field(..., description="道具名")
    fling_power: int = Field(..., description="なげつける威力")
    boost_type: Optional[str] = Field(None, description="強化タイプ")
    resist_type: Optional[str] = Field(None, description="半減タイプ")
    power_modifier: float = Field(..., description="威力補正")
    is_consumable: bool = Field(..., description="消耗品か")


class TypeEffectivenessResponse(BaseModel):
    """タイプ相性レスポンス"""
    
    attacking_type: str = Field(..., description="攻撃タイプ")
    defending_types: List[str] = Field(..., description="防御タイプ")
    effectiveness: float = Field(..., description="タイプ相性")
    description: str = Field(..., description="相性説明")


class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    
    error: str = Field(..., description="エラータイプ")
    detail: str = Field(..., description="エラー詳細")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="バリデーションエラー詳細")


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""
    
    status: str = Field(..., description="ステータス")
    service: str = Field(..., description="サービス名")
    version: str = Field(..., description="バージョン")


class APIInfoResponse(BaseModel):
    """API情報レスポンス"""
    
    message: str = Field(..., description="メッセージ")
    version: str = Field(..., description="バージョン")
    docs: str = Field(..., description="ドキュメントURL")
    health: str = Field(..., description="ヘルスチェックURL")
    api_base: str = Field(..., description="API ベースURL")


class SupportedDataResponse(BaseModel):
    """サポートデータレスポンス"""
    
    pokemon_count: int = Field(..., description="対応ポケモン数")
    move_count: int = Field(..., description="対応技数")
    item_count: int = Field(..., description="対応道具数")
    features: List[str] = Field(..., description="サポート機能")


class PokemonListResponse(BaseModel):
    """ポケモン一覧レスポンス"""
    
    pokemon: List[str] = Field(..., description="ポケモン名リスト")
    total: int = Field(..., description="総数")


class MoveListResponse(BaseModel):
    """技一覧レスポンス"""
    
    moves: List[str] = Field(..., description="技名リスト")
    total: int = Field(..., description="総数")


class ItemListResponse(BaseModel):
    """道具一覧レスポンス"""
    
    items: List[str] = Field(..., description="道具名リスト")
    total: int = Field(..., description="総数")