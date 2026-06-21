"""
API情報・ユーティリティ関連のAPIエンドポイント
"""

from fastapi import APIRouter, HTTPException

from src.damage_calculator_api.schemas.responses import (
    APIInfoResponse,
    HealthResponse,
    SupportedDataResponse,
)
from src.damage_calculator_api.utils.data_loader import get_data_loader

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    ヘルスチェック

    APIの動作状況を確認する
    """
    try:
        # データローダーが正常に動作するか確認
        data_loader = get_data_loader()

        return HealthResponse(
            status="healthy",
            service="Pokémon SV Damage Calculator API",
            version="1.0.0",
        )

    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"サービスが利用できません: {str(e)}"
        )


@router.get("/", response_model=APIInfoResponse)
async def get_api_info():
    """
    API基本情報を取得

    APIの概要と利用可能なエンドポイント情報を返す
    """
    return APIInfoResponse(
        message="Pokémon SV Damage Calculator API",
        version="1.0.0",
        docs="/docs",
        health="/health",
        api_base="/api/v1",
    )


@router.get("/supported-data", response_model=SupportedDataResponse)
async def get_supported_data():
    """
    サポートデータ情報を取得

    対応ポケモン・技・道具数と利用可能機能を返す
    """
    try:
        data_loader = get_data_loader()

        # サポート機能リスト
        features = [
            "16段階乱数ダメージ計算",
            "タイプ相性計算（テラスタル対応）",
            "STAB（タイプ一致）計算",
            "特性効果計算",
            "道具効果計算",
            "天気・フィールド効果",
            "状態異常・能力ランク補正",
            "確定数・KO確率計算",
            "技比較機能",
            "ダメージ範囲分析",
            "ポケモン・技・道具情報検索",
        ]

        return SupportedDataResponse(
            pokemon_count=len(data_loader.pokemon_data),
            move_count=len(data_loader.move_data),
            item_count=len(data_loader.item_data),
            features=features,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"サポートデータ取得中にエラーが発生しました: {str(e)}",
        )
