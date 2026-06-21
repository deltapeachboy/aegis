"""
ポケモンSVダメージ計算機API - メインアプリケーション

FastAPIを使用したREST APIサーバー
高精度なダメージ計算エンジンをWeb APIとして提供
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.damage_calculator_api.routers import damage, info, pokemon
from src.damage_calculator_api.utils.data_loader import get_data_loader

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    # 起動時: データ事前読み込み
    logger.info("Starting Pokémon SV Damage Calculator API...")
    try:
        data_loader = get_data_loader()
        logger.info(
            f"Loaded {len(data_loader.pokemon_data)} Pokémon and {len(data_loader.move_data)} moves"
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    yield

    # 終了時のクリーンアップ
    logger.info("Shutting down Pokémon SV Damage Calculator API...")


# FastAPIアプリケーション作成
app = FastAPI(
    title="Pokémon SV Damage Calculator API",
    description="""
高精度なポケモンSV対応ダメージ計算API

## 機能
- 完全なダメージ計算（16段階乱数、全補正対応）
- タイプ相性・STAB計算（テラスタル対応）
- 特性・道具・天気・状態異常効果
- 技比較・確定数計算・KO確率計算
- 688技・733ポケモン対応

## 使用例
```python
import requests

# 基本的なダメージ計算
response = requests.post("/api/v1/damage/calculate", json={
    "attacker": {
        "species": "ピカチュウ",
        "level": 50,
        "stats": {"sp_attack": 137},
        "ability": "せいでんき"
    },
    "defender": {
        "species": "ギャラドス",
        "level": 50,
        "stats": {"sp_defense": 120}
    },
    "move": {
        "name": "10まんボルト"
    }
})
```
""",
    version="1.0.0",
    contact={
        "name": "Pokémon Battle Simulator Team",
        "url": "https://github.com/your-repo/pokemon-damage-calculator",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# グローバル例外ハンドラー
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """バリデーションエラーハンドラー"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": "リクエストデータが無効です",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """値エラーハンドラー"""
    return JSONResponse(
        status_code=400, content={"error": "Invalid Value", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """一般例外ハンドラー"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "サーバー内部エラーが発生しました",
        },
    )


# ルーター登録
app.include_router(damage.router, prefix="/api/v1/damage", tags=["damage"])
app.include_router(pokemon.router, prefix="/api/v1/pokemon", tags=["pokemon"])
app.include_router(info.router, prefix="/api/v1/info", tags=["info"])


# ヘルスチェック
@app.get("/health", tags=["health"])
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "service": "Pokémon SV Damage Calculator API",
        "version": "1.0.0",
    }


# ルートエンドポイント
@app.get("/", tags=["root"])
async def root():
    """ルートエンドポイント - API情報を返す"""
    return {
        "message": "Pokémon SV Damage Calculator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api_base": "/api/v1",
    }


# 開発用サーバー起動
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
