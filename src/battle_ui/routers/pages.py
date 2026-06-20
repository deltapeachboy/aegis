from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ルーターの初期化
router = APIRouter()

# テンプレートディレクトリの設定
templates = Jinja2Templates(directory="src/battle_ui/templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    【Aegis ビジュアルダッシュボード メインページ】

    新旧の FastAPI / Starlette のシグネチャ変更（引数の順序ズレ）による
    Jinja2の 'unhashable type: dict' エラーを、キーワード引数指定によって完全回避します。
    """
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@router.get("/setup", response_class=HTMLResponse)
async def setup(request: Request):
    """セットアップ画面"""
    return templates.TemplateResponse(
        request=request,
        name="pages/setup.html",
        context={}
    )


@router.get("/battle", response_class=HTMLResponse)
async def battle(request: Request):
    """メインバトル画面"""
    return templates.TemplateResponse(
        request=request,
        name="pages/battle.html",
        context={}
    )