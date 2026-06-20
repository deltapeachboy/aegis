import sys
import types
from typing import Optional, List, Dict, Any, Set, Tuple

# =========================================================================
# 【Aegis Namespace Bridge for Web UI】
# Webサーバー(uvicorn)起動時にも古いシミュレータ名空間を pokepy へ自動リダイレクトするパッチ
# =========================================================================
# A. 空のプレースホルダーモジュールを先に sys.modules に登録する
sys.modules['src.pokemon_battle_sim'] = types.ModuleType('src.pokemon_battle_sim')
sys.modules['src.pokemon_battle_sim.pokemon'] = types.ModuleType('src.pokemon_battle_sim.pokemon')
sys.modules['src.pokemon_battle_sim.battle'] = types.ModuleType('src.pokemon_battle_sim.battle')
sys.modules['src.pokemon_battle_sim.damage'] = types.ModuleType('src.pokemon_battle_sim.damage')

# B. 依存関係を持たない純粋な便利関数モジュール(utils)を最優先でロードして結合
import pokepy.utils as utils_module
sys.modules['src.pokemon_battle_sim.utils'] = utils_module

# C. ポケモンとバトルの実体モジュールをロード
import pokepy.pokemon as pokemon_module
import pokepy.battle as battle_module

# D. プレースホルダーの内部を実体コードで同期・埋める
sys.modules['src.pokemon_battle_sim'].__dict__.update(pokemon_module.__dict__)
sys.modules['src.pokemon_battle_sim.pokemon'].__dict__.update(pokemon_module.__dict__)
sys.modules['src.pokemon_battle_sim.battle'].__dict__.update(battle_module.__dict__)
sys.modules['src.pokemon_battle_sim.damage'].__dict__.update(pokemon_module.__dict__)
# =========================================================================

# =========================================================================
# 2. 本来の FastAPI 起動処理（ここから元のコード）
# =========================================================================
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# このインポートが正常に機能するようになります
from src.battle_ui.routers import api, pages

app = FastAPI(title="Project Aegis Battle UI")

# 静的ファイルのマウント
app.mount("/static", StaticFiles(directory="src/battle_ui/static"), name="static")

# テンプレートの読み込み
templates = Jinja2Templates(directory="src/battle_ui/templates")

# ルーターの登録
app.include_router(pages.router)
app.include_router(api.router)