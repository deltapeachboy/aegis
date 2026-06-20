"""
APIルーター - HTMX用のHTMLフラグメントを返すエンドポイント
"""

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

from src.battle_ui.services.session_manager import session_manager
from src.battle_ui.services.battle_service import BattleService

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/trainers", response_class=HTMLResponse)
async def get_trainers(request: Request):
    """トレーナー一覧を取得"""
    trainers = BattleService.get_trainers()
    return templates.TemplateResponse(
        "components/trainer_select.html",
        {"request": request, "trainers": trainers}
    )


@router.get("/player-party", response_class=HTMLResponse)
async def get_player_party(request: Request):
    """プレイヤーのパーティを取得"""
    party = BattleService.get_player_party()
    return templates.TemplateResponse(
        "components/party_display.html",
        {"request": request, "party": party}
    )


@router.get("/checkpoints", response_class=HTMLResponse)
async def get_checkpoints(request: Request):
    """チェックポイント一覧を取得"""
    checkpoints = BattleService.get_checkpoints()
    return templates.TemplateResponse(
        "components/checkpoint_select.html",
        {"request": request, "checkpoints": checkpoints}
    )


@router.post("/battle/create", response_class=HTMLResponse)
async def create_battle(
    request: Request,
    trainer_index: str = Form(...),  # "debug_gliscor" などのデバッグIDも受け付ける
    ai_mode: str = Form("random"),
    checkpoint_path: Optional[str] = Form(None),
    ai_analysis_always_on: Optional[str] = Form(None),
):
    """バトルセッションを作成"""
    try:
        # trainer_indexを適切な型に変換
        # デバッグ用ID（"debug_*"）はそのまま文字列、それ以外は整数に変換
        parsed_trainer_index: int | str
        if trainer_index.startswith("debug_"):
            parsed_trainer_index = trainer_index
        else:
            parsed_trainer_index = int(trainer_index)

        # プレイヤーパーティとAIトレーナーのパーティを取得
        player_party = BattleService.get_player_party()
        ai_party = BattleService.get_trainer_party(parsed_trainer_index)

        # チェックボックスの値を処理（"true" or None）
        analysis_always_on = ai_analysis_always_on == "true"

        # セッション作成
        session = session_manager.create_session(
            player_team=player_party,
            opponent_team=ai_party,
            opponent_trainer_index=parsed_trainer_index,
            ai_mode=ai_mode,
            checkpoint_path=checkpoint_path,
            ai_analysis_always_on=analysis_always_on,
        )

        # 選出画面へリダイレクト用のレスポンス
        return HTMLResponse(
            content="",
            headers={"HX-Redirect": f"/battle/{session.id}"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/battle/{session_id}/state", response_class=HTMLResponse)
async def get_battle_state(request: Request, session_id: str):
    """バトル状態を取得"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    state = BattleService.get_battle_state(session)

    if session.phase == "selection":
        return templates.TemplateResponse(
            "components/selection_phase.html",
            {"request": request, "session_id": session_id, "state": state}
        )
    elif session.phase == "battle" or session.phase == "change":
        return templates.TemplateResponse(
            "components/battle_field.html",
            {"request": request, "session_id": session_id, "state": state}
        )
    elif session.phase == "finished":
        return templates.TemplateResponse(
            "components/battle_result.html",
            {"request": request, "session_id": session_id, "state": state}
        )


@router.post("/battle/{session_id}/select", response_class=HTMLResponse)
async def select_pokemon(
    request: Request,
    session_id: str,
    pokemon_indices: str = Form(...),  # JSON形式の配列 "[0, 1, 2]"
):
    """選出を確定"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        print(f"[DEBUG] pokemon_indices raw: {pokemon_indices!r}")
        indices = json.loads(pokemon_indices)
        print(f"[DEBUG] indices parsed: {indices}")
        BattleService.select_pokemon(session, indices)

        state = BattleService.get_battle_state(session)
        return templates.TemplateResponse(
            "components/battle_field.html",
            {"request": request, "session_id": session_id, "state": state}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
            status_code=400
        )


@router.post("/battle/{session_id}/action", response_class=HTMLResponse)
async def perform_action(
    request: Request,
    session_id: str,
    action_type: str = Form(...),
    action_index: int = Form(...),
    with_tera: bool = Form(False),
):
    """行動を実行"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        BattleService.perform_action(
            session,
            action_type=action_type,
            action_index=action_index,
            with_tera=with_tera,
        )

        state = BattleService.get_battle_state(session)

        if session.phase == "finished":
            return templates.TemplateResponse(
                "components/battle_result.html",
                {"request": request, "session_id": session_id, "state": state}
            )
        elif session.phase == "change":
            return templates.TemplateResponse(
                "components/change_pokemon.html",
                {"request": request, "session_id": session_id, "state": state}
            )
        else:
            return templates.TemplateResponse(
                "components/battle_field.html",
                {"request": request, "session_id": session_id, "state": state}
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
            status_code=400
        )


@router.post("/battle/{session_id}/surrender", response_class=HTMLResponse)
async def surrender(request: Request, session_id: str):
    """降参"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    BattleService.surrender(session)
    state = BattleService.get_battle_state(session)

    return templates.TemplateResponse(
        "components/battle_result.html",
        {"request": request, "session_id": session_id, "state": state}
    )


@router.get("/battle/{session_id}/ai-analysis", response_class=HTMLResponse)
async def get_ai_analysis(request: Request, session_id: str):
    """AI分析データを取得"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    state = BattleService.get_battle_state(session, include_ai_analysis=True)

    return templates.TemplateResponse(
        "components/ai_analysis.html",
        {"request": request, "session_id": session_id, "state": state}
    )


@router.get("/battle/{session_id}/export")
async def export_battle_log(session_id: str, format: str = "json"):
    """バトルログをエクスポート"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    battle = session.battle
    state = BattleService.get_battle_state(session)

    # エクスポートデータを構築
    export_data = {
        "session_id": session_id,
        "exported_at": datetime.now().isoformat(),
        "result": {
            "winner": "player" if session.winner == 0 else "ai",
            "turn": battle.turn,
            "ai_surrendered": session.ai_surrendered,
        },
        "teams": {
            "player": {
                "selection": session.selected_indices,
                "pokemon": [
                    {
                        "name": p["name"],
                        "item": p.get("item", ""),
                        "ability": p.get("ability", ""),
                        "tera_type": p.get("Ttype", p.get("tera_type", "")),
                        "moves": p.get("moves", []),
                        "nature": p.get("nature", ""),
                    }
                    for i, p in enumerate(session.player_team_data)
                    if session.selected_indices and i in session.selected_indices
                ],
            },
            "opponent": {
                "selection": session.ai_selected_indices,
                "pokemon": [
                    {
                        "name": p["name"],
                        "item": p.get("item", ""),
                        "ability": p.get("ability", ""),
                        "tera_type": p.get("Ttype", p.get("tera_type", "")),
                        "moves": p.get("moves", []),
                        "nature": p.get("nature", ""),
                    }
                    for i, p in enumerate(session.opponent_team_data)
                    if session.ai_selected_indices and i in session.ai_selected_indices
                ],
            },
        },
        "final_state": {
            "player_pokemon": [],
            "opponent_pokemon": [],
        },
        "battle_log": [],
    }

    # 最終状態のポケモン情報
    if state.get("player_active"):
        export_data["final_state"]["player_pokemon"].append({
            "name": state["player_active"]["name"],
            "hp_percent": state["player_active"]["hp_percent"],
            "status": "active",
        })
    for p in state.get("player_bench", []):
        export_data["final_state"]["player_pokemon"].append({
            "name": p["name"],
            "hp_percent": p["hp_percent"],
            "status": "bench",
        })

    if state.get("opponent_active"):
        export_data["final_state"]["opponent_pokemon"].append({
            "name": state["opponent_active"]["name"],
            "hp_percent": state["opponent_active"]["hp_percent"],
            "status": "active",
        })
    for p in state.get("opponent_bench", []):
        export_data["final_state"]["opponent_pokemon"].append({
            "name": p["name"],
            "hp_percent": p["hp_percent"],
            "status": "bench",
        })

    # バトルログ（累積ログを使用）
    for log_entry in session.accumulated_log:
        export_data["battle_log"].append({
            "turn": log_entry.get("turn", 0),
            "player": "player" if log_entry["player"] == 0 else "ai",
            "message": log_entry["message"],
        })

    # フォーマットに応じて出力
    if format == "txt":
        # テキスト形式
        lines = []
        lines.append("=" * 60)
        lines.append("バトルログ")
        lines.append("=" * 60)
        lines.append(f"日時: {export_data['exported_at']}")
        lines.append(f"結果: {'プレイヤー勝利' if session.winner == 0 else 'AI勝利'}")
        lines.append(f"ターン数: {battle.turn}")
        if session.ai_surrendered:
            lines.append("※AIが降参")
        lines.append("")

        lines.append("-" * 40)
        lines.append("【プレイヤーチーム】")
        for p in export_data["teams"]["player"]["pokemon"]:
            lines.append(f"  {p['name']} @ {p['item']}")
            lines.append(f"    特性: {p['ability']}, テラス: {p['tera_type']}")
            lines.append(f"    技: {', '.join(p['moves'])}")
        lines.append("")

        lines.append("-" * 40)
        lines.append("【相手チーム】")
        for p in export_data["teams"]["opponent"]["pokemon"]:
            lines.append(f"  {p['name']} @ {p['item']}")
            lines.append(f"    特性: {p['ability']}, テラス: {p['tera_type']}")
            lines.append(f"    技: {', '.join(p['moves'])}")
        lines.append("")

        lines.append("-" * 40)
        lines.append("【最終状態】")
        lines.append("プレイヤー:")
        for p in export_data["final_state"]["player_pokemon"]:
            status = "場" if p["status"] == "active" else "控え"
            lines.append(f"  {p['name']}: HP {p['hp_percent']}% ({status})")
        lines.append("相手:")
        for p in export_data["final_state"]["opponent_pokemon"]:
            status = "場" if p["status"] == "active" else "控え"
            lines.append(f"  {p['name']}: HP {p['hp_percent']}% ({status})")
        lines.append("")

        if export_data["battle_log"]:
            lines.append("-" * 40)
            lines.append("【バトルログ】")
            current_turn = -1
            for log in export_data["battle_log"]:
                turn = log.get("turn", 0)
                if turn != current_turn:
                    lines.append(f"\n  --- ターン {turn} ---")
                    current_turn = turn
                prefix = "[自]" if log["player"] == "player" else "[相]"
                lines.append(f"  {prefix} {log['message']}")

        lines.append("=" * 60)

        content = "\n".join(lines)
        filename = f"battle_log_{session_id[:8]}.txt"

        return Response(
            content=content,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    else:
        # JSON形式（デフォルト）
        filename = f"battle_log_{session_id[:8]}.json"
        return Response(
            content=json.dumps(export_data, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
