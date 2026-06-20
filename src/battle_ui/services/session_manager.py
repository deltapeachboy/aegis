"""
セッション管理 - バトルセッションの作成・取得・削除
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime

from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon


@dataclass
class BattleSession:
    """バトルセッション"""
    id: str
    battle: Battle
    player_team_data: List[Dict[str, Any]]  # 元のパーティデータ
    opponent_team_data: List[Dict[str, Any]]
    opponent_trainer_index: int | str  # intまたはデバッグ用ID（"debug_gliscor"等）
    phase: str = "selection"  # selection, battle, change, finished
    selected_indices: Optional[List[int]] = None
    ai_selected_indices: Optional[List[int]] = None
    created_at: datetime = field(default_factory=datetime.now)
    winner: Optional[int] = None
    accumulated_log: List[Dict[str, Any]] = field(default_factory=list)  # バトル全体の累積ログ
    ai_mode: str = "random"  # random, rebel
    checkpoint_path: Optional[str] = None
    rebel_ai: Optional[Any] = None  # ReBeL AI instance
    ai_analysis_always_on: bool = False  # AI分析を常に表示
    ai_surrendered: bool = False  # AIが降参したか
    # プレイヤー視点の観測情報（相手ポケモンの観測済み持ち物・特性）
    observed_items: Dict[str, str] = field(default_factory=dict)  # pokemon_name -> item
    observed_abilities: Dict[str, str] = field(default_factory=dict)  # pokemon_name -> ability
    observed_pokemon: List[str] = field(default_factory=list)  # 観測済みの相手ポケモン名リスト


class SessionManager:
    """セッション管理クラス"""

    def __init__(self):
        self._sessions: Dict[str, BattleSession] = {}

    def create_session(
        self,
        player_team: List[Dict[str, Any]],
        opponent_team: List[Dict[str, Any]],
        opponent_trainer_index: int | str,  # intまたはデバッグ用ID（"debug_gliscor"等）
        ai_mode: str = "random",
        checkpoint_path: Optional[str] = None,
        ai_analysis_always_on: bool = False,
    ) -> BattleSession:
        """新しいバトルセッションを作成"""
        session_id = str(uuid.uuid4())

        # Battleオブジェクトを作成（選出フェーズから始めるため、まだポケモンは設定しない）
        battle = Battle()

        session = BattleSession(
            id=session_id,
            battle=battle,
            player_team_data=player_team,
            opponent_team_data=opponent_team,
            opponent_trainer_index=opponent_trainer_index,
            phase="selection",
            ai_mode=ai_mode,
            checkpoint_path=checkpoint_path,
            ai_analysis_always_on=ai_analysis_always_on,
        )

        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[BattleSession]:
        """セッションを取得"""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """セッションを削除"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """古いセッションを削除"""
        now = datetime.now()
        to_delete = []

        for session_id, session in self._sessions.items():
            age = (now - session.created_at).total_seconds() / 3600
            if age > max_age_hours:
                to_delete.append(session_id)

        for session_id in to_delete:
            del self._sessions[session_id]

        return len(to_delete)


# グローバルインスタンス
session_manager = SessionManager()
