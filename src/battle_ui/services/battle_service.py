"""
バトルサービス - Battleクラスのラッパー
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon
from src.battle_ui.services.session_manager import BattleSession

logger = logging.getLogger(__name__)


# データファイルパス
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"

# トレーナーファイルのパス（環境変数でカンマ区切りで複数指定可能）
# 例: TRAINER_FILES="data/top_rankers/season_35.json,data/top_rankers/season_36.json"
_TRAINER_FILES_ENV = os.environ.get("TRAINER_FILES", None)
if _TRAINER_FILES_ENV:
    TRAINER_FILES = [Path(p.strip()) for p in _TRAINER_FILES_ENV.split(",")]
else:
    # デフォルト: season_35とseason_36を両方使用
    TRAINER_FILES = [
        DATA_DIR / "top_rankers" / "season_35.json",
        DATA_DIR / "top_rankers" / "season_36.json",
    ]

# プレイヤーパーティのパス（環境変数から取得）
PLAYER_PARTY_PATH = os.environ.get("PLAYER_PARTY_PATH", None)

# デバッグ用対戦相手（環境変数から取得）
DEBUG_OPPONENT = os.environ.get("DEBUG_OPPONENT", None)

# デバッグ用対戦相手の定義
DEBUG_OPPONENTS = {
    "gliscor": {
        "name": "【デバッグ】グライオン軸",
        "lead_pokemon": "グライオン",  # 必ず先発にするポケモン
        "pokemons": [
            {
                "name": "グライオン",
                "item": "どくどくだま",
                "ability": "ポイズンヒール",
                "Ttype": "みず",
                "nature": "わんぱく",
                "moves": ["じしん", "どくづき", "まもる", "みがわり"],
                "effort": [244, 0, 188, 0, 0, 76],
            },
            {
                "name": "ドヒドイデ",
                "item": "くろいヘドロ",
                "ability": "さいせいりょく",
                "Ttype": "みず",
                "nature": "ずぶとい",
                "moves": ["ねっとう", "どくどく", "じこさいせい", "くろいきり"],
                "effort": [252, 0, 252, 0, 4, 0],
            },
            {
                "name": "ラッキー",
                "item": "しんかのきせき",
                "ability": "しぜんかいふく",
                "Ttype": "ゴースト",
                "nature": "ずぶとい",
                "moves": ["タマゴうみ", "でんじは", "ちきゅうなげ", "ステルスロック"],
                "effort": [4, 0, 252, 0, 252, 0],
            },
            {
                "name": "エアームド",
                "item": "ゴツゴツメット",
                "ability": "がんじょう",
                "Ttype": "ひこう",
                "nature": "わんぱく",
                "moves": ["ボディプレス", "はねやすめ", "てっぺき", "まきびし"],
                "effort": [252, 0, 252, 0, 4, 0],
            },
            {
                "name": "ヘイラッシャ",
                "item": "たべのこし",
                "ability": "てんねん",
                "Ttype": "ドラゴン",
                "nature": "わんぱく",
                "moves": ["ウェーブタックル", "じしん", "ゆきなだれ", "ねむる"],
                "effort": [252, 0, 252, 0, 4, 0],
            },
            {
                "name": "ドオー",
                "item": "オボンのみ",
                "ability": "ちょすい",
                "Ttype": "どく",
                "nature": "しんちょう",
                "moves": ["どくづき", "じしん", "ステルスロック", "じこさいせい"],
                "effort": [252, 0, 4, 0, 252, 0],
            },
        ],
    },
    "stall": {
        "name": "【デバッグ】受けループ",
        "lead_pokemon": "ドヒドイデ",
        "pokemons": [
            {
                "name": "ドヒドイデ",
                "item": "くろいヘドロ",
                "ability": "さいせいりょく",
                "Ttype": "みず",
                "nature": "ずぶとい",
                "moves": ["ねっとう", "どくどく", "じこさいせい", "くろいきり"],
                "effort": [252, 0, 252, 0, 4, 0],
            },
            {
                "name": "ラッキー",
                "item": "しんかのきせき",
                "ability": "しぜんかいふく",
                "Ttype": "ゴースト",
                "nature": "ずぶとい",
                "moves": ["タマゴうみ", "でんじは", "ちきゅうなげ", "ステルスロック"],
                "effort": [4, 0, 252, 0, 252, 0],
            },
            {
                "name": "キョジオーン",
                "item": "たべのこし",
                "ability": "きよめのしお",
                "Ttype": "ゴースト",
                "nature": "わんぱく",
                "moves": ["しおづけ", "じこさいせい", "ステルスロック", "がんせきふうじ"],
                "effort": [252, 0, 252, 0, 4, 0],
            },
            {
                "name": "ヘイラッシャ",
                "item": "オボンのみ",
                "ability": "てんねん",
                "Ttype": "ドラゴン",
                "nature": "わんぱく",
                "moves": ["ウェーブタックル", "じしん", "ゆきなだれ", "ねむる"],
                "effort": [252, 0, 252, 0, 4, 0],
            },
            {
                "name": "アーマーガア",
                "item": "ゴツゴツメット",
                "ability": "ミラーアーマー",
                "Ttype": "ひこう",
                "nature": "わんぱく",
                "moves": ["ブレイブバード", "ボディプレス", "はねやすめ", "てっぺき"],
                "effort": [252, 0, 252, 0, 4, 0],
            },
            {
                "name": "ドオー",
                "item": "くろいヘドロ",
                "ability": "ちょすい",
                "Ttype": "どく",
                "nature": "しんちょう",
                "moves": ["どくづき", "じしん", "ステルスロック", "じこさいせい"],
                "effort": [252, 0, 4, 0, 252, 0],
            },
        ],
    },
}


def _get_party_signature(trainer: dict) -> str:
    """パーティのユニークな署名を生成（重複除去用）"""
    pokemons = trainer.get("pokemons", [])
    parts = []
    for p in pokemons[:6]:
        name = p.get("name", "")
        item = p.get("item", "")
        moves = sorted(p.get("moves", []))
        parts.append(f"{name}|{item}|{','.join(moves)}")
    return "||".join(sorted(parts))


def _load_all_trainers() -> List[Dict[str, Any]]:
    """全トレーナーファイルからデータを読み込み、重複除去"""
    all_trainers = []
    for trainer_file in TRAINER_FILES:
        if trainer_file.exists():
            with open(trainer_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_trainers.extend(data)
                logger.info(f"Loaded {len(data)} trainers from {trainer_file}")
        else:
            logger.warning(f"Trainer file not found: {trainer_file}")

    # パーティ構成ベースで重複除去
    seen_signatures = set()
    unique_trainers = []
    for trainer in all_trainers:
        sig = _get_party_signature(trainer)
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_trainers.append(trainer)

    logger.info(f"Total: {len(unique_trainers)} unique trainers from {len(TRAINER_FILES)} files")
    return unique_trainers


# キャッシュ
_TRAINERS_CACHE: Optional[List[Dict[str, Any]]] = None


def _get_trainers_data() -> List[Dict[str, Any]]:
    """トレーナーデータを取得（キャッシュ付き）"""
    global _TRAINERS_CACHE
    if _TRAINERS_CACHE is None:
        _TRAINERS_CACHE = _load_all_trainers()
    return _TRAINERS_CACHE

# チェックポイントディレクトリ（環境変数から取得）
CHECKPOINT_DIR = os.environ.get(
    "REBEL_CHECKPOINT_DIR",
    str(MODELS_DIR / "revel_full_state_selection_BERT")
)


# 特定のアイテムが判明するバトルログのキーワード
ITEM_LOG_PATTERNS = {
    "ふうせん": ["ふうせん", "で浮いている"],
    "きあいのタスキ": ["きあいのタスキ", "こらえた"],
    "たべのこし": ["たべのこし", "回復"],
    "くろいヘドロ": ["くろいヘドロ"],
    "いのちのたま": ["いのちのたま"],
    "ゴツゴツメット": ["ゴツゴツメット"],
    "とつげきチョッキ": ["とつげきチョッキ"],
    "ブーストエナジー": ["ブーストエナジー"],
    "オボンのみ": ["オボンのみ"],
    "ラムのみ": ["ラムのみ"],
    "イバンのみ": ["イバンのみ"],
    "カゴのみ": ["カゴのみ"],
    "クリアチャーム": ["クリアチャーム"],
    "おんみつマント": ["おんみつマント"],
    "レッドカード": ["レッドカード"],
    "だっしゅつボタン": ["だっしゅつボタン"],
    "だっしゅつパック": ["だっしゅつパック"],
    "じゃくてんほけん": ["じゃくてんほけん"],
    "しろいハーブ": ["しろいハーブ"],
}

# 特性が判明するバトルログのキーワード
ABILITY_LOG_PATTERNS = {
    "いかく": ["いかく", "威嚇"],
    "ひでり": ["ひでり", "日照り"],
    "あめふらし": ["あめふらし"],
    "すなおこし": ["すなおこし"],
    "ゆきふらし": ["ゆきふらし"],
    "エレキメイカー": ["エレキメイカー"],
    "グラスメイカー": ["グラスメイカー"],
    "サイコメイカー": ["サイコメイカー"],
    "ミストメイカー": ["ミストメイカー"],
    "ふゆう": ["ふゆう"],
    "がんじょう": ["がんじょう"],
    "ばけのかわ": ["ばけのかわ"],
    "ひらいしん": ["ひらいしん"],
    "よびみず": ["よびみず"],
    "もらいび": ["もらいび"],
    "ちょすい": ["ちょすい"],
    "かんそうはだ": ["かんそうはだ"],
    "そうしょく": ["そうしょく"],
    "でんきエンジン": ["でんきエンジン"],
    "ポイズンヒール": ["ポイズンヒール"],
    "かそく": ["かそく"],
    "てんねん": ["てんねん"],
    "マルチスケイル": ["マルチスケイル"],
    "ひひいろのこどう": ["ひひいろのこどう", "緋色の鼓動"],
    "おわりのだいち": ["おわりのだいち"],
    "こだいかっせい": ["こだいかっせい", "古代活性"],
    "クォークチャージ": ["クォークチャージ"],
    "おみとおし": ["おみとおし"],
    "トレース": ["トレース"],
    "かたやぶり": ["かたやぶり"],
    "すりぬけ": ["すりぬけ"],
    "てきおうりょく": ["てきおうりょく"],
    "マジックガード": ["マジックガード"],
}


class BattleService:
    """バトルロジックを提供するサービスクラス"""

    @staticmethod
    def get_checkpoints() -> List[Dict[str, Any]]:
        """利用可能なチェックポイント一覧を取得"""
        checkpoint_dir = Path(CHECKPOINT_DIR)
        if not checkpoint_dir.exists():
            return []

        checkpoints = []
        for item in sorted(checkpoint_dir.iterdir()):
            if item.is_dir() and item.name.startswith("checkpoint_iter"):
                # イテレーション番号を抽出
                try:
                    iter_num = int(item.name.replace("checkpoint_iter", ""))
                except ValueError:
                    continue

                # メタデータを読み込み
                meta_file = item / "checkpoint_meta.json"
                meta = {}
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)

                checkpoints.append({
                    "name": item.name,
                    "path": str(item),
                    "iteration": iter_num,
                    "win_rate": meta.get("win_rate"),
                    "games_played": meta.get("games_played"),
                })

        # イテレーション番号でソート
        checkpoints.sort(key=lambda x: x["iteration"])
        return checkpoints

    @staticmethod
    def get_trainers() -> List[Dict[str, Any]]:
        """トレーナー一覧を取得"""
        result = []

        # デバッグ用対戦相手を先頭に追加（環境変数が設定されている場合）
        if DEBUG_OPPONENT:
            for key, opponent in DEBUG_OPPONENTS.items():
                result.append({
                    "index": f"debug_{key}",  # 特別なインデックス
                    "name": opponent["name"],
                    "rank": 0,
                    "rating": 0,
                    "pokemon_names": [p["name"] for p in opponent["pokemons"]],
                    "is_debug": True,
                })

        # 通常のトレーナー
        trainers = _get_trainers_data()
        for i, t in enumerate(trainers[:50]):  # 上位50人まで表示
            result.append({
                "index": i,
                "name": t["name"],
                "rank": t.get("rank", i + 1),
                "rating": t.get("rating", 0),
                "pokemon_names": [p["name"] for p in t["pokemons"]],
                "is_debug": False,
            })

        return result

    @staticmethod
    def get_trainer_party(trainer_index: int | str) -> List[Dict[str, Any]]:
        """指定トレーナーのパーティを取得

        Args:
            trainer_index: トレーナーインデックス（int）またはデバッグ用ID（"debug_gliscor"等）
        """
        # デバッグ用対戦相手の場合
        if isinstance(trainer_index, str) and trainer_index.startswith("debug_"):
            debug_key = trainer_index.replace("debug_", "")
            if debug_key in DEBUG_OPPONENTS:
                return DEBUG_OPPONENTS[debug_key]["pokemons"]
            raise ValueError(f"Unknown debug opponent: {debug_key}")

        # 通常のトレーナー
        trainers = _get_trainers_data()

        if trainer_index < 0 or trainer_index >= len(trainers):
            raise ValueError(f"Invalid trainer index: {trainer_index}")

        return trainers[trainer_index]["pokemons"]

    @staticmethod
    def get_player_party() -> List[Dict[str, Any]]:
        """プレイヤーのパーティを取得"""
        # 環境変数で指定されたパーティファイルがあればそれを使用
        if PLAYER_PARTY_PATH:
            party_path = Path(PLAYER_PARTY_PATH)
            if party_path.exists():
                with open(party_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # {"pokemons": [...]} または [...] 形式に対応
                if isinstance(data, list):
                    return data
                elif "pokemons" in data:
                    return data["pokemons"]
                else:
                    raise ValueError(f"Invalid party file format: {party_path}")

        # デフォルト：2番目のトレーナーのパーティをプレイヤー用に
        trainers = _get_trainers_data()
        return trainers[1]["pokemons"]

    @staticmethod
    def create_pokemon_from_data(data: Dict[str, Any]) -> Pokemon:
        """データからPokemonインスタンスを作成"""
        pokemon = Pokemon(data["name"])
        pokemon.item = data.get("item", "")
        pokemon.ability = data.get("ability", "")
        pokemon.Ttype = data.get("Ttype", data.get("tera_type", ""))
        pokemon.nature = data.get("nature", "")
        pokemon.moves = data.get("moves", [])

        # 努力値
        effort = data.get("effort", [0, 0, 0, 0, 0, 0])
        if isinstance(effort, list) and len(effort) == 6:
            pokemon.effort = effort

        # ステータス再計算
        pokemon.update_status()

        return pokemon

    @staticmethod
    def _init_rebel_ai(session: BattleSession) -> None:
        """ReBeL AIを初期化"""
        if session.ai_mode != "rebel" or not session.checkpoint_path:
            return

        try:
            from src.battle_ui.services.rebel_ai_service import load_rebel_ai

            session.rebel_ai = load_rebel_ai(session.checkpoint_path)
            logger.info(f"ReBeL AI initialized from {session.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ReBeL AI: {e}")
            session.rebel_ai = None

    @staticmethod
    def select_pokemon(session: BattleSession, indices: List[int]) -> None:
        """選出を確定してバトルを開始"""
        if len(indices) != 3:
            raise ValueError("Must select exactly 3 pokemon")

        if any(i < 0 or i >= 6 for i in indices):
            raise ValueError("Invalid pokemon index")

        session.selected_indices = indices

        # ReBeL AIの初期化
        BattleService._init_rebel_ai(session)

        # AIの選出
        # デバッグ対戦相手の場合は固定選出（lead_pokemonを必ず先頭）
        trainer_index = session.opponent_trainer_index
        if isinstance(trainer_index, str) and trainer_index.startswith("debug_"):
            debug_key = trainer_index.replace("debug_", "")
            if debug_key in DEBUG_OPPONENTS:
                debug_config = DEBUG_OPPONENTS[debug_key]
                lead_pokemon_name = debug_config["lead_pokemon"]
                # lead_pokemonのインデックスを見つける
                lead_idx = None
                for i, p in enumerate(session.opponent_team_data):
                    if p["name"] == lead_pokemon_name:
                        lead_idx = i
                        break
                # lead_pokemonを先頭に、残りはランダムに2匹選ぶ
                if lead_idx is not None:
                    other_indices = [i for i in range(6) if i != lead_idx]
                    selected_others = random.sample(other_indices, 2)
                    ai_indices = [lead_idx] + selected_others
                    logger.info(f"Debug opponent ({debug_key}) selected: {ai_indices}, lead: {lead_pokemon_name}")
                else:
                    # lead_pokemonが見つからない場合はランダム
                    ai_indices = random.sample(range(6), 3)
                    logger.warning(f"Lead pokemon {lead_pokemon_name} not found, using random selection")
            else:
                ai_indices = random.sample(range(6), 3)
        elif session.rebel_ai is not None:
            # Selection BERTを使用
            my_team_names = [p["name"] for p in session.player_team_data]
            opp_team_names = [p["name"] for p in session.opponent_team_data]
            ai_indices = session.rebel_ai.get_selection(
                my_team_names=opp_team_names,  # AIから見た自分のチーム
                opponent_team_names=my_team_names,  # AIから見た相手（プレイヤー）
                deterministic=True,
            )
            logger.info(f"ReBeL AI selected: {ai_indices}")
        else:
            # ランダム選出
            ai_indices = random.sample(range(6), 3)
        session.ai_selected_indices = ai_indices

        # Battleオブジェクトを初期化
        battle = session.battle
        battle.reset_game()

        # UI待機モードを有効化（交代コマンドがなければ中断）
        battle.interactive_mode = True

        # プレイヤーのポケモンを設定
        player_pokemon = []
        for idx in indices:
            pokemon_data = session.player_team_data[idx]
            pokemon = BattleService.create_pokemon_from_data(pokemon_data)
            player_pokemon.append(pokemon)
        battle.selected[0] = player_pokemon

        # AIのポケモンを設定
        ai_pokemon = []
        for idx in ai_indices:
            pokemon_data = session.opponent_team_data[idx]
            pokemon = BattleService.create_pokemon_from_data(pokemon_data)
            ai_pokemon.append(pokemon)
        battle.selected[1] = ai_pokemon

        # ReBeL AIの信念状態を初期化
        if session.rebel_ai is not None:
            session.rebel_ai.reset()
            player_names = [p.name for p in player_pokemon]
            session.rebel_ai.init_belief_state(1, player_names)

        # バトル開始（0ターン目：先頭のポケモンを場に出す）
        battle.proceed(commands=[None, None])

        # AIに場に出たポケモンの情報を伝える（ふうせん等）
        if session.rebel_ai is not None:
            session.rebel_ai.observe_battle_state(battle, ai_player=1)

        # プレイヤー視点の観測情報を更新
        BattleService._update_observations_from_battle(session)

        # バトルログを累積
        BattleService._accumulate_battle_log(session)

        session.phase = "battle"

    @staticmethod
    def _get_ai_command(session: BattleSession, phase: str) -> Optional[int]:
        """AIのコマンドを取得"""
        battle = session.battle

        # ReBeL AIが利用可能な場合
        if session.rebel_ai is not None:
            try:
                if phase == "change":
                    return session.rebel_ai.get_change_command(battle, player=1)
                else:
                    return session.rebel_ai.get_battle_command(battle, player=1)
            except Exception as e:
                logger.error(f"ReBeL AI command failed: {e}")
                # フォールバック: ランダム

        # ランダム選択
        ai_commands = battle.available_commands(1, phase=phase)
        if ai_commands and ai_commands != [Battle.NO_COMMAND]:
            return random.choice(ai_commands)
        return Battle.SKIP if phase == "battle" else None

    @staticmethod
    def perform_action(
        session: BattleSession,
        action_type: str,
        action_index: int,
        with_tera: bool = False,
    ) -> None:
        """行動を実行"""
        import time
        start_time = time.time()

        battle = session.battle

        # プレイヤーのコマンドを決定
        if action_type == "move":
            player_command = action_index
            if with_tera:
                player_command += 10  # テラスタル込み
        elif action_type == "switch":
            player_command = 20 + action_index
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        # changeフェーズの場合（プレイヤーが交代先を選んだ）
        if session.phase == "change":
            # プレイヤーの交代コマンドを予約
            battle.reserved_change_commands[0].append(player_command)

            # AIの交代コマンドも予約（AIも交代が必要な場合）
            if battle.breakpoint[1]:
                ai_command = BattleService._get_ai_command(session, phase="change")
                if ai_command is not None:
                    battle.reserved_change_commands[1].append(ai_command)

            # proceedを再開（reserved_change_commandsから取り出される）
            battle.proceed()
        else:
            # 通常のバトルフェーズ
            ai_command = BattleService._get_ai_command(session, phase="battle")
            if ai_command is None:
                ai_command = Battle.SKIP

            cmds = [player_command, ai_command]
            battle.proceed(commands=cmds)  # type: ignore

            # ターン処理後にAI側の交代が必要になった場合は予約しておく
            # (interactive_modeなのでプレイヤーの交代は中断される)
            if battle.breakpoint[1] and not battle.breakpoint[0]:
                ai_change_cmd = BattleService._get_ai_command(session, phase="change")
                if ai_change_cmd is not None:
                    battle.reserved_change_commands[1].append(ai_change_cmd)
                    battle.proceed()  # AI交代を実行

        # AIにターン後の状態を観測させる（持ち物・特性・テラスタル等）
        if session.rebel_ai is not None:
            session.rebel_ai.observe_battle_state(battle, ai_player=1)

        # プレイヤー視点の観測情報を更新
        BattleService._update_observations_from_battle(session)

        # バトルログを累積
        BattleService._accumulate_battle_log(session)

        # 勝敗チェック
        winner = battle.winner()
        if winner is not None:
            session.phase = "finished"
            session.winner = winner
        # AIが必敗状態なら降参
        elif session.rebel_ai is not None and session.rebel_ai.should_surrender(battle, player=1):
            logger.info("ReBeL AI detected hopeless situation, surrendering")
            session.phase = "finished"
            session.winner = 0  # プレイヤー勝利
            session.ai_surrendered = True
        # プレイヤーの交代が必要な場合（breakpoint[0]が空でない文字列）
        elif battle.breakpoint[0]:
            session.phase = "change"
        else:
            session.phase = "battle"

        elapsed = time.time() - start_time
        if elapsed > 0.5:
            logger.warning(f"perform_action took {elapsed:.2f}s (action={action_type}, index={action_index})")

    @staticmethod
    def surrender(session: BattleSession) -> None:
        """降参"""
        session.phase = "finished"
        session.winner = 1  # AI勝利

    @staticmethod
    def _update_observations_from_battle(session: BattleSession) -> None:
        """
        バトル状態から観測情報を更新

        相手の場に出ているポケモン、およびバトルログから
        持ち物・特性が判明した情報を記録
        """
        battle = session.battle

        # 相手のポケモン（player=1）
        opponent_pokemon = battle.pokemon[1]
        if opponent_pokemon is None:
            return

        pokemon_name = opponent_pokemon.name

        # 1. バトルログから持ち物・特性を検出
        if hasattr(battle, "log") and battle.log:
            # battle.log は [player0のログ, player1のログ] のリスト
            for player_idx in range(2):
                if player_idx >= len(battle.log) or not battle.log[player_idx]:
                    continue

                for msg in battle.log[player_idx]:
                    if not isinstance(msg, str):
                        continue

                    # 持ち物の検出（相手のポケモン名がログに含まれている場合のみ）
                    for item, patterns in ITEM_LOG_PATTERNS.items():
                        if any(p in msg for p in patterns):
                            # ログ中のポケモン名を特定（相手の選出全員をチェック）
                            for p in battle.selected[1]:
                                if p and p.name in msg:
                                    session.observed_items[p.name] = item
                                    break
                            # ポケモン名が含まれない場合は記録しない
                            # （プレイヤーのポケモンの情報を相手に誤って紐づけるのを防ぐ）

                    # 特性の検出（相手のポケモン名がログに含まれている場合のみ）
                    for ability, patterns in ABILITY_LOG_PATTERNS.items():
                        if any(p in msg for p in patterns):
                            # ログ中のポケモン名を特定（相手の選出全員をチェック）
                            for p in battle.selected[1]:
                                if p and p.name in msg:
                                    session.observed_abilities[p.name] = ability
                                    break
                            # ポケモン名が含まれない場合は記録しない
                            # （プレイヤーのポケモンの特性を相手に誤って紐づけるのを防ぐ）

        # 2. 場に出ているポケモンを観測済みとして記録
        if pokemon_name not in session.observed_pokemon:
            session.observed_pokemon.append(pokemon_name)

        # 3. 場に出ているポケモンの持ち物が公開されるケース
        #    ふうせん、ブーストエナジーなど特定のアイテムは場に出た時点で判明
        visible_on_entry_items = {"ふうせん", "ブーストエナジー"}
        if opponent_pokemon.item in visible_on_entry_items:
            session.observed_items[pokemon_name] = opponent_pokemon.item

        # 4. テラスタル状態の観測（テラス使用済みなら判明）
        if opponent_pokemon.terastal and opponent_pokemon.Ttype:
            # テラスタイプも観測情報として保存可能（今回は省略、必要なら追加）
            pass

    # ログメッセージの変換マップ（分かりにくい表示を改善）
    LOG_MESSAGE_TRANSFORMS = {
        "行動スキップ": "（連続技継続中）",
        "行動不能 交代": "（交代のため行動なし）",
        "行動不能 反動": "（反動で動けない）",
    }

    # 表示から除外するログメッセージのパターン
    LOG_FILTER_PATTERNS = [
        "先手",
        "後手",
        "コマンド ",
        "HP ",  # "HP +48" や "HP -48" などの内部情報
    ]

    @staticmethod
    def _accumulate_battle_log(session: BattleSession) -> None:
        """
        バトルログを累積ログに追加

        Battle.logはターンごとにリセットされるため、
        ターン終了時にセッションの累積ログに追加する
        """
        battle = session.battle

        if not hasattr(battle, "log") or not battle.log:
            return

        turn = battle.turn

        # 現在のターンのログを追加
        for player_idx in range(2):
            if player_idx >= len(battle.log) or not battle.log[player_idx]:
                continue

            for msg in battle.log[player_idx]:
                if isinstance(msg, str):
                    # フィルタリング: 内部情報は除外
                    should_skip = False
                    for pattern in BattleService.LOG_FILTER_PATTERNS:
                        if msg.startswith(pattern):
                            should_skip = True
                            break
                    if should_skip:
                        continue

                    # メッセージ変換
                    display_msg = BattleService.LOG_MESSAGE_TRANSFORMS.get(msg, msg)

                    session.accumulated_log.append({
                        "turn": turn,
                        "player": player_idx,
                        "message": display_msg,
                    })

    @staticmethod
    def get_battle_state(
        session: BattleSession, include_ai_analysis: bool = False
    ) -> Dict[str, Any]:
        """バトル状態をテンプレート用のdictに変換"""
        battle = session.battle

        state = {
            "session_id": session.id,
            "phase": session.phase,
            "turn": battle.turn,
            "winner": session.winner,
            "player_team_data": session.player_team_data,
            "opponent_team_data": session.opponent_team_data,
            "selected_indices": session.selected_indices,
            "ai_mode": session.ai_mode,
            "ai_analysis_always_on": session.ai_analysis_always_on,
            "ai_surrendered": session.ai_surrendered,
        }

        # 選出フェーズ
        if session.phase == "selection":
            state["player_team"] = [
                BattleService._pokemon_data_to_view(p, i)
                for i, p in enumerate(session.player_team_data)
            ]
            state["opponent_preview"] = [
                {
                    "name": p["name"],
                    "index": i,
                    "arceus_type": BattleService.ARCEUS_PLATE_TYPE.get(p.get("item", ""), "ノーマル") if p["name"] == "アルセウス" else None,
                }
                for i, p in enumerate(session.opponent_team_data)
            ]
            return state

        # バトルフェーズ以降
        # 観測情報を取得
        observed_items = session.observed_items
        observed_abilities = session.observed_abilities

        if battle.pokemon[0] is not None:
            state["player_active"] = BattleService._pokemon_to_view(
                battle.pokemon[0], is_player=True
            )

        if battle.pokemon[1] is not None:
            state["opponent_active"] = BattleService._pokemon_to_view(
                battle.pokemon[1],
                is_player=False,
                observed_items=observed_items,
                observed_abilities=observed_abilities,
            )

        # ベンチポケモン（アクティブなポケモンは除外）
        state["player_bench"] = []
        active_player = battle.pokemon[0]
        for i, p in enumerate(battle.selected[0]):
            # アクティブポケモンと同一でないポケモン
            # finishedフェーズでは全ポケモンを表示（敗北したポケモンも含む）
            # それ以外ではHPが残っているポケモンのみ
            if p is not None and p is not active_player:
                if session.phase == "finished" or p.hp > 0:
                    state["player_bench"].append(
                        BattleService._pokemon_to_view(p, is_player=True, bench_index=i)
                    )

        state["opponent_bench"] = []
        active_opponent = battle.pokemon[1]
        observed_pokemon = session.observed_pokemon

        # 相手の控えポケモン数をカウント
        total_opponent_bench = 0
        for i, p in enumerate(battle.selected[1]):
            if p is not None and p is not active_opponent:
                # finishedフェーズでは全ポケモンを表示
                if session.phase == "finished":
                    total_opponent_bench += 1
                    state["opponent_bench"].append(
                        BattleService._pokemon_to_view(
                            p,
                            is_player=False,
                            bench_index=i,
                            observed_items=observed_items,
                            observed_abilities=observed_abilities,
                        )
                    )
                elif p.hp > 0:
                    # バトル中は生存しているポケモンのみカウント
                    total_opponent_bench += 1
                    # 観測済みのポケモンのみ詳細を表示
                    if p.name in observed_pokemon:
                        state["opponent_bench"].append(
                            BattleService._pokemon_to_view(
                                p,
                                is_player=False,
                                bench_index=i,
                                observed_items=observed_items,
                                observed_abilities=observed_abilities,
                            )
                        )

        # 未観測の控えポケモン数
        state["opponent_bench_total"] = total_opponent_bench
        state["opponent_bench_unknown"] = total_opponent_bench - len(state["opponent_bench"])

        # 利用可能なアクション
        if session.phase == "battle":
            state["available_actions"] = BattleService._get_available_actions(
                battle, player=0, phase="battle"
            )
        elif session.phase == "change":
            state["available_actions"] = BattleService._get_available_actions(
                battle, player=0, phase="change"
            )
            # 交代理由を追加（breakpointの値で判断）
            state["change_reason"] = battle.breakpoint[0] if battle.breakpoint[0] else "faint"
        else:
            state["available_actions"] = []

        # フィールド状態
        state["field"] = BattleService._get_field_state(battle)

        # ログ（累積ログを使用）
        state["log"] = session.accumulated_log

        # AI分析データ（ReBeL AIの場合のみ）
        if include_ai_analysis and session.rebel_ai is not None:
            state["ai_analysis"] = session.rebel_ai.get_analysis(battle, player=1)
        else:
            state["ai_analysis"] = None

        return state

    # アルセウスのプレートとタイプの対応
    ARCEUS_PLATE_TYPE = {
        "ひのたまプレート": "ほのお",
        "しずくプレート": "みず",
        "みどりのプレート": "くさ",
        "いかずちプレート": "でんき",
        "つららのプレート": "こおり",
        "こぶしのプレート": "かくとう",
        "もうどくプレート": "どく",
        "だいちのプレート": "じめん",
        "あおぞらプレート": "ひこう",
        "ふしぎのプレート": "エスパー",
        "たまむしプレート": "むし",
        "がんせきプレート": "いわ",
        "もののけプレート": "ゴースト",
        "りゅうのプレート": "ドラゴン",
        "こわもてプレート": "あく",
        "こうてつプレート": "はがね",
        "せいれいプレート": "フェアリー",
    }

    @staticmethod
    def _pokemon_data_to_view(data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """パーティデータをビュー用に変換"""
        name = data["name"]
        item = data.get("item", "")

        # アルセウスの場合、プレートによるタイプを計算
        arceus_type = None
        if name == "アルセウス":
            arceus_type = BattleService.ARCEUS_PLATE_TYPE.get(item, "ノーマル")

        return {
            "index": index,
            "name": name,
            "item": item,
            "ability": data.get("ability", ""),
            "tera_type": data.get("Ttype", data.get("tera_type", "")),
            "moves": data.get("moves", []),
            "nature": data.get("nature", ""),
            "arceus_type": arceus_type,  # アルセウスの場合のみ設定
        }

    @staticmethod
    def _pokemon_to_view(
        pokemon: Pokemon,
        is_player: bool,
        bench_index: Optional[int] = None,
        observed_items: Optional[Dict[str, str]] = None,
        observed_abilities: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """PokemonインスタンスをビューMに変換"""
        max_hp = pokemon.status[0]  # status[0] = HP
        current_hp = pokemon.hp
        hp_ratio = current_hp / max_hp if max_hp > 0 else 0

        view = {
            "name": pokemon.name,
            "hp_ratio": hp_ratio,
            "hp_percent": int(hp_ratio * 100),
            "current_hp": current_hp,
            "max_hp": max_hp,
            "types": pokemon.types,
            "status": pokemon.ailment if pokemon.ailment else None,
            "is_terastallized": pokemon.terastal,
            "tera_type": pokemon.Ttype,
            "bench_index": bench_index,
        }

        if is_player:
            view["item"] = pokemon.item
            view["ability"] = pokemon.ability
            view["moves"] = pokemon.moves
            # rank は [H, A, B, C, D, S, 命中, 回避] のリスト
            view["stat_changes"] = {
                "attack": pokemon.rank[1],
                "defense": pokemon.rank[2],
                "sp_attack": pokemon.rank[3],
                "sp_defense": pokemon.rank[4],
                "speed": pokemon.rank[5],
            }
        else:
            # 相手は観測済み情報のみ表示
            observed_items = observed_items or {}
            observed_abilities = observed_abilities or {}
            view["item"] = observed_items.get(pokemon.name, "")
            view["ability"] = observed_abilities.get(pokemon.name, "")

        return view

    @staticmethod
    def _get_available_actions(
        battle: Battle, player: int, phase: str
    ) -> List[Dict[str, Any]]:
        """利用可能なアクションを取得"""
        commands = battle.available_commands(player, phase=phase)
        actions = []

        pokemon = battle.pokemon[player]
        if pokemon is None:
            return actions

        for cmd in commands:
            if cmd == Battle.NO_COMMAND:
                continue
            elif cmd == Battle.STRUGGLE:
                actions.append({
                    "type": "move",
                    "index": 30,
                    "name": "わるあがき",
                    "disabled": False,
                })
            elif 0 <= cmd <= 3:
                # 通常技
                move_name = pokemon.moves[cmd] if cmd < len(pokemon.moves) else "???"
                # テラスタル可能かどうか: チーム全体でまだテラスタルしていない
                can_tera = battle.can_terastal(player)
                actions.append({
                    "type": "move",
                    "index": cmd,
                    "name": move_name,
                    "disabled": False,
                    "can_tera": can_tera,
                })
            elif 10 <= cmd <= 13:
                # テラスタル技（既に通常技として追加済みなのでスキップ）
                pass
            elif 20 <= cmd <= 25:
                # 交代
                bench_idx = cmd - 20
                if bench_idx < len(battle.selected[player]):
                    bench_pokemon = battle.selected[player][bench_idx]
                    if bench_pokemon and bench_pokemon.hp > 0:
                        max_hp = bench_pokemon.status[0]
                        actions.append({
                            "type": "switch",
                            "index": bench_idx,
                            "name": bench_pokemon.name,
                            "hp_ratio": bench_pokemon.hp / max_hp if max_hp > 0 else 0,
                            "disabled": False,
                        })

        return actions

    @staticmethod
    def _get_field_state(battle: Battle) -> Dict[str, Any]:
        """フィールド状態を取得"""
        condition = battle.condition

        # 天気
        weather = None
        if condition["sunny"]:
            weather = "はれ"
        elif condition["rainy"]:
            weather = "あめ"
        elif condition["snow"]:
            weather = "ゆき"
        elif condition["sandstorm"]:
            weather = "すなあらし"

        # フィールド
        terrain = None
        if condition["elecfield"]:
            terrain = "エレキフィールド"
        elif condition["glassfield"]:
            terrain = "グラスフィールド"
        elif condition["psycofield"]:
            terrain = "サイコフィールド"
        elif condition["mistfield"]:
            terrain = "ミストフィールド"

        return {
            "weather": weather,
            "terrain": terrain,
            "trick_room": condition["trickroom"] > 0,
            "gravity": condition["gravity"] > 0,
            "player_side": {
                "stealth_rock": condition["stealthrock"][0] > 0,
                "spikes": condition["makibishi"][0],
                "toxic_spikes": condition["dokubishi"][0],
                "sticky_web": condition["nebanet"][0] > 0,
                "reflect": condition["reflector"][0],
                "light_screen": condition["lightwall"][0],
                "tailwind": condition["oikaze"][0],
            },
            "opponent_side": {
                "stealth_rock": condition["stealthrock"][1] > 0,
                "spikes": condition["makibishi"][1],
                "toxic_spikes": condition["dokubishi"][1],
                "sticky_web": condition["nebanet"][1] > 0,
                "reflect": condition["reflector"][1],
                "light_screen": condition["lightwall"][1],
                "tailwind": condition["oikaze"][1],
            },
        }

    @staticmethod
    def _get_battle_log(battle: Battle) -> List[Dict[str, Any]]:
        """バトルログを取得"""
        logs = []

        # battle.log は [player0のログ, player1のログ] の形式
        # ターンごとにまとめる
        if hasattr(battle, "log") and battle.log:
            for player in range(2):
                if battle.log[player]:
                    for msg in battle.log[player]:
                        logs.append({
                            "player": player,
                            "message": msg,
                        })

        return logs
