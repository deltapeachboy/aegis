import sys
import os
import glob
import json
import time
import signal  # 🌟 追加：Mac環境専用の割り込み制御
import traceback
from copy import deepcopy
from pokepy.pokemon import Pokemon
from pokepy.battle import Battle
from aegis_bot import AegisAnalyzer

# =========================================================================
# 🌟 性格補正 ＆ 進行パッチ
# =========================================================================
if hasattr(Pokemon, 'nature_corrections'):
    flat_correction = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for flat_nature in ["まじめ", "がんばりや", "すなお", "てれや", "きまぐれ"]:
        if flat_nature not in Pokemon.nature_corrections:
            Pokemon.nature_corrections[flat_nature] = flat_correction

original_proceed = Battle.proceed


def patched_proceed(self, commands=None):
    restorations = []
    for player in range(2):
        p = self.pokemon[player]
        if p:
            current_moves = p.moves
            cmds = commands if commands is not None else self.command
            if cmds:
                cmd = cmds[player]
                if cmd is not None and cmd < 20:
                    move_idx = cmd % 10
                    if move_idx >= len(p.moves) or len(p.moves) < 4:
                        original_moves = list(p.moves)
                        while len(p.moves) < 4:
                            p.moves.append("テラバースト")
                        restorations.append((p, original_moves))
    try:
        res = original_proceed(self, commands=commands)
    finally:
        for p, original_moves in restorations:
            p.moves = original_moves
    return res


Battle.proceed = patched_proceed
AegisAnalyzer.update_beliefs_by_implicit_observations = lambda self: None


# =========================================================================
# 🌟 Mac専用 タイムアウト割り込みハンドラ
# =========================================================================
class SimulationTimeoutException(Exception):
    """シミュレーションの無限ループを強制遮断するための例外クラス"""
    pass


def timeout_handler(signum, frame):
    raise SimulationTimeoutException("Aegis: proceed infinite loop detected")


# アラームシグナルを登録
signal.signal(signal.SIGALRM, timeout_handler)


# =========================================================================


def run_debug_verification():
    print("\n==================================================")
    print("  🔍 Aegis 1～192 世代リプレイ無限ループ検出・検証システム 起動")
    print("==================================================")

    log_files = glob.glob(os.path.join("log", "selfplay_gen_*.jsonl"))
    if not log_files:
        print("❌ log/ 配下に対戦ログファイルが見つかりません。")
        return

    log_files.sort(key=lambda x: int(x.split("_gen_")[-1].split(".")[0]))
    print(f"📂 検出された全ログファイル数: {len(log_files)} 個")

    analyzer = AegisAnalyzer()

    for file_path in log_files:
        gen_num = file_path.split("_gen_")[-1].split(".")[0]
        print(f"\n──────────────────────────────────────────────────")
        print(f"📂 第 {gen_num} 世代のログファイル検証中... ({os.path.basename(file_path)})")
        print(f"──────────────────────────────────────────────────")

        match_count = 0
        with open(file_path, "r", encoding="utf-8") as f_in:
            for line_idx, line in enumerate(f_in, 1):
                if not line.strip():
                    continue
                try:
                    match_data = json.loads(line)
                    match_count += 1

                    battle = Battle(seed=match_data["seed"])
                    teams_data = match_data["teams"]
                    selections = match_data["selections"]

                    for pl in [0, 1]:
                        restored_team = []
                        for p_data in teams_data[pl]:
                            p = Pokemon()
                            p.name = p_data["name"]
                            p.item = p_data.get("item", "")
                            p.moves = p_data.get("moves", [])
                            p.nature = p_data.get("nature", "いじっぱり")
                            p.ability = p_data.get("ability", "とくせいなし")
                            p.update_status()
                            restored_team.append(p)
                        battle.selected[pl] = [deepcopy(restored_team[idx]) for idx in selections[pl]]

                    battle.turn = 0
                    for player in range(2):
                        battle.change_pokemon(player, idx=0, landing=False)
                    for player in battle.speed_order:
                        battle.land(player)

                    history = match_data.get("history", [])

                    # ⏳ 1試合最大 3 秒の制限時間をセット（シミュレータ内の無限ループを強制破壊します）
                    signal.alarm(3)

                    try:
                        for step_idx, step in enumerate(history):
                            turn_num = step.get("turn", step_idx + 1)
                            cmds = step.get("commands")

                            # 進行
                            battle.proceed(commands=cmds)

                        # 正常に終了したらアラームを解除
                        signal.alarm(0)

                    except SimulationTimeoutException:
                        print(
                            f"\n🚨 [無限ループ自動スキップ] 第 {gen_num} 世代 - 試合 {match_count} で進行がフリーズしました。")
                        print(f"   ┗ 対象のシード値: {match_data['seed']}")
                        print(f"   ┗ ターン数: {battle.turn} | 直前コマンド: {cmds}")
                        print("   ※ 処理を強制スキップして、安全に次の対戦に進みます。")
                        continue
                    finally:
                        # 確実にアラームを解除
                        signal.alarm(0)

                    # 10試合完了ごとに正常通知
                    if match_count % 10 == 0:
                        print(
                            f"   - [試合 {match_count}/40] 展開完了... (シード: {match_data['seed']}, ターン数: {battle.turn})")

                except Exception as e:
                    print(f"\n❌ エラー検出: 第 {gen_num} 世代 - 試合 {match_count} (行: {line_idx})")
                    print(f"   ┗ エラー内容: {e}")
                    traceback.print_exc()
                    print("   ※ エラーの試合をスキップし、処理を継続します。")
                    continue

    print("\n🏁 すべての世代ログの検証が完了しました。")


if __name__ == "__main__":
    Pokemon.init(season=22)
    run_debug_verification()