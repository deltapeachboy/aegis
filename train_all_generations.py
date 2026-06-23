import os
import glob
import sys
import torch
import warnings
import random
import signal
from pokepy.pokemon import Pokemon
from pokepy.battle import Battle
from train_value_network import train_model

# 超高速化パッチ用インポート
from aegis_bot import AegisAnalyzer


# =========================================================================
# 🌟 Mac専用 タイムアウト割り込みハンドラ
# =========================================================================
def pretrain_timeout_handler(signum, frame):
    raise RuntimeError("Aegis: pretrain proceed infinite loop detected")


signal.signal(signal.SIGALRM, pretrain_timeout_handler)


def run_all_pretraining():
    print("\n==================================================")
    print("  🔄 Aegis 1～192 世代一括プレトレーニング起動")
    print("==================================================")

    log_files = glob.glob(os.path.join("log", "selfplay_gen_*.jsonl"))
    if not log_files:
        print("❌ log/ 配下に対戦ログファイルが見つかりません。")
        return

    # 世代番号順にファイルをソート
    log_files.sort(key=lambda x: int(x.split("_gen_")[-1].split(".")[0]))
    print(f"📂 検出された全ログファイル数: {len(log_files)} 個")

    # すべての世代ログを1つの大きなマージファイルにする
    merged_log_path = "log/all_generations_merged.jsonl"
    print(f"💾 すべてのログを '{merged_log_path}' へ一括統合しています...")

    merged_count = 0
    with open(merged_log_path, "w", encoding="utf-8") as f_out:
        for file_path in log_files:
            with open(file_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if line.strip():
                        f_out.write(line.strip() + "\n")
                        merged_count += 1

    print(f"✅ 統合完了。総対戦レコード（ゲーム数）: {merged_count} ライン")
    print("🚀 価値予測AI（PyTorch）の一括オフライン・トレーニングを開始します（CPU安全学習）...")

    # 一括マージデータを用いて、エポック数「15」でじっくりと全展開をディープラーニング
    train_model(
        log_path=merged_log_path,
        epochs=15,
        batch_size=64,
        lr=1e-4
    )

    if os.path.exists(merged_log_path):
        os.remove(merged_log_path)

    print("\n==================================================")
    print("  🏆 一括プレトレーニングが完了しました！")
    print("  更新されたモデル: 'src/rebel/value_network.pth'")
    print("==================================================")


if __name__ == "__main__":
    # 1. マスターデータの初期化を確実に最優先で実行
    Pokemon.init(season=22)

    # 2. 【Aegis 性格パッチ】性格定義に無補正性格が無い場合の防壁
    if hasattr(Pokemon, 'nature_corrections'):
        flat_correction = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        for flat_nature in ["まじめ", "がんばりや", "すなお", "てれや", "きまぐれ"]:
            if flat_nature not in Pokemon.nature_corrections:
                Pokemon.nature_corrections[flat_nature] = flat_correction

    # 3. 【Aegis 属性パッチ】Pokemon クラスに不足している実数値プロパティを動的追加
    if hasattr(Pokemon, 'status'):
        if not hasattr(Pokemon, 'attack'):
            Pokemon.attack = property(lambda self: self.status[1])
        if not hasattr(Pokemon, 'defense'):
            Pokemon.defense = property(lambda self: self.status[2])
        if not hasattr(Pokemon, 'sp_attack'):
            Pokemon.sp_attack = property(lambda self: self.status[3])
        if not hasattr(Pokemon, 'sp_defense'):
            Pokemon.sp_defense = property(lambda self: self.status[4])
        if not hasattr(Pokemon, 'speed'):
            Pokemon.speed = property(lambda self: self.status[5])

    # 🌟 4. 【Battle.proceed 技インデックス限界突破＆空スロット完全防止パッチ（超堅牢版）】
    original_proceed = Battle.proceed


    def patched_proceed(self, commands=None):
        cmds = commands if commands is not None else self.command
        if cmds:
            cmds = list(cmds)
            for player in range(2):
                p = self.pokemon[player]
                if p and p.hp > 0:
                    # [無限ループ回避] ゲッターコピーによるフリーズを防ぐため、一度ローカル変数でリストを拡張する
                    temp_moves = list(p.moves) if hasattr(p, 'moves') and p.moves else []

                    if not temp_moves:
                        temp_moves = ["わるあがき"]

                    is_modified = False
                    if len(temp_moves) < 4:
                        pokemon_name = p.name
                        # 図鑑データ（本物の習得技）から安全に再サンプリング
                        learnable_moves = Pokemon.learnsets.get(pokemon_name, ["わるあがき"])
                        extra_pool = [m for m in learnable_moves if m not in temp_moves]
                        needed = 4 - len(temp_moves)

                        if extra_pool:
                            extra_moves = random.sample(extra_pool, min(needed, len(extra_pool)))
                            temp_moves.extend(extra_moves)
                            is_modified = True
                        while len(temp_moves) < 4:
                            temp_moves.append("わるあがき")
                            is_modified = True

                    if is_modified:
                        try:
                            p.moves = temp_moves
                        except AttributeError:
                            # プロパティにセッターが無い場合、名前修飾されたプライベート変数へ直接書き込みを行う
                            p._Pokemon__moves = temp_moves
                        p.update_status()

                    cmd = cmds[player]
                    if cmd is not None:
                        if cmd not in range(20, 26):
                            move_idx = cmd % 10
                            if move_idx >= len(p.moves):
                                fallback_idx = 0
                                base_offset = (cmd // 10) * 10
                                cmds[player] = base_offset + fallback_idx
            self.command = cmds

        # ⏳ 3秒のタイムリミットをセット（進行不能時のみ強制スキップ）
        signal.alarm(3)
        try:
            return original_proceed(self, commands=cmds)
        except AttributeError as ae:
            signal.alarm(0)
            raise RuntimeError(f"Aegis: Sim AttributeError: {ae}")
        except IndexError as ie:
            signal.alarm(0)
            print("\n🚨 [Aegis IndexError Debug Logger]")
            print(f"   - ターン: {self.turn}")
            for player in range(2):
                p = self.pokemon[player]
                cmd = self.command[player] if self.command else None
                if p:
                    print(
                        f"   - プレイヤー {player}: 名: '{p.name}' | 持ち物: '{p.item}' | 特性: '{p.ability}' | 技: {p.moves} | コマンド: {cmd}")
            print("==================================================\n")
            raise ie
        finally:
            signal.alarm(0)


    Battle.proceed = patched_proceed
    print("ℹ️ [Aegis Patch] 超堅牢版・本物技自動再サンプリングパッチ を適用しました。")

    # 6. 【Aegis 超高速化パッチ】一括再生中は重い「ベイズ型看破処理」を完全にバイパス
    AegisAnalyzer.update_beliefs_by_implicit_observations = lambda self: None
    print("ℹ️ [Aegis Patch] プレトレーニング用の超高速化バイパスを適用しました。")

    # 7. メイン関数の実行
    run_all_pretraining()