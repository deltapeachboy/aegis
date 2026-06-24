"""
Aegis 全世代一括プレトレーニング制御スクリプト

過去の全世代の対戦ログをマージし、
超高速化ディープコピーパッチおよびWindowsセーフガードを適用した上で、
バリューネットワークに環境全体の大局観を一括学習（プレトレーニング）させる。
"""

import os
import glob
import sys
import torch
import warnings
import random
import signal
import types
from copy import deepcopy

# =========================================================================
# 0. 【Aegis Namespace Bridge & Redirects】
# =========================================================================
import builtins
import io
import json

_original_open = builtins.open

def patched_open(file, *args, **kwargs):
    if isinstance(file, str) and "learnset.json" in file:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        custom_path = os.path.join(base_dir, "battle_data", "mb_learnset.json")
        if os.path.exists(custom_path):
            with _original_open(custom_path, "r", encoding="utf-8") as f:
                learnset = json.load(f)
            base_key = "ギルガルド" if "ギルガルド" in learnset else "ギルガルド(シールド)"
            if base_key in learnset:
                learnset["ギルガルド(ブレード)"] = learnset[base_key]
                learnset["ギルガルド(シールド)"] = learnset[base_key]
            patched_json_str = json.dumps(learnset, ensure_ascii=False)
            return io.StringIO(patched_json_str)
    return _original_open(file, *args, **kwargs)

builtins.open = patched_open

sys.modules['src.pokemon_battle_sim'] = types.ModuleType('src.pokemon_battle_sim')
sys.modules['src.pokemon_battle_sim.pokemon'] = types.ModuleType('src.pokemon_battle_sim.pokemon')
sys.modules['src.pokemon_battle_sim.battle'] = types.ModuleType('src.pokemon_battle_sim.battle')
sys.modules['src.pokemon_battle_sim.damage'] = types.ModuleType('src.pokemon_battle_sim.damage')

import pokepy.utils as utils_module
sys.modules['src.pokemon_battle_sim.utils'] = utils_module

import pokepy.pokemon as pokemon_module
import pokepy.battle as battle_module
sys.modules['src.pokemon_battle_sim'].__dict__.update(pokemon_module.__dict__)
sys.modules['src.pokemon_battle_sim.pokemon'].__dict__.update(pokemon_module.__dict__)
sys.modules['src.pokemon_battle_sim.battle'].__dict__.update(battle_module.__dict__)
sys.modules['src.pokemon_battle_sim.damage'].__dict__.update(pokemon_module.__dict__)

from pokepy.pokemon import Pokemon
from pokepy.battle import Battle
from train_value_network import train_model
from aegis_bot import AegisAnalyzer

# =========================================================================
# 🌟 [Aegis Patch] プラットフォーム互換アラームセーフティ
# =========================================================================
HAS_ALARM = hasattr(signal, "alarm")

def safe_set_alarm(seconds: int):
    if HAS_ALARM:
        signal.alarm(seconds)

def pretrain_timeout_handler(signum, frame):
    raise RuntimeError("Aegis: pretrain proceed infinite loop detected")

if HAS_ALARM:
    signal.signal(signal.SIGALRM, pretrain_timeout_handler)


# =========================================================================
# 🏆 メインの一括プレトレーニング処理
# =========================================================================
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

    # =========================================================================
    # 🚀 [Aegis Patch] 一括プレトレーニングに必須の超高速ディープコピーパッチ
    # =========================================================================
    def patched_battle_deepcopy(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        cls = self.__class__
        new_battle = cls.__new__(cls)
        memo[id(self)] = new_battle

        for k, v in self.__dict__.items():
            if k in ['solver', 'value_network', 'nn', 'model', 'w2v_model', 'analyzer', 'builder', 'beliefs', 'pbs']:
                continue
            if v.__class__.__name__ in ['ReBeLValueNetwork', 'CFRSolver', 'AegisTeamBuilder', 'AegisAnalyzer', 'Word2Vec', 'PokemonBeliefState', 'PublicBeliefState']:
                continue
            if hasattr(v, '__class__') and ('torch' in v.__class__.__module__ or 'rebel' in v.__class__.__module__):
                continue
            if isinstance(v, (types.ModuleType, types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
                continue

            try:
                setattr(new_battle, k, deepcopy(v, memo))
            except Exception:
                setattr(new_battle, k, v)

        if hasattr(new_battle, 'pokemon') and new_battle.pokemon:
            for p in new_battle.pokemon:
                if p:
                    for attr in ['battle', '_battle', 'current_battle']:
                        if hasattr(p, attr):
                            setattr(p, attr, new_battle)

        if hasattr(new_battle, 'selected') and new_battle.selected:
            for side in new_battle.selected:
                if side:
                    for p in side:
                        if p:
                            for attr in ['battle', '_battle', 'current_battle']:
                                if hasattr(p, attr):
                                    setattr(p, attr, new_battle)

        return new_battle

    def patched_pokemon_deepcopy(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        cls = self.__class__
        new_poke = cls.__new__(cls)
        memo[id(self)] = new_poke

        avoid_keys = {'battle', '_battle', 'current_battle'}
        for k, v in self.__dict__.items():
            if k in avoid_keys:
                continue

            if isinstance(v, (str, int, float, bool, type(None))):
                new_poke.__dict__[k] = v
            elif isinstance(v, list):
                new_poke.__dict__[k] = [
                    deepcopy(item, memo) if not isinstance(item, (str, int, float, bool, type(None))) else item
                    for item in v
                ]
            elif isinstance(v, dict):
                new_dict = {}
                for dk, dv in v.items():
                    new_dk = deepcopy(dk, memo) if not isinstance(dk, (str, int, float, bool, type(None))) else dk
                    new_dv = deepcopy(dv, memo) if not isinstance(dv, (str, int, float, bool, type(None))) else dv
                    new_dict[new_dk] = new_dv
                new_poke.__dict__[k] = new_dict
            elif isinstance(v, set):
                new_poke.__dict__[k] = {
                    deepcopy(item, memo) if not isinstance(item, (str, int, float, bool, type(None))) else item
                    for item in v
                }
            else:
                try:
                    new_poke.__dict__[k] = deepcopy(v, memo)
                except Exception:
                    new_poke.__dict__[k] = v

        return new_poke

    Battle.__deepcopy__ = patched_battle_deepcopy
    Pokemon.__deepcopy__ = patched_pokemon_deepcopy
    print("ℹ️ [Aegis Patch] 一括学習用・超高速ディープコピーパッチをインジェクションしました。")

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
                    temp_moves = list(p.moves) if hasattr(p, 'moves') and p.moves else []
                    if not temp_moves:
                        temp_moves = ["わるあがき"]

                    is_modified = False
                    if len(temp_moves) < 4:
                        pokemon_name = p.name
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

        safe_set_alarm(3)
        try:
            return original_proceed(self, commands=cmds)
        except AttributeError as ae:
            safe_set_alarm(0)
            raise RuntimeError(f"Aegis: Sim AttributeError: {ae}")
        except IndexError as ie:
            safe_set_alarm(0)
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
            safe_set_alarm(0)

    Battle.proceed = patched_proceed
    print("ℹ️ [Aegis Patch] 超堅牢版・本物技自動再サンプリングパッチ を適用しました。")

    # 6. 【Aegis 超高速化パッチ】一括再生中は重い「ベイズ型看破処理」を完全にバイパス
    AegisAnalyzer.update_beliefs_by_implicit_observations = lambda self: None
    print("ℹ️ [Aegis Patch] プレトレーニング用の超高速化バイパスを適用しました。")

    # 7. メイン関数の実行
    run_all_pretraining()