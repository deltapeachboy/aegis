import sys
import types
import builtins
import os
import json
import io
import time
import random
import signal  # 🌟 タイムアウト強制遮断・パッチ用に追加
from copy import deepcopy

# =========================================================================
# 0. 【File Path Redirect & Aegislash Data Patch (絶対位置対応版)】
# =========================================================================
_original_open = builtins.open


def patched_open(file, *args, **kwargs):
    if isinstance(file, str) and "learnset.json" in file:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        custom_path = os.path.join(base_dir, "battle_data", "mb_learnset.json")

        if os.path.exists(custom_path):
            print(f"ℹ️ [Redirect & Patch] '{file}' ➔ '{custom_path}' へ強制リダイレクトロードします。")

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

# =========================================================================
# 1. 【Aegis Namespace Bridge】
# =========================================================================
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

# =========================================================================
# 2. PyTorch および周辺ライブラリのロード
# =========================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from aegis_bot import AegisAnalyzer, AegisTeamBuilder
from pokepy.pokemon import Pokemon
from pokepy.battle import Battle
from src.rebel.belief_state import PokemonBeliefState
from src.rebel.public_state import PublicBeliefState
from src.rebel.value_network import ReBeLValueNetwork

device = torch.device("cpu")
print(f"ℹ️ 使用デバイス: {device} (安全なCPU学習に固定しました)")

# 🌟 独立起動時にも pokepy 内のステータス参照バグを回避するグローバルプロパティ注入パッチ
Pokemon.attack = property(lambda self: self.status[1])
Pokemon.defense = property(lambda self: self.status[2])
Pokemon.sp_attack = property(lambda self: self.status[3])
Pokemon.sp_defense = property(lambda self: self.status[4])
Pokemon.speed = property(lambda self: self.status[5])


# =========================================================================
# 3. データセット再現・リプレイ抽出クラス
# =========================================================================
class SelfPlayReplayDataset(Dataset):
    def __init__(self, log_path: str, analyzer: AegisAnalyzer):
        self.log_path = log_path
        self.analyzer = analyzer
        self.encoded_states = []
        self.targets = []
        self._load_and_replay()

    def _load_and_replay(self):
        print("📊 対戦ログをシミュレータ上でリプレイ中（特徴量抽出）...")
        t0 = time.time()

        if not os.path.exists(self.log_path):
            raise FileNotFoundError(f"対戦ログファイル '{self.log_path}' が見つかりません。")

        # タイムアウト例外のローカル定義
        class ReplayTimeoutException(Exception):
            pass

        def replay_timeout_handler(signum, frame):
            raise ReplayTimeoutException("Replay match timed out!")

        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, replay_timeout_handler)

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                match_data = json.loads(line)

                seed = match_data["seed"]
                selections = match_data["selections"]
                winner = match_data["winner"]
                history = match_data["history"]

                if winner is None or winner == -1:
                    continue

                # 特徴量リストのロールバック用バックアップ
                start_state_idx = len(self.encoded_states)
                start_target_idx = len(self.targets)

                # 🌟 [リプレイ守護神] 各試合の解読に2秒制限を適用。不整合ハングを自動で切り捨てます。
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(2)

                try:
                    battle = Battle(seed=seed)
                    team_p0 = self._rebuild_team(match_data["teams"][0])
                    team_p1 = self._rebuild_team(match_data["teams"][1])
                    battle.selected[0] = team_p0
                    battle.selected[1] = team_p1

                    battle.selected[0] = [deepcopy(team_p0[i]) for i in selections[0]]
                    battle.selected[1] = [deepcopy(team_p1[i]) for i in selections[1]]

                    beliefs = [
                        PokemonBeliefState.__new__(PokemonBeliefState),
                        PokemonBeliefState.__new__(PokemonBeliefState)
                    ]
                    for pl in [0, 1]:
                        beliefs[pl].usage_db = None
                        beliefs[pl].max_hypotheses = 30
                        beliefs[pl].min_probability = 0.01
                        beliefs[pl].observation_history = []
                        opp_names = [p.name for p in battle.selected[1 - pl]]
                        beliefs[pl].revealed_moves = {name: set() for name in opp_names}
                        beliefs[pl].revealed_items = {name: None for name in opp_names}
                        beliefs[pl].revealed_abilities = {name: None for name in opp_names}
                        beliefs[pl].revealed_tera = {name: None for name in opp_names}
                        beliefs[pl].move_use_count = {name: {} for name in opp_names}
                        beliefs[pl].beliefs = {}
                        for name in opp_names:
                            beliefs[pl].beliefs[name] = self.analyzer._build_flat_belief(name)

                    battle.turn = 0
                    for player in range(2):
                        battle.change_pokemon(player, idx=0, landing=False)
                    for player in battle.speed_order:
                        battle.land(player)

                    for turn_idx, turn_data in enumerate(history):
                        for pl in [0, 1]:
                            pbs = PublicBeliefState.from_battle(battle, perspective=pl, belief=beliefs[pl])
                            with torch.no_grad():
                                encoded_pbs = self.analyzer.cfr_solver.value_network.encoder(pbs).cpu()
                            self.encoded_states.append(encoded_pbs)

                            # 🌟 [バグ解消・高速化仕様]
                            # 価値予測モデルは純粋な勝敗（1.0 / 0.0）をターゲットとして学習させ、
                            # 中間報酬（ミミッキュ・ステロ・ランク）は推論時に動的に上乗せすることで二重補正を防ぎます。
                            my_win = 1.0 if pl == winner else 0.0
                            opp_win = 1.0 - my_win
                            self.targets.append(torch.tensor([my_win, opp_win], dtype=torch.float))

                        cmds = turn_data["commands"]
                        battle.command = cmds

                        builtins._aegis_current_battle = battle
                        battle.proceed(commands=cmds)

                except ReplayTimeoutException:
                    print(
                        f"⚠️ [Replay Guardian] 試合 {line_idx} の再現中に乱数不整合によるハングを検知したため、この試合の特徴量をスキップします。")
                    self.encoded_states = self.encoded_states[:start_state_idx]
                    self.targets = self.targets[:start_target_idx]
                    continue
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)

                if line_idx % 10 == 0:
                    print(f"   - {line_idx} 試合分の盤面データを展開完了...")

        print(f"✅ 特徴量抽出完了 (総局面数: {len(self.encoded_states)} 個, 所要時間: {time.time() - t0:.1f}秒)")

    def _rebuild_team(self, team_data: list) -> list:
        rebuilt = []
        for raw in team_data:
            p = Pokemon()
            name = raw["name"]
            if name == "ギルガルド" and "ギルガルド" not in Pokemon.zukan:
                for k in ['ギルガルド(シールド)', 'ギルガルド（シールド）']:
                    if k in Pokemon.zukan:
                        Pokemon.zukan['ギルガルド'] = deepcopy(Pokemon.zukan[k])
                        Pokemon.zukan['ギルガルド']['display_name'] = 'ギルガルド'
                        break
            p.name = name
            p.item = raw.get("item", "")
            p.moves = raw.get("moves", ["テラバースト"])
            p.indiv = raw.get("indiv", [31] * 6)

            p.nature = raw.get("nature", "いじっぱり")
            p.ability = raw.get("ability", "とくせいなし")
            p.effort = raw.get("effort", [0] * 6)

            p.update_status()
            rebuilt.append(p)
        return rebuilt

    def __len__(self):
        return len(self.encoded_states)

    def __getitem__(self, idx):
        return self.encoded_states[idx], self.targets[idx]


# =========================================================================
# 4. 訓練ループ
# =========================================================================
def train_model(log_path: str, model_save_path: str = "src/rebel/value_network.pth", epochs: int = 15,
                batch_size: int = 64, lr: float = 1e-4):
    analyzer = AegisAnalyzer()

    if getattr(analyzer.cfr_solver, 'value_network', None) is None:
        print("ℹ️ analyzer.cfr_solver.value_network が None のため、新規に ReBeLValueNetwork を生成します。")
        from src.rebel.value_network import ReBeLValueNetwork
        try:
            analyzer.cfr_solver.value_network = ReBeLValueNetwork(
                hidden_dim=256,
                num_res_blocks=4,
                dropout=0.1,
                use_move_effectiveness=True
            )
        except Exception as e:
            print(f"⚠️ ReBeLValueNetwork の生成中にエラーが発生しました: {e}")
            raise e

    dataset = SelfPlayReplayDataset(log_path, analyzer)

    # 🌟 0件時の安全な進行ガード
    if len(dataset) == 0:
        print("⚠️ 警告: 有効な特徴量が0件です。追加学習を安全にスキップして次の世代へ進行します。")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = analyzer.cfr_solver.value_network
    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"\n==================================================")
    print(f"  ReBeL Value Network 訓練サイクル開始")
    print(f"  総データ数(ターン数): {len(dataset)}")
    print(f"  バッチサイズ: {batch_size} | エポック数: {epochs}")
    print(f"==================================================\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t_epoch_start = time.time()

        for batch_idx, (states, targets) in enumerate(dataloader, 1):
            states = states.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            h = model.input_layer(states)
            for block in model.res_blocks:
                h = block(h)
            predictions = torch.sigmoid(model.value_head(h))

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * states.size(0)

        epoch_loss /= len(dataset)
        epoch_time = time.time() - t_epoch_start
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.5f} - 所要時間: {epoch_time:.1f}秒")

    torch.save(model.state_dict(), model_save_path)
    print(f"\n==================================================")
    print(f"  訓練完了 (総所要時間: {time.time() - start_time:.1f}秒)")
    print(f"  保存先: '{model_save_path}'")
    print(f"==================================================\n")


if __name__ == "__main__":
    Pokemon.init(season=22)

    original_find = Pokemon.find


    @classmethod
    def patched_find(cls, pokemon_list, name=None, display_name=None):
        res = original_find(pokemon_list, name=name, display_name=display_name)
        if res is not None:
            return res

        if name:
            base_name = name.replace("メガ", "").rstrip("XYＸＹ ").split("(")[0]
            for p in pokemon_list:
                p_base = p.name.replace("メガ", "").rstrip("XYＸＹ ").split("(")[0]
                if p_base == base_name or p.display_name == base_name:
                    return p

        if pokemon_list:
            return pokemon_list[0]
        return None


    Pokemon.find = patched_find

    original_get_mega_name = Battle.get_mega_name


    def patched_get_mega_name(self, p: Pokemon) -> str:
        if not p or not p.item:
            return ""

        if p.name == "リザードン":
            if "X" in p.item or "Ｘ" in p.item: return "メガリザードンX"
            if "Y" in p.item or "Ｙ" in p.item: return "メガリザードンY"
        elif p.name == "ミュウツー":
            if "X" in p.item or "Ｘ" in p.item: return "メガミュウツーX"
            if "Y" in p.item or "Ｙ" in p.item: return "メガミュウツーY"
        elif p.name == "ライチュウ":
            if "X" in p.item or "Ｘ" in p.item: return "メガライチュウX"
            if "Y" in p.item or "Ｙ" in p.item: return "メガライチュウY"

        return original_get_mega_name(self, p)


    Battle.get_mega_name = patched_get_mega_name

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

        return original_proceed(self, commands=cmds)


    Battle.proceed = patched_proceed

    for target_alias in ['キングズシールド', 'キング・シールド', 'キングズ・シールド']:
        if target_alias in Pokemon.all_moves:
            Pokemon.all_moves['キングシールド'] = Pokemon.all_moves[target_alias]
            break

    log_file_path = "log/selfplay_dataset.jsonl"
    if os.path.exists(log_file_path):
        train_model(log_path=log_file_path, epochs=15, batch_size=64, lr=1e-4)
    else:
        print(
            f"⚠️ 訓練データ '{log_file_path}' が見つかりません。先に run_selfplay.py や evolution ログを用意してください。")