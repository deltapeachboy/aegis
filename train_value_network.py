import sys
import types
import builtins
import os
import json
import io
import time
import random
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

                        my_win = 1.0 if pl == winner else 0.0
                        opp_win = 1.0 - my_win
                        self.targets.append(torch.tensor([my_win, opp_win], dtype=torch.float))

                    cmds = turn_data["commands"]
                    battle.command = cmds
                    battle.proceed(commands=cmds)

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
            p.item = raw["item"]
            p.moves = raw["moves"]
            p.indiv = [31] * 6
            p.effort = [252, 0, 0, 252, 0, 0] if raw["name"] in ["アシレーヌ", "サーフゴー"] else [0, 252, 0, 0, 0, 252]
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
    log_file_path = "log/selfplay_dataset.jsonl"
    if os.path.exists(log_file_path):
        train_model(log_path=log_file_path, epochs=15, batch_size=64, lr=1e-4)
    else:
        print(f"⚠️ 訓練データ '{log_file_path}' が見つかりません。先に run_selfplay.py を実行してください。")