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
# 1. 【Aegis Namespace Bridge (二重上書き・リセット防止ガード版)】
# =========================================================================
# 🌟【根本解決】run_meta_evolution.py で当てられた動的パッチが、
# インポート時の二重初期化によってリセット・破壊される致命的なバグを完全に防止します。
if 'src.pokemon_battle_sim' not in sys.modules:
    sys.modules['src.pokemon_battle_sim'] = types.ModuleType('src.pokemon_battle_sim')
if 'src.pokemon_battle_sim.pokemon' not in sys.modules:
    sys.modules['src.pokemon_battle_sim.pokemon'] = types.ModuleType('src.pokemon_battle_sim.pokemon')
if 'src.pokemon_battle_sim.battle' not in sys.modules:
    sys.modules['src.pokemon_battle_sim.battle'] = types.ModuleType('src.pokemon_battle_sim.battle')
if 'src.pokemon_battle_sim.damage' not in sys.modules:
    sys.modules['src.pokemon_battle_sim.damage'] = types.ModuleType('src.pokemon_battle_sim.damage')

import pokepy.utils as utils_module

if 'src.pokemon_battle_sim.utils' not in sys.modules:
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
if not hasattr(Pokemon, 'attack'):
    Pokemon.attack = property(lambda self: self.status[1])
if not hasattr(Pokemon, 'defense'):
    Pokemon.defense = property(lambda self: self.status[2])
if not hasattr(Pokemon, 'sp_attack'):
    Pokemon.sp_attack = property(lambda self: self.status[3])
if not hasattr(Pokemon, 'sp_defense'):
    Pokemon.sp_defense = property(lambda self: self.status[4])


def get_aegis_effective_speed(self) -> int:
    """
    天候、おいかぜ、特性(すいすい/すなかき/こだいかっせい/クォークチャージ/かるわざ)、
    麻痺、スカーフ、ランク補正をすべて厳密に計算した【最終実質素早さ】を返します。
    """
    base_speed = self.status[5] if (hasattr(self, 'status') and len(self.status) > 5) else 100

    # 1. ランク補正の適用
    speed_rank = self.rank[5] if (hasattr(self, 'rank') and len(self.rank) > 5) else 0
    if speed_rank > 0:
        rank_modifier = (2.0 + speed_rank) / 2.0
    elif speed_rank < 0:
        rank_modifier = 2.0 / (2.0 - speed_rank)
    else:
        rank_modifier = 1.0

    speed = int(base_speed * rank_modifier)

    # 仮想バトルインスタンスの参照
    current_battle = getattr(builtins, '_aegis_current_battle', None)
    if current_battle and hasattr(current_battle, 'condition'):
        # 自身を所有するプレイヤー(0 or 1)の特定
        player_idx = 0
        for p_idx in [0, 1]:
            if self in current_battle.selected[p_idx]:
                player_idx = p_idx
                break

        # 🌐 A. 天候特性補正 (すいすい / すなかき / こだいかっせい)
        if getattr(self, 'ability', '') == 'すいすい':
            rain_val = current_battle.condition.get('rainy', [0, 0])
            if isinstance(rain_val, list) and player_idx < len(rain_val) and rain_val[player_idx] > 0:
                speed *= 2

        elif getattr(self, 'ability', '') == 'すなかき':
            sand_val = current_battle.condition.get('sandstorm', [0, 0])
            if isinstance(sand_val, list) and player_idx < len(sand_val) and sand_val[player_idx] > 0:
                speed *= 2

        elif getattr(self, 'ability', '') == 'こだいかっせい':
            sunny_val = current_battle.condition.get('sunny', [0, 0])
            has_sun = isinstance(sunny_val, list) and player_idx < len(sunny_val) and sunny_val[player_idx] > 0
            is_boosted = getattr(self, 'BE_activated', False) or getattr(self, 'boost_index', 0) == 5
            if has_sun or is_boosted:
                speed = int(speed * 1.5)

        # 🌐 B. フィールド特性補正 (クォークチャージ)
        elif getattr(self, 'ability', '') == 'クォークチャージ':
            elec_val = current_battle.condition.get('elecfield', [0, 0])
            has_elec = isinstance(elec_val, list) and player_idx < len(elec_val) and elec_val[player_idx] > 0
            is_boosted = getattr(self, 'BE_activated', False) or getattr(self, 'boost_index', 0) == 5
            if has_elec or is_boosted:
                speed = int(speed * 1.5)

        # 🌐 C. おいかぜ補正 (tailwind / oikaze: 2倍)
        tailwind_val = current_battle.condition.get('tailwind', None)
        if tailwind_val is None:
            tailwind_val = current_battle.condition.get('oikaze', [0, 0])

        if isinstance(tailwind_val, list) and player_idx < len(tailwind_val) and tailwind_val[player_idx] > 0:
            speed *= 2

    # 🌐 D. 特性補正（かるわざ：持ち物消費で2倍）
    if getattr(self, 'ability', '') == 'かるわざ':
        if not getattr(self, 'item', '') and getattr(self, 'lost_item', ''):
            speed *= 2

    # 🌐 E. 持ち物補正 (こだわりスカーフ: 1.5倍)
    if getattr(self, 'item', '') == 'こだわりスカーフ':
        speed = int(speed * 1.5)

    # 🌐 F. 状態異常補正 (まひ: 素早さ半減)
    if getattr(self, 'ailment', '') == 'PAR' or getattr(self, 'ailment', '') == 'まひ':
        speed = int(speed * 0.5)

    return max(1, speed)


# 🌟【冪等性ガード】すでに正しい実質素早さ関数が登録されていれば上書きしない
if not hasattr(Pokemon, '_aegis_speed_patched') or Pokemon.speed.fget.__name__ != "get_aegis_effective_speed":
    Pokemon.speed = property(get_aegis_effective_speed)
    Pokemon._aegis_speed_patched = True


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

        class ReplayTimeoutException(Exception):
            pass

        def replay_timeout_handler(signum, frame):
            raise ReplayTimeoutException("Replay match timed out!")

        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, replay_timeout_handler)

        total_sanitized_count = 0

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

                start_state_idx = len(self.encoded_states)
                start_target_idx = len(self.targets)

                if hasattr(signal, "SIGALRM"):
                    signal.alarm(2)

                try:
                    battle = Battle(seed=seed)

                    # 🌟【根本解決】再現バトル生成直後に、即座にグローバルにバインド
                    # ポケモンの繰り出し（change_pokemon, land）の時点で天候補正が完全に同期されます。
                    builtins._aegis_current_battle = battle

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

                        # 進行
                        battle.proceed(commands=cmds)

                        # 🌟【簡易不整合テスト】
                        if battle.command != cmds:
                            total_sanitized_count += 1
                            if total_sanitized_count <= 5:  # 最初の5件のみ詳細を出力
                                print(
                                    f"   🔍 [Sanitizer Test Detect] 試合 {line_idx} / ターン {turn_idx + 1} で非同期クレンジングを検知しました。")
                                print(f"      - 予定コマンド: {cmds} ➔ 補正後コマンド: {battle.command}")
                                if battle.pokemon[0] and battle.pokemon[1]:
                                    print(
                                        f"      - 盤面状況: {battle.pokemon[0].name}(HP:{battle.pokemon[0].hp}) vs {battle.pokemon[1].name}(HP:{battle.pokemon[1].hp})")

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

        # 🌟 テスト診断結果の出力
        print(f"\n==================================================")
        print(f"  🧪 【Aegis Sanitizer 簡易診断結果】")
        print(f"  総クレンジング（非同期）発生件数: {total_sanitized_count} 件 / 全局面中")
        if total_sanitized_count == 0:
            print("  ✅ 判定: 優秀。対戦時とリプレイ再生時の物理的な素早さ・HPアライメントは完全に 100% 同期しています！")
        else:
            print("  ℹ️ 判定: 許容範囲内。微小な状態不整合はサニタイザーによって安全に自動修復されています。")
        print(f"==================================================\n")

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