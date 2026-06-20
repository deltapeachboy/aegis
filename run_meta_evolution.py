import sys
import types
import builtins
import os
import json
import io
import time
import random
from copy import deepcopy
from collections import Counter

# =========================================================================
# 0. 【File Path Redirect & Aegislash Data Patch (絶対位置対応版)】
# =========================================================================
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

# =========================================================================
# 1. 共通ライブラリ・インポート
# =========================================================================
from pokepy.pokemon import Pokemon
from pokepy.battle import Battle
from aegis_bot import AegisTeamBuilder, AegisTeamSelector, AegisAnalyzer
from src.rebel.belief_state import PokemonBeliefState
from src.rebel.public_state import PublicBeliefState
from train_value_network import train_model


# =========================================================================
# 2. 環境適応型（重み付き）チーム生成システム
# =========================================================================
def generate_evolved_team(builder: AegisTeamBuilder, weights: dict[str, float]) -> list:
    """環境の勝率（重み）に基づき、優秀な個体を引き当てて6体構築を生成する"""
    candidates = list(builder.mb_pokemon)

    # 重みリストの作成（未登録のものはデフォルト値 1.0）
    prob_weights = [weights.get(name, 1.0) for name in candidates]
    if sum(prob_weights) == 0:
        prob_weights = [1.0] * len(candidates)

    # 重み付きランダムサンプリングで軸を決定
    random_core = random.choices(candidates, weights=prob_weights, k=1)[0]
    team_dict = builder.build_team(random_core)

    # シミュレータオブジェクトへの復元とデータ型安全変換
    selected_team = []
    for s in team_dict:
        p = Pokemon()
        name = team_dict[s]['name']

        # ギルガルド図鑑の自動補正
        if name == "ギルガルド" and "ギルガルド" not in Pokemon.zukan:
            for k in ['ギルガルド(シールド)', 'ギルガルド（シールド）']:
                if k in Pokemon.zukan:
                    Pokemon.zukan['ギルガルド'] = deepcopy(Pokemon.zukan[k])
                    Pokemon.zukan['ギルガルド']['display_name'] = 'ギルガルド'
                    break

        p.name = name
        p.sex = team_dict[s]['sex']
        p.level = team_dict[s]['level']
        p.nature = team_dict[s]['nature']
        p.ability = team_dict[s]['ability']
        p.item = team_dict[s]['item']
        p.Ttype = team_dict[s]['Ttype']
        p.moves = team_dict[s]['moves']

        # 努力値・個体値の安全なリスト化解決
        ind_data = team_dict[s].get('indiv', [31] * 6)
        p.indiv = [ind_data.get(k, 31) for k in ["H", "A", "B", "C", "D", "S"]] if isinstance(ind_data,
                                                                                              dict) else ind_data

        eff_data = team_dict[s].get('effort', [0] * 6)
        p.effort = [eff_data.get(k, 0) for k in ["H", "A", "B", "C", "D", "S"]] if isinstance(eff_data,
                                                                                              dict) else eff_data

        p.update_status()
        selected_team.append(p)
    return selected_team


# =========================================================================
# 3. 世代別環境ログ解析システム
# =========================================================================
def analyze_generation_meta(log_path: str) -> dict:
    """その世代の自己対戦結果を集計し、ポケモンの勝率・持ち物・技使用率を算出する"""
    if not os.path.exists(log_path):
        return {}

    pokemon_picks = Counter()
    pokemon_wins = Counter()
    pokemon_items = {}
    pokemon_moves = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            match = json.loads(line)
            winner = match["winner"]
            if winner is None or winner == -1:
                continue

            for pl in [0, 1]:
                selections = match["selections"][pl]
                team = match["teams"][pl]

                # 選出された3体
                for idx in selections:
                    poke = team[idx]
                    name = poke["name"]
                    item = poke["item"]
                    moves = poke["moves"]

                    pokemon_picks[name] += 1
                    if pl == winner:
                        pokemon_wins[name] += 1

                    if name not in pokemon_items:
                        pokemon_items[name] = Counter()
                    pokemon_items[name][item] += 1

                    if name not in pokemon_moves:
                        pokemon_moves[name] = Counter()
                    for m in moves:
                        pokemon_moves[name][m] += 1

    meta_report = {}
    for name, picks in pokemon_picks.items():
        wins = pokemon_wins[name]
        win_rate = wins / picks if picks > 0 else 0.0

        top_item = pokemon_items[name].most_common(1)[0][0] if name in pokemon_items else ""
        top_moves = [m[0] for m in pokemon_moves[name].most_common(4)] if name in pokemon_moves else []

        meta_report[name] = {
            "picks": picks,
            "wins": wins,
            "win_rate": round(win_rate, 3),
            "preferred_item": top_item,
            "preferred_moves": top_moves
        }

    return meta_report


# =========================================================================
# 4. 世代進化パイプライン
# =========================================================================
def run_generation_match_file(match_id: int, builder: AegisTeamBuilder, selector: AegisTeamSelector, cfr_solver,
                              analyzer, weights: dict) -> dict:
    """重み（環境適合率）を反映した対戦実行"""
    match_seed = int(time.time() * 1000) % 1000000
    battle = Battle(seed=match_seed)

    # 重み付きチーム生成を適用
    team_p0 = generate_evolved_team(builder, weights)
    team_p1 = generate_evolved_team(builder, weights)

    battle.selected[0] = team_p0
    battle.selected[1] = team_p1

    # 3体選出フェーズ
    sel_p0 = selector.select(team_p0, team_p1, num_select=3)
    sel_p1 = selector.select(team_p1, team_p0, num_select=3)

    battle.selected[0] = [deepcopy(team_p0[i]) for i in sel_p0]
    battle.selected[1] = [deepcopy(team_p1[i]) for i in sel_p1]

    # 信念状態初期化
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
            beliefs[pl].beliefs[name] = analyzer._build_flat_belief(name)

    # 0ターン目の繰り出し
    battle.turn = 0
    for player in range(2):
        battle.change_pokemon(player, idx=0, landing=False)
    for player in battle.speed_order:
        battle.land(player)

    # バトルループ
    history_log = []
    while battle.winner() is None:
        battle.turn += 1
        commands = [None, None]
        for pl in [0, 1]:
            pbs = PublicBeliefState.from_battle(battle, perspective=pl, belief=beliefs[pl])
            cfr_solver.solver.num_samples = 3
            my_strategy, _ = cfr_solver.solve(pbs, battle)
            if my_strategy:
                actions = list(my_strategy.keys())
                probs = list(my_strategy.values())
                commands[pl] = random.choices(actions, weights=probs, k=1)[0]
            else:
                commands[pl] = random.choice(battle.available_commands(pl))

        battle.command = commands
        battle.proceed(commands=commands)

        # ログへの追加
        history_log.append({
            "turn": battle.turn,
            "commands": commands,
            "hp": [battle.pokemon[0].hp, battle.pokemon[1].hp] if all(battle.pokemon) else [0, 0]
        })

        if battle.turn >= 50:
            break

    winner = battle.winner()

    return {
        "match_id": match_id,
        "seed": match_seed,
        "teams": [
            [{"name": p.name, "item": p.item, "moves": p.moves} for p in team_p0],
            [{"name": p.name, "item": p.item, "moves": p.moves} for p in team_p1]
        ],
        "selections": [sel_p0, sel_p1],
        "winner": winner,
        "history": history_log
    }


def run_evolution_loop(total_generations: int = 100, matches_per_gen: int = 40):
    """世代交代の強化学習およびメタ変遷サイクルを実行する"""
    Pokemon.init(season=22)

    # ギルガルド・キングシールドの表記揺れ自動同期
    for target_alias in ['キングズシールド', 'キング・シールド', 'キングズ・シールド']:
        if target_alias in Pokemon.all_moves:
            Pokemon.all_moves['キングシールド'] = Pokemon.all_moves[target_alias]
            break

    analyzer = AegisAnalyzer()
    builder = analyzer.team_builder
    selector = analyzer.team_selector
    cfr_solver = analyzer.cfr_solver

    # 1. 世代初期重み
    pokemon_weights = {name: 1.0 for name in builder.mb_pokemon}

    weights_path = "log/meta_weights.json"
    if os.path.exists(weights_path):
        with open(weights_path, "r", encoding="utf-8") as f:
            pokemon_weights.update(json.load(f))
        print("ℹ️ 既存 of the meta weights loaded.")

    print("\n==================================================")
    print("  🚀 Aegis 環境メタ進化ループ（100世代サイクル）起動")
    print(f"  総世代数: {total_generations}世代")
    print(f"  世代ごとの対戦数: {matches_per_gen}回戦")
    print("==================================================\n")

    for gen in range(1, total_generations + 1):
        print(f"\n--- ［ 世代 {gen} / {total_generations} ］をシミュレート中 ---")

        gen_log_path = f"log/selfplay_gen_{gen}.jsonl"

        # 自己対戦の実行
        with open(gen_log_path, "w", encoding="utf-8") as f_out:
            for match_idx in range(1, matches_per_gen + 1):
                try:
                    match_data = run_generation_match_file(
                        match_id=match_idx,
                        builder=builder,
                        selector=selector,
                        cfr_solver=cfr_solver,
                        analyzer=analyzer,
                        weights=pokemon_weights
                    )
                    f_out.write(json.dumps(match_data, ensure_ascii=False) + "\n")
                except Exception as e:
                    import traceback
                    traceback.print_exc()

        # 世代メタの統計解析
        meta_report = analyze_generation_meta(gen_log_path)

        # 解析結果から勝率に基づき「重み（登場確率）」を更新 [2]
        learning_rate = 0.5
        for name, stats in meta_report.items():
            if name in pokemon_weights:
                win_rate = stats["win_rate"]
                weight_delta = 1.0 + learning_rate * (win_rate - 0.5)
                pokemon_weights[name] = max(0.1, min(10.0, pokemon_weights[name] * weight_delta))

        # 重みデータの保存
        os.makedirs("log", exist_ok=True)
        with open(weights_path, "w", encoding="utf-8") as f_out:
            json.dump(pokemon_weights, f_out, ensure_ascii=False, indent=2)

        # バリューネットワーク（価値予測モデル）のオンライン追加学習
        print(f"🔄 世代 {gen} の対戦ログを用いて価値予測AI（PyTorch）を追加学習中...")
        train_model(
            log_path=gen_log_path,
            epochs=5,
            batch_size=32,
            lr=1e-4
        )

        # 世代終了時の『環境上位10（Meta Top 10）』の出力 [2]
        sorted_meta = sorted(meta_report.items(), key=lambda x: (-x[1]["win_rate"], -x[1]["picks"]))[:10]

        print(f"\n==================================================")
        print(f"  👑 【Aegis Meta Report】世代 {gen} の最強ポケモン Top 10")
        print(f"==================================================")
        for rank, (name, info) in enumerate(sorted_meta, 1):
            preferred_moves = ", ".join(info["preferred_moves"])
            print(f"  {rank}位: 【{name}】 (勝率: {info['win_rate']:.1%}, 選出回数(Picks): {info['picks']})")
            print(f"       ┗ 最頻持ち物: {info['preferred_item']} | 頻出技: [{preferred_moves}]")
        print(f"==================================================\n")

    print("\n🏁 100世代すべての進化学習サイクルが正常に完了しました。")


# =========================================================================
# 5. エントリーポイント（X/Y分岐メガシンカ・曖昧検索完全対応パッチ） [2]
# =========================================================================
if __name__ == "__main__":
    # 1. シミュレータの初期化
    Pokemon.init(season=22)

    # -------------------------------------------------------------------------
    # A. 【Aegis 曖昧検索パッチ（X/Y分岐対応補強版）】
    # 「メガ」および末尾の「X」「Y」「全角のＸ」「全角のＹ」をすべて除去してベース名にする [2]
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # B. 【Battle.get_mega_name のX/Y分岐メガシンカ対応パッチ】 [2]
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------

    # ギルガルド・キングシールドの表記揺れ自動同期
    for target_alias in ['キングズシールド', 'キング・シールド', 'キングズ・シールド']:
        if target_alias in Pokemon.all_moves:
            Pokemon.all_moves['キングシールド'] = Pokemon.all_moves[target_alias]
            break

    # 進化サイクルの始動（1世代40試合、100世代サイクルを完全自動実行します）
    run_evolution_loop(total_generations=100, matches_per_gen=40)