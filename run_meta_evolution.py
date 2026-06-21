import sys
import types
import builtins
import os
import json
import io
import time
import random
import warnings  # 🌟 インポート漏れ修正
from typing import Any  # 🌟 インポート漏れ修正
from copy import deepcopy
from collections import Counter

# =========================================================================
# 0. 【File Path Redirect & Aegislash Data Patch】
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
# 2. 共通ライブラリ・モジュールのロード
# =========================================================================
from pokepy.pokemon import Pokemon
from pokepy.battle import Battle
from aegis_bot import AegisTeamBuilder, AegisTeamSelector, AegisAnalyzer
from src.rebel.belief_state import PokemonBeliefState
from src.rebel.public_state import PublicBeliefState
from train_value_network import train_model


# =========================================================================
# 3. 環境適応型（重み付き）チーム生成システム
# =========================================================================
def generate_evolved_team(builder: AegisTeamBuilder, weights: dict[str, Any]) -> list:
    """環境の勝率（重み）、および型重みに基づき、優秀な個体・型を引き当てて6体構築を生成する"""
    candidates = list(builder.mb_pokemon)

    # 重みリストの作成（未登録のものはデフォルト値 1.0）
    prob_weights = []
    for name in candidates:
        val = weights.get(name, 1.0)
        if isinstance(val, dict):
            prob_weights.append(val.get("weight", 1.0))
        else:
            prob_weights.append(float(val))

    if sum(prob_weights) == 0:
        prob_weights = [1.0] * len(candidates)

    # 重み付きランダムサンプリングで軸を決定
    random_core = random.choices(candidates, weights=prob_weights, k=1)[0]

    # 軸ポケモンに対応する個別の型重みをビルダーに渡してチームを決定
    team_dict = builder.build_team(random_core, pokemon_weights=weights)

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
# 4. 世代別環境ログ解析システム（型情報：技、特性、性格の勝敗集計） [2]
# =========================================================================
def analyze_generation_meta(log_path: str) -> dict:
    """その世代の自己対戦結果を集計し、勝率および技、性格、特性の勝利実績を算出する"""
    if not os.path.exists(log_path):
        return {}

    pokemon_picks = Counter()
    pokemon_wins = Counter()
    pokemon_items = {}

    # 技、特性、性格ごとの勝利数と敗北数の集計
    move_picks = {}
    move_wins = {}
    ability_picks = {}
    ability_wins = {}
    nature_picks = {}
    nature_wins = {}

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

                for idx in selections:
                    poke = team[idx]
                    name = poke["name"]
                    item = poke["item"]
                    moves = poke["moves"]

                    nature = poke.get("nature", "いじっぱり")
                    ability = poke.get("ability", "とくせいなし")

                    pokemon_picks[name] += 1
                    if pl == winner:
                        pokemon_wins[name] += 1

                    if name not in pokemon_items:
                        pokemon_items[name] = Counter()
                    pokemon_items[name][item] += 1

                    # 🌟 技・特性・性格の統計集計
                    if name not in move_picks:
                        move_picks[name] = Counter()
                        move_wins[name] = Counter()
                        ability_picks[name] = Counter()
                        ability_wins[name] = Counter()
                        nature_picks[name] = Counter()
                        nature_wins[name] = Counter()

                    nature_picks[name][nature] += 1
                    ability_picks[name][ability] += 1
                    if pl == winner:
                        nature_wins[name][nature] += 1
                        ability_wins[name][ability] += 1

                    for m in moves:
                        move_picks[name][m] += 1
                        if pl == winner:
                            move_wins[name][m] += 1

    meta_report = {}
    for name, picks in pokemon_picks.items():
        wins = pokemon_wins[name]
        win_rate = wins / picks if picks > 0 else 0.0

        top_item = pokemon_items[name].most_common(1)[0][0] if name in pokemon_items else ""
        top_moves = [m[0] for m in move_picks[name].most_common(4)] if name in move_picks else []

        moves_meta = {}
        if name in move_picks:
            for m, m_picks in move_picks[name].items():
                m_wins = move_wins[name][m]
                moves_meta[m] = m_wins / m_picks if m_picks > 0 else 0.0

        abilities_meta = {}
        if name in ability_picks:
            for ab, ab_picks in ability_picks[name].items():
                ab_wins = ability_wins[name][ab]
                abilities_meta[ab] = ab_wins / ab_picks if ab_picks > 0 else 0.0

        natures_meta = {}
        if name in nature_picks:
            for nat, nat_picks in nature_picks[name].items():
                nat_wins = nature_wins[name][nat]
                natures_meta[nat] = nat_wins / nat_picks if nat_picks > 0 else 0.0

        meta_report[name] = {
            "picks": picks,
            "wins": wins,
            "win_rate": round(win_rate, 3),
            "preferred_item": top_item,
            "preferred_moves": top_moves,
            "moves_win_rate": moves_meta,
            "abilities_win_rate": abilities_meta,
            "natures_win_rate": natures_meta
        }

    return meta_report


# =========================================================================
# 5. 自己対戦解決処理
# =========================================================================
def run_generation_match_file(match_id: int, builder: AegisTeamBuilder, selector: AegisTeamSelector, cfr_solver,
                              analyzer, weights: dict) -> dict:
    match_seed = int(time.time() * 1000) % 1000000
    battle = Battle(seed=match_seed)

    team_p0 = generate_evolved_team(builder, weights)
    team_p1 = generate_evolved_team(builder, weights)

    battle.selected[0] = team_p0
    battle.selected[1] = team_p1

    sel_p0 = selector.select(team_p0, team_p1, num_select=3)
    sel_p1 = selector.select(team_p1, team_p0, num_select=3)

    battle.selected[0] = [deepcopy(team_p0[i]) for i in sel_p0]
    battle.selected[1] = [deepcopy(team_p1[i]) for i in sel_p1]

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

    battle.turn = 0
    for player in range(2):
        battle.change_pokemon(player, idx=0, landing=False)
    for player in battle.speed_order:
        battle.land(player)

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
            [{"name": p.name, "item": p.item, "moves": p.moves, "nature": p.nature, "ability": p.ability} for p in
             team_p0],
            [{"name": p.name, "item": p.item, "moves": p.moves, "nature": p.nature, "ability": p.ability} for p in
             team_p1]
        ],
        "selections": [sel_p0, sel_p1],
        "winner": winner,
        "history": history_log
    }


# =========================================================================
# 6. 世代進化パイプラインメインループ (1000世代 & 自動再開仕様)
# =========================================================================
def run_evolution_loop(total_generations: int = 1000, matches_per_gen: int = 40):
    Pokemon.init(season=22)

    for target_alias in ['キングズシールド', 'キング・シールド', 'キングズ・シールド']:
        if target_alias in Pokemon.all_moves:
            Pokemon.all_moves['キングシールド'] = Pokemon.all_moves[target_alias]
            break

    analyzer = AegisAnalyzer()
    builder = analyzer.team_builder
    selector = analyzer.team_selector
    cfr_solver = analyzer.cfr_solver

    # 1. 世代初期重みのロードとマイグレーション
    pokemon_weights = {}
    weights_path = "log/meta_weights.json"

    for name in builder.mb_pokemon:
        pokemon_weights[name] = {
            "weight": 1.0,
            "moves": {},
            "abilities": {},
            "natures": {}
        }

    if os.path.exists(weights_path):
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                loaded_weights = json.load(f)

            for name, val in loaded_weights.items():
                if name in pokemon_weights:
                    if isinstance(val, (int, float)):
                        pokemon_weights[name]["weight"] = float(val)
                    elif isinstance(val, dict):
                        pokemon_weights[name].update(val)
            print("ℹ️ Existing meta weights loaded and migrated successfully.")
        except Exception as e:
            warnings.warn(f"重みデータのパース・移行に失敗しました(リセットして続行します): {e}")

    # 自動再開スキャン
    start_generation = 1
    for gen in range(1, total_generations + 1):
        gen_log_path = f"log/selfplay_gen_{gen}.jsonl"
        if os.path.exists(gen_log_path):
            if os.path.getsize(gen_log_path) > 0:
                start_generation = gen + 1

    print("\n==================================================")
    print("  🚀 Aegis 環境メタ進化ループ（1000世代サイクル）起動")
    print(f"  総世代数: {total_generations}世代")
    print(f"  世代ごとの対戦数: {matches_per_gen}回戦")
    if start_generation > 1:
        print(f"  ✨ 既存の 1～{start_generation - 1} 世代のデータを検出しました。")
        print(f"  🔄 第 {start_generation} 世代からシミュレーションを再開します。")
    print("==================================================\n")

    for gen in range(start_generation, total_generations + 1):
        print(f"\n--- ［ 世代 {gen} / {total_generations} ］をシミュレート中 ---")

        gen_log_path = f"log/selfplay_gen_{gen}.jsonl"

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

        # 型重みの勝率連動学習
        learning_rate = 0.5
        for name, stats in meta_report.items():
            if name in pokemon_weights:
                win_rate = stats["win_rate"]
                weight_delta = 1.0 + learning_rate * (win_rate - 0.5)
                pokemon_weights[name]["weight"] = max(0.1, min(10.0, pokemon_weights[name]["weight"] * weight_delta))

                for m, m_win_rate in stats.get("moves_win_rate", {}).items():
                    m_delta = 1.0 + learning_rate * (m_win_rate - 0.5)
                    current_m_w = pokemon_weights[name]["moves"].get(m, 1.0)
                    pokemon_weights[name]["moves"][m] = max(0.1, min(10.0, current_m_w * m_delta))

                for ab, ab_win_rate in stats.get("abilities_win_rate", {}).items():
                    ab_delta = 1.0 + learning_rate * (ab_win_rate - 0.5)
                    current_ab_w = pokemon_weights[name]["abilities"].get(ab, 1.0)
                    pokemon_weights[name]["abilities"][ab] = max(0.1, min(10.0, current_ab_w * ab_delta))

                for nat, nat_win_rate in stats.get("natures_win_rate", {}).items():
                    nat_delta = 1.0 + learning_rate * (nat_win_rate - 0.5)
                    current_nat_w = pokemon_weights[name]["natures"].get(nat, 1.0)
                    pokemon_weights[name]["natures"][nat] = max(0.1, min(10.0, current_nat_w * nat_delta))

        os.makedirs("log", exist_ok=True)
        with open(weights_path, "w", encoding="utf-8") as f_out:
            json.dump(pokemon_weights, f_out, ensure_ascii=False, indent=2)

        print(f"🔄 世代 {gen} の対戦ログを用いて価値予測AI（PyTorch）を追加学習中...")
        train_model(
            log_path=gen_log_path,
            epochs=5,
            batch_size=32,
            lr=1e-4
        )

        # 3勝以上基準での Top 3 構築保存
        filtered_meta = [item for item in meta_report.items() if item[1]["wins"] >= 3]
        if len(filtered_meta) < 3:
            filtered_meta = [item for item in meta_report.items() if item[1]["wins"] >= 2]
        if len(filtered_meta) < 3:
            filtered_meta = [item for item in meta_report.items() if item[1]["wins"] >= 1]
        if not filtered_meta:
            filtered_meta = list(meta_report.items())

        sorted_meta = sorted(filtered_meta, key=lambda x: (-x[1]["win_rate"], -x[1]["wins"]))

        print(f"💾 世代 {gen} の勝率上位3構築をファイルに保存します...")
        for rank in range(1, 4):
            if len(sorted_meta) >= rank:
                top_poke_name = sorted_meta[rank - 1][0]
                try:
                    rank_party = builder.build_team(top_poke_name, pokemon_weights=pokemon_weights)

                    rank_path = f"log/party_gen{gen}_rank{rank}.json"
                    shortcut_path = f"log/party_rank{rank}.json"

                    with open(rank_path, "w", encoding="utf-8") as f_rank:
                        json.dump(rank_party, f_rank, ensure_ascii=False, indent=2)
                    with open(shortcut_path, "w", encoding="utf-8") as f_short:
                        json.dump(rank_party, f_short, ensure_ascii=False, indent=2)

                    print(
                        f"   - {rank}位軸: 【{top_poke_name}】 (勝利数: {sorted_meta[rank - 1][1]['wins']}) ➔ {rank_path} に保存完了")
                except Exception as e:
                    print(f"   - {rank}位軸: {top_poke_name} の構築生成中にエラーが発生しました: {e}")

        display_filtered = [item for item in meta_report.items() if item[1]["wins"] >= 3]
        if len(display_filtered) < 5:
            display_filtered = [item for item in meta_report.items() if item[1]["wins"] >= 2]
        if len(display_filtered) < 5:
            display_filtered = [item for item in meta_report.items() if item[1]["wins"] >= 1]
        if not display_filtered:
            display_filtered = list(meta_report.items())

        display_meta = sorted(display_filtered, key=lambda x: (-x[1]["win_rate"], -x[1]["wins"]))[:10]

        print(f"\n==================================================")
        print(f"  👑 【Aegis Meta Report】世代 {gen} の最強ポケモン Top 10 (※勝利数基準適合)")
        print(f"==================================================")
        for rank, (name, info) in enumerate(display_meta, 1):
            preferred_moves = ", ".join(info["preferred_moves"])
            print(
                f"  {rank}位: 【{name}】 (勝率: {info['win_rate']:.1%}, 勝利数(Wins): {info['wins']}, 選出回数(Picks): {info['picks']})")
            print(f"       ┗ 最頻持ち物: {info['preferred_item']} | 頻出技: [{preferred_moves}]")
        print(f"==================================================\n")

    print(f"\n🏁 {total_generations}世代すべての進化学習サイクルが正常に完了しました。")


# =========================================================================
# 7. エントリーポイント
# =========================================================================
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

    # C. 【Battle.proceed 技インデックス限界突破防止パッチ】
    original_proceed = Battle.proceed


    def patched_proceed(self, commands=None):
        cmds = commands if commands is not None else self.command
        if cmds:
            cmds = list(cmds)
            for player in range(2):
                p = self.pokemon[player]
                if p and p.hp > 0:
                    cmd = cmds[player]
                    if cmd is not None:
                        if cmd < 20:
                            move_idx = cmd % 10
                            if move_idx >= len(p.moves):
                                fallback_idx = 0 if p.moves else 0
                                if cmd >= 10:
                                    cmds[player] = 10 + fallback_idx
                                else:
                                    cmds[player] = fallback_idx

        return original_proceed(self, commands=cmds)


    Battle.proceed = patched_proceed

    for target_alias in ['キングズシールド', 'キング・シールド', 'キングズ・シールド']:
        if target_alias in Pokemon.all_moves:
            Pokemon.all_moves['キングシールド'] = Pokemon.all_moves[target_alias]
            break

    run_evolution_loop(total_generations=1000, matches_per_gen=40)