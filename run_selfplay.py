import sys
import types
import builtins
import os
import json
import io

# =========================================================================
# 0. 【File Path Redirect & Aegislash Data Patch (絶対位置対応版)】
# シミュレータの技リストロード時に、自動で mb_learnset.json を参照し、
# かつギルガルド各フォルムに技データを動的同期（注入）するパッチ
# =========================================================================
_original_open = builtins.open


def patched_open(file, *args, **kwargs):
    # 'learnset.json' を含むファイルアクセスを検知
    if isinstance(file, str) and "learnset.json" in file:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        custom_path = os.path.join(base_dir, "battle_data", "mb_learnset.json")

        if os.path.exists(custom_path):
            print(f"ℹ️ [Redirect & Patch] '{file}' ➔ '{custom_path}' へ強制リダイレクトロードします。")

            # 1. 実際に mb_learnset.json をロード
            with _original_open(custom_path, "r", encoding="utf-8") as f:
                learnset = json.load(f)

            # 2. ギルガルド(ブレード/シールド)に技データを同期（エイリアス作成）
            base_key = "ギルガルド" if "ギルガルド" in learnset else "ギルガルド(シールド)"
            if base_key in learnset:
                learnset["ギルガルド(ブレード)"] = learnset[base_key]
                learnset["ギルガルド(シールド)"] = learnset[base_key]

            # 3. メモリ上でJSON文字列に戻し、ストリームオブジェクトとしてシミュレータに引き渡す
            patched_json_str = json.dumps(learnset, ensure_ascii=False)
            return io.StringIO(patched_json_str)

    return _original_open(file, *args, **kwargs)


builtins.open = patched_open

# =========================================================================
# 1. 【Aegis Namespace & Import Bridge】
# 自己対戦スクリプト起動時にもインポート衝突を回避するブリッジ
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
# 2. 標準ライブラリ・統合AIモジュールのロード
# =========================================================================
import time
import random
from copy import deepcopy

from pokepy.pokemon import Pokemon
from pokepy.battle import Battle
from aegis_bot import AegisTeamBuilder, AegisTeamSelector
from src.rebel.belief_state import PokemonBeliefState
from src.rebel.public_state import PublicBeliefState
from src.rebel.cfr_solver import ReBeLSolver

# 行動ログ同期に必要な Observation クラス群の安全なインポート
try:
    from src.rebel.belief_state import Observation, ObservationType
except ImportError:
    try:
        from src.hypothesis.item_belief_state import Observation, ObservationType
    except ImportError:
        # クラスが定義されていない場合のフォールバック定義
        class ObservationType:
            MOVE_USED = "move_used"
            ITEM_REVEALED = "item_revealed"


        class Observation:
            def __init__(self, type, pokemon_name, details):
                self.type = type
                self.pokemon_name = pokemon_name
                self.details = details


def generate_random_team(builder: AegisTeamBuilder) -> list:
    """mbルール適合プールからランダムな軸を選び、最適な6体構築を生成してリストで返す"""
    random_core = random.choice(list(builder.mb_pokemon))
    team_dict = builder.build_team(random_core)

    # シミュレータが読めるPokemonオブジェクトのリストに変換
    selected_team = []
    for s in team_dict:
        p = Pokemon()
        p.name = team_dict[s]['name']
        p.sex = team_dict[s]['sex']
        p.level = team_dict[s]['level']
        p.nature = team_dict[s]['nature']
        p.ability = team_dict[s]['ability']
        p.item = team_dict[s]['item']
        p.Ttype = team_dict[s]['Ttype']
        p.moves = team_dict[s]['moves']

        # 辞書型・リスト型の双方に自動適合させる二重保険
        ind_data = team_dict[s].get('indiv', [31] * 6)
        if isinstance(ind_data, dict):
            p.indiv = [
                ind_data.get("H", 31),
                ind_data.get("A", 31),
                ind_data.get("B", 31),
                ind_data.get("C", 31),
                ind_data.get("D", 31),
                ind_data.get("S", 31)
            ]
        else:
            p.indiv = ind_data

        eff_data = team_dict[s].get('effort', [0] * 6)
        if isinstance(eff_data, dict):
            p.effort = [
                eff_data.get("H", 0),
                eff_data.get("A", 0),
                eff_data.get("B", 0),
                eff_data.get("C", 0),
                eff_data.get("D", 0),
                eff_data.get("S", 0)
            ]
        else:
            p.effort = eff_data

        p.update_status()
        selected_team.append(p)
    return selected_team


def run_single_selfplay_match(match_id: int, builder: AegisTeamBuilder, selector: AegisTeamSelector,
                              cfr_solver: ReBeLSolver, analyzer) -> dict:
    """CFR同士の自己対戦を1試合実行し、学習用ログデータを返す"""

    # 1. 乱数シードの設定
    match_seed = int(time.time() * 1000) % 1000000
    battle = Battle(seed=match_seed)

    # 2. お互いの手持ち（6体）をランダム自動構築
    team_p0 = generate_random_team(builder)
    team_p1 = generate_random_team(builder)

    # シミュレータへセット
    battle.selected[0] = team_p0
    battle.selected[1] = team_p1

    print(f"\n==================================================")
    print(f"  [Self-Play Match {match_id}] 試合開始 (Seed: {match_seed})")
    print(f"  - Player 0 構築: {[p.name for p in team_p0]}")
    print(f"  - Player 1 構築: {[p.name for p in team_p1]}")
    print(f"==================================================")

    # 3. 選出フェーズ（相性セレクターによる自動選出）
    sel_p0 = selector.select(team_p0, team_p1, num_select=3)
    sel_p1 = selector.select(team_p1, team_p0, num_select=3)

    battle.selected[0] = [deepcopy(team_p0[i]) for i in sel_p0]
    battle.selected[1] = [deepcopy(team_p1[i]) for i in sel_p1]

    # 4. 双方の「信念状態（相手の型へのベイズ推測）」を初期化（等確率 Prior）
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

        # フラット予測の仮説群をロード
        beliefs[pl].beliefs = {}
        for name in opp_names:
            beliefs[pl].beliefs[name] = analyzer._build_flat_belief(name)

    # 5. 0ターン目の解決（繰り出し）
    battle.turn = 0
    for player in range(2):
        battle.change_pokemon(player, idx=0, landing=False)
    for player in battle.speed_order:
        battle.land(player)

    # だっしゅつパック判定 (0ターン目)
    for pl in battle.speed_order:
        if battle.pokemon[pl].item == 'だっしゅつパック' and battle.pokemon[
            pl].rank_dropped and battle.changeable_indexes(pl):
            # 交代コマンドをランダム選択
            rand_cmd = random.choice(battle.available_commands(pl, phase='change'))
            battle.change_pokemon(pl, command=rand_cmd)
            battle.pokemon[pl].rank_dropped = False

    # 6. バトルループ
    history_log = []

    while battle.winner() is None:
        battle.turn += 1
        print(
            f"  ・Turn {battle.turn} - {battle.pokemon[0].name} (H:{battle.pokemon[0].hp_ratio:.0%}) vs {battle.pokemon[1].name} (H:{battle.pokemon[1].hp_ratio:.0%})")

        # 双方のCFRによるナッシュ最適手の決定
        commands = [None, None]
        for pl in [0, 1]:
            # 公開信念状態(PBS)の構築
            pbs = PublicBeliefState.from_battle(battle, perspective=pl, belief=beliefs[pl])

            # CFRソルバーによる戦略の解決（ワールドサンプリング数を3に抑えて高速化）
            cfr_solver.solver.num_samples = 3
            my_strategy, _ = cfr_solver.solve(pbs, battle)

            # 確率分布に従ってコマンドを決定
            if my_strategy:
                actions = list(my_strategy.keys())
                probs = list(my_strategy.values())
                commands[pl] = random.choices(actions, weights=probs, k=1)[0]
            else:
                commands[pl] = random.choice(battle.available_commands(pl))

        # ターンを解決
        battle.command = commands
        battle.proceed(commands=commands)

        # お互いの行動結果（ログ）をベイズ推論に同期・反映
        for pl in [0, 1]:
            opp = 1 - pl
            opp_poke = battle.pokemon[opp]
            if opp_poke and opp_poke.last_used_move:
                obs = Observation(
                    type=ObservationType.MOVE_USED,
                    pokemon_name=opp_poke.name,
                    details={"move": opp_poke.last_used_move}
                )
                beliefs[pl].update(obs)

        # ターン結果のダンプ
        history_log.append({
            "turn": battle.turn,
            "commands": commands,
            "hp": [battle.pokemon[0].hp, battle.pokemon[1].hp] if all(battle.pokemon) else [0, 0]
        })

        # 安全のための無限ループガード（最大50ターン）
        if battle.turn >= 50:
            print("  ⚠️ 50ターンに達したため引き分け終了します。")
            break

    winner = battle.winner()
    print(f"🏆 試合終了！ 勝者: Player {winner} (経過ターン: {battle.turn})")

    # 7. 学習用データセットの生成
    match_data = {
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
    return match_data


if __name__ == "__main__":
    from aegis_bot import AegisAnalyzer

    # 1. シミュレータの初期化
    Pokemon.init(season=22)

    # =========================================================================
    # 【図鑑補正パッチ】Pokemon.zukan における 'ギルガルド' のキー欠損補正
    # =========================================================================
    if hasattr(Pokemon, 'zukan'):
        if 'ギルガルド' not in Pokemon.zukan:
            # ギルガルド(シールド) または ギルガルド（シールド）（全角半角の表記揺れ対応）を探す
            for target_key in ['ギルガルド(シールド)', 'ギルガルド（シールド）']:
                if target_key in Pokemon.zukan:
                    # データをディープコピーして 'ギルガルド' キーを生成
                    Pokemon.zukan['ギルガルド'] = deepcopy(Pokemon.zukan[target_key])
                    Pokemon.zukan['ギルガルド']['display_name'] = 'ギルガルド'
                    print(f"ℹ️ [Patch] Pokemon.zukan に '{target_key}' から 'ギルガルド' のエイリアスを生成しました。")
                    break

    # アナライザーから共通設定をインポート
    print("Aegis環境データを統合中...")
    analyzer = AegisAnalyzer()

    # =========================================================================
    # 【動的補正】analyzer.belief_state が None の場合の初期化補正
    # =========================================================================
    if getattr(analyzer, 'belief_state', None) is None:
        print("ℹ️ analyzer.belief_state が None のため、動的に代替インスタンスを割り当てます。")
        from src.rebel.belief_state import PokemonBeliefState

        try:
            analyzer.belief_state = PokemonBeliefState()
        except Exception:
            dummy_belief = PokemonBeliefState.__new__(PokemonBeliefState)
            dummy_belief.usage_db = None
            dummy_belief.max_hypotheses = 30
            dummy_belief.min_probability = 0.01
            analyzer.belief_state = dummy_belief

    builder = analyzer.team_builder
    selector = analyzer.team_selector
    cfr_solver = analyzer.cfr_solver

    # 2. 自動自己対戦ループの設定
    num_matches = 10
    output_path = "log/selfplay_dataset.jsonl"
    os.makedirs("log", exist_ok=True)

    print(f"\n==================================================")
    print(f"  Aegis Self-Play 学習サイクル 起動")
    print(f"  総対戦回数: {num_matches}回戦")
    print(f"==================================================\n")

    start_time = time.time()

    with open(output_path, "w", encoding="utf-8") as f_out:
        for match_idx in range(1, num_matches + 1):
            print(f"\n--- ［試合 {match_idx}/{num_matches}］をシミュレート中 ---")

            try:
                # 自己対戦を実行 (analyzer インスタンスを引数に追加)
                match_data = run_single_selfplay_match(
                    match_id=match_idx,
                    builder=builder,
                    selector=selector,
                    cfr_solver=cfr_solver,
                    analyzer=analyzer
                )

                f_out.write(json.dumps(match_data, ensure_ascii=False) + "\n")
                print(f"➔ 試合 {match_idx} の対戦ログを '{output_path}' に保存完了。")
            except Exception as e:
                print(f"⚠️ 試合 {match_idx} の進行中にエラーが発生したためスキップします: {e}")
                import traceback

                traceback.print_exc()

    print(f"\n==================================================")
    print(f"  自己対戦サイクル 完了 (総所要時間: {time.time() - start_time:.1f}秒)")
    print(f"  生成データ: '{output_path}'")
    print(f"==================================================\n")