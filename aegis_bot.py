import sys
import types
from typing import Optional, List, Dict, Any, Set, Tuple

# =========================================================================
# 【Aegis Namespace Bridge for Web UI】
# Webサーバー(uvicorn)起動時にも古いシミュレータ名空間を pokepy へ自動リダイレクトするパッチ
# =========================================================================
# A. 空のプレースホルダーモジュールを先に sys.modules に登録する
sys.modules['src.pokemon_battle_sim'] = types.ModuleType('src.pokemon_battle_sim')
sys.modules['src.pokemon_battle_sim.pokemon'] = types.ModuleType('src.pokemon_battle_sim.pokemon')
sys.modules['src.pokemon_battle_sim.battle'] = types.ModuleType('src.pokemon_battle_sim.battle')
sys.modules['src.pokemon_battle_sim.damage'] = types.ModuleType('src.pokemon_battle_sim.damage')

# B. 依存関係を持たない純粋な便利関数モジュール(utils)を最優先でロードして結合
import pokepy.utils as utils_module
sys.modules['src.pokemon_battle_sim.utils'] = utils_module

# C. ポケモンとバトルの実体モジュールをロード
import pokepy.pokemon as pokemon_module
import pokepy.battle as battle_module

# D. プレースホルダーの内部を実体コードで同期・埋める
sys.modules['src.pokemon_battle_sim'].__dict__.update(pokemon_module.__dict__)
sys.modules['src.pokemon_battle_sim.pokemon'].__dict__.update(pokemon_module.__dict__)
sys.modules['src.pokemon_battle_sim.battle'].__dict__.update(battle_module.__dict__)  # 正しくbattle_moduleを割り当て
sys.modules['src.pokemon_battle_sim.damage'].__dict__.update(pokemon_module.__dict__)
# =========================================================================

# =========================================================================
# 2. 標準・外部ライブラリのロード
# =========================================================================
import os
import time
import warnings
import json
import numpy as np
import cv2
import mss  # 高速画面キャプチャライブラリ
import urllib.request  # SDK不要でGemini APIと直接通信

# =========================================================================
# 3. 共通モジュールからのクリーンインポート
# =========================================================================
from pokepy.pokemon import Pokemon
from pokepy.battle import Battle
from pokepy.pokebot import Pokebot
from src.rebel.belief_state import PokemonBeliefState, ObservationType, Observation, PokemonTypeHypothesis
from src.rebel.public_state import PublicBeliefState
from src.rebel.cfr_solver import ReBeLSolver, CFRConfig
from src.llm.state_representation import battle_to_llm_state


class AegisTeamBuilder:
    """
    Project Aegis 構築自動生成システム (Layer 15)

    指定された「軸(エース)」のポケモンをベースに、mbルールのプール(149種)から
    タイプ補完(サイクル)とアイテム重複制限(Item Clause)を満たした6体構築を自動で構築・保存する。
    """

    def __init__(self, learnsets: Dict[str, List[str]], mb_pokemon: Set[str], mb_items: Set[str]):
        self.learnsets = learnsets
        self.mb_pokemon = mb_pokemon
        self.mb_items = mb_items

    def calculate_weaknesses(self, types: List[str]) -> List[str]:
        """指定されたタイプの組み合わせに対する弱点タイプ（2倍以上）をリストアップする"""
        weaknesses = []
        for t_atk in Pokemon.type_id.keys():
            eff = 1.0
            for t_def in types:
                atk_id = Pokemon.type_id[t_atk]
                def_id = Pokemon.type_id[t_def]
                eff *= Pokemon.type_corrections[atk_id][def_id]
            if eff > 1.0:
                weaknesses.append(t_atk)
        return weaknesses

    def calculate_resistances(self, types: List[str]) -> List[str]:
        """指定されたタイプの組み合わせに対する耐性タイプ（0.5倍以下）をリストアップする"""
        resistances = []
        for t_atk in Pokemon.type_id.keys():
            eff = 1.0
            for t_def in types:
                atk_id = Pokemon.type_id[t_atk]
                def_id = Pokemon.type_id[t_def]
                eff *= Pokemon.type_corrections[atk_id][def_id]
            if eff <= 0.5:
                resistances.append(t_atk)
        return resistances

    def build_team(self, core_name: str) -> Dict[str, Any]:
        """指定された『軸(エース)』から、mbルールに適合した6体構築を自動生成する"""
        if core_name not in Pokemon.zukan:
            # 入力揺れの修正を試みる
            core_name = Pokemon.japanese_display_name.get(core_name, core_name)
            if core_name not in Pokemon.zukan and Pokemon.zukan_name.get(core_name):
                core_name = Pokemon.zukan_name[core_name][0]
            else:
                raise ValueError(f"指定されたポケモン '{core_name}' は図鑑データに存在しません。")

        team_members = [core_name]

        # サイクル補完評価に基づき、残りの5体を順に選出
        while len(team_members) < 6:
            current_weaknesses = []
            for member in team_members:
                current_weaknesses += self.calculate_weaknesses(Pokemon.zukan[member]["type"])

            best_candidate = None
            max_complement_score = -999.0

            # プール内の149体から探索
            for candidate in self.mb_pokemon:
                if candidate in team_members:
                    continue
                # 同一のディスプレネーム（重複種）を避ける
                if any(Pokemon.zukan[candidate]["display_name"] == Pokemon.zukan[m]["display_name"] for m in
                       team_members):
                    continue

                # 候補ポケモンの耐性を取得
                cand_res = self.calculate_resistances(Pokemon.zukan[candidate]["type"])

                # スコア計算: 現在のチームの弱点を、候補ポケモンがどれだけ半減以下でカバーできるか
                score = sum(2.0 if w in cand_res else 0.0 for w in current_weaknesses)

                # 種族値の高さも加味する
                score += sum(Pokemon.zukan[candidate]["base"]) * 0.001

                if score > max_complement_score:
                    max_complement_score = score
                    best_candidate = candidate

            if best_candidate:
                team_members.append(best_candidate)

        # 3. 構築メンバーへのアイテムの最適配分（Item Clauseの遵守）
        standard_items = ["ちからのハチマキ", "いのちのたま", "たべのこし", "とつげきチョッキ", "こだわりスカーフ",
                          "きあいのタスキ"]
        assigned_items = {}

        for member in team_members:
            # メガシンカ可能なポケモンかチェック
            mega_stone_name = member.split("(")[0] + "ナイト"
            if mega_stone_name in self.mb_items:
                assigned_items[member] = mega_stone_name
            else:
                # 汎用アイテムから重複を避けて割り当て
                for item in standard_items:
                    if item not in assigned_items.values() and item in self.mb_items:
                        assigned_items[member] = item
                        break
                else:
                    assigned_items[member] = ""

        # 4. 完成した6体のステータス、技、努力値を構築
        generated_party = {}
        for i, name in enumerate(team_members):
            base = Pokemon.zukan[name]["base"]
            is_phy = base[1] >= base[3]  # A >= C
            is_fast = base[5] >= 90

            nature = "いじっぱり" if is_phy else "ひかえめ"
            if is_fast:
                nature = "ようき" if is_phy else "おくびょう"

            effort = [0, 252, 0, 0, 0, 252] if is_phy else [252, 0, 0, 252, 0, 0]

            moves = self.learnsets.get(name, ["テラバースト"])[:4]
            while len(moves) < 4:
                moves.append("ままもる" if "ままもる" in self.learnsets.get(name, []) else "まもる")

            generated_party[str(i)] = {
                "name": name,
                "sex": 1 if i % 2 == 0 else -1,
                "level": 50,
                "nature": nature,
                "ability": Pokemon.zukan[name]["ability"][0],
                "item": assigned_items[name],
                "Ttype": Pokemon.zukan[name]["type"][0],
                "moves": moves,
                "indiv": [31, 31, 31, 31, 31, 31],
                "effort": effort
            }

        # 5. 生成されたパーティを party.log に保存（即時適用）
        os.makedirs("log", exist_ok=True)
        with open("log/party.log", "w", encoding="utf-8") as fout:
            json.dump(generated_party, fout, ensure_ascii=False, indent=2)

        return generated_party


class AegisTeamSelector:
    """
    Project Aegis 相性補完型チームセレクター（選出最適化エンジン）
    """

    def __init__(self, learnsets: Dict[str, List[str]]):
        self.learnsets = learnsets

    def evaluate_matchup(self, my_poke: Pokemon, opp_poke: Pokemon) -> float:
        """自分と相手のポケモンの1vs1における簡易有利度スコア"""
        score = 0.0
        my_moves = self.learnsets.get(my_poke.name, ["テラバースト"])
        opp_types = opp_poke.types

        best_my_eff = 0.0
        for move_name in my_moves:
            move_data = Pokemon.all_moves.get(move_name)
            if not move_data:
                continue
            move_type = move_data.get("type", "ノーマル")
            if move_data.get("class") == "sta":
                continue

            eff = 1.0
            for opp_type in opp_types:
                atk_id = Pokemon.type_id.get(move_type, 0)
                def_id = Pokemon.type_id.get(opp_type, 0)
                eff *= Pokemon.type_corrections[atk_id][def_id]

            if move_type == "じめん" and ("ひこう" in opp_types or opp_poke.ability == "ふゆう"):
                eff = 0.0

            if eff > best_my_eff:
                best_my_eff = eff

        opp_moves = self.learnsets.get(opp_poke.name, ["テラバースト"])
        my_types = my_poke.types

        best_opp_eff = 0.0
        for move_name in opp_moves:
            move_data = Pokemon.all_moves.get(move_name)
            if not move_data:
                continue
            move_type = move_data.get("type", "ノーマル")
            if move_data.get("class") == "sta":
                continue

            eff = 1.0
            for my_type in my_types:
                atk_id = Pokemon.type_id.get(move_type, 0)
                def_id = Pokemon.type_id.get(my_type, 0)
                eff *= Pokemon.type_corrections[atk_id][def_id]

            if move_type == "じめん" and ("ひこう" in my_types or my_poke.ability == "ふゆう"):
                eff = 0.0

            if eff > best_opp_eff:
                best_opp_eff = eff

        score = best_my_eff - best_opp_eff
        return score

    def select(self, my_team: List[Pokemon], opp_team: List[Pokemon], num_select: int = 3) -> List[int]:
        """最適な3体（先発1、控え2）を選出するインデックスのリストを返す"""
        if len(my_team) <= num_select:
            return list(range(len(my_team)))

        import itertools
        all_combinations = list(itertools.combinations(range(len(my_team)), num_select))

        best_combination = all_combinations[0]
        max_total_score = -999.0

        for combo in all_combinations:
            combo_score = 0.0
            for my_idx in combo:
                my_poke = my_team[my_idx]
                poke_score = 0.0
                for opp_poke in opp_team:
                    poke_score += self.evaluate_matchup(my_poke, opp_poke)
                combo_score += poke_score

            if combo_score > max_total_score:
                max_total_score = combo_score
                best_combination = combo

        selected_indices = list(best_combination)
        best_lead_idx = selected_indices[0]
        max_lead_score = -999.0

        for idx in selected_indices:
            lead_poke = my_team[idx]
            lead_score = sum(self.evaluate_matchup(lead_poke, opp) for opp in opp_team)
            if lead_score > max_lead_score:
                max_lead_score = lead_score
                best_lead_idx = idx

        selected_indices.remove(best_lead_idx)
        final_selection = [best_lead_idx] + selected_indices

        return final_selection


class AegisAnalyzer(Pokebot):
    """
    Project Aegis 戦略解析・配信観測AI（アナライザー）
    """

    def __init__(self, capture_box: Optional[Dict[str, int]] = None):
        # 1. ハードウェア操作（nxbt）接続処理を自動バイパス
        original_name = os.name
        os.name = 'nt'  # Windows扱いにして nxbt のロードを回避
        super().__init__()
        os.name = original_name

        # 2. 画面キャプチャ(mss)のセットアップ
        self.sct = mss.mss()
        self.capture_box = capture_box

        # 3. mbルール環境設定のロード（初期シーズンのカスタム設定）
        self.mb_pokemon: Set[str] = set()
        self.mb_items: Set[str] = set()
        self.mb_learnset: Dict[str, List[str]] = {}
        self._load_mb_rules()

        self.belief_state: Optional[PokemonBeliefState] = None
        self.last_processed_buffer_len = 0
        self.current_turn_processed = -1

        # 4. CFR/ReBeL ソルバーの初期化
        self.cfr_solver = ReBeLSolver(use_simplified=True, use_lightweight=False)

        # 5. Aegis チームセレクター（選出最適化エンジン）の初期化
        self.team_selector = AegisTeamSelector(learnsets=self.mb_learnset)

        # 6. Aegis チームビルダー（構築自動生成エンジン）の初期化 (Layer 15)
        self.team_builder = AegisTeamBuilder(
            learnsets=self.mb_learnset,
            mb_pokemon=self.mb_pokemon,
            mb_items=self.mb_items
        )

        # 7. Gemini API キーの確認
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

        # 8. デフォルトパーティファイルの自動生成（TeamBuilderを使用して生成）
        self._ensure_default_party_exists()

    def selection_command(self, player=0) -> List[int]:
        """選出画面で最適な選出インデックスのリストを自動計算"""
        if player == 0:
            print("[Aegis Selection] 相手のパーティに対する最適な選出パターンを計算しています...")
            t0 = time.time()
            best_selection = self.team_selector.select(self.party[0], self.party[1], num_select=3)

            lead_poke = self.party[0][best_selection[0]]
            bench_pokes = [self.party[0][best_selection[1]], self.party[0][best_selection[2]]]
            print(f"\n==================================================")
            print(f"  [Aegis SELECTION ANALYZER] 最適選出パターン決定 (計算時間: {time.time() - t0:.3f}秒)")
            print(f"==================================================")
            print(f" 🌟 先発推奨: 【{lead_poke.name}】")
            print(f"   (相手の並びに対し、最も出し勝ちしやすく展開を作りやすい先発です)")
            print(f" 👥 控え推奨: 【{bench_pokes[0].name}】 ＆ 【{bench_pokes[1].name}】")
            print(f"==================================================\n")

            return best_selection
        else:
            return list(range(3))

    def _ensure_default_party_exists(self) -> None:
        """手持ちパーティログが存在しない場合、チームビルダーで自動構築する"""
        os.makedirs("log", exist_ok=True)
        path = "log/party.log"
        if not os.path.exists(path):
            print("⚠️ log/party.log が見つかりません。Aegis TeamBuilder で自動構築します。")
            # 「ガブリアス」を軸とした最適な6体チームを自動設計
            self.team_builder.build_team("ガブリアス")
            print("✅ チームビルダーによる最強構築(1世代目)の自動生成が完了しました。")

    def _load_mb_rules(self) -> None:
        """mbルールの定義ファイル群をロードし、シミュレータに注入する"""
        try:
            with open("battle_data/mb_pokemon.txt", encoding="utf-8") as f:
                self.mb_pokemon = {line.strip() for line in f if line.strip()}
            Pokemon.permitted_pool = self.mb_pokemon

            with open("battle_data/mb_items.txt", encoding="utf-8") as f:
                self.mb_items = {line.strip() for line in f if line.strip()}
            Pokemon.permitted_items = self.mb_items
            Pokemon.mb_items = self.mb_items

            with open("battle_data/mb_learnset.json", encoding="utf-8") as f:
                self.mb_learnset = json.load(f)
            Pokemon.learnsets = self.mb_learnset

            print(f"✅ [Aegis Custom Rule] mbルールを適用しました。")
            print(f"   - 登録ポケモン数: {len(self.mb_pokemon)}種")
            print(f"   - 登録アイテム数: {len(self.mb_items)}種")
            print(f"   - 技習得データ数: {len(self.mb_learnset)}種")
        except FileNotFoundError as e:
            warnings.warn(f"mbルールの定義ファイルが見つかりません。デフォルト確率にフォールバックします: {e}")

    def capture(self, filename=''):
        """PCの画面を直接キャプチャして1080p画像に変換"""
        try:
            monitor = self.capture_box if self.capture_box else self.sct.monitors[1]
            screenshot = self.sct.grab(monitor)
            self.img = np.array(screenshot)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGRA2BGR)

            if self.img.shape[0] != 1080 or self.img.shape[1] != 1920:
                self.img = cv2.resize(self.img, (1920, 1080), interpolation=cv2.INTER_LINEAR)

            if filename:
                cv2.imwrite(filename, self.img)
        except Exception as e:
            warnings.warn(f"画面キャプチャに失敗しました: {e}")

    def init_belief_state(self) -> None:
        """相手のパーティが判明したタイミングでベイズモデルを初期化"""
        if not self.party[1]:
            return

        opponent_names = [p.name for p in self.party[1]]
        print(f"\n==================================================")
        print(f"[Aegis Initializing] mbルール用ベイズ推論を構築します。")
        print(f"対戦相手: {opponent_names}")
        print(f"==================================================\n")

        self.belief_state = PokemonBeliefState.__new__(PokemonBeliefState)
        self.belief_state.usage_db = None
        self.belief_state.max_hypotheses = 50
        self.belief_state.min_probability = 0.005
        self.belief_state.observation_history = []

        self.belief_state.revealed_moves = {name: set() for name in opponent_names}
        self.belief_state.revealed_items = {name: None for name in opponent_names}
        self.belief_state.revealed_tera = {name: None for name in opponent_names}
        self.belief_state.revealed_abilities = {name: None for name in opponent_names}
        self.belief_state.move_use_count = {name: {} for name in opponent_names}

        self.belief_state.beliefs = {}
        for name in opponent_names:
            self.belief_state.beliefs[name] = self._build_flat_belief(name)

    def _build_flat_belief(self, pokemon_name: str) -> Dict[PokemonTypeHypothesis, float]:
        """【初期シーズン専用】先入観ゼロのフラットな初期仮説群を生成する"""
        hypotheses: Dict[PokemonTypeHypothesis, float] = {}

        moves_pool = self.mb_learnset.get(pokemon_name, ["テラバースト"])
        item_pool = list(self.mb_items) if self.mb_items else [""]
        tera_pool = list(Pokemon.type_id.keys())
        abilities_pool = Pokemon.zukan.get(pokemon_name, {}).get("ability", [""])

        num_samples = 200
        rng = sys.modules['random']
        for _ in range(num_samples):
            moves = rng.sample(moves_pool, min(4, len(moves_pool)))

            mega_stone_name = pokemon_name.split("(")[0] + "ナイト"
            if mega_stone_name in item_pool and rng.random() < 0.5:
                item = mega_stone_name
            else:
                item = rng.choice(item_pool)

            tera = rng.choice(tera_pool)
            nature = rng.choice(
                ["いじっぱり", "ひかえめ", "ようき", "おくびょう", "わんぱく", "しんちょう", "おだやか", "ずぶとい"])
            ability = rng.choice(abilities_pool)

            hypothesis = PokemonTypeHypothesis.from_lists(
                moves=moves,
                item=item,
                tera_type=tera,
                nature=nature,
                ability=ability,
                base_stats=Pokemon.zukan.get(pokemon_name, {}).get("base")
            )
            hypotheses[hypothesis] = 1.0

        return self.belief_state._prune_and_normalize(hypotheses)

    def translate_buffer_to_observation(self, dict_event: Dict[str, Any]) -> Optional[Observation]:
        """OCRバッファをベイズ観測イベントに変換"""
        if dict_event.get("player") != 1:
            return None

        display_name = dict_event.get("display_name")
        if not display_name:
            return None

        pokemon_names = Pokemon.zukan_name.get(display_name, [])
        if not pokemon_names:
            return None
        pokemon_name = pokemon_names[0]

        if "item" in dict_event:
            return Observation(
                type=ObservationType.ITEM_REVEALED,
                pokemon_name=pokemon_name,
                details={"item": dict_event["item"]}
            )
        elif "lost_item" in dict_event:
            return Observation(
                type=ObservationType.ITEM_REVEALED,
                pokemon_name=pokemon_name,
                details={"item": ""}
            )
        elif "move" in dict_event:
            return Observation(
                type=ObservationType.MOVE_USED,
                pokemon_name=pokemon_name,
                details={"move": dict_event["move"]}
            )
        elif "ability" in dict_event:
            return Observation(
                type=ObservationType.ABILITY_REVEALED,
                pokemon_name=pokemon_name,
                details={"ability": dict_event["ability"]}
            )

        return None

    def update_beliefs(self) -> None:
        """ログバッファから信念を自動更新"""
        if self.belief_state is None:
            if self.party[1]:
                self.init_belief_state()
            else:
                return

        current_buffer_len = len(self.process_buffer)
        if current_buffer_len > self.last_processed_buffer_len:
            new_events = self.process_buffer[self.last_processed_buffer_len:]

            for event in new_events:
                observation = self.translate_buffer_to_observation(event)
                if observation:
                    self.belief_state.update(observation)

            self.last_processed_buffer_len = current_buffer_len

    def clone(self, player: int = None) -> Battle:
        """脳内クローン（推論された世界を1つサンプリングして複製）"""
        battle_clone = deepcopy(self)
        if player is None or self.belief_state is None:
            return battle_clone

        sampled_opponent_team = self.belief_state.sample_world()

        opponent_player = not player
        for p_name, hypothesis in sampled_opponent_team.items():
            p = Pokemon.find(battle_clone.selected[opponent_player], name=p_name)
            if p:
                p.item = hypothesis.item
                p.moves = list(hypothesis.moves)
                p.nature = hypothesis.nature
                p.ability = hypothesis.ability
                p.effort = hypothesis.get_evs()
                p.indiv = hypothesis.get_ivs()
                p.update_status(keep_damage=True)

        return battle_clone

    def request_gemini_commentary(self, prompt: str) -> str:
        """【Layer 17】Gemini APIから直接実況解説テキストを取得する"""
        if not self.gemini_api_key or self.gemini_api_key == "YOUR_GEMINI_API_KEY":
            return "※ [Aegis] APIキーが設定されていません。戦略数値のみを出力します。"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 250
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                commentary = response_data['contents'][0]['parts'][0]['text']
                return commentary.strip()
        except Exception as e:
            return f"※ [Aegis Live Commentary] 解説の取得に失敗しました: {e}"

    def run_strategy_analysis(self) -> None:
        """CFRゲーム理論解析 ＆ Geminiによるプロ級実況解説の出力"""
        self.update_beliefs()
        if self.belief_state is None or self.pokemon[0] is None or self.pokemon[1] is None:
            return

        if self.turn == self.current_turn_processed:
            return
        self.current_turn_processed = self.turn

        active_enemy = self.pokemon[1].name
        item_dist = self.belief_state.get_item_distribution(active_enemy)
        top_items = sorted(item_dist.items(), key=lambda x: -x[1])[:3]
        item_str = " | ".join(f"{item}: {prob:.1%}" for item, prob in top_items if prob > 0.05)

        ability_dist = self.belief_state.get_ability_distribution(active_enemy)
        top_abilities = sorted(ability_dist.items(), key=lambda x: -x[1])[:2]
        ability_str = " | ".join(f"{ab}: {prob:.1%}" for ab, prob in top_abilities if prob > 0.05)

        pbs = PublicBeliefState.from_battle(self, perspective=0, belief=self.belief_state)

        try:
            my_strategy, opp_strategy = self.cfr_solver.solve(pbs, self)
        except Exception as e:
            warnings.warn(f"CFRソルバーの計算中にエラーが発生しました。フォールバックします: {e}")
            my_strategy, opp_strategy = {}, {}

        llm_state = battle_to_llm_state(self, player=0)
        state_yaml = llm_state.state_text

        my_strategy_text = ""
        sorted_my_strat = sorted(my_strategy.items(), key=lambda x: -x[1]) if my_strategy else []
        for cmd, prob in sorted_my_strat:
            if prob > 0.01:
                my_strategy_text += f"  - {self._get_command_name(cmd, self.pokemon[0])} (推奨確率: {prob:.1%})\n"

        opp_strategy_text = ""
        sorted_opp_strat = sorted(opp_strategy.items(), key=lambda x: -x[1]) if opp_strategy else []
        for cmd, prob in sorted_opp_strat:
            if prob > 0.01:
                opp_strategy_text += f"  - {self._get_command_name(cmd, self.pokemon[1])} (予測確率: {prob:.1%})\n"

        commentary_prompt = (
            f"あなたはポケモンの公式世界大会の日本語解説者です。\n"
            f"現在の盤面データ（YAML）と、ゲーム理論AIが計算した「両プレイヤーの最適行動（混合戦略）」を元に、\n"
            f"現在のターンにおける戦況の解説と、AIがなぜその技を推奨しているのか、その理由（タイプ相性、サイクル戦、リスクの最小化など）を、\n"
            f"熱く、かつ論理的（知的）に実況解説してください。\n\n"
            f"【現在の盤面情報】\n{state_yaml}\n"
            f"【相手の型予測（ベイズ推定）】\n  持ち物候補: {item_str}\n  特性候補: {ability_str}\n\n"
            f"【AIが計算した自分の推奨戦略一覧（CFR）】\n{my_strategy_text}\n"
            f"【AIが予測した相手の最適戦略一覧（CFR）】\n{opp_strategy_text}\n"
            f"※回答は、150文字〜200文字程度の短い解説スピーチとして日本語で出力してください。Markdownは使わずプレーンテキストにしてください。"
        )

        commentary_text = self.request_gemini_commentary(commentary_prompt)

        print(f"\n==================================================")
        print(f"  [Aegis ReBeL COMMENTARY] Turn {self.turn} の戦略実況解説")
        print(f"==================================================")
        print(f"▼ 対面状況: {self.pokemon[0].name}  vs  {self.pokemon[1].name}")
        print(
            f"▼ 自分の残りHP: {self.pokemon[0].hp}/{self.pokemon[0].status[0]}  |  相手の残りHP割合: {int(self.pokemon[1].hp_ratio * 100)}%")
        print(f"▼ 相手の持ち物予測: {item_str}")
        print(f"\n【🎙️ AI解説者による状況分析と解説】")
        print(f"  {commentary_text}")
        print(f"--------------------------------------------------")
        if sorted_my_strat:
            best_desc = self._get_command_name(sorted_my_strat[0][0], self.pokemon[0])
            print(f"👉 AI推奨の最善手: 『 {best_desc} 』 (均衡確率: {sorted_my_strat[0][1]:.1%})")
        print(f"==================================================\n")

    def _get_command_name(self, cmd: int, perspective_pokemon: Pokemon) -> str:
        """コマンドIDから実況出力用の技名・交代先名を取得する"""
        if cmd == Battle.SURRENDER:
            return "降参 (Surrender)"
        elif cmd == Battle.STRUGGLE:
            return "わるあがき"
        elif cmd == Battle.SKIP:
            return "行動スキップ"
        elif cmd < 10:
            return f"技: 【{perspective_pokemon.moves[cmd]}】"
        elif cmd < 20:
            return f"テラスタル ➔ 技: 【{perspective_pokemon.moves[cmd % 10]}】"
        elif cmd in range(20, 26):
            target_poke = self.selected[perspective_pokemon.sex != 1][cmd - 20]  # プレイヤーインデックス
            return f"交代 ➔ 【{target_poke.name}】"
        return "様子見 / その他"

    def run_observer_loop(self) -> None:
        """画面を監視し続け、解説を出力する常時監視ループ"""
        print("\n==================================================")
        print("  Project Aegis 観測・解説システム 起動中...")
        print("  対象ルール: カスタム mbルール (初期シーズン仕様)")
        print("  思考エンジン: CFR (Counterfactual Regret Minimization)")
        print(r"  選出エンジン: Aegis 相性補完セレクター ( $_6 \mathrm{C}_3 $ 総当たり評価)")
        print("  解説エンジン: Gemini 1.5 Flash (urllib 直結)")
        print("  監視モード: Observer Mode")
        print("==================================================\n")

        self.load_party()  # 自分のパーティ情報をロード
        print("画面を監視しています。対戦画面が表示されるのを待っています...")

        while True:
            self.capture()
            phase = self.read_phase(capture=False)

            if phase == 'selection' and not self.selection_finished:
                print("[Aegis Vision] 選出画面を検知しました。相手パーティを読み込みます...")
                time.sleep(1)
                self.capture()
                self.reset_game()
                self.read_enemy_party(capture=False)

                # 選出最適化エンジンの実行 ➔ 相性に基づいた最適な3体を決定
                self.selection_command(player=0)

                # ベイズ確率モデルの自動初期化
                self.init_belief_state()
                self.selection_finished = True
                self.turn = 0

            elif phase == 'battle':
                self.selection_finished = False

                if self.read_battle_situlation():
                    self.update_beliefs()

                    self.turn = self.turn_numbers() if hasattr(self, 'turn_numbers') else (self.turn + 1)
                    if self.turn == 0:
                        self.turn = 1

                    self.run_strategy_analysis()
                else:
                    time.sleep(0.5)

            elif phase == 'change':
                for i in range(len(self.selected[0])):
                    hp = self.read_party_hp(i, capture=(i == 0))
                    p = Pokemon.find(self.selected[0], display_name=self.read_party_display_name(i))
                    if p:
                        p.hp = hp
                self.update_beliefs()

            else:
                if all(self.pokemon):
                    if self.read_bottom_text(capture=False):
                        self.update_beliefs()
                    for player in range(2):
                        if self.read_ability_text(player, capture=False):
                            self.update_beliefs()

                if (result := self.read_win_lose(capture=False)):
                    print(f"\n【対戦終了検出】 結果は '{result}' でした。次の対戦を待機します。")
                    self.reset_game()

            time.sleep(0.1)


# デバッグ実行用エントリーポイント
if __name__ == "__main__":
    # シミュレータの初期化
    Pokemon.init(season=22)

    # 画面キャプチャのターゲット設定（Noneの場合は全画面監視）
    my_box = None

    # ----------------------------------------------------
    # 【Aegis TeamBuilder デバッグ用コマンド】
    # 新しい軸（エース）から、いつでもオリジナルの最強mbルール構築（6体）を
    # 自動作成し、log/party.log に上書き保存できます。
    #
    # 例：カバルドン軸を作りたい場合、以下のコメントアウトを解除して1度だけ実行します。
    # ----------------------------------------------------
    # test_builder = AegisAnalyzer()
    # test_builder.team_builder.build_team("カバルドン")
    # print("カバルドン軸の構築を自動作成しました。")
    # sys.exit(0)
    # ----------------------------------------------------

    analyzer = AegisAnalyzer(capture_box=my_box)
    analyzer.run_observer_loop()