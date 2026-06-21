import sys
import types
from typing import Optional, List, Dict, Any, Set, Tuple
from copy import deepcopy
import builtins
import random
import os
import json
import io
import time
import warnings
import numpy as np
import cv2
import mss  # 高速画面キャプチャライブラリ
import urllib.request  # SDK不要でGemini APIと直接通信

# =========================================================================
# 0. 外部高度推論ライブラリの安全ロード（Safe Fallback）
# =========================================================================
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None
except Exception:
    Word2Vec = None

try:
    from src.damage_calculator_api.calculators.damage_calculator import calculate_damage
except ImportError:
    calculate_damage = None

# =========================================================================
# 1. 【File Path Redirect & Aegislash Data Patch】
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
# 2. 【Aegis Namespace Bridge】
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

from pokepy.pokemon import Pokemon
from pokepy.battle import Battle
from pokepy.pokebot import Pokebot
from src.rebel.belief_state import PokemonBeliefState, ObservationType, Observation, PokemonTypeHypothesis
from src.rebel.public_state import PublicBeliefState
from src.rebel.cfr_solver import ReBeLSolver, CFRConfig
from src.llm.state_representation import battle_to_llm_state

Battle.can_terastal = lambda self, player: False


# =========================================================================
# 3. 【高度化】AegisTeamBuilder (重み付き型構築システム搭載) [2]
# =========================================================================
class AegisTeamBuilder:
    """
    Project Aegis 構築自動生成システム (Layer 15)
    Word2Vecによるシナジーと、性格・努力値・持ち物・特性・技の「重み付き対戦特化サンプリング」を統合。
    """

    # 1. メガシンカ確率の重み付け (通常が強すぎる枠は33%, メガ必須枠は66%)
    MEGA_PROBABILITIES = {
        "スピアー": 0.66, "クチート": 0.66, "ガルーラ": 0.66, "ボスゴドラ": 0.66,
        "チャーレム": 0.66, "ヤミラミ": 0.66, "ライボルト": 0.66, "ジュペッタ": 0.66,
        "アブソル": 0.66, "オニゴーリ": 0.66, "ピジョット": 0.66, "ヘルガー": 0.66,
        "エルレイド": 0.66, "サメハダー": 0.66, "バクーダ": 0.66, "チルタリス": 0.66,
        "ガブリアス": 0.33, "ボーマンダ": 0.33, "バンギラス": 0.33, "ギャラドス": 0.33,
        "メタグロス": 0.33, "ゲンガー": 0.33, "ハッサム": 0.33, "カイリュー": 0.33,
        "ヘラクロス": 0.33, "エアームド": 0.33, "シビルドン": 0.33, "シャンデラ": 0.33,
        "ドリュウズ": 0.33, "ブリガロン": 0.33, "マフォクシー": 0.33, "カラマネロ": 0.33,
        "ゲッコウガ": 0.33, "ドラミドロ": 0.33, "ルチャブル": 0.33, "タイレーツ": 0.33,
        "スコヴィラン": 0.33, "キラフロル": 0.33
    }

    # 2. 強力な持ち物の傾斜重み (強い持ち物を高確率で優先配分)
    ITEM_TIERS = {
        # S Tier (重み 5.0)
        "きあいのタスキ": 5.0, "こだわりハチマキ": 5.0, "こだわりメガネ": 5.0, "こだわりスカーフ": 5.0,
        "とつげきチョッキ": 5.0, "いのちのたま": 5.0, "たべのこし": 5.0, "オボンのみ": 5.0, "ラムのみ": 5.0,
        # A Tier (重み 2.0)
        "ゴツゴツメット": 2.0, "あつぞこブーツ": 2.0, "しんかのきせき": 2.0, "おうじゃのしるし": 2.0,
        "しろいハーブ": 2.0, "たつじんのおび": 2.0, "ピントレンズ": 2.0, "おおきなねっこ": 2.0,
        "ヨプのみ": 2.0, "ヤチェのみ": 2.0, "シュカのみ": 2.0, "オッカのみ": 2.0, "ロゼルのみ": 2.0,
        "リンドのみ": 2.0, "バコウのみ": 2.0, "ソクノのみ": 2.0, "ハバンのみ": 2.0, "ビアーのみ": 2.0,
        "イトケのみ": 2.0, "ホズのみ": 2.0, "カシブのみ": 2.0, "ナモのみ": 2.0, "タンガのみ": 2.0,
        # B Tier (重み 1.0)
        "メタルコート": 1.0, "しんぴのしずく": 1.0, "やわらかいすな": 1.0, "きせきのタネ": 1.0,
        "もくたん": 1.0, "じしゃく": 1.0, "とけないこおり": 1.0, "まがったスプーン": 1.0,
        "かたいいし": 1.0, "のろいのおふだ": 1.0, "りゅうのキバ": 1.0, "くろおび": 1.0,
        "どくばり": 1.0, "するどいくちばし": 1.0, "ようせいのハネ": 1.0, "きれいなぬけがら": 1.0,
        "ヒメリのみ": 1.0, "クラボのみ": 1.0, "チーゴのみ": 1.0, "ナナシのみ": 1.0, "オレンのみ": 1.0,
    }

    # 3. 強特性の選定バイアス (強特性は66% / その他は33%の確率比で選択)
    POWERFUL_ABILITIES = {
        "てんねん", "へんげんじざい", "かそく", "いかく", "マルチスケイル", "いたずらごころ",
        "ちからもち", "ポイズンヒール", "テクニシャン", "きれあじ", "おうごんのからだ",
        "ちょすい", "ひでり", "あめふらし", "すいすい", "すなかき", "ゆきかき", "ようりょくそ",
        "ダウンロード", "トレース", "バトルスイッチ", "おやこあい", "マジックミラー", "かげふみ"
    }

    # 4. 実戦性格の傾斜配分 (主要アタッカー性格に72%のウェイトを配分)
    NATURE_WEIGHTS = {
        "いじっぱり": 0.18, "ひかえめ": 0.18, "ようき": 0.18, "おくびょう": 0.18,
        "わんぱく": 0.07, "しんちょう": 0.07, "ずぶとい": 0.07, "おだやか": 0.07
    }

    # 5. 強力な変化・サポート・展開技の定義 (技構成の重み付けに使用)
    POWERFUL_MOVES_KEYWORDS = {
        "ステルスロック", "あくび", "ちょうはつ", "おにび", "でんじは", "へびにらみ", "ねばねばネット",
        "つるぎのまい", "りゅうのまい", "ちょうのまい", "めいそん", "てっぺき", "ビルドアップ",
        "なまける", "じこさいせい", "はねやすめ", "こうごうせい", "ちからをすいとる",
        "キングシールド", "トーチカ", "ニードルガード", "アンコール", "いたみわけ", "しっぽきり"
    }

    def __init__(self, learnsets: Dict[str, List[str]], mb_pokemon: Set[str], mb_items: Set[str]):
        self.learnsets = learnsets
        self.mb_pokemon = mb_pokemon
        self.mb_items = mb_items

        self.w2v_model = None
        w2v_path = "data/pokemon_word2vec.model"
        if Word2Vec and os.path.exists(w2v_path):
            try:
                self.w2v_model = Word2Vec.load(w2v_path)
                print(f"ℹ️ [Aegis Builder] Word2Vec 構築共起モデル '{w2v_path}' をロードしました。")
            except Exception as e:
                warnings.warn(f"Word2Vecモデルのロードに失敗しました: {e}")

    def get_w2v_synergy(self, member_name: str, candidate_name: str) -> float:
        if self.w2v_model is None:
            return 0.0
        try:
            return float(self.w2v_model.wv.similarity(member_name, candidate_name))
        except KeyError:
            return 0.0

    def calculate_weaknesses(self, types: List[str]) -> List[str]:
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
        if core_name == "ギルガルド" and "ギルガルド" not in Pokemon.zukan:
            for k in ['ギルガルド(シールド)', 'ギルガルド（シールド）']:
                if k in Pokemon.zukan:
                    Pokemon.zukan['ギルガルド'] = deepcopy(Pokemon.zukan[k])
                    Pokemon.zukan['ギルガルド']['display_name'] = 'ギルガルド'
                    break

        if core_name not in Pokemon.zukan:
            core_name = Pokemon.japanese_display_name.get(core_name, core_name)
            if core_name not in Pokemon.zukan and Pokemon.zukan_name.get(core_name):
                core_name = Pokemon.zukan_name[core_name][0]
            else:
                raise ValueError(f"指定されたポケモン '{core_name}' は図鑑データに存在しません。")

        team_members = [core_name]

        # 1. 軸に基づいたメンバー5体の決定 (補完 ＆ 共起シナジー)
        while len(team_members) < 6:
            current_weaknesses = []
            for member in team_members:
                current_weaknesses += self.calculate_weaknesses(Pokemon.zukan[member]["type"])

            best_candidate = None
            max_total_score = -999.0

            for candidate in self.mb_pokemon:
                if candidate in team_members:
                    continue

                if candidate == "ギルガルド" and "ギルガルド" not in Pokemon.zukan:
                    for k in ['ギルガルド(シールド)', 'ギルガルド（シールド）']:
                        if k in Pokemon.zukan:
                            Pokemon.zukan['ギルガルド'] = deepcopy(Pokemon.zukan[k])
                            Pokemon.zukan['ギルガルド']['display_name'] = 'ギルガルド'
                            break

                if not Pokemon.zukan.get(candidate):
                    continue

                if any(Pokemon.zukan[candidate]["display_name"] == Pokemon.zukan[m]["display_name"] for m in
                       team_members):
                    continue

                cand_res = self.calculate_resistances(Pokemon.zukan[candidate]["type"])
                type_score = sum(2.0 if w in cand_res else 0.0 for w in current_weaknesses)
                type_score += sum(Pokemon.zukan[candidate]["base"]) * 0.001

                w2v_score = 0.0
                if self.w2v_model:
                    synergies = [self.get_w2v_synergy(m, candidate) for m in team_members]
                    w2v_score = sum(synergies) / len(team_members) if synergies else 0.0

                total_score = type_score + (w2v_score * 5.0)

                if total_score > max_total_score:
                    max_total_score = total_score
                    best_candidate = candidate

            if best_candidate:
                team_members.append(best_candidate)

        # 2. 重み付きアイテム決定ロジック [2]
        assigned_items = {}
        mega_stones_in_pool = {item for item in self.mb_items if "ナイト" in item}
        normal_items_pool = list(self.mb_items - mega_stones_in_pool)

        # メンバーごとにメガシンカ可能なら「設定された確率重み」でストーンを割り当て
        for member in team_members:
            mega_stone_name = member.split("(")[0] + "ナイト"

            if mega_stone_name in self.mb_items:
                # ユーザー要求: メガシンカ確率を33%～66%の確率重みに動的設定 [2]
                mega_prob = self.MEGA_PROBABILITIES.get(member, 0.50)

                if random.random() < mega_prob:
                    assigned_items[member] = mega_stone_name
                    continue

            # メガに選ばれなかった場合、通常の持ち物から重複なしで重み付き選定
            available_items = [item for item in normal_items_pool if item not in assigned_items.values()]
            if available_items:
                # ユーザー要求: 持ち物の強さに合わせて重み(確率)付きサンプリング [2]
                item_weights = [self.ITEM_TIERS.get(itm, 0.1) for itm in available_items]
                chosen_item = random.choices(available_items, weights=item_weights, k=1)[0]
                assigned_items[member] = chosen_item
            else:
                assigned_items[member] = ""

        # 3. 特性・性格・努力値・技構成の重み付き組み立て [2]
        generated_party = {}
        for i, name in enumerate(team_members):
            zukan_entry = Pokemon.zukan[name]

            # ユーザー要求: 性格の実戦重み付きランダム配分
            natures = list(self.NATURE_WEIGHTS.keys())
            nature_w = list(self.NATURE_WEIGHTS.values())
            nature = random.choices(natures, weights=nature_w, k=1)[0]

            # ユーザー要求: 特性の強さに応じた重み付き選定 (強特性=2.0、その他=1.0)
            abilities = zukan_entry.get("ability", ["とくせいなし"])
            if abilities:
                ability_weights = [2.0 if ab in self.POWERFUL_ABILITIES else 1.0 for ab in abilities]
                ability = random.choices(abilities, weights=ability_weights, k=1)[0]
            else:
                ability = "とくせいなし"

            # 努力値配分 (50%極振り / 50%完全ランダム)
            if random.random() < 0.5:
                effort = [0] * 6
                all_indices = [0, 1, 2, 3, 4, 5]
                max_two = random.sample(all_indices, 2)
                for idx in max_two:
                    effort[idx] = 252
                remaining = [idx for idx in all_indices if idx not in max_two]
                last_four = random.choice(remaining)
                effort[last_four] = 4
            else:
                effort = [0] * 6
                total_units = 127
                for _ in range(total_units):
                    valid_indices = [idx for idx in range(6) if effort[idx] < 252]
                    if not valid_indices:
                        break
                    idx = random.choice(valid_indices)
                    effort[idx] += 4

            # ユーザー要求: 技構成の強さ重み付きランダム配分
            learnable = self.learnsets.get(name, ["テラバースト"])

            # 各技の強さ(威力、先制、強力補助)をスコアリングして重み付け
            move_weights = []
            for move_name in learnable:
                move_data = Pokemon.all_moves.get(move_name)
                weight = 1.0  # デフォルト
                if move_data:
                    power = move_data.get("power", 0)
                    priority = move_data.get("priority", 0)
                    move_class = move_data.get("class", "phy")

                    # 威力80以上の攻撃技、または先制技、または主要な補助技は重み 3.0
                    if power >= 80 or priority > 0 or move_name in self.POWERFUL_MOVES_KEYWORDS:
                        weight = 3.0
                    # 使用されないゴミ変化技は重みを極小 0.1 にして徹底排除
                    elif move_class == "sta" and move_name not in self.POWERFUL_MOVES_KEYWORDS:
                        weight = 0.1
                move_weights.append(weight)

            # 重複なしでの重み付きサンプリング (Pure Python 高速版)
            chosen_moves = []
            temp_pool = list(learnable)
            temp_weights = list(move_weights)
            num_to_select = min(4, len(temp_pool))

            for _ in range(num_to_select):
                if sum(temp_weights) <= 0:
                    temp_weights = [1.0] * len(temp_pool)
                chosen = random.choices(temp_pool, weights=temp_weights, k=1)[0]
                chosen_moves.append(chosen)
                idx = temp_pool.index(chosen)
                temp_pool.pop(idx)
                temp_weights.pop(idx)

            generated_party[str(i)] = {
                "name": name,
                "sex": 1 if i % 2 == 0 else -1,
                "level": 50,
                "nature": nature,
                "ability": ability,
                "item": assigned_items.get(name, ""),
                "Ttype": zukan_entry["type"][0],
                "moves": chosen_moves,
                "indiv": [31, 31, 31, 31, 31, 31],
                "effort": effort
            }

        os.makedirs("log", exist_ok=True)
        with open("log/party.log", "w", encoding="utf-8") as fout:
            json.dump(generated_party, fout, ensure_ascii=False, indent=2)

        return generated_party


# =========================================================================
# 4. 【高度化】AegisTeamSelector (BERT選出予測の安全な統合)
# =========================================================================
class AegisTeamSelector:
    """
    Project Aegis 相性補完型＆BERT予測型チームセレクター（選出最適化エンジン）
    """

    def __init__(self, learnsets: Dict[str, List[str]]):
        self.learnsets = learnsets

        self.bert_model = None
        bert_path = "src/selection_bert/selection_bert.pth"

        if torch and os.path.exists(bert_path):
            try:
                import importlib

                base_dir = os.path.dirname(os.path.abspath(__file__))
                src_dir = os.path.join(base_dir, "src")
                if src_dir not in sys.path:
                    sys.path.append(src_dir)
                if base_dir not in sys.path:
                    sys.path.append(base_dir)

                model_module = importlib.import_module("selection_bert.model")
                target_class = None

                if hasattr(model_module, "SelectionBERT"):
                    target_class = getattr(model_module, "SelectionBERT")
                else:
                    for attr_name in dir(model_module):
                        attr = getattr(model_module, attr_name)
                        if isinstance(attr, type) and nn and issubclass(attr, nn.Module) and attr.__name__ != "Module":
                            target_class = attr
                            break

                if target_class is not None:
                    self.bert_model = target_class()
                    self.bert_model.load_state_dict(torch.load(bert_path, map_location="cpu"))
                    self.bert_model.eval()
                    print(
                        f"ℹ️ [Aegis Selector] 検出された選出予測モデル '{target_class.__name__}' を正常にロードしました。")
                else:
                    warnings.warn("selection_bert/model.py 内に有効な PyTorch モデルクラスが見つかりません。")

            except Exception as e:
                warnings.warn(f"SelectionBERTの動的ロードに失敗しました(相性総当たり評価にフォールバックします): {e}")

    def evaluate_matchup(self, my_poke: Pokemon, opp_poke: Pokemon) -> float:
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

    def get_bert_prob_score(self, combo: Tuple[int, ...], my_team: List[Pokemon], opp_team: List[Pokemon]) -> float:
        if self.bert_model is None or torch is None:
            return 0.0
        try:
            my_ids = torch.tensor([[Pokemon.zukan_name.get(p.name, [0])[0] for p in my_team]], dtype=torch.long)
            opp_ids = torch.tensor([[Pokemon.zukan_name.get(p.name, [0])[0] for p in opp_team]], dtype=torch.long)

            with torch.no_grad():
                probs = self.bert_model(my_ids, opp_ids)
                score = float(sum(probs[0][idx].item() for idx in combo))
                return score
        except Exception:
            return 0.0

    def select(self, my_team: List[Pokemon], opp_team: List[Pokemon], num_select: int = 3) -> List[int]:
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

            bert_score = self.get_bert_prob_score(combo, my_team, opp_team)
            total_score = combo_score + (bert_score * 10.0)

            if total_score > max_total_score:
                max_total_score = total_score
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


# =========================================================================
# 5. 【高度化】AegisAnalyzer (行動順・被ダメージからのベイズ逆算看破)
# =========================================================================
class AegisAnalyzer(Pokebot):
    """
    Project Aegis 戦略解析・配信観測AI（アナライザー）
    """

    def __init__(self, capture_box: Optional[Dict[str, int]] = None):
        original_name = os.name
        os.name = 'nt'
        super().__init__()
        os.name = original_name

        try:
            self.sct = mss.MSS()
        except Exception:
            self.sct = None
        self.capture_box = capture_box

        self.mb_pokemon: Set[str] = set()
        self.mb_items: Set[str] = set()
        self.mb_learnset: Dict[str, List[str]] = {}
        self._load_mb_rules()

        self.belief_state: Optional[PokemonBeliefState] = None
        self.last_processed_buffer_len = 0
        self.current_turn_processed = -1

        self.cfr_solver = ReBeLSolver(use_simplified=True, use_lightweight=False)
        self.team_selector = AegisTeamSelector(learnsets=self.mb_learnset)

        self.team_builder = AegisTeamBuilder(
            learnsets=self.mb_learnset,
            mb_pokemon=self.mb_pokemon,
            mb_items=self.mb_items
        )

        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
        self._ensure_default_party_exists()

    def selection_command(self, player=0) -> List[int]:
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
        os.makedirs("log", exist_ok=True)
        path = "log/party.log"
        if not os.path.exists(path):
            print("⚠️ log/party.log が見つかりません。Aegis TeamBuilder で自動構築します。")
            self.team_builder.build_team("ガブリアス")
            print("✅ チームビルダーによる最強構築(1世代目)の自動生成が完了しました。")

    def _load_mb_rules(self) -> None:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            pokemon_path = os.path.join(base_dir, "battle_data", "mb_pokemon.txt")
            items_path = os.path.join(base_dir, "battle_data", "mb_items.txt")
            learnset_path = os.path.join(base_dir, "battle_data", "mb_learnset.json")

            with open(pokemon_path, encoding="utf-8") as f:
                self.mb_pokemon = {line.strip() for line in f if line.strip()}
            Pokemon.permitted_pool = self.mb_pokemon

            with open(items_path, encoding="utf-8") as f:
                self.mb_items = {line.strip() for line in f if line.strip()}
            Pokemon.permitted_items = self.mb_items
            Pokemon.mb_items = self.mb_items

            with open(learnset_path, encoding="utf-8") as f:
                self.mb_learnset = json.load(f)
            Pokemon.learnsets = self.mb_learnset

            print(f"✅ [Aegis Custom Rule] mbルールを適用しました。")
            print(f"   - 登録ポケモン数: {len(self.mb_pokemon)}種")
            print(f"   - 登録アイテム数: {len(self.mb_items)}種")
            print(f"   - 技習得データ数: {len(self.mb_learnset)}種")
        except FileNotFoundError as e:
            warnings.warn(f"mbルールの定義ファイルが見つかりません。デフォルト確率にフォールバックします: {e}")

    def capture(self, filename=''):
        try:
            if self.sct is None:
                return
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

        belief_state = self.belief_state
        if belief_state is None:
            belief_state = PokemonBeliefState.__new__(PokemonBeliefState)
            belief_state.usage_db = None
            belief_state.max_hypotheses = 50
            belief_state.min_probability = 0.005

        if hasattr(belief_state, '_prune_and_normalize'):
            return belief_state._prune_and_normalize(hypotheses)
        else:
            total = sum(h for h in hypotheses.values())
            for h in hypotheses:
                hypotheses[h] /= (total if total > 0 else 1.0)
            return hypotheses

    def translate_buffer_to_observation(self, dict_event: Dict[str, Any]) -> Optional[Observation]:
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

    def update_beliefs_by_implicit_observations(self) -> None:
        if self.belief_state is None or self.pokemon[0] is None or self.pokemon[1] is None:
            return

        active_enemy = self.pokemon[1].name
        if active_enemy not in self.belief_state.beliefs:
            return

        if hasattr(self, 'speed_order') and self.speed_order:
            if self.turn > 1 and len(self.speed_order) >= 2:
                fast_player = self.speed_order[0]
                slow_player = self.speed_order[1]

                my_s = self.pokemon[0].status[5]

                if fast_player == 0 and slow_player == 1:
                    for hyp in list(self.belief_state.beliefs[active_enemy].keys()):
                        hyp_s = hyp.get_stats()[5]
                        if hyp_s >= my_s:
                            self.belief_state.beliefs[active_enemy][hyp] *= 0.1

                elif fast_player == 1 and slow_player == 0:
                    for hyp in list(self.belief_state.beliefs[active_enemy].keys()):
                        hyp_s = hyp.get_stats()[5]
                        if hyp_s <= my_s:
                            self.belief_state.beliefs[active_enemy][hyp] *= 0.1

        if calculate_damage and self.process_buffer:
            last_events = self.process_buffer[-5:]
            dmg_events = [e for e in last_events if e.get("type") == "damage" and e.get("player") == 1]

            for event in dmg_events:
                dmg_percent = event.get("damage_percent")
                move_used = event.get("move")

                if dmg_percent and move_used:
                    for hyp in list(self.belief_state.beliefs[active_enemy].keys()):
                        min_dmg, max_dmg = calculate_damage(
                            attacker=self.pokemon[0],
                            defender_hyp=hyp,
                            move_name=move_used
                        )
                        if min_dmg <= dmg_percent <= max_dmg:
                            self.belief_state.beliefs[active_enemy][hyp] *= 1.5
                        else:
                            self.belief_state.beliefs[active_enemy][hyp] *= 0.2

        total = sum(self.belief_state.beliefs[active_enemy].values())
        if total > 0:
            for hyp in self.belief_state.beliefs[active_enemy]:
                self.belief_state.beliefs[active_enemy][hyp] /= total

    def update_beliefs(self) -> None:
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

        self.update_beliefs_by_implicit_observations()

    def clone(self, player: int = None) -> Battle:
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
            target_poke = self.selected[perspective_pokemon.sex != 1][cmd - 20]
            return f"交代 ➔ 【{target_poke.name}】"
        return "様子見 / その他"

    def run_observer_loop(self) -> None:
        print("\n==================================================")
        print("  Project Aegis 観測・解説システム 起動中...")
        print("  対象ルール: カスタム mbルール (初期シーズン仕様)")
        print("  思考エンジン: CFR (Counterfactual Regret Minimization)")
        print(r"  選出エンジン: Aegis 相性補完セレクター ( $_6 \mathrm{C}_3 $ 総当たり評価)")
        print("  解説エンジン: Gemini 1.5 Flash (urllib 直結)")
        print("  監視モード: Observer Mode")
        print("==================================================\n")

        self.load_party()
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

                self.selection_command(player=0)
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
                    print(f"\n【対戦終了検出】 結果は '{result}' ででした。次の対戦を待機します。")
                    self.reset_game()

            time.sleep(0.1)


if __name__ == "__main__":
    Pokemon.init(season=22)

    for target_alias in ['キングズシールド', 'キング・シールド']:
        if target_alias in Pokemon.all_moves:
            Pokemon.all_moves['キングシールド'] = Pokemon.all_moves[target_alias]
            print(
                f"ℹ️ [Aegis Patch] 表記揺れ '{target_alias}' を本物の 'キングシールド' データとしてエイリアスマッピングしました。")
            break

    if hasattr(Pokemon, 'zukan'):
        if 'ギルガルド' not in Pokemon.zukan:
            for target_key in ['ギルガルド(シールド)', 'ギルガルド（シールド）']:
                if target_key in Pokemon.zukan:
                    Pokemon.zukan['ギルガルド'] = deepcopy(Pokemon.zukan[target_key])
                    Pokemon.zukan['ギルガルド']['display_name'] = 'ギルガルド'
                    print(f"ℹ️ [Patch] Pokemon.zukan に '{target_key}' から 'ギルガルド' のエイリアスを生成しました。")
                    break

    my_box = None
    analyzer = AegisAnalyzer(capture_box=my_box)
    analyzer.run_observer_loop()