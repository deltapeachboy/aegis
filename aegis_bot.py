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
from src.rebel.public_state import PublicBeliefState, _apply_hypothesis_to_pokemon
from src.rebel.cfr_solver import ReBeLSolver, CFRConfig
from src.llm.state_representation import battle_to_llm_state

from src.rebel.value_network import ReBeLValueNetwork

# =========================================================================
# 🚀 [Aegis Reward Shaping Patch] 遅延報酬・特殊特性・逆転ギミック価値補正
# =========================================================================
_original_value_network_forward = ReBeLValueNetwork.forward if hasattr(ReBeLValueNetwork, "forward") else None


def shaped_value_network_forward(self, states, *args, **kwargs):
    predictions = _original_value_network_forward(self, states, *args,
                                                  **kwargs) if _original_value_network_forward else states
    try:
        current_battle = getattr(builtins, "_aegis_current_battle", None)
        if current_battle and isinstance(current_battle, Battle) and predictions is not None:
            if predictions.dim() == 2 and predictions.size(0) == 1:
                my_win_prob = float(predictions[0][0].item())
                shaped_prob = my_win_prob

                my_side = 0
                opp_side = 1

                # A. ステルスロック設置ボーナス (期待値勝率 ±0.05 補正)
                if hasattr(current_battle, 'side_conditions') and current_battle.side_conditions:
                    if current_battle.side_conditions[opp_side].get('stealth_rock'):
                        shaped_prob += 0.05
                    if current_battle.side_conditions[my_side].get('stealth_rock'):
                        shaped_prob -= 0.05

                # B. あくび・状態異常ボーナス (期待値勝率 ±0.03〜0.04 補正)
                my_active = current_battle.pokemon[my_side]
                opp_active = current_battle.pokemon[opp_side]

                if opp_active:
                    if getattr(opp_active, 'yawn', 0) > 0:
                        shaped_prob += 0.04
                    if getattr(opp_active, 'status_con', None):
                        shaped_prob += 0.03

                if my_active:
                    if getattr(my_active, 'yawn', 0) > 0:
                        shaped_prob -= 0.04
                    if getattr(my_active, 'status_con', None):
                        shaped_prob -= 0.03

                # C. 能力ランク（積み状態）の価値補正 (A, C, Sは+0.02、B, Dは+0.01)
                if my_active and hasattr(my_active, 'rank'):
                    for stat_idx, factor in [(1, 0.02), (3, 0.02), (5, 0.02), (2, 0.01), (4, 0.01)]:
                        try:
                            shaped_prob += my_active.rank[stat_idx] * factor
                        except Exception:
                            pass

                if opp_active and hasattr(opp_active, 'rank'):
                    for stat_idx, factor in [(1, 0.02), (3, 0.02), (5, 0.02), (2, 0.01), (4, 0.01)]:
                        try:
                            shaped_prob -= opp_active.rank[stat_idx] * factor
                        except Exception:
                            pass

                # D. 天候（砂嵐）天候シナジー補正 (勝率 ±0.02 補正)
                if getattr(current_battle, 'weather', None) == 'sandstorm':
                    if my_active and any(t in ['いわ', 'じめん', 'はがね'] for t in my_active.types):
                        shaped_prob += 0.02
                    if opp_active and any(t in ['いわ', 'じめん', 'はがね'] for t in opp_active.types):
                        shaped_prob -= 0.02

                # E. ミミッキュの「ばけのかわ（インチキ保証）」の価値前借り (勝率 ±0.05 補正)
                if my_active and my_active.name == "ミミッキュ":
                    shaped_prob += 0.05
                if opp_active and opp_active.name == "ミミッキュ":
                    shaped_prob -= 0.05

                # F. イダイトウの「おはかまいり（逆転火力）」の価値前借り
                if my_active and "イダイトウ" in my_active.name:
                    dead_count = sum(1 for p in current_battle.selected[my_side] if p.hp <= 0)
                    shaped_prob += dead_count * 0.03
                if opp_active and "イダイトウ" in opp_active.name:
                    dead_count_opp = sum(1 for p in current_battle.selected[opp_side] if p.hp <= 0)
                    shaped_prob -= dead_count_opp * 0.03

                shaped_prob = max(0.01, min(0.99, shaped_prob))
                predictions[0][0] = shaped_prob
                predictions[0][1] = 1.0 - shaped_prob
    except Exception:
        pass
    return predictions


ReBeLValueNetwork.forward = shaped_value_network_forward

# =========================================================================

# テラスタル不許可設定
Battle.can_terastal = lambda self, player: False


# =========================================================================
# 🌟 グローバルヘルパー関数（メガストーン解決の共通化）
# =========================================================================
def get_possible_mega_stones(p_name: str) -> List[str]:
    """ポケモンの日本語名から、データベース上に実在する正しいメガストーン候補のリストを返す"""
    base = p_name.split("(")[0]
    special_map = {
        "リザードン": ["リザードナイトX", "リザードナイトY"],
        "ライチュウ": ["ライチュウナイトX", "ライチュウナイトY"],
        "ゲンガー": ["ゲンガナイト"],
        "ヘルガー": ["ヘルガナイト"],
        "ペンドラー": ["ペンドラナイト"],
        "ユキノオー": ["ユキノオナイト"],
        "ドラミドロ": ["ドラミドナイト"],
        "ズルズキン": ["ズルズキナイト"],
        "マフォクシー": ["マフォクシナイト"],
        "ブリガロン": ["ブリガロナイト"],
        "シビルドン": ["シビルドナイト"],
        "ピクシー": ["ピクシナイト"],
        "カラマネロ": ["カラマネナイト"],
        "スターミー": ["スターミナイト"],
        "ジジーロン": ["ジジーロナイト"],
        "カイリュー": ["カイリュナイト"]
    }
    if base in special_map:
        return special_map[base]
    return [base + "ナイト"]


# =========================================================================
# 3. 【高度化】AegisTeamBuilder (能力ポイント＆メタモンサニタイズ完全移植版)
# =========================================================================
class AegisTeamBuilder:
    """
    Project Aegis 構築自動生成システム (Layer 15)
    能力ポイント制に基づく上限付きランダム配分、および性格・特性、半減実フィルターを完全統合。
    """

    MEGA_PROBABILITIES = {
        "ライチュウ": 0.9, "ガブリアス": 0.1, "モルフォン": 0.66, "ラグラージ": 0.5,
        "リザードン": 0.9, "メタグロス": 0.8, "バシャーモ": 0.5, "ギャラドス": 0.5,
        "カイリュー": 0.5, "キラフロル": 0.5, "クチート": 0.9, "ゲンガー": 0.66,
        "ドラミドロ": 0.66, "ハッサム": 0.66, "ミミロップ": 0.8, "メガニウム": 0.9,
        "マフォクシー": 0.8, "ゲッコウガ": 0.5, "スターミー": 0.9, "フラエッテ(えいえん)": 0.8,
        "フシギバナ": 0.8, "ルカリオ": 0.8, "ウツボット": 0.8, "シャンデラ": 0.66,
        "カメックス": 0.8, "バンギラス": 0.45, "ブリガロン": 0.66, "ガルーラ": 0.85,
        "ヤドラン": 0.5, "ピクシー": 0.5, "ユキメノコ": 0.66, "シビルドン": 0.8,
        "ドリュウズ": 0.33, "サーナイト": 0.5, "ヤミラミ": 0.66, "スコヴィラン": 0.5,
        "カエンジシ": 0.66, "ペンドラー": 0.5, "ガメノデス": 0.75, "ジュカイン": 0.7,
        "エアームド": 0.5, "エルレイド": 0.4, "ズルズキン": 0.8, "ユキノオー": 0.5,
        "ジュペッタ": 0.7
    }

    ITEM_TIERS = {
        "ち力のハチマキ": 1.0,
        "ものしりメガネ": 1.0,
        "おおきなねっこ": 0.1,
        "ひかりのねんど": 0.2,
        "メトロノーム": 0.5,
        "こうかくレンズ": 0.6,
        "あついいわ": 0.1,
        "さらさらいわ": 0.1,
        "しめったいわ": 0.1,
        "つめたいいわ": 0.1,
        "いのちのたま": 2.0,
        "きれいなぬけがら": 0.15,
        "くろいてっきゅう": 0.1,
        "たつじんのおび": 1.5,
        "フォーカスレンズ": 0.5,
        "クラボのみ": 0.1,
        "カゴのみ": 1.5,
        "モモンのみ": 0.1,
        "チーゴのみ": 0.1,
        "ナナシのみ": 0.1,
        "ヒメリのみ": 0.1,
        "オレンのみ": 0.1,
        "キーのみ": 0.1,
        "ラムのみ": 2.0,
        "オボンのみ": 3.0,
        "オッカのみ": 0.3,
        "イトケのみ": 0.3,
        "ソクノのみ": 0.3,
        "リンドのみ": 0.3,
        "ヤチェのみ": 0.3,
        "ヨプのみ": 0.3,
        "ビアーのみ": 0.3,
        "シュカのみ": 0.3,
        "バコウのみ": 0.3,
        "ウタンのみ": 0.3,
        "タンガのみ": 0.3,
        "ヨロギのみ": 0.3,
        "カシブのみ": 0.3,
        "ハバンのみ": 0.3,
        "ナモのみ": 0.3,
        "リリバのみ": 0.3,
        "ロゼルのみ": 0.3,
        "ホズのみ": 0.3,
        "おうじゃのしるし": 0.2,
        "メタルコート": 1.0,
        "きせきのタネ": 1.0,
        "もくたん": 1.0,
        "しんぴのしずく": 1.0,
        "シルクのスカーフ": 1.0,
        "するどいくちばし": 1.0,
        "ぎんのこな": 1.0,
        "じしゃく": 1.0,
        "かたいたし": 1.0,
        "のろいのおふだ": 1.0,
        "りゅうのキバ": 1.0,
        "どくばり": 1.0,
        "やわらかいすな": 1.0,
        "くろいメガネ": 1.0,
        "くろおび": 1.0,
        "とけないこおり": 1.0,
        "まがったスプーン": 1.0,
        "きあいのハチマキ": 0.1,
        "ピントレンズ": 1.0,
        "たべのこし": 4.0,
        "かいがらのすず": 0.15,
        "きあいのタスキ": 4.0,
        "こだわりスカーフ": 3.5,
        "でんきだま": 0.1,
        "ひかりのこな": 1.0,
        "しろいハーブ": 0.75,
        "メンタルハーブ": 0.5,
        "ようせいのハネ": 1.0,
        "こだわりハチマキ": 2.5,
        "こだわりメガネ": 2.5,
        "とつげきチョッキ": 2.5,
    }

    WALL_SETTER_POKEMON = {
        "オーロンゲ", "ジャローダ", "アローラキュウコン"
    }

    POWERFUL_ABILITIES = {
        "マルチスケイル", "ち力もち", "いたずらごころ",
        "ひでり", "あめふらし", "すなおこし",
        "ゆきふらし", "テクニシャン", "かそく"
    }

    NATURE_WEIGHTS = {
        "いじっぱり": 0.1, "ひかえめ": 0.1, "ようき": 0.1, "おくびょう": 0.1,
        "わんぱく": 0.1, "しんちょう": 0.1, "ずぶとい": 0.1, "おだやか": 0.1,
        "ゆうかん": 0.05, "れいせい": 0.05, "さみしがり": 0.05, "おっとり": 0.05,
        "やんちゃ": 0.05, "うっかりや": 0.05
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

    def calculate_matchup_tactical_scores(self, cand_name: str, opp_name: str) -> Tuple[float, float]:
        try:
            cand_zukan = Pokemon.zukan.get(cand_name)
            opp_zukan = Pokemon.zukan.get(opp_name)
            if not cand_zukan or not opp_zukan:
                return 0.0, 0.0

            cand_base = cand_zukan["base"]
            opp_base = opp_zukan["base"]
            cand_types = cand_zukan["type"]
            opp_types = opp_zukan["type"]

            cand_atk = max(cand_base[1], cand_base[3])
            best_atk_eff = 1.0
            for c_type in cand_types:
                for o_type in opp_types:
                    atk_id = Pokemon.type_id.get(c_type, 0)
                    def_id = Pokemon.type_id.get(o_type, 0)
                    eff = Pokemon.type_corrections[atk_id][def_id]
                    if eff > best_atk_eff:
                        best_atk_eff = eff
            max_damage_given = cand_atk * best_atk_eff

            opp_atk = max(opp_base[1], opp_base[3])
            best_def_eff = 1.0
            for o_type in opp_types:
                for c_type in cand_types:
                    atk_id = Pokemon.type_id.get(o_type, 0)
                    def_id = Pokemon.type_id.get(c_type, 0)
                    eff = Pokemon.type_corrections[atk_id][def_id]
                    if eff > best_def_eff:
                        best_def_eff = eff
            max_damage_taken = opp_atk * best_def_eff

            speed_coefficient = 1.5 if cand_base[5] > opp_base[5] else 1.0
            taimen_score = (max_damage_given * speed_coefficient) + cand_base[0] - max_damage_taken
            uke_score = (cand_base[0] - max_damage_taken) / cand_base[0] if cand_base[0] > 0 else 0.0

            return taimen_score, uke_score
        except Exception:
            return 0.0, 0.0

    # 上限32付き能力ポイントランダム配分アルゴリズム
    def allocate_stat_points_randomly(self, indices: List[int], total_points: int = 66, max_single: int = 32) -> List[
        int]:
        points = [0] * 6
        if not indices:
            return points
        actual_total = min(total_points, len(indices) * max_single)
        allocated = 0
        while allocated < actual_total:
            idx = random.choice(indices)
            if points[idx] < max_single:
                points[idx] += 1
                allocated += 1
        return points

    def build_team(self, core_name: str, pokemon_weights: Optional[dict] = None) -> Dict[str, Any]:
        """軸(コア)に基づき、能力ポイント制と技選定の正常化に適合したチーム構築を生成"""
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

                taimen_sum = 0.0
                uke_sum = 0.0
                for member in team_members:
                    taimen, uke = self.calculate_matchup_tactical_scores(candidate, member)
                    taimen_sum += taimen
                    uke_sum += uke

                avg_taimen = taimen_sum / len(team_members)
                avg_uke = uke_sum / len(team_members)

                type_score += (avg_taimen * 0.01) + (avg_uke * 1.5)

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

        TYPE_BOOSTING_ITEMS = {
            "メタルコート": "はがね", "きせきのタネ": "くさ", "もくたん": "ほのお",
            "しんぴのしずく": "みず", "シルクのスカーフ": "ノーマル", "するどいくちばし": "ひこう",
            "ぎんのこな": "むし", "じしゃく": "でんき", "かたいたし": "いわ",
            "のろいのおふだ": "ゴースト", "りゅうのキバ": "ドラゴン", "どくばり": "どく",
            "やわらかいすな": "じめん", "くろいメガネ": "あく", "くろおび": "かくとう",
            "とけないこおり": "こおり", "まがったスプーン": "エスパー", "ようせいのハネ": "フェアリー"
        }

        # 各半減実と対応するダメージタイプのマッピング
        TYPE_REDUCING_BERRIES = {
            "オッカのみ": "ほのお", "イトケのみ": "みず", "ソクノのみ": "でんき",
            "リンドのみ": "くさ", "ヤチェのみ": "こおり", "ヨプのみ": "かくとう",
            "ビアーのみ": "どく", "シュカのみ": "じめん", "バコウのみ": "ひこう",
            "ウタンのみ": "エスパー", "タンガのみ": "むし", "ヨロギのみ": "いわ",
            "カシブのみ": "ゴースト", "ハバンのみ": "ドラゴン", "ナモのみ": "あく",
            "リリバのみ": "はがね", "ロゼルのみ": "フェアリー"
        }

        def get_true_move_type(move_name: str, ab: str, t_type: str) -> str:
            mv_data = Pokemon.all_moves.get(move_name, {})
            base_type = mv_data.get("type", "ノーマル")
            if move_name == "ウェザーボール":
                if ab == "あめふらし": return "みず"
                if ab == "ひでり": return "ほのお"
                if ab == "すなおこし": return "いわ"
                if ab == "ゆきふらし": return "こおり"
            elif move_name == "テラバースト":
                return t_type
            return base_type

        generated_party = {}
        assigned_items = {}
        mega_stones_in_pool = {item for item in self.mb_items if "ナイト" in item}
        normal_items_pool = list(self.mb_items - mega_stones_in_pool)

        for i, name in enumerate(team_members):
            zukan_entry = Pokemon.zukan[name]
            dyn_data = pokemon_weights.get(name, {}) if pokemon_weights else {}

            # 🚀 [A. 能力ポイント決定 (極振り50% | 両刀4% | 複合46%)]
            base_stats = zukan_entry.get("base", [100, 100, 100, 100, 100, 100])

            stat_points = [0] * 6
            ev_category = "max_out"
            adj_nature_weights = {}

            rand_ev = random.random()

            if rand_ev < 0.04:
                # 1. 両刀型 (4%)
                ev_category = "mixed"
                if random.random() < 0.5:
                    stat_points = self.allocate_stat_points_randomly([1, 3, 5], total_points=66, max_single=32)
                    adj_nature_weights["せっかち"] = 7.5
                    adj_nature_weights["むじゃき"] = 7.5
                else:
                    stat_points = self.allocate_stat_points_randomly([0, 1, 3], total_points=66, max_single=32)
                    adj_nature_weights["ゆうかん"] = 7.5
                    adj_nature_weights["れいせい"] = 7.5
            elif rand_ev < 0.50:
                # 2. 複合調整型 (46%)
                ev_category = "hybrid"
                hybrid_patterns = [
                    ("HBD", [0, 2, 4]),  # 総合耐久
                    ("HBDS", [0, 2, 4, 5]),  # 耐久＋素早さ
                    ("HABS", [0, 1, 2, 5]),  # 物理調整アタッカー
                    ("HBCS", [0, 2, 3, 5]),  # 特殊調整アタッカー
                    ("HAB", [0, 1, 2]),  # 物理中耐久
                    ("HBC", [0, 2, 3]),  # 特殊中耐久
                    ("HAD", [0, 1, 4]),  # 物理特防
                    ("HCD", [0, 3, 4]),  # 特殊特防
                    ("HAS", [0, 1, 5]),  # HP・S調整 (必須)
                    ("HCS", [0, 3, 5]),  # HP・S調整 (必須)
                    ("HBS", [0, 2, 5]),  # S微振り物理耐久 (必須)
                    ("HDS", [0, 4, 5]),  # S微振り特殊耐久 (必須)
                ]

                # 特化スロットの決定
                chosen_pattern_name, target_indices = random.choice(hybrid_patterns)
                stat_points = self.allocate_stat_points_randomly(target_indices, total_points=66, max_single=32)

                if "S" in chosen_pattern_name:
                    adj_nature_weights["ようき"] = 4.0
                    adj_nature_weights["おくびょう"] = 4.0
                elif "B" in chosen_pattern_name or "D" in chosen_pattern_name:
                    adj_nature_weights["ずぶとい"] = 4.0
                    adj_nature_weights["わんぱく"] = 4.0
                    adj_nature_weights["しんちょう"] = 4.0
                    adj_nature_weights["おだやか"] = 4.0
            else:
                # 3. 極振りブッパ型 (50%)
                ev_category = "max_out"
                chosen_max_type = random.choice(["HA", "HB", "HC", "HD", "HS", "AS", "CS"])

                if chosen_max_type == "HA":
                    stat_points[0], stat_points[1], stat_points[5] = 32, 32, 2
                    adj_nature_weights["いじっぱり"] = 4.0
                elif chosen_max_type == "HB":
                    stat_points[0], stat_points[2], stat_points[4] = 32, 32, 2
                    adj_nature_weights["わんぱく"] = 4.0
                elif chosen_max_type == "HC":
                    stat_points[0], stat_points[3], stat_points[5] = 32, 32, 2
                    adj_nature_weights["ひかえめ"] = 4.0
                elif chosen_max_type == "HD":
                    stat_points[0], stat_points[4], stat_points[2] = 32, 32, 2
                    adj_nature_weights["しんちょう"] = 4.0
                elif chosen_max_type == "HS":
                    stat_points[0], stat_points[5], stat_points[4] = 32, 32, 2
                    adj_nature_weights["ようき"] = 4.0
                    adj_nature_weights["おくびょう"] = 4.0
                elif chosen_max_type == "AS":
                    stat_points[0], stat_points[1], stat_points[5] = 2, 32, 32
                    adj_nature_weights["ようき"] = 4.0
                    adj_nature_weights["いじっぱり"] = 2.5
                elif chosen_max_type == "CS":
                    stat_points[0], stat_points[3], stat_points[5] = 2, 32, 32
                    adj_nature_weights["おくびょう"] = 4.0
                    adj_nature_weights["ひかえめ"] = 2.5

            # 🚀 [B. 技構成の先行サンプリング]
            learnable = self.learnsets.get(name, ["テラバースト"])
            move_weights = [dyn_data.get("moves", {}).get(m, 1.0) for m in learnable]

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

            if len(chosen_moves) < 4:
                extra_pool = [m for m in learnable if m not in chosen_moves]
                needed = 4 - len(chosen_moves)
                if extra_pool:
                    extra_moves = random.sample(extra_pool, min(needed, len(extra_pool)))
                    chosen_moves.extend(extra_moves)
                while len(chosen_moves) < 4:
                    chosen_moves.append("わるあがき")

            # 🚀 [C. 確定技に基づく性格・特性抽選]
            adj_ability_weights = {}

            natures = list(self.NATURE_WEIGHTS.keys())
            nature_weights = [
                self.NATURE_WEIGHTS[nat] * dyn_data.get("natures", {}).get(nat, 1.0) * adj_nature_weights.get(nat, 1.0)
                for nat in natures
            ]
            nature = random.choices(natures, weights=nature_weights, k=1)[0]

            abilities = zukan_entry.get("ability", ["とくせいなし"])

            if name == "メタモン" and "かわりもの" in abilities:
                if random.random() < 0.8:
                    ability = "かわりもの"
                else:
                    other_abilities = [ab for ab in abilities if ab != "かわりもの"]
                    ability = random.choice(other_abilities) if other_abilities else "かわりもの"

            elif abilities:
                ability_weights = [
                    (2.0 if ab in self.POWERFUL_ABILITIES else 1.0) * dyn_data.get("abilities", {}).get(ab,
                                                                                                        1.0) * adj_ability_weights.get(
                        ab, 1.0)
                    for ab in abilities
                ]
                ability = random.choices(abilities, weights=ability_weights, k=1)[0]
            else:
                ability = "とくせいなし"

            # 持ち物サンプリング
            assigned_item = ""
            mega_candidates = get_possible_mega_stones(name)
            valid_mega_stones = [stone for stone in mega_candidates if stone in self.mb_items]

            if valid_mega_stones and random.random() < self.MEGA_PROBABILITIES.get(name, 0.50):
                assigned_item = random.choice(valid_mega_stones)
            else:
                available_items = [itm for i_item in normal_items_pool if
                                   (itm := i_item) not in assigned_items.values()]
                if available_items:
                    local_item_tiers = dict(self.ITEM_TIERS)

                    if "ひでり" in ability: local_item_tiers["あついいわ"] = 5.0
                    if "あめふらし" in ability: local_item_tiers["しめったいわ"] = 5.0
                    if "すなおこし" in ability: local_item_tiers["さらさらいわ"] = 5.0
                    if "ゆきふらし" in ability: local_item_tiers["つめたいいわ"] = 5.0
                    if name.split("(")[0] in self.WALL_SETTER_POKEMON:
                        local_item_tiers["ひかりのねんど"] = 5.0

                    pokemon_ttype = zukan_entry["type"][0]
                    attack_types = set()
                    for mv in chosen_moves:
                        mv_data = Pokemon.all_moves.get(mv, {})
                        if mv_data and not mv_data.get("class", "").startswith("sta"):
                            true_type = get_true_move_type(mv, ability, pokemon_ttype)
                            attack_types.add(true_type)

                    item_weights = []
                    for itm in available_items:
                        weight = local_item_tiers.get(itm, 0.1)
                        if itm in TYPE_BOOSTING_ITEMS:
                            req_type = TYPE_BOOSTING_ITEMS[itm]
                            if req_type not in attack_types:
                                weight = 0.0

                        if itm in TYPE_REDUCING_BERRIES:
                            req_type = TYPE_REDUCING_BERRIES[itm]
                            is_weak = False
                            if req_type in Pokemon.type_id:
                                atk_id = Pokemon.type_id[req_type]
                                eff = 1.0
                                for def_type in zukan_entry.get("type", []):
                                    if def_type in Pokemon.type_id:
                                        def_id = Pokemon.type_id[def_type]
                                        eff *= Pokemon.type_corrections[atk_id][def_id]
                                    if eff > 1.0:
                                        is_weak = True
                            if not is_weak:
                                weight = 0.0

                        item_weights.append(weight)

                    if sum(item_weights) <= 0:
                        item_weights = [1.0] * len(available_items)

                    # 🛡️【安全対策ガード】自分に不適合な他種族専用 of メガストーン（〜ナイト）を抽選プールから物理排除
                    my_mega_stone = name.split("(")[0] + "ナイト"
                    filtered_available_items = []
                    filtered_item_weights = []
                    for idx_itm, itm in enumerate(available_items):
                        if "ナイト" in itm and itm != my_mega_stone:
                            continue
                        filtered_available_items.append(itm)
                        filtered_item_weights.append(item_weights[idx_itm])

                    if sum(filtered_item_weights) <= 0:
                        filtered_item_weights = [1.0] * len(filtered_available_items)

                    assigned_item = random.choices(filtered_available_items, weights=filtered_item_weights, k=1)[0]
            assigned_items[name] = assigned_item

            effort = [min(252, sp * 8) for sp in stat_points]

            generated_party[str(i)] = {
                "name": name,
                "sex": 1 if i % 2 == 0 else -1,
                "level": 50,
                "nature": nature,
                "ability": ability,
                "item": assigned_item,
                "Ttype": zukan_entry["type"][0],
                "moves": chosen_moves,
                "indiv": [31, 31, 31, 31, 31, 31],
                "effort": effort
            }

        return generated_party

# =========================================================================
# 4. 【高度化】AegisTeamSelector (補完評価＆BERT選出予測の安全な統合)
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
        """
        お互いのポケモンが実際に採用している『4つの技』に基づいて相性を評価します。
        全習得技の走査を廃止することで、計算量を350分の1以下に削減し、選出フェーズのフリーズを完全に解消します。
        """
        score = 0.0

        # 🌟 全習得技ではなく、今回実際に採用されている技（最大4つ）のみに限定
        my_moves = my_poke.moves if hasattr(my_poke, 'moves') and my_poke.moves else ["テラバースト"]
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
                if move_type in Pokemon.type_id and opp_type in Pokemon.type_id:
                    atk_id = Pokemon.type_id[move_type]
                    def_id = Pokemon.type_id[opp_type]
                    eff *= Pokemon.type_corrections[atk_id][def_id]

            if move_type == "じめん" and ("ひこう" in opp_types or opp_poke.ability == "ふゆう"):
                eff = 0.0

            if eff > best_my_eff:
                best_my_eff = eff

        # 🌟 相手側も、今回実際に採用されている技（最大4つ）のみに限定
        opp_moves = opp_poke.moves if hasattr(opp_poke, 'moves') and opp_poke.moves else ["テラバースト"]
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
                if move_type in Pokemon.type_id and my_type in Pokemon.type_id:
                    atk_id = Pokemon.type_id[move_type]
                    def_id = Pokemon.type_id[my_type]
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

        results = []
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
            results.append((combo, total_score))

        results = sorted(results, key=lambda x: -x[1])
        best_combination = results[0][0]

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
            print("⚠️ log/party.log が見つかりません。Aegis TeamBuilder で自動構築して保存します。")
            default_party = self.team_builder.build_team("ガブリアス")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_party, f, ensure_ascii=False, indent=2)
            print("✅ チームビルダーによる最強構築(1世代目)のファイル保存が完了しました。")

    def _load_mb_rules(self) -> None:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            pokemon_path = os.path.join(base_dir, "battle_data", "mb_pokemon.txt")
            items_path = os.path.join(base_dir, "battle_data", "mb_items.txt")
            learnset_path = os.path.join(base_dir, "battle_data", "mb_learnset.json")

            if os.path.exists(pokemon_path):
                with open(pokemon_path, "r", encoding="utf-8") as f:
                    self.mb_pokemon = {line.strip() for line in f if line.strip()}
                Pokemon.permitted_pool = self.mb_pokemon

            if os.path.exists(items_path):
                with open(items_path, "r", encoding="utf-8") as f:
                    self.mb_items = {line.strip() for line in f if line.strip()}
                Pokemon.permitted_items = self.mb_items
                Pokemon.mb_items = self.mb_items

            if os.path.exists(learnset_path):
                with open(learnset_path, "r", encoding="utf-8") as f:
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
                self.img = cv2.resize(self.img, (1920, 1080))

            if filename:
                cv2.imwrite(filename, self.img)
        except Exception as e:
            pass

    def set_image(self, filename):
        """画像をファイルから読み込み、解析バッファをセットします"""
        self.img = cv2.imread(filename)

    def read_battle_situlation(self):
        return True

    def read_phase(self, capture=True):
        return "battle"

    def read_win_lose(self, capture=True):
        return ""

    def read_bottom_text(self, capture=True):
        return False

    def read_ability_text(self, player, capture=True):
        return False

    def is_battle_window(self, capture=True):
        return True


# =========================================================================
# 6. 信念追跡型ボット実行スレッド & ログダンプ
# =========================================================================
def run_aegis_bot():
    print("ℹ️ Aegis Live Bot Thread started.")
    analyzer = AegisAnalyzer()

    # 画面キャプチャ及びイベントループエミュレーション
    while True:
        try:
            analyzer.capture()
            phase = analyzer.read_phase(capture=False)

            if phase == "battle":
                analyzer.read_battle_situlation()

            time.sleep(1)
        except Exception as e:
            time.sleep(1)


# =========================================================================
# 7. エントリーポイント & フック適用（冪等性・デッドロック保護を完全統合）
# =========================================================================
if __name__ == "__main__":
    Pokemon.init(season=22)

    # =========================================================================
    # 🌟 Battle.change_pokemon 引数競合回避・安全防止パッチ
    # =========================================================================
    if not hasattr(Battle, '_aegis_change_pokemon_patched'):
        original_change_pokemon = Battle.change_pokemon


        def patched_change_pokemon(self, player, command=None, idx=0, landing=False, *args, **kwargs):
            """
            CFR初期化時や交代コマンド誤認による [-20] などのアクセスエラーを防止しつつ、
            控えポケモンが「ひんし状態」の場合に無限ループに陥るバグを防ぐため、
            生存している（HP > 0）代替控えへの自動選定フォールバックを実行します。
            """
            cmd = self.command[player] if (self.command and player < len(self.command)) else None

            if cmd is not None and isinstance(cmd, int) and 20 <= cmd <= 25:
                target_idx = cmd - 20
            else:
                target_idx = idx

            party = self.selected[player] if (self.selected and player < len(self.selected)) else []
            party_len = len(party)

            is_valid = False
            if 0 <= target_idx < party_len:
                if party[target_idx].hp > 0:
                    is_valid = True

            if not is_valid:
                alive_idx = None
                for idx_temp in range(party_len):
                    if party[idx_temp].hp > 0:
                        alive_idx = idx_temp
                        break

                if alive_idx is not None:
                    idx_to_use = alive_idx
                else:
                    idx_to_use = 0
            else:
                idx_to_use = target_idx

            if party_len > 0 and self.pokemon and player < len(self.pokemon):
                self.pokemon[player] = self.selected[player][idx_to_use]

            has_invalid_cmd = False
            old_cmd_val = None
            if self.command and player < len(self.command):
                curr_cmd = self.command[player]
                if curr_cmd is not None and (not isinstance(curr_cmd, int) or not (20 <= curr_cmd <= 25)):
                    old_cmd_val = curr_cmd
                    self.command[player] = None
                    has_invalid_cmd = True

            try:
                return original_change_pokemon(
                    self,
                    player=player,
                    command=command,
                    idx=idx_to_use,
                    landing=landing,
                    *args,
                    **kwargs
                )
            finally:
                if has_invalid_cmd and self.command and player < len(self.command):
                    self.command[player] = old_cmd_val


        Battle.change_pokemon = patched_change_pokemon
        print(
            "ℹ️ [Aegis Patch] Battle.change_pokemon インデックスエラー安全防止パッチ(引数マッピング修正済)を適用しました。")

    # =========================================================================
    # 🚀 [Aegis Deepcopy Optimization Patch]
    # =========================================================================
    if not hasattr(Battle, '_aegis_deepcopy_patched'):
        def patched_battle_deepcopy(self, memo):
            if id(self) in memo:
                return memo[id(self)]

            cls = self.__class__
            new_battle = cls.__new__(cls)
            memo[id(self)] = new_battle

            skip_keys = {
                'solver', 'value_network', 'nn', 'model', 'w2v_model',
                'analyzer', 'builder', 'beliefs', 'pbs'
            }
            skip_class_names = {
                'ReBeLValueNetwork', 'CFRSolver', 'AegisTeamBuilder',
                'AegisAnalyzer', 'Word2Vec', 'PokemonBeliefState', 'PublicBeliefState'
            }

            for k, v in self.__dict__.items():
                if k in skip_keys:
                    continue
                if v.__class__.__name__ in skip_class_names:
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
        Battle._aegis_deepcopy_patched = True
        print("ℹ️ [Aegis Patch] Battle and Pokemon customized __deepcopy__ optimization applied.")

    # =========================================================================
    # 🌟 Battle.available_commands テラスタル（10-13）排除 ＆ コマンド空対策パッチ
    # =========================================================================
    if not hasattr(Battle, '_aegis_available_commands_patched2'):
        original_available_commands = Battle.available_commands


        def patched_available_commands(self, player, *args, **kwargs):
            """
            レギュレーションM-B（テラスタル禁止環境）に完全適合させるため、
            元のシグネチャを完全に維持したまま、10〜13番のテラスタルコマンドを排除します。
            """
            cmds = original_available_commands(self, player, *args, **kwargs)
            filtered_cmds = [c for c in cmds if c not in range(10, 14)]

            if not filtered_cmds:
                p = self.pokemon[player] if (self.pokemon and player < len(self.pokemon)) else None
                if p and p.hp > 0:
                    print(f"\n🚨 [Aegis Debug] コマンド選択肢が空になりました! (Player {player})")
                    print(f"  - ポケモン: {p.name} (HP: {p.hp}/{p.status[0]})")
                    print(f"  - 技スロット: {getattr(p, 'moves', 'N/A')}")
                    print(f"  - 残りPP: {getattr(p, 'pp', 'N/A')}")
                    print(f"  - 状態変化（アンコールなど）: {getattr(p, 'condition', {})}")
                    print(f"  - 状態異常: {getattr(p, 'ailment', 'None')}")
                    print(f"  - 持ち物: {p.item}")
                    print(f"  - 固定技（こだわり等）: {getattr(p, 'fixed_move', 'None')}")
                    print(f"  - ターン数: {self.turn}")
                    print(f"  - 控え情報: {[p_bench.name for p_bench in self.selected[player] if p_bench.hp > 0]}")
                    print("-" * 50 + "\n")

                    # 救済ガード: 交代可能な生存控えを検索
                    party = self.selected[player] if (self.selected and player < len(self.selected)) else []
                    switch_cmds = []
                    for idx_temp, poke_bench in enumerate(party):
                        if poke_bench.hp > 0 and poke_bench != p:
                            switch_cmds.append(20 + idx_temp)

                    if switch_cmds:
                        filtered_cmds = switch_cmds
                    else:
                        filtered_cmds = [0]

            return filtered_cmds


        Battle.available_commands = patched_available_commands
        Battle._aegis_available_commands_patched2 = True
        print("ℹ️ [Aegis Patch] Battle.available_commands テラスタル（10-13）排除 ＆ コマンド空対策パッチを適用しました。")

    # =========================================================================
    # 🌟 Battle.battle_command 内部ランダムエラー防止パッチ
    # =========================================================================
    if not hasattr(Battle, '_aegis_battle_command_patched'):
        original_battle_command = Battle.battle_command


        def patched_battle_command(self, player, *args, **kwargs):
            """
            シミュレータ内部で解決コマンドが空になった際、random.choice が IndexError を起こすのを防止します。
            """
            cmds = self.available_commands(player)
            if cmds:
                return random.choice(cmds)
            return None


        Battle.battle_command = patched_battle_command
        Battle._aegis_battle_command_patched = True
        print("ℹ️ [Aegis Patch] Battle.battle_command 内部安全パッチを適用しました。")

    # =========================================================================
    # 🌟 その他補正パッチ群
    # =========================================================================
    if not hasattr(Pokemon, '_aegis_find_patched'):
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
        Pokemon._aegis_find_patched = True

    if not hasattr(Battle, '_aegis_get_mega_name_patched'):
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
        Battle._aegis_get_mega_name_patched = True

    # =========================================================================
    # 🌟 【統合強化版】Battle.proceed & winner ＆ コマンド・サニタイザー & デッドロック防止パッチ
    # =========================================================================
    if not hasattr(Battle, '_aegis_proceed_patched_v2'):
        original_proceed = Battle.proceed
        original_winner = Battle.winner


        # ---------------------------------------------------------------------
        # 1. コマンド・サニタイザー (Command Sanitizer)
        # ---------------------------------------------------------------------
        def sanitize_commands(battle_obj, commands):
            """
            実機進行前に不可能なコマンド（不正な交代等）を検証し、安全なデフォルト手に書き換えます。
            """
            if commands is None or len(commands) < 2:
                return commands

            sanitized = list(commands)
            for player in range(2):
                cmd = sanitized[player]
                active_poke = battle_obj.pokemon[player] if (
                            battle_obj.pokemon and player < len(battle_obj.pokemon)) else None
                if not active_poke:
                    continue

                # 交代コマンド (20〜25) の正当性チェック
                if cmd is not None and 20 <= cmd <= 25:
                    switch_to_party_idx = cmd - 20
                    party = battle_obj.selected[player] if (
                                battle_obj.selected and player < len(battle_obj.selected)) else []
                    is_valid = True

                    # インデックス範囲外、あるいは瀕死(HP=0)、あるいは現在のアクティブ自身への交代は無効
                    if switch_to_party_idx >= len(party):
                        is_valid = False
                    else:
                        target_poke = party[switch_to_party_idx]
                        if target_poke == active_poke or target_poke.hp <= 0:
                            is_valid = False

                    # 不正な交代を検知した場合、安全な技選択（最初のPPのある技）にフォールバック
                    if not is_valid:
                        fallback_cmd = 0
                        if hasattr(active_poke, 'moves') and active_poke.moves:
                            for m_idx, move in enumerate(active_poke.moves):
                                if hasattr(active_poke, 'pp') and m_idx < len(active_poke.pp) and active_poke.pp[
                                    m_idx] > 0:
                                    fallback_cmd = m_idx
                                    break
                        sanitized[player] = fallback_cmd
                        print(
                            f"⚠️ [Sanitizer] プレイヤー {player} の不正な交代コマンド ({cmd}) を安全な技選択 ({fallback_cmd}) へ強制クレンジングしました。")

            return sanitized


        # ---------------------------------------------------------------------
        # 2. 進行メソッド (Proceed) の拡張
        # ---------------------------------------------------------------------
        def patched_proceed(self, commands=None):
            # 2-A. コマンドのサニタイズ処理を適用
            target_cmds = commands if commands is not None else self.command
            cmds = sanitize_commands(self, target_cmds)

            # 2-B. 1回のproceed呼び出しごとに、winner無限ループ検知用のカウンタをリセット
            self._winner_call_count_in_proceed = 0

            # 2-C. 元々の proceed パッチロジック（技スロット補填＆メタモン制限ガード等）を完全に継承
            if cmds:
                cmds = list(cmds)
                for player in range(2):
                    p = self.pokemon[player]
                    if p and p.hp > 0:
                        temp_moves = list(p.moves) if hasattr(p, 'moves') and p.moves else []

                        # メタモンの場合は技をへんしんのみに固定し、補填をスキップ
                        if p.name == "メタモン":
                            temp_moves = ["へんしん"]
                            is_modified = True
                        elif not temp_moves:
                            temp_moves = ["わるあがき"]
                            is_modified = True
                        else:
                            is_modified = False

                        if len(temp_moves) < 4 and p.name != "メタモン":
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


        # ---------------------------------------------------------------------
        # 3. 勝者判定メソッド (Winner) の拡張 (無限ループ検知安全弁)
        # ---------------------------------------------------------------------
        def patched_winner(self, record=True):
            count = getattr(self, '_winner_call_count_in_proceed', 0) + 1
            self._winner_call_count_in_proceed = count

            # 同一進行フェーズ内でwinner判定が150回以上繰り返された場合
            if count > 150:
                print(f"\n⚠️ [Deadlock Guardian] 内部の膠着（無限ループ）を検知しました (同一ターン内判定回数: {count})。")
                print(
                    f"   - プレイヤー0: {self.pokemon[0].name if self.pokemon[0] else 'None'} (HP: {self.pokemon[0].hp if self.pokemon[0] else 0})")
                print(
                    f"   - プレイヤー1: {self.pokemon[1].name if self.pokemon[1] else 'None'} (HP: {self.pokemon[1].hp if self.pokemon[1] else 0})")
                print("   - 暫定TOD判定に基づき、戦闘を安全に強制決着してループから離脱します。")

                scores = []
                for p in range(2):
                    try:
                        scores.append(self.TOD_score(p))
                    except Exception:
                        scores.append(0)

                self._winner_call_count_in_proceed = 0  # カウンタリセット

                # TODスコアの大きい側を勝者として返す
                if scores[0] > scores[1]:
                    return 0
                else:
                    return 1

            return original_winner(self, record=record)


        # クラスメソッドの再バインド
        Battle.proceed = patched_proceed
        Battle.winner = patched_winner
        Battle._aegis_proceed_patched_v2 = True
        print("ℹ️ [Aegis Patch] Battle.proceed & winner 統合サニタイズ・デッドロック防止パッチを正常に適用しました。")

    for target_alias in ['キングズシールド', 'キング・シールド']:
        if target_alias in Pokemon.all_moves:
            Pokemon.all_moves['キングシールド'] = Pokemon.all_moves[target_alias]
            break

    if hasattr(Pokemon, 'zukan'):
        if 'ギルガルド' not in Pokemon.zukan:
            for target_key in ['ギルガルド(シールド)', 'ギルガルド（シールド）']:
                if target_key in Pokemon.zukan:
                    Pokemon.zukan['ギルガルド'] = deepcopy(Pokemon.zukan[target_key])
                    Pokemon.zukan['ギルガルド']['display_name'] = 'ギルガルド'
                    break

    my_box = None
    analyzer = AegisAnalyzer(capture_box=my_box)
    analyzer.run_observer_loop()