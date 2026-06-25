import sys
import types
import builtins
import os
import json
import io
import time
import random
import warnings
import signal  # タイムアウト強制遮断・パッチ用に追加
import dataclasses
from typing import Any, Optional, Dict, List, Tuple
from copy import deepcopy
from collections import Counter
from pathlib import Path

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
# 2. 共通ライブラリ・モジュールのロード
# =========================================================================
from pokepy.pokemon import Pokemon
from pokepy.battle import Battle
from aegis_bot import AegisTeamBuilder, AegisTeamSelector, AegisAnalyzer, get_possible_mega_stones
from src.rebel.belief_state import PokemonBeliefState
from src.rebel.public_state import PublicBeliefState
from train_value_network import train_model

from src.rebel.value_network import ReBeLValueNetwork

# プラットフォームアラームの安全セーフガード
HAS_ALARM = hasattr(signal, "alarm")


def safe_set_alarm(seconds: int):
    if HAS_ALARM:
        signal.alarm(seconds)


# =========================================================================
# 🚀 [Aegis Reward Shaping Patch Ver 16.0] 設置・状態異常・特殊価値補正
# =========================================================================
_original_value_network_forward = ReBeLValueNetwork.forward if hasattr(ReBeLValueNetwork, "forward") else None


def shaped_value_network_forward(self, states, *args, **kwargs):
    """
    ステルスロック等の設置技と『あくび・睡眠』が同時に成立した際の
    『起点コンボ・ループハメ展開』に強力な相乗価値補正を適用します。
    """
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

                # A. 設置技(ステロ, まきびし, どくびし)の有無を監査
                my_has_hazards = False
                opp_has_hazards = False
                if hasattr(current_battle, 'side_conditions') and current_battle.side_conditions:
                    opp_cond = current_battle.side_conditions[opp_side]
                    my_cond = current_battle.side_conditions[my_side]

                    # 相手側のコートに設置されている場合
                    if opp_cond.get('stealth_rock') or opp_cond.get('spikes') or opp_cond.get('toxic_spikes'):
                        opp_has_hazards = True
                        shaped_prob += 0.05
                    # 自分側のコートに設置されている場合
                    if my_cond.get('stealth_rock') or my_cond.get('spikes') or my_cond.get('toxic_spikes'):
                        my_has_hazards = True
                        shaped_prob -= 0.05

                my_active = current_battle.pokemon[my_side]
                opp_active = current_battle.pokemon[opp_side]

                # B. あくび・睡眠状態の有無を監査
                my_is_yawned_or_asleep = False
                opp_is_yawned_or_asleep = False

                if opp_active:
                    opp_ailment = getattr(opp_active, 'ailment', 'None')
                    if getattr(opp_active, 'yawn', 0) > 0 or opp_ailment in ['slp', '眠り']:
                        opp_is_yawned_or_asleep = True
                        shaped_prob += 0.04
                    if getattr(opp_active, 'status_con', None) or opp_ailment not in ['None', '']:
                        shaped_prob += 0.03

                if my_active:
                    my_ailment = getattr(my_active, 'ailment', 'None')
                    if getattr(my_active, 'yawn', 0) > 0 or my_ailment in ['slp', '眠り']:
                        my_is_yawned_or_asleep = True
                        shaped_prob -= 0.04
                    if getattr(my_active, 'status_con', None) or my_ailment not in ['None', '']:
                        shaped_prob -= 0.03

                # 🌟 C. 【設置技 ＋ あくびコンボ相乗効果】の価値前借り
                # 相手側に設置があり、かつ相手があくび/睡眠の場合（居座れば眠り、交代すればステロダメージを受ける絶望状況）
                if opp_has_hazards and opp_is_yawned_or_asleep:
                    shaped_prob += 0.08  # コンボ評価としてさらに +8% 加算

                # 逆に自分が設置技とあくび/睡眠を同時に押し付けられている場合
                if my_has_hazards and my_is_yawned_or_asleep:
                    shaped_prob -= 0.08  # 不利評価として -8% 減算

                # D. 能力ランク（積み状態）の価値補正 (A, C, Sは+0.02、B, Dは+0.01)
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

                # E. 天候（砂嵐）天候シナジー補正 (勝率 ±0.02 補正)
                if getattr(current_battle, 'weather', None) == 'sandstorm':
                    if my_active and any(t in ['いわ', 'じめん', 'はがね'] for t in my_active.types):
                        shaped_prob += 0.02
                    if opp_active and any(t in ['いわ', 'じめん', 'はがね'] for t in opp_active.types):
                        shaped_prob -= 0.02

                # F. ミミッキュの「ばけのかわ」の価値前借り (勝率 ±0.05 補正)
                if my_active and my_active.name == "ミミッキュ":
                    shaped_prob += 0.05
                if opp_active and opp_active.name == "ミミッキュ":
                    shaped_prob -= 0.05

                # G. イダイトウの「おはかまいり」の価値前借り
                if my_active and "イダイトウ" in my_active.name:
                    dead_count = sum(1 for p in current_battle.selected[my_side] if p.hp <= 0)
                    shaped_prob += dead_count * 0.03
                if opp_active and "イダイトウ" in opp_active.name:
                    dead_count_opp = sum(1 for p in current_battle.selected[opp_side] if p.hp <= 0)
                    shaped_prob -= dead_count_opp * 0.03

                # 勝率予測を 1%〜99% の範囲にクリッピングして確率の破綻を防止
                shaped_prob = max(0.01, min(0.99, shaped_prob))
                predictions[0][0] = shaped_prob
                predictions[0][1] = 1.0 - shaped_prob
    except Exception:
        pass
    return predictions


ReBeLValueNetwork.forward = shaped_value_network_forward


# =========================================================================
# 3. 【高度化】AegisTeamBuilder (基本選出 ＆ メタカウンター枠統合 Ver 16.0)
# =========================================================================
def calculate_matchup_tactical_scores(cand_name: str, opp_name: str) -> Tuple[float, float]:
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


def allocate_stat_points_randomly(indices: list, total_points: int = 66, max_single: int = 32) -> list:
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


def patched_build_team(self, core_name: str, pokemon_weights: Optional[dict] = None) -> Dict[str, Any]:
    """
    [Aegis Build Ver 16.0]
    1〜3体目(基本選出)を固め、4〜6体目は基本選出が苦手とする共通弱点タイプを補完し、
    かつ主要メタに強いカウンター要員を配備する二段階選定システム。
    """
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

    # 🚀 【第1段階】 1〜3体目（基本選出）の決定
    while len(team_members) < 3:
        current_weaknesses = []
        for member in team_members:
            current_weaknesses += self.calculate_weaknesses(Pokemon.zukan[member]["type"])

        best_candidate = None
        max_total_score = -999.0

        for candidate in self.mb_pokemon:
            if candidate in team_members:
                continue
            if not Pokemon.zukan.get(candidate):
                continue
            if any(Pokemon.zukan[candidate]["display_name"] == Pokemon.zukan[m]["display_name"] for m in team_members):
                continue

            cand_res = self.calculate_resistances(Pokemon.zukan[candidate]["type"])
            type_score = sum(2.0 if w in cand_res else 0.0 for w in current_weaknesses)
            type_score += sum(Pokemon.zukan[candidate]["base"]) * 0.001

            taimen_sum = 0.0
            uke_sum = 0.0
            for member in team_members:
                taimen, uke = calculate_matchup_tactical_scores(candidate, member)
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
        else:
            break

    # 🚀 【第2段階】 4〜6体目（カウンター・弱点補完選出）の決定
    ENVIRONMENT_METAS = ["ハバタクカミ", "サーフゴー", "カイリュー", "テツノブジン", "パオジアン"]

    while len(team_members) < 6:
        # 1. 基本選出(1〜3体)が共通して苦手（弱点）とするタイプを算出
        basic_weaknesses = []
        for member in team_members[:3]:
            basic_weaknesses += self.calculate_weaknesses(Pokemon.zukan[member]["type"])

        # 最も被っている弱点タイプ（最優先補完タイプ）を特定
        from collections import Counter
        weak_counts = Counter(basic_weaknesses)
        priority_types = [item[0] for item in weak_counts.most_common(3)]  # 上位3つの苦手タイプ

        best_counter_candidate = None
        max_counter_score = -999.0

        for candidate in self.mb_pokemon:
            if candidate in team_members:
                continue
            if not Pokemon.zukan.get(candidate):
                continue
            if any(Pokemon.zukan[candidate]["display_name"] == Pokemon.zukan[m]["display_name"] for m in team_members):
                continue

            # A. 弱点補完スコア
            cand_res = self.calculate_resistances(Pokemon.zukan[candidate]["type"])
            type_shield_score = sum(3.0 if t in cand_res else 0.0 for t in priority_types)

            # B. 主要メタへの対面性能スコア
            meta_taimen_sum = 0.0
            actual_targets = [t for t in ENVIRONMENT_METAS if t in Pokemon.zukan]
            if actual_targets:
                for target in actual_targets:
                    taimen, _ = calculate_matchup_tactical_scores(candidate, target)
                    meta_taimen_sum += taimen
                avg_meta_taimen = meta_taimen_sum / len(actual_targets)
            else:
                avg_meta_taimen = 0.0

            # C. 基本選出メンバーとの最低限のWord2Vec共起
            w2v_score = 0.0
            if self.w2v_model:
                synergies = [self.get_w2v_synergy(m, candidate) for m in team_members[:3]]
                w2v_score = sum(synergies) / len(synergies) if synergies else 0.0

            total_counter_score = type_shield_score + (avg_meta_taimen * 0.02) + (w2v_score * 3.0)

            if total_counter_score > max_counter_score:
                max_counter_score = total_counter_score
                best_counter_candidate = candidate

        if best_counter_candidate:
            team_members.append(best_counter_candidate)
        else:
            # フォールバック
            fallback_pool = [c for c in self.mb_pokemon if c not in team_members and Pokemon.zukan.get(c)]
            if fallback_pool:
                team_members.append(random.choice(fallback_pool))
            else:
                break

    # 持ち物適合定義
    TYPE_BOOSTING_ITEMS = {
        "メタルコート": "はがね", "きせきのタネ": "くさ", "もくたん": "ほのお",
        "しんぴのしずく": "みず", "シルクのスカーフ": "ノーマル", "するどいくちばし": "ひこう",
        "ぎんのこな": "むし", "じしゃく": "でんき", "かたいたし": "いわ",
        "のろいのおふだ": "ゴースト", "りゅうのキバ": "ドラゴン", "どくばり": "どく",
        "やわらかいすな": "じめん", "くろいメガネ": "あく", "くろおび": "かくとう",
        "とけないこおり": "こおり", "まがったスプーン": "エスパー", "ようせいのハネ": "フェアリー"
    }

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

    # 事前割り当てループ (メガストーン優先配置)
    for member in team_members:
        zukan_entry = Pokemon.zukan.get(member, {})
        abilities = zukan_entry.get("ability", [])

        mega_stone_name = member.split("(")[0] + "ナイト"

        if mega_stone_name in self.mb_items:
            mega_prob = self.MEGA_PROBABILITIES.get(member, 0.50)
            if random.random() < mega_prob:
                assigned_items[member] = mega_stone_name
                continue

        available_items = [item for item in normal_items_pool if item not in assigned_items.values()]
        if available_items:
            local_item_tiers = dict(self.ITEM_TIERS)

            if "ひでり" in abilities:
                local_item_tiers["あついいわ"] = 5.0
            if "あめふらし" in abilities:
                local_item_tiers["しめったいわ"] = 5.0
            if "すなおこし" in abilities:
                local_item_tiers["さらさらいわ"] = 5.0
            if "ゆきふらし" in abilities:
                local_item_tiers["つめたいいわ"] = 5.0

            base_member_name = member.split("(")[0]
            if base_member_name in self.WALL_SETTER_POKEMON:
                local_item_tiers["ひかりのねんど"] = 5.0

            item_weights = [local_item_tiers.get(itm, 0.1) for itm in available_items]
            chosen_item = random.choices(available_items, weights=item_weights, k=1)[0]
            assigned_items[member] = chosen_item
        else:
            assigned_items[member] = ""

    for i, name in enumerate(team_members):
        zukan_entry = Pokemon.zukan[name]
        dyn_data = pokemon_weights.get(name, {}) if pokemon_weights else {}

        # 🚀 [ステップ1: 技構成の選定を最優先で行う]
        learnable = self.learnsets.get(name, ["テラバースト"])
        move_weights = []
        for move_name in learnable:
            static_w = 1.0
            dynamic_w = dyn_data.get("moves", {}).get(move_name, 1.0)
            move_weights.append(static_w * dynamic_w)

        chosen_moves = []

        if name == "メタモン":
            chosen_moves = ["へんしん"]
        else:
            temp_pool = list(learnable)
            temp_weights = list(move_weights)

            attack_moves_pool = []
            for m in learnable:
                m_data = Pokemon.all_moves.get(m)
                if m_data and m_data.get("class", "sta") != "sta":
                    attack_moves_pool.append(m)

            force_attack = (random.random() < 0.80) and len(attack_moves_pool) > 0
            if force_attack:
                atk_weights = [move_weights[learnable.index(m)] for m in attack_moves_pool]
                if sum(atk_weights) <= 0:
                    atk_weights = [1.0] * len(attack_moves_pool)

                chosen_atk = random.choices(attack_moves_pool, weights=atk_weights, k=1)[0]
                chosen_moves.append(chosen_atk)
                idx = temp_pool.index(chosen_atk)
                temp_pool.pop(idx)
                temp_weights.pop(idx)

            num_to_select = min(4 - len(chosen_moves), len(temp_pool))
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

        # 🚀 [ステップ2: 確定した技から物理・特殊判定（からをやぶる等）]
        has_physical_attack = False
        has_special_attack = False
        for mv in chosen_moves:
            mv_data = Pokemon.all_moves.get(mv)
            if mv_data:
                mv_class = mv_data.get("class", "sta")
                if mv_class == "phy":
                    has_physical_attack = True
                elif mv_class == "spc":
                    has_special_attack = True

        if any(m in chosen_moves for m in ["つるぎのまい", "りゅうのまい", "ビルドアップ"]):
            has_physical_attack = True
        if any(m in chosen_moves for m in ["わるだくみ", "めいそう"]):
            has_special_attack = True

        if "からをやぶる" in chosen_moves:
            if has_physical_attack and not has_special_attack:
                has_physical_attack = True
            elif has_special_attack and not has_physical_attack:
                has_special_attack = True
            else:
                has_physical_attack = True
                has_special_attack = True

        # 🚀 [ステップ3: 努力値配分テンプレート]
        base_stats = zukan_entry.get("base", [100, 100, 100, 100, 100, 100])
        stat_points = [0] * 6
        ev_category = "max_out"
        adj_nature_weights = {}

        rand_ev = random.random()

        if rand_ev < 0.04 or (has_physical_attack and has_special_attack and random.random() < 0.3):
            ev_category = "mixed"
            if random.random() < 0.5:
                stat_points = allocate_stat_points_randomly([1, 3, 5], total_points=66, max_single=32)
                adj_nature_weights["せっかち"] = 7.5
                adj_nature_weights["むじゃき"] = 7.5
            else:
                stat_points = allocate_stat_points_randomly([0, 1, 3], total_points=66, max_single=32)
                adj_nature_weights["ゆうかん"] = 7.5
                adj_nature_weights["れいせい"] = 7.5

        elif rand_ev < 0.50:
            ev_category = "hybrid"
            hybrid_patterns = [
                ("HBD", [0, 2, 4]),
                ("HBDS", [0, 2, 4, 5]),
                ("HABS", [0, 1, 2, 5]),
                ("HBCS", [0, 2, 3, 5]),
                ("HAB", [0, 1, 2]),
                ("HBC", [0, 2, 3]),
                ("HAD", [0, 1, 4]),
                ("HCD", [0, 3, 4]),
                ("HAS", [0, 1, 5]),
                ("HCS", [0, 3, 5]),
                ("HBS", [0, 2, 5]),
                ("HDS", [0, 4, 5]),
            ]

            valid_patterns = []
            for name_pat, idx_list in hybrid_patterns:
                if "A" in name_pat and has_special_attack and not has_physical_attack: continue
                if "C" in name_pat and has_physical_attack and not has_special_attack: continue
                valid_patterns.append((name_pat, idx_list))

            if not valid_patterns:
                valid_patterns = hybrid_patterns

            chosen_pattern_name, target_indices = random.choice(valid_patterns)
            stat_points = allocate_stat_points_randomly(target_indices, total_points=66, max_single=32)

            if "S" in chosen_pattern_name:
                adj_nature_weights["ようき"] = 4.0
                adj_nature_weights["おくびょう"] = 4.0
            elif "B" in chosen_pattern_name or "D" in chosen_pattern_name:
                adj_nature_weights["ずぶとい"] = 4.0
                adj_nature_weights["わんぱく"] = 4.0
                adj_nature_weights["しんちょう"] = 4.0
                adj_nature_weights["おだやか"] = 4.0

        else:
            ev_category = "max_out"
            max_out_candidates = ["HB", "HD", "HS"]
            if has_physical_attack or not has_special_attack:
                max_out_candidates += ["HA", "AS"]
            if has_special_attack or not has_physical_attack:
                max_out_candidates += ["HC", "CS"]

            chosen_max_type = random.choice(max_out_candidates)

            if chosen_max_type == "HA":
                stat_points[0], stat_points[1], stat_points[5] = 32, 32, 2
                adj_nature_weights["いじっぱり"] = 4.0
            elif chosen_max_type == "HB":
                stat_points[0], stat_points[2], stat_points[4] = 32, 32, 2
                adj_nature_weights["わんぱく"] = 4.0
                adj_nature_weights["ずぶとい"] = 4.0
            elif chosen_max_type == "HC":
                stat_points[0], stat_points[3], stat_points[5] = 32, 32, 2
                adj_nature_weights["ひかえめ"] = 4.0
            elif chosen_max_type == "HD":
                stat_points[0], stat_points[4], stat_points[2] = 32, 32, 2
                adj_nature_weights["しんちょう"] = 4.0
                adj_nature_weights["おだやか"] = 4.0
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

        # ----------------------------------------------------
        # 🚀 [ステップ4: 致命的矛盾を生まないマイルド性格抽選]
        # ----------------------------------------------------
        adj_ability_weights = {}

        if "オーロラベール" in chosen_moves or "ふぶき" in chosen_moves:
            if "ゆきふらし" in zukan_entry.get("ability", []):
                adj_ability_weights["ゆきふらし"] = 5.0
            elif "ゆきがくれ" in zukan_entry.get("ability", []):
                adj_ability_weights["ゆきがくれ"] = 2.0
            assigned_items[name] = "ひかりのねんど"

        if "ステルスロック" in chosen_moves or "あくび" in chosen_moves:
            if "すなおこし" in zukan_entry.get("ability", []):
                adj_ability_weights["すなおこし"] = 5.0

        natures = list(self.NATURE_WEIGHTS.keys())
        nature_weights = []
        for nat in natures:
            static_w = self.NATURE_WEIGHTS[nat]
            dynamic_w = dyn_data.get("natures", {}).get(nat, 1.0)
            synergy_w = adj_nature_weights.get(nat, 1.0)

            # 物理・特殊アタッカーの「ねじれデメリット性格」を完全に排除
            if has_physical_attack and not has_special_attack:
                if nat in ["ひかえめ", "おくびょう", "ずぶとい", "おだやか"]:
                    static_w = 0.0
            if has_special_attack and not has_physical_attack:
                if nat in ["いじっぱり", "ようき", "わんぱく", "しんちょう"]:
                    static_w = 0.0

            nature_weights.append(static_w * dynamic_w * synergy_w)

        if sum(nature_weights) <= 0:
            nature_weights = [1.0] * len(natures)
        nature = random.choices(natures, weights=nature_weights, k=1)[0]

        # 特性の最終決定
        abilities = zukan_entry.get("ability", ["とくせいなし"])

        # メタモン専用特性ロック
        if name == "メタモン" and "かわりもの" in abilities:
            ability = "かわりもの"
        elif abilities:
            ability_weights = []
            for ab in abilities:
                static_w = 2.0 if ab in self.POWERFUL_ABILITIES else 1.0
                dynamic_w = dyn_data.get("abilities", {}).get(ab, 1.0)
                synergy_w = adj_ability_weights.get(ab, 1.0)
                ability_weights.append(static_w * dynamic_w * synergy_w)
            ability = random.choices(abilities, weights=ability_weights, k=1)[0]
        else:
            ability = "とくせいなし"

        # 🌟 持ち物適合選定
        pre_assigned = assigned_items.get(name, "")
        if "ナイト" not in pre_assigned:
            assigned_item = ""
            available_items = [itm for itm in normal_items_pool if
                               itm not in assigned_items.values() or itm == pre_assigned]
            if available_items:
                local_item_tiers = dict(self.ITEM_TIERS)

                # 天候岩および粘土 of 動的ブースト
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

                    # 半減実弱点適合フィルター
                    elif itm in TYPE_REDUCING_BERRIES:
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

                # 🛡️【安全対策ガード】自分に不適合な他種族専用のメガストーン（〜ナイト）を抽選プールから物理排除
                my_mega_stone = name.split("(")[0] + "ナイト"
                filtered_available_items = []
                filtered_item_weights = []
                for idx_itm, itm in enumerate(available_items):
                    if "ナイト" in itm and itm != my_mega_stone:
                        continue  # 不適合メガストーンをスキップ
                    filtered_available_items.append(itm)
                    filtered_item_weights.append(item_weights[idx_itm])

                if sum(filtered_item_weights) <= 0:
                    filtered_item_weights = [1.0] * len(filtered_available_items)

                assigned_item = random.choices(filtered_available_items, weights=filtered_item_weights, k=1)[0]
            assigned_items[name] = assigned_item
        else:
            assigned_item = pre_assigned  # 🌟 事前割り当て済みメガストーン等を引き継ぎ

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


AegisTeamBuilder.build_team = patched_build_team
AegisTeamBuilder.calculate_matchup_tactical_scores = calculate_matchup_tactical_scores
AegisTeamBuilder.allocate_stat_points_randomly = allocate_stat_points_randomly


# =========================================================================
# 3. 環境適応型（重み付き）チーム生成システム (ソフトリミッター搭載型)
# =========================================================================
def generate_evolved_team(builder: AegisTeamBuilder, weights: dict[str, Any]) -> list:
    """環境の勝率（重み）、および型重みに基づき、優秀な個体・型を引き当てて6体構築を生成する"""
    candidates = list(builder.mb_pokemon)

    prob_weights = []
    for name in candidates:
        val = weights.get(name, 1.0)
        if isinstance(val, dict):
            prob_weights.append(val.get("weight", 1.0))
        else:
            prob_weights.append(float(val))

    if sum(prob_weights) == 0:
        prob_weights = [1.0] * len(candidates)

    # ----------------------------------------------------
    # 🌟 特定の強キャラ(サーフゴー等)の出現比率を「最大15%」に制限し、溢れ分を他へ再分配
    # ----------------------------------------------------
    total_w = sum(prob_weights)
    if total_w > 0:
        max_allowed_w = total_w * 0.15
        adjusted_weights = []
        overflow = 0.0

        for w in prob_weights:
            if w > max_allowed_w:
                overflow += (w - max_allowed_w)
                adjusted_weights.append(max_allowed_w)
            else:
                adjusted_weights.append(w)

        under_limit_count = sum(1 for w in adjusted_weights if w < max_allowed_w)
        if under_limit_count > 0 and overflow > 0:
            redistribute_share = overflow / under_limit_count
            adjusted_weights = [
                (w + redistribute_share if w < max_allowed_w else w)
                for w in adjusted_weights
            ]
        prob_weights = adjusted_weights

    random_core = random.choices(candidates, weights=prob_weights, k=1)[0]
    team_dict = builder.build_team(random_core, pokemon_weights=weights)

    selected_team = []
    for s in team_dict:
        p = Pokemon()
        name = team_dict[s]['name']

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
# 4. 世代別環境ログ解析システム
# =========================================================================
def analyze_generation_meta(log_path: str) -> dict:
    """その世代の自己対戦結果を集計し、勝率および技、性格、特性の勝利実績を算出する"""
    if not os.path.exists(log_path):
        return {}

    pokemon_picks = Counter()
    pokemon_wins = Counter()
    pokemon_items = {}

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
                              analyzer, weights: dict, generation: int, selection_predictor=None) -> dict:
    match_seed = int(time.time() * 1000) % 1000000
    battle = Battle(seed=match_seed)

    team_p0 = generate_evolved_team(builder, weights)
    team_p1 = generate_evolved_team(builder, weights)

    battle.selected[0] = team_p0
    battle.selected[1] = team_p1

    opp_bert_prob_p0 = None
    opp_bert_prob_p1 = None
    if selection_predictor is not None:
        try:
            team_p0_names = [p.name for p in team_p0]
            team_p1_names = [p.name for p in team_p1]
            p0_pred, p1_pred = selection_predictor.predict(team_p0_names, team_p1_names)
            opp_bert_prob_p0 = p1_pred  # Player0から見たPlayer1（相手）の選出予測
            opp_bert_prob_p1 = p0_pred  # Player1から見たPlayer0（相手）の選出予測
        except Exception as e:
            print(f"⚠️ [Aegis BERT] 選出予測の事前取得に失敗しました(フラット信念で補填します): {e}")

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

        # BERTの選出確率予測を適用
        active_bert_prob = opp_bert_prob_p0 if pl == 0 else opp_bert_prob_p1

        for name in opp_names:
            flat_belief = analyzer._build_flat_belief(name)

            if active_bert_prob is not None:
                try:
                    opp_idx = opp_names.index(name)
                    weight_factor = max(0.01, active_bert_prob.selection_probs[opp_idx])
                except ValueError:
                    weight_factor = 1.0

                weighted_belief = {}
                for h, p in flat_belief.items():
                    weighted_belief[h] = p * weight_factor

                total_p = sum(weighted_belief.values())
                if total_p > 0:
                    beliefs[pl].beliefs[name] = {h: p / total_p for h, p in weighted_belief.items()}
                else:
                    beliefs[pl].beliefs[name] = flat_belief
            else:
                beliefs[pl].beliefs[name] = flat_belief

    # 前の対戦や初期値によるコマンドの残存（0等）をリセットし、command - 20 のインデックス暴走を完全に防ぐ
    battle.command = [None, None]

    battle.turn = 0
    for player in range(2):
        # キーワード引数を明示して、第3引数の command に landing=False がマッピングされるバグを防ぐ
        battle.change_pokemon(player=player, idx=0, command=None, landing=False)

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException("CFR solve timed out!")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)

    history_log = []
    while battle.winner() is None:
        battle.turn += 1
        commands = [None, None]
        turn_strategies = [None, None]

        for pl in [0, 1]:
            pbs = PublicBeliefState.from_battle(battle, perspective=pl, belief=beliefs[pl])
            cfr_solver.solver.num_samples = 3

            builtins._aegis_current_battle = battle

            my_strategy = None
            safe_set_alarm(5)  # 5秒タイマースタート

            try:
                my_strategy, _ = cfr_solver.solve(pbs, battle)
            except TimeoutException:
                print(
                    f"⚠️ [Timeout Guardian] ターン {battle.turn} (プレイヤー {pl}) のCFR計算が5秒を超過したため、ハング防止目的で一時的に遮断してランダム手へフォールバックします。")
                my_strategy = None
            finally:
                safe_set_alarm(0)  # タイマークリア

            if my_strategy:
                strat_sum = sum(my_strategy.values())
                if strat_sum <= 0:
                    strat_sum = 1.0
                normalized_strategy = {act: (val / strat_sum) for act, val in my_strategy.items()}

                formatted_strat = {}
                for action, prob in sorted(normalized_strategy.items(), key=lambda x: -x[1]):
                    action_name = str(action)
                    if isinstance(action, int):
                        if action in range(20, 26):
                            target_idx = action - 20
                            try:
                                target_name = battle.selected[pl][target_idx].name
                                action_name = f"switch_to_{target_name}"
                            except:
                                action_name = f"switch_{target_idx}"
                        else:
                            move_idx = action % 10
                            try:
                                move_name = battle.pokemon[pl].moves[move_idx]
                                action_name = f"move_{move_name}"
                            except:
                                action_name = f"move_slot_{move_idx}"
                    formatted_strat[action_name] = round(prob, 4)
                turn_strategies[pl] = formatted_strat

                actions = list(my_strategy.keys())
                probs = list(my_strategy.values())

                temperature = max(0.2, 1.5 - (generation / 200.0))

                adjusted_probs = []
                for p in probs:
                    p_safe = max(p, 1e-9)
                    adjusted_probs.append(p_safe ** (1.0 / temperature))

                sum_adj = sum(adjusted_probs)
                if sum_adj > 0:
                    probs = [ap / sum_adj for ap in adjusted_probs]

                commands[pl] = random.choices(actions, weights=probs, k=1)[0]
            else:
                avail = battle.available_commands(pl)
                commands[pl] = random.choice(avail) if avail else None

        battle.command = commands
        battle.proceed(commands=commands)

        history_log.append({
            "turn": battle.turn,
            "commands": commands,
            "hp": [battle.pokemon[0].hp, battle.pokemon[1].hp] if all(battle.pokemon) else [0, 0],
            "strategies": turn_strategies
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
    print("ℹ️ 使用デバイス: cpu (安全なCPU学習に固定しました)")
    Pokemon.init(season=22)

    for target_alias in ['キングズシールド', 'キング・シールド', 'キングズ・シールド']:
        if target_alias in Pokemon.all_moves:
            Pokemon.all_moves['キングシールド'] = Pokemon.all_moves[target_alias]
            break

    analyzer = AegisAnalyzer()
    builder = analyzer.team_builder
    selector = analyzer.team_selector
    cfr_solver = analyzer.cfr_solver

    try:
        from src.selection_bert.selection_belief import SelectionBeliefPredictor
        selection_predictor = SelectionBeliefPredictor.load(Path("log/selection_bert"))
        print("ℹ️ [Aegis BERT] Pretrained team selection predictor loaded successfully.")
    except Exception as e:
        selection_predictor = None
        print(f"ℹ️ [Aegis BERT] Pretrained selection predictor not found or failed to load: {e}")

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
        print(f"  ✨ 既存 of 1～{start_generation - 1} 世代的データを検出しました。")
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
                        weights=pokemon_weights,
                        generation=gen,
                        selection_predictor=selection_predictor
                    )
                    f_out.write(json.dumps(match_data, ensure_ascii=False) + "\n")
                    f_out.flush()  # 即時書き出しを強制して中断時のデータ破損（0バイト）を完全防止
                except Exception as e:
                    import traceback
                    traceback.print_exc()

        meta_report = analyze_generation_meta(gen_log_path)

        boss_meta = None
        if meta_report:
            sorted_by_tactical = sorted(meta_report.items(), key=lambda x: (-x[1]["wins"], -x[1]["picks"]))
            if sorted_by_tactical:
                boss_meta = sorted_by_tactical[0][0]

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

        if boss_meta:
            print(f"🎯 [MetaPoke Search] 世代 {gen} のトップメタ 【{boss_meta}】 に対するカウンターポケモンを特定中...")
            meta_candidates = []
            for candidate in builder.mb_pokemon:
                if candidate == boss_meta:
                    continue
                taimen, uke = calculate_matchup_tactical_scores(candidate, boss_meta)
                total_counter_score = taimen + (uke * 100.0)
                meta_candidates.append((candidate, total_counter_score))

            top_counters = sorted(meta_candidates, key=lambda x: -x[1])[:5]

            print(f"   ┗ 検出された対策ポケモン（次世代出現重み1.5倍ブースト対象）:")
            for rank_idx, (counter_name, score) in enumerate(top_counters, 1):
                old_w = pokemon_weights[counter_name]["weight"]

                if old_w < 4.0:
                    pokemon_weights[counter_name]["weight"] = max(0.1, min(10.0, old_w * 1.5))
                    print(
                        f"     {rank_idx}位: 【{counter_name}】 (補正前重み: {old_w:.2f} ➔ ブースト適用 | 補正後重み: {pokemon_weights[counter_name]['weight']:.2f})")
                else:
                    print(
                        f"     {rank_idx}位: 【{counter_name}】 (補前重み: {old_w:.2f} ➔ 現状維持（4.0超過によるブースト制限） | 補正後重み: {old_w:.2f})")

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

    # [動的インポート解決ガード]
    if not hasattr(AegisAnalyzer, '_aegis_flat_belief_patched'):
        def _build_flat_belief_fallback_patch(self, pokemon_name: str) -> Dict[Any, float]:
            from src.rebel.belief_state import PokemonTypeHypothesis
            hypotheses: Dict[PokemonTypeHypothesis, float] = {}

            moves_pool = self.mb_learnset.get(pokemon_name, ["テラバースト"])
            item_pool = list(self.mb_items) if self.mb_items else [""]
            tera_pool = list(Pokemon.type_id.keys())
            abilities_pool = Pokemon.zukan.get(pokemon_name, {}).get("ability", [""])

            num_samples = 200
            rng = random
            for _ in range(num_samples):
                moves = rng.sample(moves_pool, min(4, len(moves_pool)))

                mega_candidates = get_possible_mega_stones(pokemon_name)
                valid_mega_stones = [stone for stone in mega_candidates if stone in item_pool]

                if valid_mega_stones and rng.random() < 0.5:
                    item = rng.choice(valid_mega_stones)
                else:
                    item = rng.choice(item_pool)

                tera = rng.choice(tera_pool)
                nature = rng.choice(
                    ["いじっぱり", "ひかえめ", "ようき", "おくびょう", "わんぱく", "しんちょう", "おだやか",
                     "ずぶとい"])
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

            total = sum(hypotheses.values())
            for h in hypotheses:
                hypotheses[h] /= (total if total > 0 else 1.0)
            return hypotheses


        AegisAnalyzer._build_flat_belief = _build_flat_belief_fallback_patch
        AegisAnalyzer._aegis_flat_belief_patched = True
        print("ℹ️ [Aegis Patch] AegisAnalyzer._build_flat_belief 動的注入を完了しました。")

    # Battle.change_pokemon 引数競合・タイマン交代無効化・安全防止パッチ
    if not hasattr(Battle, '_aegis_change_pokemon_patched'):
        original_change_pokemon = Battle.change_pokemon


        def patched_change_pokemon(self, player, command=None, idx=0, landing=False, *args, **kwargs):
            player_int = int(player)
            party = self.selected[player_int] if (self.selected and player_int < len(self.selected)) else []
            party_len = len(party)
            active_p = self.pokemon[player_int] if (self.pokemon and player_int < len(self.pokemon)) else None
            alive_benches = [p for p in party if p.hp > 0 and p != active_p]

            if not alive_benches:
                return None

            cmd = self.command[player_int] if (self.command and player_int < len(self.command)) else None
            if cmd is not None and isinstance(cmd, int) and 20 <= cmd <= 25:
                target_idx = cmd - 20
            else:
                target_idx = idx

            is_valid = False
            if 0 <= target_idx < party_len:
                if party[target_idx].hp > 0 and party[target_idx] != active_p:
                    is_valid = True

            if not is_valid:
                idx_to_use = party.index(alive_benches[0]) if alive_benches else 0
            else:
                idx_to_use = target_idx

            if party_len > 0 and self.pokemon and player_int < len(self.pokemon):
                self.pokemon[player_int] = self.selected[player_int][idx_to_use]

            has_invalid_cmd = False
            old_cmd_val = None
            if self.command and player_int < len(self.command):
                curr_cmd = self.command[player_int]
                if curr_cmd is not None and (not isinstance(curr_cmd, int) or not (20 <= curr_cmd <= 25)):
                    old_cmd_val = curr_cmd
                    self.command[player_int] = None
                    has_invalid_cmd = True

            try:
                return original_change_pokemon(
                    self,
                    player=player_int,
                    command=command,
                    idx=idx_to_use,
                    landing=landing,
                    *args,
                    **kwargs
                )
            finally:
                if has_invalid_cmd and self.command and player_int < len(self.command):
                    self.command[player_int] = old_cmd_val


        Battle.change_pokemon = patched_change_pokemon
        Battle._aegis_change_pokemon_patched = True
        print(
            "ℹ️ [Aegis Patch] Battle.change_pokemon インデックスエラー安全防止パッチ(引数マッピング修正済)を適用しました。")

    # [Aegis Deepcopy Optimization Patch]
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
    # 🌟 Battle.available_commands テラスタル排除 ＆ 瀕死交代完全排除フィルタ
    # =========================================================================
    if not hasattr(Battle, '_aegis_available_commands_patched3'):
        original_available_commands = Battle.available_commands


        def patched_available_commands(self, player, *args, **kwargs):
            """
            レギュレーションM-B（テラスタル禁止環境）適合に加え、
            シミュレータ本来のバグである「瀕死のポケモンへの交代コマンド(20〜25)」を
            合法手リストから物理的に完全に排除します。

            これにより、CFRソルバーやランダムフォールバックが、
            そもそも『不正な交代』を脳内評価・選択すること自体を根本から防止します。
            """
            import warnings

            # オリジナルの呼び出し時に発生する "No available commands" 警告のみを一時的に非表示にする
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*No available commands.*")
                cmds = original_available_commands(self, player, *args, **kwargs)

            player_int = int(player)
            party = self.selected[player_int] if (self.selected and player_int < len(self.selected)) else []
            p_active = self.pokemon[player_int] if (self.pokemon and player_int < len(self.pokemon)) else None

            # 🌟 【クレンジング】テラスタル除外 ＆ 瀕死への交代コマンドの完全排除
            cleaned_cmds = []
            for c in cmds:
                if c in range(10, 14):  # テラスタル除外
                    continue
                if 20 <= c <= 25:  # 交代コマンドの検証
                    target_idx = c - 20
                    if target_idx >= len(party):
                        continue  # 範囲外除外
                    target_poke = party[target_idx]
                    # 交代先が自分自身、またはすでに瀕死（HP <= 0）の場合は選択肢から完全に抹殺
                    if target_poke == p_active or target_poke.hp <= 0:
                        continue
                cleaned_cmds.append(c)

            filtered_cmds = cleaned_cmds

            if not filtered_cmds:
                if p_active and p_active.hp > 0:
                    phase = "battle"
                    if len(args) > 0:
                        phase = args[0]
                    elif "phase" in kwargs:
                        phase = kwargs["phase"]

                    alive_benches = [pb.name for pb in party if pb.hp > 0 and pb != p_active]
                    active_conditions = [k for k, v in getattr(p_active, 'condition', {}).items() if v > 0]

                    if phase == "battle" or (phase == "change" and len(alive_benches) > 0):
                        print(f"\n🚨 [Aegis Available Commands Debug] --- 警告発生時の戦況診断ダンプ ---")
                        print(f"  - プレイヤー   : Player {player_int}")
                        print(f"  - フェーズ     : {phase}")
                        print(f"  - ポケモン     : {p_active.name} (HP: {p_active.hp}/{p_active.status[0]})")
                        print(f"  - 技スロット   : {getattr(p_active, 'moves', 'N/A')}")
                        print(f"  - 各技残りPP   : {getattr(p_active, 'pp', 'N/A')}")
                        print(f"  - 状態変化     : {active_conditions if active_conditions else 'なし'}")
                        print(f"  - 控えの生存者 : {alive_benches if alive_benches else 'なし (タイマン状態)'}")
                        print("-" * 60 + "\n")

                    # 生存している安全な控えを再計算
                    switch_cmds = []
                    for idx_temp, poke_bench in enumerate(party):
                        if poke_bench.hp > 0 and poke_bench != p_active:
                            switch_cmds.append(20 + idx_temp)

                    if switch_cmds:
                        filtered_cmds = switch_cmds
                    else:
                        filtered_cmds = [0]  # 通常技0（わるあがき）にフォールバック

            return filtered_cmds


        Battle.available_commands = patched_available_commands
        Battle._aegis_available_commands_patched3 = True
        print("ℹ️ [Aegis Patch] Battle.available_commands 瀕死交代完全排除パッチが適用されました。")

    # Battle.battle_command 内部ランダムエラー防止パッチ
    if not hasattr(Battle, '_aegis_battle_command_patched'):
        original_battle_command = Battle.battle_command


        def patched_battle_command(self, player, *args, **kwargs):
            cmds = self.available_commands(player)
            if cmds:
                return random.choice(cmds)
            return None


        Battle.battle_command = patched_battle_command
        Battle._aegis_battle_command_patched = True
        print("ℹ️ [Aegis Patch] Battle.battle_command 内部安全パッチを適用しました。")

    # その他補正パッチ群
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
    # 🌟 【根本解決】Battle.is_float & proceed & TOD_score 安全防壁統合パッチ
    # =========================================================================
    if not hasattr(Battle, '_aegis_tod_lock_guard_v4'):
        original_proceed = Battle.proceed
        original_tod_score = Battle.TOD_score if hasattr(Battle, 'TOD_score') else Battle.tod_score
        original_is_float = Battle.is_float if hasattr(Battle, 'is_float') else None


        # -------------------------------------------------------------------------
        # A. is_float (浮遊判定) の無限ループ・AttributeError 完全防止ガード
        # -------------------------------------------------------------------------
        def patched_is_float(self, player: int) -> bool:
            try:
                player_int = int(player)
                if player_int not in [0, 1]:
                    return False

                # 場にポケモンがいない（None）、または瀕死の場合は浮いていない(False)として即時返却
                p = self.pokemon[player_int] if (self.pokemon and player_int < len(self.pokemon)) else None
                if not p or p.hp <= 0:
                    return False

                # 通常判定を実行
                if original_is_float:
                    return original_is_float(self, player_int)
                return False
            except Exception:
                # 内部でいかなるエラーが発生した場合も、ハングを防ぐため安全に False を返して脱出
                return False


        # ---------------------------------------------------------------------
        # 1. コマンド・サニタイザー (Command Sanitizer)
        # ---------------------------------------------------------------------
        def sanitize_commands(battle_obj, commands):
            """
            実機進行前に不可能なコマンド（不正な交代等）を検証し、安全なデフォルト手に書き換えます。
            """
            if commands is None or len(commands) < 2:
                return commands

            # 🌟【根本解決】シミュレータ内部の交代/行動解決ループ中（change, action）は、
            # 内部の自動遷移を阻害しないためにサニタイズ（書き換え）を完全にバイパスします。
            current_phase = getattr(battle_obj, 'phase', 'battle')
            if current_phase in ['change', 'action']:
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


        # 進行メソッド (Proceed) の拡張
        def patched_proceed(self, commands=None):
            target_cmds = commands if commands is not None else self.command
            cmds = sanitize_commands(self, target_cmds)

            self._tod_debug_count = 0  # カウンタリセット

            if cmds:
                cmds = list(cmds)
                for player in range(2):
                    p = self.pokemon[player]
                    if p and p.hp > 0:
                        temp_moves = list(p.moves) if hasattr(p, 'moves') and p.moves else []
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

            res = original_proceed(self, commands=cmds)
            return res


        # 予備安全弁としてのTOD_scoreフック（150回空回りした場合は自動的に相手を勝者にして無限ループから脱出）
        def patched_tod_score(self, player, *args, **kwargs):
            count = getattr(self, '_tod_debug_count', 0) + 1
            self._tod_debug_count = count

            if count > 150:
                self._tod_debug_count = 0  # カウンタリセット
                if player == 0:
                    return 999999.0
                else:
                    return 0.0

            return original_tod_score(self, player, *args, **kwargs)


        # クラスメソッドの再マッピング（強制上書き）
        Battle.proceed = patched_proceed
        if hasattr(Battle, 'is_float'):
            Battle.is_float = patched_is_float
        if hasattr(Battle, 'TOD_score'):
            Battle.TOD_score = patched_tod_score
        else:
            Battle.tod_score = patched_tod_score

        Battle._aegis_tod_lock_guard_v4 = True
        print(
            "ℹ️ [Project Aegis] インデント修復および Battle.is_float, proceed, TOD_score 統合安全防壁パッチの強制適用が完了しました。")

    for target_alias in ['キングズシールド', 'キング・シールド', 'キングズ・シールド']:
        if target_alias in Pokemon.all_moves:
            Pokemon.all_moves['キングシールド'] = Pokemon.all_moves[target_alias]
            break

    # 進化ループの実行
    run_evolution_loop(total_generations=1000, matches_per_gen=40)