"""
タイプ相性・STAB計算専用モジュール

既存のbattle.py の attack_type_correction() と defence_type_correction() を独立化
"""

from typing import List, Dict, Optional
import logging

from src.damage_calculator_api.models.pokemon_models import PokemonState, MoveInput, BattleConditions
from src.damage_calculator_api.utils.data_loader import get_data_loader

logger = logging.getLogger(__name__)


class TypeCalculator:
    """
    タイプ相性とSTAB（タイプ一致）計算を担当するクラス
    
    既存のBattle.attack_type_correction() と Battle.defence_type_correction() を参考に実装
    """
    
    def __init__(self):
        self.data_loader = get_data_loader()
        
        # タイプID辞書（既存のPokemon.type_idと同等）
        self.type_ids = {
            'ノーマル': 0, 'ほのお': 1, 'みず': 2, 'でんき': 3, 'くさ': 4, 'こおり': 5, 'かくとう': 6,
            'どく': 7, 'じめん': 8, 'ひこう': 9, 'エスパー': 10, 'むし': 11, 'いわ': 12, 'ゴースト': 13,
            'ドラゴン': 14, 'あく': 15, 'はがね': 16, 'フェアリー': 17, 'ステラ': 18
        }
        
        # 完全なタイプ相性表を初期化
        self._init_type_chart()
    
    def _init_type_chart(self):
        """
        完全なタイプ相性表を初期化
        
        既存のPokemon.type_correctionsと同等の19x19マトリックス
        """
        # 19x19のタイプ相性表を初期化（デフォルト1.0倍）
        self.type_chart = [[1.0 for _ in range(19)] for _ in range(19)]
        
        # タイプ相性を設定
        # 効果抜群（2.0倍）
        super_effective_pairs = [
            # ノーマル（攻撃側ID: 0）
            # 特になし
            
            # ほのお（攻撃側ID: 1）
            (1, 4),   # ほのお → くさ
            (1, 5),   # ほのお → こおり
            (1, 11),  # ほのお → むし
            (1, 16),  # ほのお → はがね
            
            # みず（攻撃側ID: 2）
            (2, 1),   # みず → ほのお
            (2, 8),   # みず → じめん
            (2, 12),  # みず → いわ
            
            # でんき（攻撃側ID: 3）
            (3, 2),   # でんき → みず
            (3, 9),   # でんき → ひこう
            
            # くさ（攻撃側ID: 4）
            (4, 2),   # くさ → みず
            (4, 8),   # くさ → じめん
            (4, 12),  # くさ → いわ
            
            # こおり（攻撃側ID: 5）
            (5, 4),   # こおり → くさ
            (5, 8),   # こおり → じめん
            (5, 9),   # こおり → ひこう
            (5, 14),  # こおり → ドラゴン
            
            # かくとう（攻撃側ID: 6）
            (6, 0),   # かくとう → ノーマル
            (6, 5),   # かくとう → こおり
            (6, 12),  # かくとう → いわ
            (6, 15),  # かくとう → あく
            (6, 16),  # かくとう → はがね
            
            # どく（攻撃側ID: 7）
            (7, 4),   # どく → くさ
            (7, 17),  # どく → フェアリー
            
            # じめん（攻撃側ID: 8）
            (8, 1),   # じめん → ほのお
            (8, 3),   # じめん → でんき
            (8, 7),   # じめん → どく
            (8, 12),  # じめん → いわ
            (8, 16),  # じめん → はがね
            
            # ひこう（攻撃側ID: 9）
            (9, 3),   # ひこう → でんき
            (9, 4),   # ひこう → くさ
            (9, 6),   # ひこう → かくとう
            (9, 11),  # ひこう → むし
            
            # エスパー（攻撃側ID: 10）
            (10, 6),  # エスパー → かくとう
            (10, 7),  # エスパー → どく
            
            # むし（攻撃側ID: 11）
            (11, 4),  # むし → くさ
            (11, 10), # むし → エスパー
            (11, 15), # むし → あく
            
            # いわ（攻撃側ID: 12）
            (12, 1),  # いわ → ほのお
            (12, 5),  # いわ → こおり
            (12, 9),  # いわ → ひこう
            (12, 11), # いわ → むし
            
            # ゴースト（攻撃側ID: 13）
            (13, 10), # ゴースト → エスパー
            (13, 13), # ゴースト → ゴースト
            
            # ドラゴン（攻撃側ID: 14）
            (14, 14), # ドラゴン → ドラゴン
            
            # あく（攻撃側ID: 15）
            (15, 10), # あく → エスパー
            (15, 13), # あく → ゴースト
            
            # はがね（攻撃側ID: 16）
            (16, 5),  # はがね → こおり
            (16, 12), # はがね → いわ
            (16, 17), # はがね → フェアリー
            
            # フェアリー（攻撃側ID: 17）
            (17, 6),  # フェアリー → かくとう
            (17, 14), # フェアリー → ドラゴン
            (17, 15), # フェアリー → あく
        ]
        
        # 効果今ひとつ（0.5倍）
        not_very_effective_pairs = [
            # ほのお（攻撃側ID: 1）
            (1, 1),   # ほのお → ほのお
            (1, 2),   # ほのお → みず
            (1, 12),  # ほのお → いわ
            (1, 14),  # ほのお → ドラゴン
            
            # みず（攻撃側ID: 2）
            (2, 2),   # みず → みず
            (2, 4),   # みず → くさ
            (2, 14),  # みず → ドラゴン
            
            # でんき（攻撃側ID: 3）
            (3, 3),   # でんき → でんき
            (3, 4),   # でんき → くさ
            (3, 14),  # でんき → ドラゴン
            
            # くさ（攻撃側ID: 4）
            (4, 1),   # くさ → ほのお
            (4, 4),   # くさ → くさ
            (4, 7),   # くさ → どく
            (4, 9),   # くさ → ひこう
            (4, 11),  # くさ → むし
            (4, 14),  # くさ → ドラゴン
            (4, 16),  # くさ → はがね
            
            # こおり（攻撃側ID: 5）
            (5, 1),   # こおり → ほのお
            (5, 2),   # こおり → みず
            (5, 5),   # こおり → こおり
            (5, 16),  # こおり → はがね
            
            # かくとう（攻撃側ID: 6）
            (6, 7),   # かくとう → どく
            (6, 9),   # かくとう → ひこう
            (6, 10),  # かくとう → エスパー
            (6, 11),  # かくとう → むし
            (6, 17),  # かくとう → フェアリー
            
            # どく（攻撃側ID: 7）
            (7, 7),   # どく → どく
            (7, 8),   # どく → じめん
            (7, 12),  # どく → いわ
            (7, 13),  # どく → ゴースト
            
            # じめん（攻撃側ID: 8）
            (8, 4),   # じめん → くさ
            (8, 11),  # じめん → むし
            
            # ひこう（攻撃側ID: 9）
            (9, 12),  # ひこう → いわ
            (9, 16),  # ひこう → はがね
            
            # エスパー（攻撃側ID: 10）
            (10, 10), # エスパー → エスパー
            (10, 16), # エスパー → はがね
            
            # むし（攻撃側ID: 11）
            (11, 1),  # むし → ほのお
            (11, 6),  # むし → かくとう
            (11, 7),  # むし → どく
            (11, 9),  # むし → ひこう
            (11, 13), # むし → ゴースト
            (11, 16), # むし → はがね
            (11, 17), # むし → フェアリー
            
            # いわ（攻撃側ID: 12）
            (12, 6),  # いわ → かくとう
            (12, 8),  # いわ → じめん
            (12, 16), # いわ → はがね
            
            # ゴースト（攻撃側ID: 13）
            (13, 15), # ゴースト → あく
            
            # ドラゴン（攻撃側ID: 14）
            (14, 16), # ドラゴン → はがね
            
            # あく（攻撃側ID: 15）
            (15, 6),  # あく → かくとう
            (15, 15), # あく → あく
            (15, 17), # あく → フェアリー
            
            # はがね（攻撃側ID: 16）
            (16, 1),  # はがね → ほのお
            (16, 2),  # はがね → みず
            (16, 3),  # はがね → でんき
            (16, 16), # はがね → はがね
            
            # フェアリー（攻撃側ID: 17）
            (17, 1),  # フェアリー → ほのお
            (17, 7),  # フェアリー → どく
            (17, 16), # フェアリー → はがね
        ]
        
        # 効果なし（0.0倍）
        no_effect_pairs = [
            (0, 13),  # ノーマル → ゴースト
            (3, 8),   # でんき → じめん
            (6, 13),  # かくとう → ゴースト
            (7, 16),  # どく → はがね
            (8, 9),   # じめん → ひこう
            (10, 15), # エスパー → あく
            (13, 0),  # ゴースト → ノーマル
            (17, 14), # フェアリー → ドラゴン（実際は等倍だが、フェアリーの場合ドラゴンに効果抜群）
        ]
        
        # タイプ相性表に値を設定
        for atk, def_ in super_effective_pairs:
            self.type_chart[atk][def_] = 2.0
            
        for atk, def_ in not_very_effective_pairs:
            self.type_chart[atk][def_] = 0.5
            
        for atk, def_ in no_effect_pairs:
            self.type_chart[atk][def_] = 0.0
    
    def calculate_stab_modifier(
        self,
        attacker: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions
    ) -> float:
        """
        STAB（タイプ一致）補正を計算
        
        既存のBattle.attack_type_correction() を参考に実装
        テラスタル時の特殊処理も含む
        """
        attacking_type = move.move_type or move_data.move_type
        attacker_species = self.data_loader.get_pokemon_data(attacker.species)
        
        if not attacker_species:
            return 1.0
        
        original_types = attacker_species.types
        
        # テラスタル時の処理
        if attacker.is_terastalized and attacker.terastal_type:
            terastal_type = attacker.terastal_type
            
            # ステラタイプの特殊処理
            if terastal_type == "ステラ":
                # ステラテラスタル時は元のタイプと一致する技のみ1.2倍
                if attacking_type in original_types:
                    return 1.2
                else:
                    return 1.0
            
            # 通常のテラスタルタイプ
            if attacking_type == terastal_type:
                # テラスタルタイプと一致: 2.0倍
                return 2.0
            elif attacking_type in original_types:
                # 元のタイプと一致: 1.5倍
                return 1.5
            else:
                # タイプ不一致
                return 1.0
        else:
            # 通常時のタイプ一致
            if attacking_type in original_types:
                return 1.5
            else:
                return 1.0
    
    def calculate_type_effectiveness(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions
    ) -> float:
        """
        タイプ相性を計算
        
        既存のBattle.defence_type_correction() を参考に実装
        テラスタル時の処理と特殊効果も含む
        """
        attacking_type = move.move_type or move_data.move_type
        defender_species = self.data_loader.get_pokemon_data(defender.species)
        
        if not defender_species:
            return 1.0
        
        # 防御側のタイプを取得
        defending_types = defender_species.types.copy()
        
        # テラスタル時のタイプ変更
        if defender.is_terastalized and defender.terastal_type:
            if defender.terastal_type == "ステラ":
                # ステラテラスタル時は元のタイプを保持
                pass
            else:
                # 通常のテラスタルタイプに変更
                defending_types = [defender.terastal_type]
        
        # 基本的なタイプ相性を計算
        effectiveness = 1.0
        for defending_type in defending_types:
            type_multiplier = self._get_type_multiplier(attacking_type, defending_type)
            effectiveness *= type_multiplier
        
        # 特殊効果の処理
        effectiveness = self._apply_special_type_interactions(
            attacker, defender, move, move_data, attacking_type, defending_types, effectiveness, conditions
        )
        
        return effectiveness
    
    def _get_type_multiplier(self, attacking_type: str, defending_type: str) -> float:
        """タイプ相性の倍率を取得"""
        atk_id = self.type_ids.get(attacking_type, 0)
        def_id = self.type_ids.get(defending_type, 0)
        
        return self.type_chart[atk_id][def_id]
    
    def _apply_special_type_interactions(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        attacking_type: str,
        defending_types: List[str],
        base_effectiveness: float,
        conditions: BattleConditions
    ) -> float:
        """
        特殊なタイプ相性処理を適用
        
        特性、道具、技の特殊効果によるタイプ相性変更
        """
        effectiveness = base_effectiveness
        
        # フリーズドライの特殊処理
        if move.name == "フリーズドライ" and "みず" in defending_types:
            # みずタイプに効果抜群
            effectiveness = 2.0 if effectiveness == 1.0 else effectiveness * 2.0
        
        # フライングプレスの特殊処理
        if move.name == "フライングプレス":
            # かくとう + ひこうタイプとして扱う
            fighting_effectiveness = 1.0
            flying_effectiveness = 1.0
            
            for defending_type in defending_types:
                fighting_effectiveness *= self._get_type_multiplier("かくとう", defending_type)
                flying_effectiveness *= self._get_type_multiplier("ひこう", defending_type)
            
            effectiveness = fighting_effectiveness * flying_effectiveness
        
        # 特性による無効化・変更
        defender_ability = defender.ability
        
        # ふゆう（じめん技無効）
        if defender_ability == "ふゆう" and attacking_type == "じめん":
            effectiveness = 0.0
        
        # ちくでん（でんき技無効化）
        if defender_ability == "ちくでん" and attacking_type == "でんき":
            effectiveness = 0.0
        
        # もらいび（ほのお技無効化）
        if defender_ability == "もらいび" and attacking_type == "ほのお":
            effectiveness = 0.0
        
        # そうしょく（くさ技無効化）
        if defender_ability == "そうしょく" and attacking_type == "くさ":
            effectiveness = 0.0
        
        # よびみず（みず技無効化）
        if defender_ability == "よびみず" and attacking_type == "みず":
            effectiveness = 0.0
        
        # でんきエンジン（でんき技無効化）
        if defender_ability == "でんきエンジン" and attacking_type == "でんき":
            effectiveness = 0.0
        
        # フィルター（効果抜群軽減）
        # Note: ハードロックは stat_calculator.py の final damage modifier で処理
        if defender_ability == "フィルター" and effectiveness > 1.0:
            effectiveness *= 0.75
        
        # たいねつ（ほのお技軽減）
        if defender_ability == "たいねつ" and attacking_type == "ほのお":
            effectiveness *= 0.5
        
        # あついしぼう（ほのお・こおり技軽減）
        if defender_ability == "あついしぼう" and attacking_type in ["ほのお", "こおり"]:
            effectiveness *= 0.5
        
        # 道具による効果
        defender_item = defender.item
        
        # リングターゲット（無効タイプに等倍で当たる）
        if defender_item == "リングターゲット" and effectiveness == 0.0:
            effectiveness = 1.0
        
        # TODO: その他の特性・道具効果を追加
        
        return effectiveness