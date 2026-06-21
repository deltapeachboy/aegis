"""
能力値・補正計算専用モジュール

既存のbattle.py の attack_correction(), defence_correction(), power_correction() 等を独立化
"""

import logging

from src.damage_calculator_api.models.pokemon_models import (
    BattleConditions,
    MoveInput,
    PokemonState,
    StatusAilment,
    TerrainCondition,
    WeatherCondition,
)
from src.damage_calculator_api.utils.data_loader import get_data_loader

logger = logging.getLogger(__name__)


class StatCalculator:
    """
    能力値と各種補正を計算するクラス

    既存のBattle.attack_correction(), Battle.defence_correction(),
    Battle.power_correction(), Battle.damage_correction() を参考に実装
    """

    def __init__(self):
        self.data_loader = get_data_loader()

    def calculate_attack_stat(
        self,
        attacker: PokemonState,
        move_data,
        conditions: BattleConditions,
        opponent: PokemonState = None,
    ) -> int:
        """
        攻撃実数値を計算（特性・道具・状況補正込み）

        既存のBattle.attack_correction() を参考に実装
        """
        # 基本攻撃実数値を取得
        if move_data.is_physical:
            base_stat = attacker.stats.get("attack", 105)
            rank_boost = attacker.stat_boosts.get("attack", 0)
        else:
            base_stat = attacker.stats.get("sp_attack", 105)
            rank_boost = attacker.stat_boosts.get("sp_attack", 0)

        # ランク補正を適用
        rank_multiplier = self._get_rank_multiplier(rank_boost)
        stat_with_rank = int(base_stat * rank_multiplier)

        # 特性による補正
        ability_multiplier = self._get_attack_ability_multiplier(
            attacker, move_data, conditions
        )

        # 道具による補正
        item_multiplier = self._get_attack_item_multiplier(
            attacker, move_data, conditions
        )

        # 状態による補正
        status_multiplier = self._get_attack_status_multiplier(attacker, move_data)

        # その他の補正
        other_multiplier = self._get_attack_other_multiplier(
            attacker, move_data, conditions
        )

        # 相手の災いの特性による補正
        disaster_multiplier = self._get_attack_disaster_multiplier(
            attacker, move_data, opponent
        )

        # 最終攻撃実数値
        final_stat = int(
            stat_with_rank
            * ability_multiplier
            * item_multiplier
            * status_multiplier
            * other_multiplier
            * disaster_multiplier
        )

        return max(1, final_stat)  # 最低1

    def calculate_defense_stat(
        self,
        defender: PokemonState,
        move_data,
        conditions: BattleConditions,
        opponent: PokemonState = None,
    ) -> int:
        """
        防御実数値を計算（特性・道具・状況補正込み）

        既存のBattle.defence_correction() を参考に実装
        """
        # 基本防御実数値を取得
        if move_data.is_physical:
            base_stat = defender.stats.get("defense", 105)
            rank_boost = defender.stat_boosts.get("defense", 0)
        else:
            base_stat = defender.stats.get("sp_defense", 105)
            rank_boost = defender.stat_boosts.get("sp_defense", 0)

        # ランク補正を適用
        rank_multiplier = self._get_rank_multiplier(rank_boost)
        stat_with_rank = int(base_stat * rank_multiplier)

        # 特性による補正
        ability_multiplier = self._get_defense_ability_multiplier(
            defender, move_data, conditions
        )

        # 道具による補正
        item_multiplier = self._get_defense_item_multiplier(
            defender, move_data, conditions
        )

        # 壁による補正
        wall_multiplier = self._get_wall_multiplier(move_data, conditions)

        # その他の補正
        other_multiplier = self._get_defense_other_multiplier(
            defender, move_data, conditions
        )

        # 相手の災いの特性による補正
        disaster_multiplier = self._get_defense_disaster_multiplier(
            defender, move_data, opponent
        )

        # 最終防御実数値
        final_stat = int(
            stat_with_rank
            * ability_multiplier
            * item_multiplier
            * wall_multiplier
            * other_multiplier
            * disaster_multiplier
        )

        return max(1, final_stat)  # 最低1

    def calculate_move_power(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions,
    ) -> int:
        """
        技威力を計算（特性・道具・状況補正込み）

        既存のBattle.power_correction() を参考に実装
        """
        base_power = move_data.power
        if base_power <= 0:
            return 0

        # 基本威力修正
        power_modifier = move.power_modifier

        # 天気による補正
        weather_modifier = self._get_weather_power_modifier(move_data, conditions)

        # テラインによる補正
        terrain_modifier = self._get_terrain_power_modifier(move_data, conditions)

        # 特性による補正
        ability_modifier = self._get_power_ability_modifier(
            attacker, defender, move, move_data, conditions
        )

        # 道具による補正
        item_modifier = self._get_power_item_modifier(attacker, move_data)

        # 状況による補正
        situation_modifier = self._get_power_situation_modifier(
            attacker, defender, move, move_data, conditions
        )

        # 最終威力
        final_power = int(
            base_power
            * power_modifier
            * weather_modifier
            * terrain_modifier
            * ability_modifier
            * item_modifier
            * situation_modifier
        )

        return max(1, final_power)  # 最低1

    def calculate_final_damage_modifier(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions,
    ) -> float:
        """
        最終ダメージ補正を計算

        既存のBattle.damage_correction() を参考に実装
        """
        modifier = 1.0

        # 特性による最終ダメージ補正
        modifier *= self._get_final_damage_ability_modifier(
            attacker, defender, move, move_data, conditions
        )

        # 道具による最終ダメージ補正
        modifier *= self._get_final_damage_item_modifier(
            attacker, defender, move, move_data
        )

        # 技の特殊効果による補正
        modifier *= self._get_move_special_modifier(move, move_data, conditions)

        return modifier

    def _get_rank_multiplier(self, rank: int) -> float:
        """ランク補正の倍率を取得"""
        rank = max(-6, min(6, rank))  # -6〜+6に制限

        if rank >= 0:
            return (2 + rank) / 2
        else:
            return 2 / (2 - rank)

    def _get_attack_ability_multiplier(
        self, attacker: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """攻撃特性による補正"""
        ability = attacker.ability
        multiplier = 1.0

        # ちからもち（物理攻撃2倍）
        if ability == "ちからもち" and move_data.is_physical:
            multiplier *= 2.0

        # ヨガパワー（物理攻撃2倍）
        if ability == "ヨガパワー" and move_data.is_physical:
            multiplier *= 2.0

        # こんじょう（状態異常時物理攻撃1.5倍）
        if (
            ability == "こんじょう"
            and move_data.is_physical
            and attacker.status_ailment != StatusAilment.NONE
        ):
            multiplier *= 1.5

        # サンパワー（晴れ時特殊攻撃1.5倍）
        if (
            ability == "サンパワー"
            and move_data.is_special
            and conditions.weather == WeatherCondition.SUN
        ):
            multiplier *= 1.5

        # ソーラーパワー（晴れ時特殊攻撃1.5倍）
        if (
            ability == "ソーラーパワー"
            and move_data.is_special
            and conditions.weather == WeatherCondition.SUN
        ):
            multiplier *= 1.5

        # はりきり（物理攻撃1.5倍）
        if ability == "はりきり" and move_data.is_physical:
            multiplier *= 1.5

        # スロースタート（5ターン間攻撃・素早さ半減）
        # TODO: ターン管理が必要

        # もうか（HP1/3以下でほのお技威力1.5倍）
        if (
            ability == "もうか"
            and move_data.move_type == "ほのお"
            and attacker.hp_ratio <= 1 / 3
        ):
            multiplier *= 1.5

        # しんりょく（HP1/3以下でくさ技威力1.5倍）
        if (
            ability == "しんりょく"
            and move_data.move_type == "くさ"
            and attacker.hp_ratio <= 1 / 3
        ):
            multiplier *= 1.5

        # げきりゅう（HP1/3以下でみず技威力1.5倍）
        if (
            ability == "げきりゅう"
            and move_data.move_type == "みず"
            and attacker.hp_ratio <= 1 / 3
        ):
            multiplier *= 1.5

        # むしのしらせ（HP1/3以下でむし技威力1.5倍）
        if (
            ability == "むしのしらせ"
            and move_data.move_type == "むし"
            and attacker.hp_ratio <= 1 / 3
        ):
            multiplier *= 1.5

        # ひひいろのこどう（晴れ時攻撃1.33倍）
        if ability == "ひひいろのこどう" and conditions.weather == WeatherCondition.SUN:
            multiplier *= 5461 / 4096  # ≈ 1.33x

        # ハドロンエンジン（エレキフィールド時特攻1.33倍）
        if (
            ability == "ハドロンエンジン"
            and not move_data.is_physical
            and conditions.terrain == TerrainCondition.ELECTRIC
        ):
            multiplier *= 5461 / 4096  # ≈ 1.33x

        # クォークチャージ（エレキフィールド時最も高い能力値1.3倍）
        if (
            ability == "クォークチャージ"
            and conditions.terrain == TerrainCondition.ELECTRIC
            and attacker.paradox_boost_stat
        ):
            # 指定された能力値が攻撃系の場合のみ適用
            if move_data.is_physical and attacker.paradox_boost_stat == "attack":
                multiplier *= 1.3
            elif (
                not move_data.is_physical and attacker.paradox_boost_stat == "sp_attack"
            ):
                multiplier *= 1.3

        # 古代活性（晴れ時最も高い能力値1.3倍）
        if (
            ability == "こだいかっせい"
            and conditions.weather == WeatherCondition.SUN
            and attacker.paradox_boost_stat
        ):
            # 指定された能力値が攻撃系の場合のみ適用
            if move_data.is_physical and attacker.paradox_boost_stat == "attack":
                multiplier *= 1.3
            elif (
                not move_data.is_physical and attacker.paradox_boost_stat == "sp_attack"
            ):
                multiplier *= 1.3

        # すいほう（みず技威力2倍）
        if ability == "すいほう" and move_data.move_type == "みず":
            multiplier *= 2.0

        # ごりむちゅう（物理攻撃1.5倍）
        if ability == "ごりむちゅう" and move_data.is_physical:
            multiplier *= 1.5

        # フェアリースキン（ノーマル技がフェアリータイプになり威力1.2倍）
        if ability == "フェアリースキン" and move_data.move_type == "ノーマル":
            # Note: タイプ変更は別途TypeCalculatorで処理
            multiplier *= 1.2

        # スカイスキン（ノーマル技がひこうタイプになり威力1.2倍）
        if ability == "スカイスキン" and move_data.move_type == "ノーマル":
            multiplier *= 1.2

        # エレキスキン（ノーマル技がでんきタイプになり威力1.2倍）
        if ability == "エレキスキン" and move_data.move_type == "ノーマル":
            multiplier *= 1.2

        # フリーズスキン（ノーマル技がこおりタイプになり威力1.2倍）
        if ability == "フリーズスキン" and move_data.move_type == "ノーマル":
            multiplier *= 1.2

        # ダウンロード（相手の防御・特防の低い方を判定し、対応する攻撃を1段階上昇相当）
        # 既にランク補正は適用済みのため、ここでは実数値で1.5倍
        # TODO: より正確には相手の防御・特防実数値比較が必要

        # ふしぎなまもり（効果バツグンでない技の威力無効化）
        # TypeCalculatorで処理される

        # いたずらごころ（変化技先制+1）
        # 先制度は戦闘順序に影響するため、攻撃力には直接影響なし

        # ひとでなし（相手が状態異常時攻撃1.25倍）
        # TODO: 実装予定

        # 力ずく（追加効果のある技の威力1.3倍、追加効果なし）
        # TODO: 技の追加効果判定が必要

        return multiplier

    def _get_defense_ability_multiplier(
        self, defender: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """防御特性による補正"""
        ability = defender.ability
        multiplier = 1.0
        # ファーコート（物理防御2倍）
        if ability == "ファーコート" and move_data.is_physical:
            multiplier *= 2.0

        # マルチスケイル（HP満タン時ダメージ半減）
        if ability == "マルチスケイル" and defender.hp_ratio >= 1.0:
            multiplier *= 2.0  # 防御側なので防御実数値を2倍

        # シャドーシールド（HP満タン時ダメージ半減）
        if ability == "シャドーシールド" and defender.hp_ratio >= 1.0:
            multiplier *= 2.0

        # ふしぎなうろこ（状態異常時防御1.5倍）
        if (
            ability == "ふしぎなうろこ"
            and move_data.is_physical
            and defender.status_ailment != StatusAilment.NONE
        ):
            multiplier *= 1.5

        # あついしぼう、たいねつ等のタイプ技軽減は TypeCalculator で処理済み

        # クォークチャージ（エレキフィールド時最も高い能力値1.3倍）
        if (
            ability == "クォークチャージ"
            and conditions.terrain == TerrainCondition.ELECTRIC
            and defender.paradox_boost_stat
        ):
            # 指定された能力値が防御系の場合のみ適用
            if move_data.is_physical and defender.paradox_boost_stat == "defense":
                multiplier *= 1.3
            elif (
                not move_data.is_physical
                and defender.paradox_boost_stat == "sp_defense"
            ):
                multiplier *= 1.3

        # 古代活性（晴れ時最も高い能力値1.3倍）
        if (
            ability == "こだいかっせい"
            and conditions.weather == WeatherCondition.SUN
            and defender.paradox_boost_stat
        ):
            # 指定された能力値が防御系の場合のみ適用
            if move_data.is_physical and defender.paradox_boost_stat == "defense":
                multiplier *= 1.3
            elif (
                not move_data.is_physical
                and defender.paradox_boost_stat == "sp_defense"
            ):
                multiplier *= 1.3

        # きせき（進化前ポケモンの防御・特防1.5倍）
        # TODO: 進化前判定が必要

        # あついしぼう（ほのお・こおり技半減）
        if ability == "あついしぼう" and move_data.move_type in ["ほのお", "こおり"]:
            multiplier *= 2.0  # 防御実数値2倍で半減効果

        # たいねつ（ほのお技半減）
        if ability == "たいねつ" and move_data.move_type == "ほのお":
            multiplier *= 2.0

        # もらいび（ほのお技無効化、特攻上昇）
        # 無効化はTypeCalculatorで処理

        # ちょすい（みず技無効化、HP回復）
        # 無効化はTypeCalculatorで処理

        # よびみず（みず技を自分に誘導、特攻上昇）
        # 誘導効果は戦闘システムで処理

        # でんきエンジン（でんき技無効化、素早さ上昇）
        # 無効化はTypeCalculatorで処理

        # ひらいしん（でんき技を自分に誘導、特攻上昇）
        # 誘導効果は戦闘システムで処理

        # そうしょく（くさ技無効化、攻撃上昇）
        # 無効化はTypeCalculatorで処理

        # こおりのりんぷん（特殊技のダメージ半減）
        if ability == "こおりのりんぷん" and not move_data.is_physical:
            multiplier *= 2.0

        # ファントムガード（HP満タン時ダメージ半減）
        if ability == "ファントムガード" and defender.hp_ratio >= 1.0:
            multiplier *= 2.0

        # もふもふ（接触技半減、ほのお技2倍）
        if ability == "もふもふ":
            # 接触技の判定
            contact_moves = [
                "10まんばりき", "DDラリアット", "Vジェネレート", "アームハンマー",
                "アイアンテール", "アイアンヘッド", "アイアンローラー", "アイススピナー",
                "アイスハンマー", "アイスボール", "アクアジェット", "アクアステップ",
                "アクアテール", "アクアブレイク", "アクセルブレイク", "アクセルロック",
                "アクロバット", "あてみなげ", "あなをほる", "あばれる", "アフロブレイク",
                "アンカーショット", "あんこくきょうだ", "いあいぎり", "イカサマ", "いかり",
                "いかりのまえば", "イナズマドライブ", "いわくだき", "インファイト",
                "ウェーブタックル", "ウッドハンマー", "ウッドホーン", "うっぷんばらし",
                "えだづき", "エラがみ", "おいうち", "おうふくビンタ", "おしおき",
                "おどろかす", "おんがえし", "かいりき", "カウンター", "かえんぐるま",
                "かかとおとし", "かげうち", "かたきうち", "がまん", "かみくだく",
                "かみつく", "かみなりのキバ", "かみなりパンチ", "がむしゃら", "からげんき",
                "からてチョップ", "からではさむ", "からみつく", "ガリョウテンセイ",
                "かわらわり", "がんせきアックス", "きあいパンチ", "ギアソーサー",
                "ギガインパクト", "きしかいせい", "きつけ", "きゅうけつ", "きょけんとつげき",
                "きょじゅうざん", "きょじゅうだん", "キラースピン", "きりさく", "きりふだ",
                "クイックターン", "くさむすび", "くさわけ", "くらいつく", "グラススライダー",
                "クラブハンマー", "グロウパンチ", "クロスチョップ", "クロスポイズン",
                "げきりん", "けたぐり", "こうそくスピン", "ゴーストダイブ", "こおりのキバ",
                "コメットパンチ", "ころがる", "サイコファング", "サイコブレイド",
                "サンダーダイブ", "ジェットパンチ", "シェルアームズ", "シェルブレード",
                "じごくぐるま", "シザークロス", "したでなめる", "じたばた", "じだんだ",
                "しっぺがえし", "しねんのずつき", "しぼりとる", "しめつける", "ジャイロボール",
                "シャドークロー", "シャドースチール", "シャドーダイブ", "シャドーパンチ",
                "じゃれつく", "しんそく", "スイープビンタ", "すいりゅうれんだ",
                "スカイアッパー", "ずつき", "すてみタックル", "スパーク", "スマートホーン",
                "せいなるつるぎ", "ソウルクラッシュ", "ソーラーブレード", "そらをとぶ",
                "ダークラッシュ", "たいあたり", "ダイビング", "たきのぼり", "たたきつける",
                "ダブルアタック", "ダブルウイング", "ダブルチョップ", "ダブルパンツァー",
                "だましうち", "ダメおし", "ちきゅうなげ", "ついばむ", "つけあがる",
                "つじぎり", "つつく", "つっぱり", "つのでつく", "つのドリル",
                "つばさでうつ", "つばめがえし", "つるのムチ", "であいがしら", "てかげん",
                "でんげきくちばし", "でんこうせっか", "でんこうそうげき", "どくづき",
                "どくどくのきば", "どげざつき", "ドゲザン", "とっしん", "とっておき",
                "とどめばり", "とびかかる", "とびげり", "とびつく", "とびはねる",
                "とびひざげり", "ともえなげ", "ドラゴンクロー", "ドラゴンダイブ",
                "ドラゴンテール", "ドラゴンハンマー", "トラバサミ", "トリプルアクセル",
                "トリプルキック", "トリプルダイブ", "ドリルくちばし", "ドリルライナー",
                "ドレインキッス", "ドレインパンチ", "トロピカルキック", "どろぼう",
                "とんぼがえり", "なしくずし", "ニードルアーム", "にぎりつぶす", "にどげり",
                "ニトロチャージ", "ねこだまし", "ネズミざん", "のしかかり", "ハートスタンプ",
                "ハードプレス", "ハードローラー", "ハイパードリル", "はいよるいちげき",
                "ばかぢから", "はがねのつばさ", "ばくれつパンチ", "ハサミギロチン",
                "はさむ", "はたきおとす", "はたく", "はっけい", "はなびらのまい",
                "はやてがえし", "バリアーラッシュ", "バレットパンチ", "パワーウィップ",
                "パワフルエッジ", "ヒートスタンプ", "ひけん・ちえなみ", "ひっかく",
                "ひっさつまえば", "ピヨピヨパンチ", "びりびりちくちく", "ふいうち",
                "フェイタルクロー", "ぶちかまし", "ふみつけ", "フライングプレス",
                "プラズマフィスト", "フリーフォール", "フレアドライブ", "ブレイククロー",
                "ブレイズキック", "ブレイブバード", "ふんどのこぶし", "ぶんまわす",
                "ヘビーボンバー", "ホイールスピン", "ポイズンテール", "ほうふく",
                "ほしがる", "ほっぺすりすり", "ボディプレス", "ほのおのキバ",
                "ほのおのパンチ", "ほのおのムチ", "ボルテッカー", "まきつく",
                "マッハパンチ", "まとわりつく", "マルチアタック", "まわしげり",
                "みだれづき", "みだれひっかき", "みねうち", "むしくい", "むねんのつるぎ",
                "メガトンキック", "メガトンパンチ", "メガホーン", "めざましビンタ",
                "メタルクロー", "もろはのずつき", "やけっぱち", "やつあたり", "やまあらし",
                "ゆきなだれ", "らいげき", "らいめいげり", "リーフブレード", "リベンジ",
                "レイジングブル", "れいとうパンチ", "れんぞくぎり", "れんぞくパンチ",
                "ローキック", "ロケットずつき", "ロッククライム", "ワイドブレイカー",
                "ワイルドボルト", "わるあがき"
            ]
            if move_data.name in contact_moves:
                multiplier *= 2.0  # 接触技半減
            elif move_data.move_type == "ほのお":
                multiplier *= 0.5  # ほのお技2倍ダメージ

        # かんそうはだ（ほのお技1.25倍ダメージ、みず技無効）
        if ability == "かんそうはだ" and move_data.move_type == "ほのお":
            multiplier *= 0.8  # ほのお技で1.25倍ダメージを受ける（防御側なので0.8倍）

        return multiplier

    def _get_attack_item_multiplier(
        self, attacker: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """攻撃道具による補正"""
        item = attacker.item
        if not item:
            return 1.0

        multiplier = 1.0

        # こだわりハチマキ（物理攻撃1.5倍）
        if item == "こだわりハチマキ" and move_data.is_physical:
            multiplier *= 1.5

        # こだわりメガネ（特殊攻撃1.5倍）
        if item == "こだわりメガネ" and move_data.is_special:
            multiplier *= 1.5

        # タイプ強化アイテム
        item_data = self.data_loader.get_item_data(item)
        if item_data and item_data.boost_type == move_data.move_type:
            multiplier *= 1.2

        return multiplier

    def _get_defense_item_multiplier(
        self, defender: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """防御道具による補正"""
        item = defender.item
        if not item:
            return 1.0

        multiplier = 1.0

        # しんかのきせき（進化前ポケモンの防御・特防1.5倍）
        if item == "しんかのきせき":
            # TODO: 進化前判定が必要
            multiplier *= 1.5

        # とつげきチョッキ（特防1.5倍、変化技使用不可）
        if item == "とつげきチョッキ" and move_data.is_special:
            multiplier *= 1.5

        # メタルパウダー（メタモン専用、防御2倍）
        if (
            item == "メタルパウダー"
            and defender.species == "メタモン"
            and move_data.is_physical
        ):
            multiplier *= 2.0

        return multiplier

    def _get_attack_status_multiplier(self, attacker: PokemonState, move_data) -> float:
        """状態異常による攻撃補正"""
        # やけど状態での物理攻撃半減（特性「こんじょう」等は除く）
        if (
            attacker.status_ailment == StatusAilment.BURN
            and move_data.is_physical
            and attacker.ability not in ["こんじょう", "からかい"]
        ):
            return 0.5

        return 1.0

    def _get_wall_multiplier(self, move_data, conditions: BattleConditions) -> float:
        """壁による防御補正"""
        # リフレクター（物理技半減）
        if conditions.reflect and move_data.is_physical:
            return 2.0  # 防御実数値を2倍

        # ひかりのかべ（特殊技半減）
        if conditions.light_screen and move_data.is_special:
            return 2.0

        # オーロラベール（物理・特殊両方半減）
        if conditions.aurora_veil:
            return 2.0

        return 1.0

    def _get_weather_power_modifier(
        self, move_data, conditions: BattleConditions
    ) -> float:
        """天気による威力補正"""
        multiplier = 1.0

        if conditions.weather == WeatherCondition.SUN:
            if move_data.move_type == "ほのお":
                multiplier *= 1.5
            elif move_data.move_type == "みず":
                multiplier *= 0.5
        elif conditions.weather == WeatherCondition.RAIN:
            if move_data.move_type == "みず":
                multiplier *= 1.5
            elif move_data.move_type == "ほのお":
                multiplier *= 0.5
        elif conditions.weather == WeatherCondition.SANDSTORM:
            if move_data.move_type == "いわ":
                # すなあらしで特防1.5倍（いわタイプ）は別処理
                pass

        return multiplier

    def _get_terrain_power_modifier(
        self, move_data, conditions: BattleConditions
    ) -> float:
        """テラインによる威力補正"""
        multiplier = 1.0

        if (
            conditions.terrain == TerrainCondition.ELECTRIC
            and move_data.move_type == "でんき"
        ):
            multiplier *= 1.3
        elif (
            conditions.terrain == TerrainCondition.GRASSY
            and move_data.move_type == "くさ"
        ):
            multiplier *= 1.3
        elif (
            conditions.terrain == TerrainCondition.PSYCHIC
            and move_data.move_type == "エスパー"
        ):
            multiplier *= 1.3
        elif (
            conditions.terrain == TerrainCondition.MISTY
            and move_data.move_type == "フェアリー"
        ):
            multiplier *= 1.3

        return multiplier

    def _get_power_ability_modifier(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions,
    ) -> float:
        """特性による威力補正"""
        ability = attacker.ability
        multiplier = 1.0

        # てきおうりょく（タイプ一致技の威力2倍→1.5倍からさらに1.33倍）
        if ability == "てきおうりょく":
            attacker_species = self.data_loader.get_pokemon_data(attacker.species)
            if attacker_species and move_data.move_type in attacker_species.types:
                multiplier *= 4 / 3  # 1.5 → 2.0にするため4/3倍

        # アナライズ（後攻時威力1.3倍）
        # TODO: 行動順判定が必要

        # テクニシャン（威力60以下の技1.5倍）
        if ability == "テクニシャン" and move_data.power <= 60:
            multiplier *= 1.5

        # すなのちから（すなあらし時じめん・いわ・はがね技1.3倍）
        if (
            ability == "すなのちから"
            and conditions.weather == WeatherCondition.SANDSTORM
            and move_data.move_type in ["じめん", "いわ", "はがね"]
        ):
            multiplier *= 1.3

        # いわはこび（いわ技威力1.5倍）
        if ability == "いわはこび" and move_data.move_type == "いわ":
            multiplier *= 1.5

        # トランジスタ（でんき技威力1.3倍）
        if ability == "トランジスタ" and move_data.move_type == "でんき":
            multiplier *= 1.3

        # りゅうのあぎと（ドラゴン技威力1.5倍）
        if ability == "りゅうのあぎと" and move_data.move_type == "ドラゴン":
            multiplier *= 1.5

        # はがねつかい（はがね技威力1.5倍）
        if ability == "はがねつかい" and move_data.move_type == "はがね":
            multiplier *= 1.5

        # パンクロック（音技威力1.3倍）
        if ability == "パンクロック":
            # TODO: 音技判定が必要（技名での判定）
            sound_moves = [
                "ハイパーボイス",
                "ばくおんぱ",
                "りんしょう",
                "いびき",
                "エコーボイス",
            ]
            if move_data.name in sound_moves:
                multiplier *= 1.3

        # どくぼうそう（毒状態時物理技威力1.5倍）
        if (
            ability == "どくぼうそう"
            and move_data.is_physical
            and attacker.status_ailment == StatusAilment.POISON
        ):
            multiplier *= 1.5

        # ねつぼうそう（やけど状態時特殊技威力1.5倍）
        if (
            ability == "ねつぼうそう"
            and move_data.is_special
            and attacker.status_ailment == StatusAilment.BURN
        ):
            multiplier *= 1.5

        # ちからずく（追加効果のある技の威力1.3倍）
        if ability == "ちからずく":
            # 追加効果のある技の判定（技の分類を確認）
            # effect カテゴリの技は追加効果あり
            # TODO: より正確な追加効果判定が必要
            effect_moves = [
                "かみつく",
                "かみくだく",
                "あくのはどう",
                "げんしのちから",
                "がんせきふうじ",
                "いわなだれ",
                "ねんりき",
                "じんつうりき",
                "しねんのずつき",
                "サイコファング",
                "サイコキネシス",
                "ローキック",
                "はっけい",
                "ばくれつパンチ",
                "けたぐり",
                "きあいだま",
                "エナジーボール",
                "シャドーボール",
                "おどろかす",
                "れいとうビーム",
                "れいとうパンチ",
                "フリーズドライ",
                "ふぶき",
                "つららおとし",
                "こごえるかぜ",
                "こおりのキバ",
                "オーロラビーム",
                "マッドショット",
                "どろかけ",
                "だいちのちから",
                "じならし",
                "ほっぺすりすり",
                "ほうでん",
                "びりびりちくちく",
                "でんじほう",
                "でんきショック",
                "チャージビーム",
                "スパーク",
                "かみなりパンチ",
                "かみなりのキバ",
                "かみなり",
                "エレキネット",
                "10まんボルト",
                "ようかいえき",
                "ポイズンテール",
                "ヘドロばくだん",
                "ヘドロこうげき",
                "ヘドロウェーブ",
                "どくばり",
                "どくどくのキバ",
                "どくづき",
                "ダストシュート",
                "スモッグ",
                "クロスポイズン",
                "ワイドブレイカー",
                "りゅうのいぶき",
                "ドラゴンダイブ",
                "たつまき",
                "ブレイククロー",
                "ふみつけ",
                "のしかかり",
                "ねこだまし",
                "トライアタック",
                "ずつき",
                "こうそくスピン",
                "いびき",
                "ラスターカノン",
                "メタルクロー",
                "はがねのつばさ",
                "コメットパンチ",
                "アイアンヘッド",
                "アイアンテール",
                "とびはねる",
                "ゴッドバード",
                "エアスラッシュ",
                "ムーンフォース",
                "ソウルクラッシュ",
                "じゃれつく",
                "れんごく",
                "マジカルフレイム",
                "ほのおのムチ",
                "ほのおのまい",
                "ほのおのパンチ",
                "ほのおのキバ",
                "ふんえん",
                "ひのこ",
                "ねっぷう",
                "だいもんじ",
                "かえんほうしゃ",
                "みずのはどう",
                "バブルこうせん",
                "だくりゅう",
                "たきのぼり",
                "シェルブレード",
                "アクアブレイク",
                "むしのていこう",
                "むしのさざめき",
                "はいよるいちげき",
                "とびかかる",
            ]
            if move_data.name in effect_moves:
                multiplier *= 1.3

        # ひとでなし（相手が状態異常時攻撃1.25倍）
        # TODO: 相手の状態異常情報が必要

        # きもったま（ノーマル・かくとう技がゴーストタイプに当たる）
        # タイプ相性はTypeCalculatorで処理

        # アナライズ（後攻時技威力1.3倍）
        if ability == "アナライズ" and attacker.moves_last:
            multiplier *= 1.3

        # とうそうしん（同性1.25倍、異性0.75倍）
        if ability == "とうそうしん" and attacker.gender and defender.gender:
            if attacker.gender == defender.gender:
                multiplier *= 1.25  # 同性
            elif attacker.gender != defender.gender and attacker.gender != "genderless" and defender.gender != "genderless":
                multiplier *= 0.75  # 異性

        # そうだいしょう（倒れた味方1匹につき威力10%上昇、最大5匹まで）
        if ability == "そうだいしょう" and attacker.fainted_teammates > 0:
            boost_count = min(attacker.fainted_teammates, 5)  # 最大5匹まで
            multiplier *= 1.0 + (boost_count * 0.1)  # 1匹につき10%上昇

        # スクリューおびれ（回転技の威力1.5倍）
        if ability == "スクリューおびれ":
            # 回転技の判定
            spin_moves = [
                "こうそくスピン",
                "からではさむ",
                "ギアソーサー",
                "ホイールスピン",
                "アイススピナー",
                "ローラースケート",
                "ドリルライナー",
                "つのドリル",
                "ハイパードリル",
                "エレキボール",
                "ジャイロボール",
                "ロケットずつき",
                "ローリングアタック",
                "ころがる",
                "ロックブラスト",
            ]
            if move_data.name in spin_moves:
                multiplier *= 1.5

        # がんじょうあご（噛み技の威力1.5倍）
        if ability == "がんじょうあご":
            # 噛み技の判定（bite カテゴリ）
            bite_moves = [
                "かみくだく",
                "かみつく",
                "かみなりのきば",
                "くらいつく",
                "こおりのきば",
                "どくどくのきば",
                "ひっさつまえば",
                "ほのおのきば",
                "エラがみ",
                "サイコファング",
            ]
            if move_data.name in bite_moves:
                multiplier *= 1.5

        # メガランチャー（波動技の威力1.5倍）
        if ability == "メガランチャー":
            # 波動技の判定（wave カテゴリ）
            wave_moves = [
                "あくのはどう",
                "こんげんのはどう",
                "だいちのはどう",
                "はどうだん",
                "みずのはどう",
                "りゅうのはどう",
            ]
            if move_data.name in wave_moves:
                multiplier *= 1.5

        # かたいつめ（接触技の威力1.3倍）
        if ability == "かたいつめ":
            # 接触技の判定（contact カテゴリ）
            contact_moves = [
                "10まんばりき", "DDラリアット", "Vジェネレート", "アームハンマー",
                "アイアンテール", "アイアンヘッド", "アイアンローラー", "アイススピナー",
                "アイスハンマー", "アイスボール", "アクアジェット", "アクアステップ",
                "アクアテール", "アクアブレイク", "アクセルブレイク", "アクセルロック",
                "アクロバット", "あてみなげ", "あなをほる", "あばれる", "アフロブレイク",
                "アンカーショット", "あんこくきょうだ", "いあいぎり", "イカサマ", "いかり",
                "いかりのまえば", "イナズマドライブ", "いわくだき", "インファイト",
                "ウェーブタックル", "ウッドハンマー", "ウッドホーン", "うっぷんばらし",
                "えだづき", "エラがみ", "おいうち", "おうふくビンタ", "おしおき",
                "おどろかす", "おんがえし", "かいりき", "カウンター", "かえんぐるま",
                "かかとおとし", "かげうち", "かたきうち", "がまん", "かみくだく",
                "かみつく", "かみなりのキバ", "かみなりパンチ", "がむしゃら", "からげんき",
                "からてチョップ", "からではさむ", "からみつく", "ガリョウテンセイ",
                "かわらわり", "がんせきアックス", "きあいパンチ", "ギアソーサー",
                "ギガインパクト", "きしかいせい", "きつけ", "きゅうけつ", "きょけんとつげき",
                "きょじゅうざん", "きょじゅうだん", "キラースピン", "きりさく", "きりふだ",
                "クイックターン", "くさむすび", "くさわけ", "くらいつく", "グラススライダー",
                "クラブハンマー", "グロウパンチ", "クロスチョップ", "クロスポイズン",
                "げきりん", "けたぐり", "こうそくスピン", "ゴーストダイブ", "こおりのキバ",
                "コメットパンチ", "ころがる", "サイコファング", "サイコブレイド",
                "サンダーダイブ", "ジェットパンチ", "シェルアームズ", "シェルブレード",
                "じごくぐるま", "シザークロス", "したでなめる", "じたばた", "じだんだ",
                "しっぺがえし", "しねんのずつき", "しぼりとる", "しめつける", "ジャイロボール",
                "シャドークロー", "シャドースチール", "シャドーダイブ", "シャドーパンチ",
                "じゃれつく", "しんそく", "スイープビンタ", "すいりゅうれんだ",
                "スカイアッパー", "ずつき", "すてみタックル", "スパーク", "スマートホーン",
                "せいなるつるぎ", "ソウルクラッシュ", "ソーラーブレード", "そらをとぶ",
                "ダークラッシュ", "たいあたり", "ダイビング", "たきのぼり", "たたきつける",
                "ダブルアタック", "ダブルウイング", "ダブルチョップ", "ダブルパンツァー",
                "だましうち", "ダメおし", "ちきゅうなげ", "ついばむ", "つけあがる",
                "つじぎり", "つつく", "つっぱり", "つのでつく", "つのドリル",
                "つばさでうつ", "つばめがえし", "つるのムチ", "であいがしら", "てかげん",
                "でんげきくちばし", "でんこうせっか", "でんこうそうげき", "どくづき",
                "どくどくのきば", "どげざつき", "ドゲザン", "とっしん", "とっておき",
                "とどめばり", "とびかかる", "とびげり", "とびつく", "とびはねる",
                "とびひざげり", "ともえなげ", "ドラゴンクロー", "ドラゴンダイブ",
                "ドラゴンテール", "ドラゴンハンマー", "トラバサミ", "トリプルアクセル",
                "トリプルキック", "トリプルダイブ", "ドリルくちばし", "ドリルライナー",
                "ドレインキッス", "ドレインパンチ", "トロピカルキック", "どろぼう",
                "とんぼがえり", "なしくずし", "ニードルアーム", "にぎりつぶす", "にどげり",
                "ニトロチャージ", "ねこだまし", "ネズミざん", "のしかかり", "ハートスタンプ",
                "ハードプレス", "ハードローラー", "ハイパードリル", "はいよるいちげき",
                "ばかぢから", "はがねのつばさ", "ばくれつパンチ", "ハサミギロチン",
                "はさむ", "はたきおとす", "はたく", "はっけい", "はなびらのまい",
                "はやてがえし", "バリアーラッシュ", "バレットパンチ", "パワーウィップ",
                "パワフルエッジ", "ヒートスタンプ", "ひけん・ちえなみ", "ひっかく",
                "ひっさつまえば", "ピヨピヨパンチ", "びりびりちくちく", "ふいうち",
                "フェイタルクロー", "ぶちかまし", "ふみつけ", "フライングプレス",
                "プラズマフィスト", "フリーフォール", "フレアドライブ", "ブレイククロー",
                "ブレイズキック", "ブレイブバード", "ふんどのこぶし", "ぶんまわす",
                "ヘビーボンバー", "ホイールスピン", "ポイズンテール", "ほうふく",
                "ほしがる", "ほっぺすりすり", "ボディプレス", "ほのおのキバ",
                "ほのおのパンチ", "ほのおのムチ", "ボルテッカー", "まきつく",
                "マッハパンチ", "まとわりつく", "マルチアタック", "まわしげり",
                "みだれづき", "みだれひっかき", "みねうち", "むしくい", "むねんのつるぎ",
                "メガトンキック", "メガトンパンチ", "メガホーン", "めざましビンタ",
                "メタルクロー", "もろはのずつき", "やけっぱち", "やつあたり", "やまあらし",
                "ゆきなだれ", "らいげき", "らいめいげり", "リーフブレード", "リベンジ",
                "レイジングブル", "れいとうパンチ", "れんぞくぎり", "れんぞくパンチ",
                "ローキック", "ロケットずつき", "ロッククライム", "ワイドブレイカー",
                "ワイルドボルト", "わるあがき"
            ]
            if move_data.name in contact_moves:
                multiplier *= 1.3

        # てつのこぶし（パンチ技の威力1.2倍）
        if ability == "てつのこぶし":
            # パンチ技の判定（punch カテゴリ）
            punch_moves = [
                "あんこくきょうだ", "かみなりパンチ", "きあいパンチ", "すいりゅうれんだ",
                "ばくれつパンチ", "ふんどのこぶし", "ぶちかまし", "ほのおのパンチ",
                "れいとうパンチ", "れんぞくパンチ", "アイスハンマー", "アームハンマー",
                "グロウパンチ", "コメットパンチ", "シャドーパンチ", "ジェットパンチ",
                "スカイアッパー", "ダブルパンツァー", "ドレインパンチ", "バレットパンチ",
                "ピヨピヨパンチ", "プラズマフィスト", "マッハパンチ", "メガトンパンチ"
            ]
            if move_data.name in punch_moves:
                multiplier *= 1.2

        # きれあじ（切断技の威力1.5倍）
        if ability == "きれあじ":
            # 切断技の判定（cut カテゴリ）
            cut_moves = [
                "いあいぎり", "がんせきアックス", "きょじゅうざん", "きりさく",
                "しんぴのつるぎ", "せいなるつるぎ", "つじぎり", "つばめがえし",
                "はっぱカッター", "むねんのつるぎ", "れんぞくぎり", "アクアカッター",
                "エアカッター", "エアスラッシュ", "クロスポイズン", "サイコカッター",
                "シェルブレード", "シザークロス", "ソーラーブレード", "ドゲザン",
                "ネズミざん", "リーフブレード"
            ]
            if move_data.name in cut_moves:
                multiplier *= 1.5

        # ノーマルスキン（すべての技がノーマルタイプになり威力1.2倍）
        if ability == "ノーマルスキン":
            # 技タイプをノーマルに変更する効果はTypeCalculatorで処理
            multiplier *= 1.2

        # もらい火（ほのお技威力1.5倍）
        if ability == "もらいび" and move_data.move_type == "ほのお":
            multiplier *= 1.5

        return multiplier

    def _get_power_item_modifier(self, attacker: PokemonState, move_data) -> float:
        """道具による威力補正"""
        item = attacker.item
        if not item:
            return 1.0

        multiplier = 1.0

        # いのちのたま（全技威力1.3倍、反動あり）
        if item == "いのちのたま":
            multiplier *= 1.3

        # たつじんのおび（効果抜群技威力1.2倍）
        # TODO: タイプ相性情報が必要

        # ノーマルジュエル等（該当タイプの技威力1.3倍、一度のみ）
        # TODO: 消費アイテム管理が必要

        return multiplier

    def _get_power_situation_modifier(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions,
    ) -> float:
        """状況による威力補正"""
        multiplier = 1.0

        # ウェザーボール（天気により威力・タイプ変化）
        if (
            move.name == "ウェザーボール"
            and conditions.weather != WeatherCondition.NONE
        ):
            multiplier *= 2.0

        # ジャイロボール（相手より遅いほど威力アップ）
        if move.name == "ジャイロボール":
            # TODO: 素早さ比較が必要
            pass

        # エレキボール（相手より速いほど威力アップ）
        if move.name == "エレキボール":
            # TODO: 素早さ比較が必要
            pass

        # アクセルブレイク/イナズマドライブ（効果バツグンの時威力1.3倍）
        if move.name == "アクセルブレイク" or move.name == "イナズマドライブ":
            # タイプ相性を確認（効果バツグンかどうか）
            type_effectiveness = self._check_type_effectiveness_for_move(
                attacker, defender, move_data
            )
            if type_effectiveness > 1.0:  # 効果バツグンの場合
                multiplier *= 5461 / 4096  # ≈ 1.33x

        return multiplier

    def _check_type_effectiveness_for_move(
        self, attacker: PokemonState, defender: PokemonState, move_data
    ) -> float:
        """技のタイプ相性を簡易チェック"""
        # TypeCalculatorを使用してタイプ相性を取得
        from src.damage_calculator_api.calculators.type_calculator import TypeCalculator

        type_calc = TypeCalculator()

        # 簡易的なダミーのMoveInputとBattleConditionsを作成
        from src.damage_calculator_api.models.pokemon_models import (
            BattleConditions,
            MoveInput,
        )

        dummy_move = MoveInput(name=move_data.name)
        dummy_conditions = BattleConditions()

        return type_calc.calculate_type_effectiveness(
            attacker, defender, dummy_move, move_data, dummy_conditions
        )

    def _get_final_damage_ability_modifier(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions,
    ) -> float:
        """特性による最終ダメージ補正"""
        multiplier = 1.0

        # スナイパー（急所時ダメージ1.5倍→2.25倍）
        if attacker.ability == "スナイパー" and move.is_critical:
            multiplier *= 1.5  # 1.5倍 × 1.5倍 = 2.25倍

        # いろめがね（効果今ひとつ技を等倍にする）
        if attacker.ability == "いろめがね":
            type_effectiveness = self._check_type_effectiveness_for_move(
                attacker, defender, move_data
            )
            if type_effectiveness < 1.0:  # 効果今ひとつの場合
                # 効果今ひとつを等倍にするため、逆数を掛ける
                multiplier *= 1.0 / type_effectiveness

        # ハードロック（効果バツグンの技威力3/4）
        if defender.ability == "ハードロック":
            type_effectiveness = self._check_type_effectiveness_for_move(
                attacker, defender, move_data
            )
            if type_effectiveness > 1.0:  # 効果バツグンの場合
                multiplier *= 0.75

        # フィルター（効果バツグンの技威力3/4）
        if defender.ability == "フィルター":
            type_effectiveness = self._check_type_effectiveness_for_move(
                attacker, defender, move_data
            )
            if type_effectiveness > 1.0:  # 効果バツグンの場合
                multiplier *= 0.75

        # プリズムアーマー（効果バツグンの技威力3/4）
        if defender.ability == "プリズムアーマー":
            type_effectiveness = self._check_type_effectiveness_for_move(
                attacker, defender, move_data
            )
            if type_effectiveness > 1.0:  # 効果バツグンの場合
                multiplier *= 0.75

        return multiplier

    def _get_final_damage_item_modifier(
        self, attacker: PokemonState, defender: PokemonState, move: MoveInput, move_data
    ) -> float:
        """道具による最終ダメージ補正"""
        multiplier = 1.0

        # メトロノーム（連続使用で威力アップ）
        # TODO: 連続使用回数管理が必要

        return multiplier

    def _get_move_special_modifier(
        self, move: MoveInput, move_data, conditions: BattleConditions
    ) -> float:
        """技の特殊効果による補正"""
        multiplier = 1.0

        # 2回攻撃技（ダブルアタック等）
        # TODO: 技の特殊効果管理が必要

        return multiplier

    def _get_attack_other_multiplier(
        self, attacker: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """その他の攻撃補正"""
        multiplier = 1.0

        # おいかぜ（味方の素早さ2倍）
        if conditions.tailwind:
            # 攻撃に直接影響なし
            pass

        return multiplier

    def _get_defense_other_multiplier(
        self, defender: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """その他の防御補正"""
        multiplier = 1.0

        # すなあらしでいわタイプの特防1.5倍
        if conditions.weather == WeatherCondition.SANDSTORM and move_data.is_special:
            defender_species = self.data_loader.get_pokemon_data(defender.species)
            if defender_species and "いわ" in defender_species.types:
                multiplier *= 1.5

        return multiplier

    def _get_attack_disaster_multiplier(
        self, attacker: PokemonState, move_data, opponent: PokemonState
    ) -> float:
        """相手の災いの特性による攻撃実数値補正"""
        if opponent is None:
            return 1.0

        multiplier = 1.0
        opponent_ability = opponent.ability

        # わざわいのうつわ（相手の特殊攻撃25%減）
        if opponent_ability == "わざわいのうつわ" and not move_data.is_physical:
            multiplier *= 3072 / 4096  # ≈ 0.75x

        # わざわいのおふだ（相手の物理攻撃25%減）
        if opponent_ability == "わざわいのおふだ" and move_data.is_physical:
            multiplier *= 3072 / 4096  # ≈ 0.75x

        return multiplier

    def _get_defense_disaster_multiplier(
        self, defender: PokemonState, move_data, opponent: PokemonState
    ) -> float:
        """相手の災いの特性による防御実数値補正"""
        if opponent is None:
            return 1.0

        multiplier = 1.0
        opponent_ability = opponent.ability

        # わざわいのつるぎ（攻撃側がこの特性を持つ時、防御側の物理防御25%減）
        if opponent_ability == "わざわいのつるぎ" and move_data.is_physical:
            multiplier *= 3072 / 4096  # ≈ 0.75x

        # わざわいのたま（攻撃側がこの特性を持つ時、防御側の特殊防御25%減）
        if opponent_ability == "わざわいのたま" and not move_data.is_physical:
            multiplier *= 3072 / 4096  # ≈ 0.75x

        return multiplier
