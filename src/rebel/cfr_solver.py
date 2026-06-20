"""
CFR (Counterfactual Regret Minimization) サブゲーム解決

ReBeL において、現在のターンのサブゲームを解くために使用する。
信念状態からワールドをサンプリングし、CFR でナッシュ均衡に近い戦略を求める。
"""

from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Protocol

from src.pokemon_battle_sim.battle import Battle

from .belief_state import PokemonBeliefState, PokemonTypeHypothesis
from .public_state import PublicBeliefState, instantiate_battle_from_hypothesis


class ValueEstimator(Protocol):
    """終端状態の価値を推定するプロトコル"""

    def estimate(self, battle: Battle, player: int) -> float:
        """
        バトル状態の価値を推定

        Args:
            battle: バトル状態
            player: 価値を求めるプレイヤー

        Returns:
            [0, 1] の勝率
        """
        ...


def can_deal_damage(battle: Battle, attacker: int) -> bool:
    """
    攻撃側が防御側にダメージを与える手段があるか判定

    以下の状況を「ダメージ手段なし」と判定：
    - 全ての攻撃技がタイプ相性で無効
    - 状態異常技も効かない（はがねにどくどく等）

    Args:
        battle: バトル状態
        attacker: 攻撃側のプレイヤー番号

    Returns:
        True: ダメージ手段がある, False: ない
    """
    from src.pokemon_battle_sim.pokemon import Pokemon

    defender = 1 - attacker

    # 攻撃側の全ポケモン
    attacker_pokemon_list = [
        p for p in battle.selected[attacker] if p is not None and p.hp > 0
    ]
    # 防御側の全ポケモン
    defender_pokemon_list = [
        p for p in battle.selected[defender] if p is not None and p.hp > 0
    ]

    if not attacker_pokemon_list or not defender_pokemon_list:
        return False

    # 攻撃側のいずれかのポケモンが、防御側のいずれかにダメージを与えられるか
    for atk_poke in attacker_pokemon_list:
        for def_poke in defender_pokemon_list:
            for move in atk_poke.moves:
                if move is None:
                    continue
                if _move_can_hit(atk_poke, def_poke, move, battle):
                    return True

    return False


def _move_can_hit(
    attacker: "Pokemon", defender: "Pokemon", move: str, battle: Battle
) -> bool:
    """
    特定の技が相手に効くか判定

    Args:
        attacker: 攻撃側ポケモン
        defender: 防御側ポケモン
        move: 技名
        battle: バトル状態

    Returns:
        True: ダメージを与えられる可能性がある
    """
    from src.pokemon_battle_sim.pokemon import Pokemon

    # 技データを取得
    move_data = Pokemon.all_moves.get(move)
    if move_data is None:
        return False

    move_type = move_data.get("type", "ノーマル")
    move_class = move_data.get("class", "phy")
    power = move_data.get("power", 0)

    # 変化技は基本的にダメージ手段とみなさない（一部例外あり）
    if move_class == "status":
        # どくどく、やどりぎのタネ等は定数ダメージを与える可能性
        damaging_status_moves = {
            "やどりぎのタネ",
            "のろい",  # ゴーストタイプが使う場合
        }
        if move in damaging_status_moves:
            # やどりぎのタネはくさタイプには効かない
            if move == "やどりぎのタネ" and "くさ" in defender.types:
                return False
            return True

        # どくどく系：はがね、どくタイプには効かない
        if move in {"どくどく", "どくのこな", "どくガス"}:
            if "はがね" in defender.types or "どく" in defender.types:
                return False
            return True

        # その他の状態異常技はダメージ手段にならない
        return False

    # 威力0の技は特殊処理（固定ダメージ技など）
    if power == 0:
        # わるあがきはダメージを与える
        if move == "わるあがき":
            return True
        # ちきゅうなげ、ナイトヘッド等の固定ダメージ技
        fixed_damage_moves = {"ちきゅうなげ", "ナイトヘッド", "がんせきおとし"}
        if move in fixed_damage_moves:
            # ノーマル/かくとう技がゴーストに効かない
            if move_type == "ノーマル" and "ゴースト" in defender.types:
                return False
            if move_type == "かくとう" and "ゴースト" in defender.types:
                return False
            return True
        return False

    # タイプ相性チェック
    defender_types = list(defender.types)
    if defender.terastal and defender.Ttype:
        defender_types = [defender.Ttype]

    type_effectiveness = 1.0
    for def_type in defender_types:
        atk_type_id = Pokemon.type_id.get(move_type, 0)
        def_type_id = Pokemon.type_id.get(def_type, 0)
        if atk_type_id < len(Pokemon.type_corrections) and def_type_id < len(
            Pokemon.type_corrections[atk_type_id]
        ):
            type_effectiveness *= Pokemon.type_corrections[atk_type_id][def_type_id]

    # じめん技は浮いているポケモンに効かない（タイプ相性より先にチェック）
    if move_type == "じめん":
        is_defender_floating = (
            "ひこう" in defender.types
            or defender.ability == "ふゆう"
            or defender.item == "ふうせん"
        )
        if is_defender_floating:
            return False

    # 完全無効の場合
    if type_effectiveness == 0:
        # 特性「きもったま」等で無効化を貫通する場合があるがここでは簡略化
        return False

    return True


def check_hopeless_situation(battle: Battle, player: int) -> bool:
    """
    プレイヤーが必敗状態かどうか判定

    必敗状態の定義：
    1. 相手にダメージを与える手段が一切ない
    2. TOD（時間切れ勝ち）も見込めない状況

    Args:
        battle: バトル状態
        player: 判定対象のプレイヤー

    Returns:
        True: 必敗状態, False: まだ勝ち筋がある
    """
    # 相手にダメージ手段がない場合は必敗
    if not can_deal_damage(battle, player):
        # 相手もダメージ手段がない場合は引き分けの可能性（PPなくなるまで続く）
        opponent = 1 - player
        if not can_deal_damage(battle, opponent):
            # 両者ダメージ手段なし → PP切れまで粘れるが、最終的には引き分け
            return False
        return True

    return False


def default_value_estimator(battle: Battle, player: int) -> float:
    """
    デフォルトの価値推定関数

    勝敗が確定していれば 1.0/0.0、未決着なら HP 比率ベースの評価
    必敗状態の検出も行う
    """
    winner = battle.winner()

    if winner is not None:
        return 1.0 if winner == player else 0.0

    # 必敗状態チェック
    if check_hopeless_situation(battle, player):
        return 0.0  # 必敗

    opponent = 1 - player
    if check_hopeless_situation(battle, opponent):
        return 1.0  # 相手が必敗 = こちらの勝ち

    # HP比率ベースの中間評価
    def calc_strength(p: int) -> float:
        alive = 0
        hp_sum = 0.0
        for pokemon in battle.selected[p]:
            if pokemon is not None and pokemon.hp > 0:
                alive += 1
                max_hp = pokemon.status[0] if pokemon.status[0] > 0 else 1
                hp_sum += pokemon.hp / max_hp
        return alive + 0.3 * hp_sum

    my_strength = calc_strength(player)
    opp_strength = calc_strength(1 - player)
    total = my_strength + opp_strength

    if total < 1e-6:
        return 0.5
    return my_strength / total


@dataclass
class CFRConfig:
    """CFR の設定"""

    num_iterations: int = 100  # CFR イテレーション数
    num_world_samples: int = 10  # サンプリングするワールド数
    depth_limit: int = 1  # 探索深さ（ターン数）
    use_linear_cfr: bool = True  # Linear CFR を使用
    regret_matching_plus: bool = True  # Regret Matching+ を使用


class CFRSubgameSolver:
    """
    1ターンのサブゲームを CFR で解く

    外部サンプリング CFR (External Sampling MCCFR) を使用。
    信念状態からワールドをサンプリングし、各ワールドで CFR を実行、
    結果を集約して戦略を求める。
    """

    def __init__(
        self,
        config: Optional[CFRConfig] = None,
        value_estimator: Optional[ValueEstimator] = None,
    ):
        """
        Args:
            config: CFR の設定
            value_estimator: 終端状態の価値推定器
        """
        self.config = config or CFRConfig()
        self.value_fn = value_estimator or default_value_estimator

    def solve(
        self,
        pbs: PublicBeliefState,
        original_battle: Battle,
        include_surrender: bool = True,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """
        CFR でサブゲームを解く

        Args:
            pbs: Public Belief State
            original_battle: 元の Battle オブジェクト
            include_surrender: 降参を選択肢に含めるか

        Returns:
            (my_strategy, opp_strategy): 両プレイヤーの平均戦略
        """
        perspective = pbs.public_state.perspective
        opponent = 1 - perspective

        # 利用可能なアクション
        my_actions = list(original_battle.available_commands(perspective))
        opp_actions = list(original_battle.available_commands(opponent))

        if not my_actions or not opp_actions:
            # 行動がない場合
            return ({}, {})

        # 必敗状態の場合、降参を選択肢に追加
        if include_surrender:
            if check_hopeless_situation(original_battle, perspective):
                my_actions.append(Battle.SURRENDER)
            if check_hopeless_situation(original_battle, opponent):
                opp_actions.append(Battle.SURRENDER)

        # 累積リグレットと累積戦略
        # regrets[player][action] = cumulative regret
        regrets: list[dict[int, float]] = [
            {a: 0.0 for a in my_actions},
            {a: 0.0 for a in opp_actions},
        ]

        # 累積戦略（重み付き）
        strategy_sum: list[dict[int, float]] = [
            {a: 0.0 for a in my_actions},
            {a: 0.0 for a in opp_actions},
        ]

        # ワールドをサンプリング
        worlds = pbs.belief.sample_worlds(self.config.num_world_samples)

        # CFR イテレーション
        for t in range(1, self.config.num_iterations + 1):
            # 現在の戦略を計算（Regret Matching）
            current_strategies = [
                self._regret_matching(regrets[0], self.config.regret_matching_plus),
                self._regret_matching(regrets[1], self.config.regret_matching_plus),
            ]

            # 各ワールドで CFR 更新
            for world in worlds:
                # 具体的な Battle を構築
                battle = instantiate_battle_from_hypothesis(pbs, world, original_battle)

                # 両プレイヤーの CFR 更新
                for player in [perspective, opponent]:
                    player_idx = 0 if player == perspective else 1

                    # サンプリングする相手の行動
                    opp_player = 1 - player
                    opp_idx = 1 - player_idx
                    opp_strategy = current_strategies[opp_idx]
                    opp_action = self._sample_action(opp_strategy)

                    # 各行動のリグレットを計算
                    actions = my_actions if player == perspective else opp_actions
                    action_values = {}

                    for action in actions:
                        # 降参の場合は即座に敗北
                        if action == Battle.SURRENDER:
                            action_values[action] = 0.0
                            continue

                        # 行動を実行
                        test_battle = deepcopy(battle)
                        if player == 0:
                            commands = [action, opp_action]
                        else:
                            commands = [opp_action, action]

                        test_battle.proceed(commands=commands)

                        # 価値を推定
                        action_values[action] = self.value_fn(test_battle, player)

                    # 期待価値を計算
                    player_strategy = current_strategies[player_idx]
                    expected_value = sum(
                        player_strategy.get(a, 0) * action_values[a] for a in actions
                    )

                    # リグレット更新
                    for action in actions:
                        regret = action_values[action] - expected_value
                        regrets[player_idx][action] += regret

            # 戦略累積（Linear CFR: 後の方の重みを大きく）
            weight = t if self.config.use_linear_cfr else 1
            for player_idx in [0, 1]:
                strategy = current_strategies[player_idx]
                for action, prob in strategy.items():
                    strategy_sum[player_idx][action] += weight * prob

        # 平均戦略を計算
        my_avg = self._normalize(strategy_sum[0])
        opp_avg = self._normalize(strategy_sum[1])

        return my_avg, opp_avg

    def _regret_matching(
        self, regrets: dict[int, float], use_plus: bool = True
    ) -> dict[int, float]:
        """
        Regret Matching で現在の戦略を計算

        Args:
            regrets: 累積リグレット
            use_plus: Regret Matching+ を使用（負のリグレットを0にクリップ）

        Returns:
            行動確率分布
        """
        if use_plus:
            # Regret Matching+: 負のリグレットを0にクリップ
            positive_regrets = {a: max(0, r) for a, r in regrets.items()}
        else:
            positive_regrets = {a: max(0, r) for a, r in regrets.items()}

        total = sum(positive_regrets.values())

        if total > 0:
            return {a: r / total for a, r in positive_regrets.items()}
        else:
            # 全てのリグレットが非正の場合、均等分布
            n = len(regrets)
            return {a: 1.0 / n for a in regrets}

    def _sample_action(self, strategy: dict[int, float]) -> int:
        """戦略から行動をサンプリング"""
        if not strategy:
            return -1
        actions = list(strategy.keys())
        probs = list(strategy.values())
        return random.choices(actions, weights=probs, k=1)[0]

    def _normalize(self, strategy: dict[int, float]) -> dict[int, float]:
        """戦略を正規化"""
        total = sum(strategy.values())
        if total > 0:
            return {a: p / total for a, p in strategy.items()}
        n = len(strategy)
        return {a: 1.0 / n for a in strategy} if n > 0 else {}


class SimplifiedCFRSolver:
    """
    簡略化 CFR ソルバー

    完全な CFR の代わりに、仮説サンプリングと期待値計算による
    より高速な近似解法。
    """

    def __init__(
        self,
        num_samples: int = 30,
        value_estimator: Optional[ValueEstimator] = None,
    ):
        self.num_samples = num_samples
        self.value_fn = value_estimator or default_value_estimator

    def solve(
        self,
        pbs: PublicBeliefState,
        original_battle: Battle,
        include_surrender: bool = True,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """
        簡略化 CFR で戦略を計算

        各行動について、相手の全行動に対する期待値を計算し、
        最大最小の観点で戦略を決定する。
        """
        perspective = pbs.public_state.perspective
        opponent = 1 - perspective

        my_actions = list(original_battle.available_commands(perspective))
        opp_actions = list(original_battle.available_commands(opponent))

        if not my_actions:
            return ({}, {})
        if not opp_actions:
            return ({a: 1.0 / len(my_actions) for a in my_actions}, {})

        # 必敗状態の場合、降参を選択肢に追加
        if include_surrender:
            if check_hopeless_situation(original_battle, perspective):
                my_actions.append(Battle.SURRENDER)
            if check_hopeless_situation(original_battle, opponent):
                opp_actions.append(Battle.SURRENDER)

        # ワールドをサンプリング
        worlds = pbs.belief.sample_worlds(self.num_samples)

        # 行動ペアごとの期待値を計算
        # payoff_matrix[my_action][opp_action] = list of values
        payoff_matrix: dict[int, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for world in worlds:
            battle = instantiate_battle_from_hypothesis(pbs, world, original_battle)

            # 仮説適用後の利用可能コマンドを再計算
            hyp_opp_actions = battle.available_commands(opponent)
            if not hyp_opp_actions:
                hyp_opp_actions = [Battle.SKIP]

            for my_action in my_actions:
                for opp_action in opp_actions:
                    # 降参の処理
                    if my_action == Battle.SURRENDER:
                        # 自分が降参 → 自分の価値は0
                        payoff_matrix[my_action][opp_action].append(0.0)
                        continue
                    if opp_action == Battle.SURRENDER:
                        # 相手が降参 → 自分の価値は1
                        payoff_matrix[my_action][opp_action].append(1.0)
                        continue

                    test_battle = deepcopy(battle)

                    # 相手の行動が仮説適用後も有効か確認
                    actual_opp_action = opp_action if opp_action in hyp_opp_actions else hyp_opp_actions[0]

                    if perspective == 0:
                        commands = [my_action, actual_opp_action]
                    else:
                        commands = [actual_opp_action, my_action]

                    try:
                        test_battle.proceed(commands=commands)
                        value = self.value_fn(test_battle, perspective)
                    except (IndexError, AttributeError, KeyError, TypeError, ValueError):
                        # バトル進行でエラーが発生した場合はデフォルト値
                        value = 0.5

                    payoff_matrix[my_action][opp_action].append(value)

        # 平均ペイオフを計算
        avg_payoff: dict[int, dict[int, float]] = {}
        for my_action in my_actions:
            avg_payoff[my_action] = {}
            for opp_action in opp_actions:
                values = payoff_matrix[my_action][opp_action]
                avg_payoff[my_action][opp_action] = sum(values) / len(values) if values else 0.5

        # 自分の戦略: 各行動の最小期待値を最大化（maximin）
        my_scores = {}
        for my_action in my_actions:
            min_value = min(avg_payoff[my_action].values())
            my_scores[my_action] = min_value

        # Softmax で戦略に変換
        my_strategy = self._softmax_strategy(my_scores, temperature=0.5)

        # 相手の戦略: 同様に maximin（自分から見た minimax）
        opp_scores = {}
        for opp_action in opp_actions:
            # 相手視点では価値が反転
            min_value = min(1 - avg_payoff[my_a][opp_action] for my_a in my_actions)
            opp_scores[opp_action] = min_value

        opp_strategy = self._softmax_strategy(opp_scores, temperature=0.5)

        return my_strategy, opp_strategy

    def _softmax_strategy(
        self, scores: dict[int, float], temperature: float = 1.0
    ) -> dict[int, float]:
        """スコアを Softmax で戦略に変換"""
        if not scores:
            return {}

        max_score = max(scores.values())
        exp_scores = {a: pow(2.718, (s - max_score) / temperature) for a, s in scores.items()}
        total = sum(exp_scores.values())

        if total > 0:
            return {a: e / total for a, e in exp_scores.items()}
        n = len(scores)
        return {a: 1.0 / n for a in scores}


class LightweightCFRSolver:
    """
    超軽量 CFR ソルバー

    最小限のワールドサンプリングと単純なダメージベース評価で
    高速に近似戦略を計算する。精度は低いが学習初期には十分。
    """

    def __init__(
        self,
        num_samples: int = 3,
        value_estimator: Optional[ValueEstimator] = None,
    ):
        self.num_samples = num_samples
        self.value_fn = value_estimator or default_value_estimator

    def solve(
        self,
        pbs: PublicBeliefState,
        original_battle: Battle,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """
        軽量版 CFR で戦略を計算

        行動ごとの期待ダメージに基づく簡易評価
        """
        perspective = pbs.public_state.perspective
        opponent = 1 - perspective

        my_actions = original_battle.available_commands(perspective)
        opp_actions = original_battle.available_commands(opponent)

        if not my_actions:
            return ({}, {})
        if len(my_actions) == 1:
            return ({my_actions[0]: 1.0}, {a: 1.0 / len(opp_actions) for a in opp_actions} if opp_actions else {})
        if not opp_actions:
            return ({a: 1.0 / len(my_actions) for a in my_actions}, {})

        # 少数のワールドをサンプリング
        worlds = pbs.belief.sample_worlds(self.num_samples)

        # 各行動の期待価値を計算（相手の行動は均等分布を仮定）
        action_values: dict[int, list[float]] = {a: [] for a in my_actions}

        for world in worlds:
            battle = instantiate_battle_from_hypothesis(pbs, world, original_battle)

            for my_action in my_actions:
                # 相手は1つだけサンプリング（高速化）
                opp_action = random.choice(opp_actions)
                test_battle = deepcopy(battle)

                if perspective == 0:
                    commands = [my_action, opp_action]
                else:
                    commands = [opp_action, my_action]

                try:
                    test_battle.proceed(commands=commands)
                    value = self.value_fn(test_battle, perspective)
                except Exception:
                    value = 0.5

                action_values[my_action].append(value)

        # 平均値を計算
        avg_values = {a: sum(v) / len(v) if v else 0.5 for a, v in action_values.items()}

        # Softmax で戦略に変換
        my_strategy = self._softmax_strategy(avg_values, temperature=0.3)

        # 相手戦略は均等分布
        opp_strategy = {a: 1.0 / len(opp_actions) for a in opp_actions}

        return my_strategy, opp_strategy

    def _softmax_strategy(
        self, scores: dict[int, float], temperature: float = 1.0
    ) -> dict[int, float]:
        """スコアを Softmax で戦略に変換"""
        if not scores:
            return {}

        max_score = max(scores.values())
        exp_scores = {a: pow(2.718, (s - max_score) / temperature) for a, s in scores.items()}
        total = sum(exp_scores.values())

        if total > 0:
            return {a: e / total for a, e in exp_scores.items()}
        n = len(scores)
        return {a: 1.0 / n for a in scores}


class ReBeLSolver:
    """
    ReBeL スタイルのソルバー

    Value Network を使用して終端価値を推定し、
    CFR でサブゲームを解く。
    """

    def __init__(
        self,
        value_network: Optional["ReBeLValueNetwork"] = None,
        cfr_config: Optional[CFRConfig] = None,
        use_simplified: bool = True,
        use_lightweight: bool = False,
    ):
        """
        Args:
            value_network: Value Network（None の場合はヒューリスティック使用）
            cfr_config: CFR の設定
            use_simplified: 簡略化 CFR を使用
            use_lightweight: 超軽量CFRを使用（最高速、精度低）
        """
        from .value_network import ReBeLValueNetwork

        self.value_network = value_network
        self.use_simplified = use_simplified
        self.use_lightweight = use_lightweight

        # Value Estimator を構築
        if value_network is not None:
            # TODO: ネットワーク推論を使用した価値推定
            value_estimator = default_value_estimator
        else:
            value_estimator = default_value_estimator

        if use_lightweight:
            self.solver = LightweightCFRSolver(
                num_samples=min(3, cfr_config.num_world_samples) if cfr_config else 3,
                value_estimator=value_estimator,
            )
        elif use_simplified:
            self.solver = SimplifiedCFRSolver(
                num_samples=cfr_config.num_world_samples if cfr_config else 30,
                value_estimator=value_estimator,
            )
        else:
            self.solver = CFRSubgameSolver(
                config=cfr_config,
                value_estimator=value_estimator,
            )

    def solve(
        self,
        pbs: PublicBeliefState,
        battle: Battle,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """サブゲームを解く"""
        return self.solver.solve(pbs, battle)

    def get_action(
        self,
        pbs: PublicBeliefState,
        battle: Battle,
        explore: bool = False,
        temperature: float = 1.0,
    ) -> int:
        """
        戦略に従って行動を選択

        Args:
            pbs: Public Belief State
            battle: Battle オブジェクト
            explore: 探索的に行動を選択（確率的）
            temperature: 探索時の温度パラメータ

        Returns:
            選択した行動
        """
        my_strategy, _ = self.solve(pbs, battle)

        if not my_strategy:
            actions = battle.available_commands(pbs.public_state.perspective)
            return actions[0] if actions else Battle.SKIP

        if explore:
            # 確率的に選択
            actions = list(my_strategy.keys())
            probs = list(my_strategy.values())

            # 温度適用
            if temperature != 1.0:
                probs = [p ** (1 / temperature) for p in probs]
                total = sum(probs)
                probs = [p / total for p in probs]

            return random.choices(actions, weights=probs, k=1)[0]
        else:
            # 最大確率の行動
            return max(my_strategy.items(), key=lambda x: x[1])[0]
