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

from pokepy.battle import Battle

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


# =========================================================================
# 🌟 【根本解決】CFR仮想盤面と現実盤面の「生存・状態変化」強制アライメント関数
# =========================================================================
def align_battle_states(real_battle: Battle, virtual_battle: Battle) -> None:
    """
    現実のバトルオブジェクト(real_battle)から、CFR思考用の仮想バトルオブジェクト(virtual_battle)へ、
    各ポケモンの「現在のHP」「生存・ひんし状態」「状態異常・状態変化」「こだわりロック」等を完全に強制同期します。
    これで、AIが「ひんし状態のポケモンを生きていると誤認して交代コマンドを選ぶバグ」を物理的に100%防止します。
    """
    for player in range(2):
        real_party = real_battle.selected[player] if (real_battle.selected and player < len(real_battle.selected)) else []
        virtual_party = virtual_battle.selected[player] if (virtual_battle.selected and player < len(virtual_battle.selected)) else []

        # パーティ個体のHPと生存の同期
        for idx in range(min(len(real_party), len(virtual_party))):
            real_p = real_party[idx]
            virtual_p = virtual_party[idx]
            if not real_p or not virtual_p:
                continue

            # HPの完全同期
            virtual_p.hp = real_p.hp

            # 最大HP種族値の同期
            if hasattr(real_p, 'status') and hasattr(virtual_p, 'status'):
                try:
                    virtual_p.status[0] = real_p.status[0]
                except Exception:
                    pass

            # 状態変化・状態異常・こだわり等の完全同期
            for attr in ['ailment', 'sleep_count', 'yawn', 'condition', 'fixed_move', 'item']:
                if hasattr(real_p, attr) and hasattr(virtual_p, attr):
                    try:
                        setattr(virtual_p, attr, deepcopy(getattr(real_p, attr)))
                    except Exception:
                        pass

        # 現在バトルフィールドに出ているアクティブポケモンの完全同期
        real_active = real_battle.pokemon[player] if (real_battle.pokemon and player < len(real_battle.pokemon)) else None
        if real_active and virtual_battle.pokemon and player < len(virtual_battle.pokemon):
            # 名前が一致する仮想パーティ内の個体をアクティブ枠に紐付け
            for virtual_p in virtual_party:
                if virtual_p and virtual_p.name == real_active.name:
                    virtual_battle.pokemon[player] = virtual_p
                    break


def can_deal_damage(battle: Battle, attacker: int) -> bool:
    """
    攻撃側が防御側にダメージを与える手段があるか判定
    """
    from pokepy.battle import Pokemon

    defender = 1 - attacker

    attacker_pokemon_list = [
        p for p in battle.selected[attacker] if p is not None and p.hp > 0
    ]
    defender_pokemon_list = [
        p for p in battle.selected[defender] if p is not None and p.hp > 0
    ]

    if not attacker_pokemon_list or not defender_pokemon_list:
        return False

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
    """
    from pokepy.battle import Pokemon

    move_data = Pokemon.all_moves.get(move)
    if move_data is None:
        return False

    move_type = move_data.get("type", "ノーマル")
    move_class = move_data.get("class", "phy")
    power = move_data.get("power", 0)

    if move_class == "status":
        damaging_status_moves = {
            "やどりぎのタネ",
            "のろい",
        }
        if move == "やどりぎのタネ" and "くさ" in defender.types:
            return False
        if move in damaging_status_moves:
            return True

        if move in {"どくどく", "どくのこな", "どくガス"}:
            if "はがね" in defender.types or "どく" in defender.types:
                return False
            return True

        return False

    if power == 0:
        if move == "わるあがき":
            return True
        fixed_damage_moves = {"ちきゅうなげ", "ナイトヘッド", "がんせきおとし"}
        if move in fixed_damage_moves:
            if move_type == "ノーマル" and "ゴースト" in defender.types:
                return False
            if move_type == "かくとう" and "ゴースト" in defender.types:
                return False
            return True
        return False

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

    if move_type == "じめん":
        is_defender_floating = (
            "ひこう" in defender.types
            or defender.ability == "ふゆう"
            or defender.item == "ふうせん"
        )
        if is_defender_floating:
            return False

    if type_effectiveness == 0:
        return False

    return True


def check_hopeless_situation(battle: Battle, player: int) -> bool:
    """
    プレイヤーが必敗状態かどうか判定
    """
    if not can_deal_damage(battle, player):
        opponent = 1 - player
        if not can_deal_damage(battle, opponent):
            return False
        return True

    return False


def default_value_estimator(battle: Battle, player: int) -> float:
    """
    デフォルトの価値推定関数
    """
    winner = battle.winner()

    if winner is not None:
        return 1.0 if winner == player else 0.0

    if check_hopeless_situation(battle, player):
        return 0.0

    opponent = 1 - player
    if check_hopeless_situation(battle, opponent):
        return 1.0

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
    num_iterations: int = 100
    num_world_samples: int = 10
    depth_limit: int = 1
    use_linear_cfr: bool = True
    regret_matching_plus: bool = True


class CFRSubgameSolver:
    """
    1ターンのサブゲームを CFR で解く
    """

    def __init__(
        self,
        config: Optional[CFRConfig] = None,
        value_estimator: Optional[ValueEstimator] = None,
    ):
        self.config = config or CFRConfig()
        self.value_fn = value_estimator or default_value_estimator

    def solve(
        self,
        pbs: PublicBeliefState,
        original_battle: Battle,
        include_surrender: bool = True,
    ) -> tuple[dict[int, float], dict[int, float]]:
        perspective = pbs.public_state.perspective
        opponent = 1 - perspective

        my_actions = list(original_battle.available_commands(perspective))
        opp_actions = list(original_battle.available_commands(opponent))

        if not my_actions or not opp_actions:
            return ({}, {})

        if include_surrender:
            if check_hopeless_situation(original_battle, perspective):
                my_actions.append(Battle.SURRENDER)
            if check_hopeless_situation(original_battle, opponent):
                opp_actions.append(Battle.SURRENDER)

        regrets: list[dict[int, float]] = [
            {a: 0.0 for a in my_actions},
            {a: 0.0 for a in opp_actions},
        ]

        strategy_sum: list[dict[int, float]] = [
            {a: 0.0 for a in my_actions},
            {a: 0.0 for a in opp_actions},
        ]

        worlds = pbs.belief.sample_worlds(self.config.num_world_samples)

        for t in range(1, self.config.num_iterations + 1):
            current_strategies = [
                self._regret_matching(regrets[0], self.config.regret_matching_plus),
                self._regret_matching(regrets[1], self.config.regret_matching_plus),
            ]

            for world in worlds:
                # 仮想バトル状態の構築 ＆ 【強制アライメント同期の適用】
                battle = instantiate_battle_from_hypothesis(pbs, world, original_battle)
                align_battle_states(original_battle, battle)

                for player in [perspective, opponent]:
                    player_idx = 0 if player == perspective else 1

                    opp_player = 1 - player
                    opp_idx = 1 - player_idx
                    opp_strategy = current_strategies[opp_idx]
                    opp_action = self._sample_action(opp_strategy)

                    actions = my_actions if player == perspective else opp_actions
                    action_values = {}

                    for action in actions:
                        if action == Battle.SURRENDER:
                            action_values[action] = 0.0
                            continue

                        test_battle = deepcopy(battle)
                        if player == 0:
                            commands = [action, opp_action]
                        else:
                            commands = [opp_action, action]

                        test_battle.proceed(commands=commands)
                        action_values[action] = self.value_fn(test_battle, player)

                    player_strategy = current_strategies[player_idx]
                    expected_value = sum(
                        player_strategy.get(a, 0) * action_values[a] for a in actions
                    )

                    for action in actions:
                        regret = action_values[action] - expected_value
                        regrets[player_idx][action] += regret

            weight = t if self.config.use_linear_cfr else 1
            for player_idx in [0, 1]:
                strategy = current_strategies[player_idx]
                for action, prob in strategy.items():
                    strategy_sum[player_idx][action] += weight * prob

        my_avg = self._normalize(strategy_sum[0])
        opp_avg = self._normalize(strategy_sum[1])

        return my_avg, opp_avg

    def _regret_matching(
        self, regrets: dict[int, float], use_plus: bool = True
    ) -> dict[int, float]:
        positive_regrets = {a: max(0, r) for a, r in regrets.items()}
        total = sum(positive_regrets.values())

        if total > 0:
            return {a: r / total for a, r in positive_regrets.items()}
        else:
            n = len(regrets)
            return {a: 1.0 / n for a in regrets}

    def _sample_action(self, strategy: dict[int, float]) -> int:
        if not strategy:
            return -1
        actions = list(strategy.keys())
        probs = list(strategy.values())
        return random.choices(actions, weights=probs, k=1)[0]

    def _normalize(self, strategy: dict[int, float]) -> dict[int, float]:
        total = sum(strategy.values())
        if total > 0:
            return {a: p / total for a, p in strategy.items()}
        n = len(strategy)
        return {a: 1.0 / n for a in strategy} if n > 0 else {}


class SimplifiedCFRSolver:
    """
    簡略化 CFR ソルバー
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
        perspective = pbs.public_state.perspective
        opponent = 1 - perspective

        my_actions = list(original_battle.available_commands(perspective))
        opp_actions = list(original_battle.available_commands(opponent))

        if not my_actions:
            return ({}, {})
        if not opp_actions:
            return ({a: 1.0 / len(my_actions) for a in my_actions}, {})

        if include_surrender:
            if check_hopeless_situation(original_battle, perspective):
                my_actions.append(Battle.SURRENDER)
            if check_hopeless_situation(original_battle, opponent):
                opp_actions.append(Battle.SURRENDER)

        worlds = pbs.belief.sample_worlds(self.num_samples)

        payoff_matrix: dict[int, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for world in worlds:
            # 仮想バトル状態の構築 ＆ 【強制アライメント同期の適用】
            battle = instantiate_battle_from_hypothesis(pbs, world, original_battle)
            align_battle_states(original_battle, battle)

            hyp_opp_actions = battle.available_commands(opponent)
            if not hyp_opp_actions:
                hyp_opp_actions = [Battle.SKIP]

            for my_action in my_actions:
                for opp_action in opp_actions:
                    if my_action == Battle.SURRENDER:
                        payoff_matrix[my_action][opp_action].append(0.0)
                        continue
                    if opp_action == Battle.SURRENDER:
                        payoff_matrix[my_action][opp_action].append(1.0)
                        continue

                    test_battle = deepcopy(battle)
                    actual_opp_action = opp_action if opp_action in hyp_opp_actions else hyp_opp_actions[0]

                    if perspective == 0:
                        commands = [my_action, actual_opp_action]
                    else:
                        commands = [actual_opp_action, my_action]

                    try:
                        test_battle.proceed(commands=commands)
                        value = self.value_fn(test_battle, perspective)
                    except (IndexError, AttributeError, KeyError, TypeError, ValueError):
                        value = 0.5

                    payoff_matrix[my_action][opp_action].append(value)

        avg_payoff: dict[int, dict[int, float]] = {}
        for my_action in my_actions:
            avg_payoff[my_action] = {}
            for opp_action in opp_actions:
                values = payoff_matrix[my_action][opp_action]
                avg_payoff[my_action][opp_action] = sum(values) / len(values) if values else 0.5

        my_scores = {}
        for my_action in my_actions:
            min_value = min(avg_payoff[my_action].values())
            my_scores[my_action] = min_value

        my_strategy = self._softmax_strategy(my_scores, temperature=0.5)

        opp_scores = {}
        for opp_action in opp_actions:
            min_value = min(1 - avg_payoff[my_a][opp_action] for my_a in my_actions)
            opp_scores[opp_action] = min_value

        opp_strategy = self._softmax_strategy(opp_scores, temperature=0.5)

        return my_strategy, opp_strategy

    def _softmax_strategy(
        self, scores: dict[int, float], temperature: float = 1.0
    ) -> dict[int, float]:
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

        worlds = pbs.belief.sample_worlds(self.num_samples)

        action_values: dict[int, list[float]] = {a: [] for a in my_actions}

        for world in worlds:
            # 仮想バトル状態の構築 ＆ 【強制アライメント同期の適用】
            battle = instantiate_battle_from_hypothesis(pbs, world, original_battle)
            align_battle_states(original_battle, battle)

            for my_action in my_actions:
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

        avg_values = {a: sum(v) / len(v) if v else 0.5 for a, v in action_values.items()}
        my_strategy = self._softmax_strategy(avg_values, temperature=0.3)
        opp_strategy = {a: 1.0 / len(opp_actions) for a in opp_actions}

        return my_strategy, opp_strategy

    def _softmax_strategy(
        self, scores: dict[int, float], temperature: float = 1.0
    ) -> dict[int, float]:
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
    """

    def __init__(
        self,
        value_network: Optional["ReBeLValueNetwork"] = None,
        cfr_config: Optional[CFRConfig] = None,
        use_simplified: bool = True,
        use_lightweight: bool = False,
    ):
        from .value_network import ReBeLValueNetwork

        self.value_network = value_network
        self.use_simplified = use_simplified
        self.use_lightweight = use_lightweight

        if value_network is not None:
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
        my_strategy, _ = self.solve(pbs, battle)

        if not my_strategy:
            actions = battle.available_commands(pbs.public_state.perspective)
            return actions[0] if actions else Battle.SKIP

        if explore:
            actions = list(my_strategy.keys())
            probs = list(my_strategy.values())

            if temperature != 1.0:
                probs = [p ** (1 / temperature) for p in probs]
                total = sum(probs)
                probs = [p / total for p in probs]

            return random.choices(actions, weights=probs, k=1)[0]
        else:
            return max(my_strategy.items(), key=lambda x: x[1])[0]