"""
仮説ベースMCTS (完全型推測対応版)

相手の特性・持ち物・技構成・努力値が不明な状況で、総合的な型仮説（PokemonTypeHypothesis）
を信念状態からサンプリングして複数のMCTSを実行し、結果を集約して最適行動を決定する。
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Any

# 🌟 pokepy インポート
from pokepy.battle import Battle
from pokepy.battle import Pokemon

# 🌟 src.rebel から信念状態と仮説適用関数を絶対パスインポート
from src.rebel.belief_state import PokemonBeliefState, PokemonTypeHypothesis
from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase

# 🌟 MCTSNode 関連の絶対パスインポート
from src.mcts.mcts_battle import MCTSNode, MCTSNodeForChangeCommand, MyMCTSBattle


def _calculate_battle_score(battle: Battle, player: int) -> float:
    """
    勝敗ベース + 中間評価のハイブリッドスコア
    """
    winner = battle.winner()

    if winner is not None:
        # 勝敗確定: 勝ち=1.0, 負け=0.0
        return 1.0 if winner == player else 0.0

    # 未決着: HP比率と生存数で評価
    def calc_team_strength(p: int) -> float:
        alive_count = 0
        hp_ratio_sum = 0.0

        for p_obj in battle.selected[p]:
            if p_obj is not None and p_obj.hp > 0:
                alive_count += 1
                max_hp = p_obj.status[0] if p_obj.status[0] > 0 else 1
                hp_ratio_sum += p_obj.hp / max_hp

        return alive_count + 0.3 * hp_ratio_sum

    my_strength = calc_team_strength(player)
    opp_strength = calc_team_strength(1 - player)

    total = my_strength + opp_strength
    if total < 1e-6:
        return 0.5

    return my_strength / total


def _ensure_score_method(battle: Battle) -> Battle:
    """
    Battleインスタンスにscoreメソッドがなければ改善版スコア関数を追加
    """
    if not hasattr(battle, "score"):
        def score(player: int) -> float:
            return _calculate_battle_score(battle, player)

        battle.score = score  # type: ignore[attr-defined]
    return battle


# =============================================================================
# 改善版スコア関数を使うカスタムMCTS
# =============================================================================

def _hypothesis_default_policy(state: Battle, player: int) -> float:
    """
    改善版スコア関数を使うランダムプレイアウト
    """
    simulation_state = deepcopy(state)
    while simulation_state.winner() is None:
        moves0 = simulation_state.available_commands(0)
        moves1 = simulation_state.available_commands(1)
        cmd0 = random.choice(moves0) if moves0 else Battle.SKIP
        cmd1 = random.choice(moves1) if moves1 else Battle.SKIP
        simulation_state.proceed(commands=[cmd0, cmd1])
    return _calculate_battle_score(simulation_state, player)


def _hypothesis_tree_policy(node: MCTSNode, player: int) -> MCTSNode:
    """
    ノードが完全展開されていなければ展開、そうでなければUCT値が高い子ノードを選択
    """
    while node.state.winner() is None:
        if not node.is_fully_expanded():
            return _hypothesis_expand(node, player)
        else:
            node = node.best_child()
    return node


def _hypothesis_expand(node: MCTSNode, player: int) -> MCTSNode:
    """
    未展開の候補手の中から1つ選び、その手を適用した新たなノードを作成
    """
    tried_moves = [child.move for child in node.children]
    available_moves = node.state.available_commands(player)
    for move in available_moves:
        if move not in tried_moves:
            new_state = deepcopy(node.state)
            opp = 1 - player
            opp_moves = new_state.available_commands(opp)
            opp_move = random.choice(opp_moves) if opp_moves else Battle.SKIP
            if player == 0:
                commands = [move, opp_move]
            else:
                commands = [opp_move, move]
            new_state.proceed(commands=commands)
            child_node = MCTSNode(
                state=new_state, parent=node, move=move, player=player
            )
            node.children.append(child_node)
            return child_node
    return node


def _hypothesis_backup(node: MCTSNode, reward: float) -> None:
    """
    シミュレーション結果を逆伝播
    """
    while node is not None:
        node.visits += 1
        node.total_score += reward
        node = node.parent


def _hypothesis_mcts(
    root_state: Battle, player: int, iterations: int = 1000
) -> tuple[int, MCTSNode]:
    """
    改善版スコア関数を使うMCTS
    """
    root = MCTSNode(state=deepcopy(root_state), player=player)
    for _ in range(iterations):
        leaf = _hypothesis_tree_policy(root, player)
        reward = _hypothesis_default_policy(leaf.state, player)
        _hypothesis_backup(leaf, reward)

    if not root.children:
        available = root_state.available_commands(player)
        return available[0] if available else Battle.SKIP, root

    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.move, root


@dataclass
class PolicyValue:
    """Policy（行動確率分布）とValue（勝率）のペア"""

    policy: dict[int, float]  # {command: probability}
    value: float  # 勝率 [0, 1]

    def __repr__(self) -> str:
        top_actions = sorted(self.policy.items(), key=lambda x: -x[1])[:3]
        actions_str = ", ".join(f"{cmd}:{prob:.2f}" for cmd, prob in top_actions)
        return f"PolicyValue(value={self.value:.3f}, policy=[{actions_str}, ...])"


class HypothesisMCTS:
    """
    仮説ベースのMCTS (完全型推測対応版)

    相手の特性・持ち物・技構成・努力値について複数の仮説をサンプリングし、
    各仮説に対してMCTSを実行して結果を集約する。
    """

    def __init__(
        self,
        usage_db: PokemonUsageDatabase,
        n_hypotheses: int = 30,
        mcts_iterations: int = 200,
    ):
        """
        Args:
            usage_db: 環境使用率・事前確率データベース
            n_hypotheses: サンプリングする仮説（世界線）の数
            mcts_iterations: 各仮説でのMCTSイテレーション数
        """
        self.usage_db = usage_db
        self.n_hypotheses = n_hypotheses
        self.mcts_iterations = mcts_iterations

    def search(
        self,
        battle: Battle,
        player: int,
        belief_state: PokemonBeliefState,
        phase: str = "battle",
    ) -> PolicyValue:
        """
        仮説ベースMCTSを実行
        """
        available_commands = battle.available_commands(player, phase=phase)

        if not available_commands:
            return PolicyValue(policy={}, value=0.5)

        if len(available_commands) == 1:
            return PolicyValue(policy={available_commands[0]: 1.0}, value=0.5)

        # 🌟 最新の信念状態から総合的な型仮説をサンプリング
        hypotheses = belief_state.sample_worlds(self.n_hypotheses)

        all_results = []
        for world in hypotheses:
            # 具体的な Battle を構築（仮説の適用）
            hypo_battle = self._apply_hypothesis(battle, player, world)

            # MCTS実行（改善版スコア関数を使用）
            best_move, root = _hypothesis_mcts(
                hypo_battle, player, iterations=self.mcts_iterations
            )

            # 訪問回数を記録
            visit_counts = {child.move: child.visits for child in root.children}

            # 勝率を計算（訪問回数で重み付けした平均スコア）
            total_visits = sum(child.visits for child in root.children)
            if total_visits > 0:
                value = sum(
                    child.total_score / max(child.visits, 1) * child.visits
                    for child in root.children
                ) / total_visits
            else:
                value = 0.5

            all_results.append((visit_counts, value))

        # 結果を集約
        return self._aggregate_results(all_results, available_commands)

    def apply_hypothesis(self, battle: Battle, player: int, hypothesis: dict[str, PokemonTypeHypothesis]) -> Battle:
        # 1. 使用するメソッドの先頭で安全にインポート（循環インポート対策）
        from src.rebel.public_state import _apply_hypothesis_to_pokemon

        """
        仮説（持ち物の組み合わせ）をBattleに適用
        """
        hypo_battle = deepcopy(battle)
        opponent = 1 - player

        # 相手の選出ポケモンに持ち物などの型仮説を一括適用
        for pokemon in hypo_battle.selected[opponent]:
            # 2. ポケモンの名前をキーに、その個体向けの仮説を取得
            h = hypothesis.get(pokemon.name) or hypothesis.get(pokemon.display_name)

            if h is not None:
                # 3. ループ内で定義された 'pokemon' と、その仮説 'h' を適用する
                _apply_hypothesis_to_pokemon(pokemon, h)

        return hypo_battle

    def _aggregate_results(
        self,
        results: list[tuple[dict[int, int], float]],
        available_commands: list[int],
    ) -> PolicyValue:
        """
        複数の仮説からの結果を集約
        """
        total_visits: dict[int, int] = defaultdict(int)
        total_value = 0.0

        for visit_counts, value in results:
            for cmd, visits in visit_counts.items():
                total_visits[cmd] += visits
            total_value += value

        # Policyを正規化
        visit_sum = sum(total_visits.values())
        if visit_sum > 0:
            policy = {cmd: total_visits[cmd] / visit_sum for cmd in available_commands}
        else:
            n = len(available_commands)
            policy = {cmd: 1.0 / n for cmd in available_commands}

        avg_value = total_value / len(results) if results else 0.5

        return PolicyValue(policy=policy, value=avg_value)

    def get_best_action(
        self,
        battle: Battle,
        player: int,
        belief_state: PokemonBeliefState,
        phase: str = "battle",
    ) -> int:
        """
        最も推奨される行動を取得
        """
        pv = self.search(battle, player, belief_state, phase)
        if not pv.policy:
            return Battle.SKIP
        return max(pv.policy.items(), key=lambda x: x[1])[0]


class HypothesisMCTSBattle(Battle):
    """
    仮説ベースMCTSを使用するBattleクラス (完全型推測対応版)
    """

    def __init__(
        self,
        usage_db: PokemonUsageDatabase,
        n_hypotheses: int = 30,
        mcts_iterations: int = 200,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)  # type: ignore[arg-type]
        self.hypothesis_mcts = HypothesisMCTS(
            usage_db=usage_db,
            n_hypotheses=n_hypotheses,
            mcts_iterations=mcts_iterations,
        )
        self.belief_states: dict[int, PokemonBeliefState] = {}

    def set_belief_state(self, player: int, belief_state: PokemonBeliefState) -> None:
        self.belief_states[player] = belief_state

    def init_belief_state(self, player: int) -> PokemonBeliefState:
        opponent = 1 - player
        opponent_pokemon_names = [p.name for p in self.selected[opponent]]

        belief_state = PokemonBeliefState(
            opponent_pokemon_names=opponent_pokemon_names,
            usage_db=self.hypothesis_mcts.usage_db,
        )
        self.belief_states[player] = belief_state
        return belief_state

    def get_belief_state(self, player: int) -> Optional[PokemonBeliefState]:
        return self.belief_states.get(player)

    def battle_command(self, player: int) -> int:
        if player not in self.belief_states:
            self.init_belief_state(player)

        return self.hypothesis_mcts.get_best_action(
            battle=self,
            player=player,
            belief_state=self.belief_states[player],
            phase="battle",
        )

    def change_command(self, player: int) -> int:
        if player not in self.belief_states:
            self.init_belief_state(player)

        return self.hypothesis_mcts.get_best_action(
            battle=self,
            player=player,
            belief_state=self.belief_states[player],
            phase="change",
        )

    def get_policy_value(self, player: int, phase: str = "battle") -> PolicyValue:
        if player not in self.belief_states:
            self.init_belief_state(player)

        return self.hypothesis_mcts.search(
            battle=self,
            player=player,
            belief_state=self.belief_states[player],
            phase=phase,
        )