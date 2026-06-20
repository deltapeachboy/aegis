import math
import random

from src.pokemon_battle_sim.battle import Battle


# --- MCTS のノードクラス ---
class MCTSNode:
    def __init__(self, state, parent=None, move=None, player=0):
        """
        state: 現在の対戦状態（Battle インスタンス）
        parent: 親ノード
        move: このノードに到達するために適用された手（コマンド）
        player: このノードで行動するプレイヤー（0 または 1）
        """
        self.state = state
        self.parent = parent
        self.move = move
        self.player = player
        self.children = []
        self.visits = 0
        self.total_score = 0

    def is_fully_expanded(self):
        # 対戦終了しているなら展開不要
        if self.state.winner() is not None:
            return True
        possible_moves = self.state.available_commands(self.player)
        return len(self.children) == len(possible_moves)

    def uct_value(self, exploration=math.sqrt(2)):
        if self.visits == 0:
            return float("inf")
        return (self.total_score / self.visits) + exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, exploration=math.sqrt(2)):
        return max(self.children, key=lambda child: child.uct_value(exploration))


class MCTSNodeForChangeCommand:
    def __init__(self, state, parent=None, move=None, player=0):
        """
        state: 現在の対戦状態（Battle インスタンス）
        parent: 親ノード
        move: このノードに到達するために適用された手（コマンド）
        player: このノードで行動するプレイヤー（0 または 1）
        """
        self.state = state
        self.parent = parent
        self.move = move
        self.player = player
        self.children = []
        self.visits = 0
        self.total_score = 0

    def is_fully_expanded(self):
        # 対戦終了しているなら展開不要
        if self.state.winner() is not None:
            return True
        possible_moves = self.state.available_commands(self.player, phase="change")
        return len(self.children) == len(possible_moves)

    def uct_value(self, exploration=math.sqrt(2)):
        if self.visits == 0:
            return float("inf")
        return (self.total_score / self.visits) + exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, exploration=math.sqrt(2)):
        return max(self.children, key=lambda child: child.uct_value(exploration))


# --- MCTS の探索・展開関数 ---
def tree_policy(node, player):
    """
    ノードが完全展開されていなければ展開、そうでなければUCT値が高い子ノードを選択する。
    """
    while node.state.winner() is None:
        if not node.is_fully_expanded():
            return expand(node, player)
        else:
            node = node.best_child()
    return node


def tree_policy_for_change_command(node, player):
    """
    ノードが完全展開されていなければ展開、そうでなければUCT値が高い子ノードを選択する。
    """
    while node.state.winner() is None:
        if not node.is_fully_expanded():
            return expand_for_change_command(node, player)
        else:
            node = node.best_child()
    return node


def expand(node, player):
    """
    未展開の候補手の中から1つ選び、その手を適用した新たなノードを作成する。
    """
    tried_moves = [child.move for child in node.children]
    available_moves = node.state.available_commands(player)
    for move in available_moves:
        if move not in tried_moves:
            # clone() を使って状態を複製
            new_state = node.state.clone(player)
            # 相手の手はランダム選択
            opp = not player
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
    return node  # 通常はここに到達しない


def expand_for_change_command(node, player):
    """
    未展開の候補手の中から1つ選び、その手を適用した新たなノードを作成する。
    """
    tried_moves = [child.move for child in node.children]
    available_moves = node.state.available_commands(player, phase="change")
    for move in available_moves:
        if move not in tried_moves:
            # clone() を使って状態を複製
            new_state = node.state.clone(player)
            # 相手の手はランダム選択
            opp = not player
            opp_moves = new_state.available_commands(opp, phase="change")
            opp_move = random.choice(opp_moves) if opp_moves else Battle.SKIP
            if player == 0:
                commands = [move, opp_move]
            else:
                commands = [opp_move, move]
            new_state.proceed(commands=commands)
            child_node = MCTSNodeForChangeCommand(
                state=new_state, parent=node, move=move, player=player
            )
            node.children.append(child_node)
            return child_node
    return node  # 通常はここに到達しない


def default_policy(state, player):
    """
    終局までランダムプレイアウトを実施し、最終状態の評価値を返す。
    """
    simulation_state = state.clone(player)
    while simulation_state.winner() is None:
        moves0 = simulation_state.available_commands(0)
        moves1 = simulation_state.available_commands(1)
        cmd0 = random.choice(moves0) if moves0 else Battle.SKIP
        cmd1 = random.choice(moves1) if moves1 else Battle.SKIP
        simulation_state.proceed(commands=[cmd0, cmd1])
    return simulation_state.score(player)


def default_policy_for_change_command(state, player):
    """
    終局までランダムプレイアウトを実施し、最終状態の評価値を返す。
    """
    simulation_state = state.clone(player)
    while simulation_state.winner() is None:
        moves0 = simulation_state.available_commands(0, phase="change")
        moves1 = simulation_state.available_commands(1, phase="change")
        cmd0 = random.choice(moves0) if moves0 else Battle.SKIP
        cmd1 = random.choice(moves1) if moves1 else Battle.SKIP
        simulation_state.proceed(commands=[cmd0, cmd1])
    return simulation_state.score(player)


def backup(node, reward):
    """
    シミュレーション結果（評価値）を逆伝播して、各ノードの訪問回数と合計評価値を更新する。
    """
    while node is not None:
        node.visits += 1
        node.total_score += reward
        node = node.parent


def mcts(root_state, player, iterations=1000):
    """
    MCTS を指定回数実行し、最も訪問回数の多い子ノードの手を最善手とする。
    """
    root = MCTSNode(state=root_state.clone(player), player=player)
    for _ in range(iterations):
        leaf = tree_policy(root, player)
        reward = default_policy(leaf.state, player)
        backup(leaf, reward)
    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.move, root


def mcts_for_change_command(root_state, player, iterations=1000):
    """
    MCTS を指定回数実行し、最も訪問回数の多い子ノードの手を最善手とする。
    """
    root = MCTSNodeForChangeCommand(state=root_state.clone(player), player=player)
    for _ in range(iterations):
        leaf = tree_policy_for_change_command(root, player)
        reward = default_policy_for_change_command(leaf.state, player)
        backup(leaf, reward)
    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.move, root


# --- MCTS を利用した Battle クラスの拡張 ---
class MyMCTSBattle(Battle):
    def __init__(self):
        super().__init__()
        self.mcts_iterations = 1000  # MCTS の探索回数（必要に応じて調整）

    def battle_command(self, player: int) -> int:
        """
        MCTS を用いて各候補手を評価し、最も有望な手を選択する。
        """
        best_move, _ = mcts(self, player, iterations=self.mcts_iterations)
        return best_move

    def change_command(self, player: int) -> int:
        """{player}の任意交代時に呼ばれる方策関数"""

        # import pdb; pdb.set_trace()

        # 選択可能なコマンドの一覧を取得
        available_commands = self.available_commands(player, phase="change")

        # print('\t'+'-'*30 + ' 交代の方策関数 ' + '-'*30)
        # print('\tここまでの展開')
        # for pl in self.action_order:
        #     print(f'\t\tPlayer {pl} {self.log[pl]}')

        scores = []

        # 自分のコマンドのループ
        for cmd in available_commands:
            # コマンドごとに仮想盤面を生成
            battle = self.clone(player)

            # コマンドを指定して、交代の直前から仮想盤面を再開し、ターンの終わりまで進める
            battle.proceed(change_commands=[cmd, None] if player == 0 else [None, cmd])

            # print(f'\tコマンド{cmd}を指定して仮想盤面を再開')
            # print(f'\t\tPlayer {player} {battle.log[pl]}')

            # 交代後、さらにターンを進めることも可能
            # battle.proceed()

            # 盤面の評価値を記録
            scores.append(battle.score(player))

        # print('\t'+'-'*76)

        # スコアが最も高いコマンドを返す
        return available_commands[scores.index(max(scores))]

    def score(self, player: int) -> float:
        """
        盤面評価関数の例。
        内部評価値 TOD_score を利用し、自分と相手の評価比を返す。
        """
        my_score = self.TOD_score(player)
        opp_score = self.TOD_score(not player)
        return (my_score + 1e-3) / (opp_score + 1e-3)
