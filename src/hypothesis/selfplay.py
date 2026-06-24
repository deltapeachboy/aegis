"""
Self-Play データ生成モジュール (完全型推測適合版)

仮説ベースMCTS同士を対戦させ、各ターンの盤面・Policy・Value、および
完全な信念状態（持ち物・技・テラス・特性の確率分布）を記録する。
生成されたデータはValue Networkの学習に使用される。
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
import dataclasses  # 🌟 名前衝突を回避するため、モジュールごとインポート
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# 🌟 pokepy インポート
from pokepy.battle import Battle
from pokepy.battle import Pokemon

from .hypothesis_mcts import HypothesisMCTS, PolicyValue, _calculate_battle_score

# 🌟 最新の信念状態・統計データベース（絶対パスインポート）
from src.rebel.belief_state import PokemonBeliefState
from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase


@dataclass
class PokemonState:
    """ポケモンの状態"""

    name: str
    hp: int
    max_hp: int
    hp_ratio: float
    ailment: str  # 状態異常（"", "どく", "もうどく", "やけど", "まひ", "ねむり", "こおり"）
    rank: list[int]  # ランク変化 [HP, A, B, C, D, S, 命中, 回避] (各-6〜+6)
    types: list[str]  # 現在のタイプ
    ability: str  # 特性
    item: str  # 持ち物（判明している場合）
    moves: list[str]  # 技（判明している場合）
    terastallized: bool  # テラスタル済みか
    tera_type: str  # テラスタイプ

    # 状態異常の詳細情報（オプション、後方互換性のためデフォルト設定）
    bad_poison_counter: int = 0  # もうどくカウンター（1から開始、毎ターン+1）
    sleep_counter: int = 0  # ねむりの残りターン数
    # PP情報（🌟 構文バグを dataclasses.field を使って修正完了）
    pp: list[int] = dataclasses.field(default_factory=list)  # 各技のPP残量（[pp1, pp2, pp3, pp4]）


@dataclass
class FieldCondition:
    """場の状態"""

    # 天候（残りターン数、0なら無し）
    sunny: int  # はれ
    rainy: int  # あめ
    snow: int  # ゆき
    sandstorm: int  # すなじらし

    # フィールド（残りターン数）
    electric_field: int  # エレキフィールド
    grass_field: int  # グラスフィールド
    psychic_field: int  # サイコフィールド
    mist_field: int  # ミストフィールド

    # その他の場の効果
    gravity: int  # じゅうりょく
    trick_room: int  # トリックルーム

    # プレイヤー別の場の効果 [player0, player1]
    reflector: list[int]  # リフレクター
    light_screen: list[int]  # ひかりのかべ
    tailwind: list[int]  # おいかぜ
    safeguard: list[int]  # しんぴのまもり
    mist: list[int]  # しろいきり

    # 設置技 [player0, player1]
    spikes: list[int]  # まきびし（段階数）
    toxic_spikes: list[int]  # どくびし（段階数）
    stealth_rock: list[int]  # ステルスロック（0 or 1）
    sticky_web: list[int]  # ねばねばネット（0 or 1）


@dataclass
class TurnRecord:
    """1ターンの記録"""

    turn: int
    player: int  # 行動したプレイヤー

    # 自分の場のポケモン詳細
    my_pokemon: PokemonState
    # 自分の控えポケモン
    my_bench: list[PokemonState]

    # 相手の場のポケモン詳細
    opp_pokemon: PokemonState
    # 相手の控えポケモン（観測情報のみ）
    opp_bench: list[PokemonState]

    # 場の状態（この変数名 'field' と関数名 'field' の衝突を解消）
    field: FieldCondition

    # 持ち物信念状態（既存学習パイプラインとの互換性のためにここに維持）
    item_beliefs: dict[str, dict[str, float]]

    # MCTSの出力
    policy: dict[str, float]  # {action_str: probability}
    value: float  # 勝率予測

    # 実際に選択した行動
    action: str  # 行動の文字列表現
    action_id: int  # コマンドID

    # 🌟 シンタックスエラーを解消するため、かつ名前衝突を防ぐために 'dataclasses.field' を使って最下部に定義します
    move_beliefs: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)  # 技信念
    tera_beliefs: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)  # テラス信念
    ability_beliefs: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)  # 特性信念


@dataclass
class GameRecord:
    """1試合の記録"""

    game_id: str
    player0_trainer: str
    player1_trainer: str
    player0_team: list[str]
    player1_team: list[str]
    winner: Optional[int]
    total_turns: int
    turns: list[TurnRecord] = dataclasses.field(default_factory=list)


def action_id_to_str(battle: Battle, player: int, action_id: int) -> str:
    """コマンドIDを人間が読める文字列に変換"""
    if action_id < 0:
        return "SKIP"
    elif action_id < 4:
        pokemon = battle.pokemon[player]
        if pokemon and action_id < len(pokemon.moves):
            return f"MOVE:{pokemon.moves[action_id]}"
        return f"MOVE:{action_id}"
    elif action_id >= 20 and action_id < 30:
        idx = action_id - 20
        if idx < len(battle.selected[player]):
            return f"SWITCH:{battle.selected[player][idx].name}"
        return f"SWITCH:{idx}"
    elif action_id == 30:
        return "STRUGGLE"
    else:
        return f"CMD:{action_id}"


def policy_to_str_dict(
    battle: Battle, player: int, policy: dict[int, float]
) -> dict[str, float]:
    """Policy辞書のキーを文字列に変換"""
    return {action_id_to_str(battle, player, k): v for k, v in policy.items()}


class SelfPlayGenerator:
    """
    Self-Playデータ生成器 (完全型推測適合版)
    """

    def __init__(
        self,
        usage_db: PokemonUsageDatabase,  # 🌟 統一されたデータベースクラスを使用
        n_hypotheses: int = 20,
        mcts_iterations: int = 150,
    ):
        self.usage_db = usage_db
        self.n_hypotheses = n_hypotheses
        self.mcts_iterations = mcts_iterations

        # 各プレイヤー用のMCTSエージェントを、最新の完全型推測DBで初期化
        self.mcts_agents = [
            HypothesisMCTS(usage_db, n_hypotheses, mcts_iterations),
            HypothesisMCTS(usage_db, n_hypotheses, mcts_iterations),
        ]

    def generate_game(
        self,
        trainer0_pokemons: list[dict],
        trainer1_pokemons: list[dict],
        trainer0_name: str = "Player0",
        trainer1_name: str = "Player1",
        game_id: str = "game_0",
        max_turns: int = 100,
        record_every_n_turns: int = 1,
    ) -> GameRecord:
        """
        1試合をシミュレートしてデータを生成
        """
        # バトル初期化
        battle = Battle(seed=random.randint(0, 2**31))
        battle.reset_game()

        # ポケモン設定
        for i, pokemons_data in enumerate([trainer0_pokemons, trainer1_pokemons]):
            for p_data in pokemons_data[:3]:
                p = Pokemon(p_data["name"])
                p.item = p_data.get("item", "")
                p.nature = p_data.get("nature", "まじめ")
                p.ability = p_data.get("ability", "")
                p.Ttype = p_data.get("Ttype", "")
                p.moves = p_data.get("moves", [])
                p.effort = p_data.get("effort", [0, 0, 0, 0, 0, 0])
                battle.selected[i].append(p)

        # 初期ポケモンを場に出す
        battle.pokemon = [battle.selected[0][0], battle.selected[1][0]]

        # 信念状態の初期化を最新の PokemonBeliefState にアップデート
        belief_states = [
            PokemonBeliefState(
                [p.name for p in battle.selected[1]], self.usage_db
            ),
            PokemonBeliefState(
                [p.name for p in battle.selected[0]], self.usage_db
            ),
        ]

        # 記録の初期化
        game_record = GameRecord(
            game_id=game_id,
            player0_trainer=trainer0_name,
            player1_trainer=trainer1_name,
            player0_team=[p.name for p in battle.selected[0]],
            player1_team=[p.name for p in battle.selected[1]],
            winner=None,
            total_turns=0,
            turns=[],
        )

        turn = 0
        while battle.winner() is None and turn < max_turns:
            turn += 1

            commands = [Battle.SKIP, Battle.SKIP]
            policies = [{}, {}]
            values = [0.5, 0.5]

            for player in range(2):
                available = battle.available_commands(player)
                if not available:
                    continue

                # MCTSで探索（最新の信念状態をそのまま引き渡し）
                pv = self.mcts_agents[player].search(
                    battle, player, belief_states[player], phase="battle"
                )

                # MCTSの探索結果を展開
                if isinstance(pv, tuple):
                    policies[player] = pv[0]
                    values[player] = pv[1].get(0, 0.5) if hasattr(pv[1], 'get') else 0.5
                else:
                    policies[player] = pv.policy
                    values[player] = pv.value

                # 最も確率の高い行動を選択
                if policies[player]:
                    commands[player] = max(policies[player].items(), key=lambda x: x[1])[0]
                elif available:
                    commands[player] = random.choice(available)

            # 記録（指定ターンごと）
            if turn % record_every_n_turns == 0:
                for player in range(2):
                    if policies[player]:
                        turn_record = self._create_turn_record(
                            battle,
                            player,
                            turn,
                            policies[player],
                            values[player],
                            commands[player],
                            belief_states[player],
                        )
                        game_record.turns.append(turn_record)

            # ターン進行
            battle.proceed(commands=commands)

        # 試合結果
        game_record.winner = battle.winner()
        game_record.total_turns = turn

        # 最終結果で各ターンのValueを補正
        if game_record.winner is not None:
            self._adjust_values_by_outcome(game_record)

        return game_record

    def _create_pokemon_state(
        self, pokemon: Pokemon, is_opponent: bool = False
    ) -> PokemonState:
        """ポケモンの状態を記録"""
        if pokemon is None:
            return PokemonState(
                name="", hp=0, max_hp=0, hp_ratio=0.0, ailment="",
                rank=[0] * 8, types=[], ability="", item="", moves=[],
                terastallized=False, tera_type="",
            )

        max_hp = pokemon.status[0] if pokemon.status[0] > 0 else 1

        return PokemonState(
            name=pokemon.name,
            hp=pokemon.hp,
            max_hp=max_hp,
            hp_ratio=pokemon.hp / max_hp,
            ailment=pokemon.ailment if hasattr(pokemon, "ailment") else "",
            rank=list(pokemon.rank) if hasattr(pokemon, "rank") else [0] * 8,
            types=list(pokemon.types) if hasattr(pokemon, "types") else [],
            ability=pokemon.ability if hasattr(pokemon, "ability") else "",
            item=pokemon.item if hasattr(pokemon, "item") else "",
            moves=list(pokemon.moves) if hasattr(pokemon, "moves") else [],
            terastallized=pokemon.terastal if hasattr(pokemon, "terastal") else False,
            tera_type=pokemon.Ttype if hasattr(pokemon, "Ttype") else "",
        )

    def _create_field_condition(self, battle: Battle) -> FieldCondition:
        """場の状態を記録"""
        cond = battle.condition

        return FieldCondition(
            sunny=cond.get("sunny", 0),
            rainy=cond.get("rainy", 0),
            snow=cond.get("snow", 0),
            sandstorm=cond.get("sandstorm", 0),
            electric_field=cond.get("elecfield", 0),
            grass_field=cond.get("glassfield", 0),
            psychic_field=cond.get("psycofield", 0),
            mist_field=cond.get("mistfield", 0),
            gravity=cond.get("gravity", 0),
            trick_room=cond.get("trickroom", 0),
            reflector=list(cond.get("reflector", [0, 0])),
            light_screen=list(cond.get("lightwall", [0, 0])),
            tailwind=list(cond.get("oikaze", [0, 0])),
            safeguard=list(cond.get("safeguard", [0, 0])),
            mist=list(cond.get("whitemist", [0, 0])),
            spikes=list(cond.get("makibishi", [0, 0])),
            toxic_spikes=list(cond.get("dokubishi", [0, 0])),
            stealth_rock=list(cond.get("stealthrock", [0, 0])),
            sticky_web=list(cond.get("nebanet", [0, 0])),
        )

    def _create_turn_record(
        self,
        battle: Battle,
        player: int,
        turn: int,
        policy: dict[int, float],
        value: float,
        action_id: int,
        belief_state: PokemonBeliefState,
    ) -> TurnRecord:
        """ターン記録を作成"""
        opp = 1 - player

        my_pokemon_state = self._create_pokemon_state(battle.pokemon[player])

        my_bench = []
        for p in battle.selected[player]:
            if p != battle.pokemon[player]:
                my_bench.append(self._create_pokemon_state(p))

        opp_pokemon_state = self._create_pokemon_state(
            battle.pokemon[opp], is_opponent=True
        )

        opp_bench = []
        for p in battle.selected[opp]:
            if p != battle.pokemon[opp]:
                opp_bench.append(self._create_pokemon_state(p, is_opponent=True))

        field_condition = self._create_field_condition(battle)

        # 🌟 最新の信念状態から、持ち物、技、テラス、特性の周辺分布を全次元シリアライズ
        opp_team = [p.name for p in battle.selected[opp]]
        item_beliefs = {}
        move_beliefs = {}
        tera_beliefs = {}
        ability_beliefs = {}

        for name in opp_team:
            # 1. 持ち物分布
            item_beliefs[name] = belief_state.get_item_distribution(name)

            # 2. 技分布（各ワールド仮説の確率の総和から算出）
            hypo_belief = belief_state.beliefs.get(name, {})
            move_probs = {}
            for hypo, prob in hypo_belief.items():
                for m in hypo.moves:
                    move_probs[m] = move_probs.get(m, 0.0) + prob
            move_beliefs[name] = move_probs

            # 3. テラスタイプ分布
            tera_beliefs[name] = belief_state.get_tera_distribution(name)

            # 4. 特性分布
            ability_beliefs[name] = belief_state.get_ability_distribution(name)

        return TurnRecord(
            turn=turn,
            player=player,
            my_pokemon=my_pokemon_state,
            my_bench=my_bench,
            opp_pokemon=opp_pokemon_state,
            opp_bench=opp_bench,
            field=field_condition,
            item_beliefs=item_beliefs,
            policy=policy_to_str_dict(battle, player, policy),
            value=value,
            action=action_id_to_str(battle, player, action_id),
            action_id=action_id,
            move_beliefs=move_beliefs,  # 🌟 シンタックス制約により最下部で引き渡し
            tera_beliefs=tera_beliefs,  # 🌟 シンタックス制約により最下部で引き渡し
            ability_beliefs=ability_beliefs,  # 🌟 シンタックス制約により最下部で引き渡し
        )

    def _adjust_values_by_outcome(self, game_record: GameRecord) -> None:
        """
        試合結果に基づいてValueを補正
        """
        if game_record.winner is None:
            return

        alpha = 0.7
        for turn_record in game_record.turns:
            outcome = 1.0 if turn_record.player == game_record.winner else 0.0
            turn_record.value = alpha * turn_record.value + (1 - alpha) * outcome


def save_records_to_jsonl(records: list[GameRecord], output_path: str | Path) -> None:
    """GameRecordをJSONL形式で保存"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            record_dict = asdict(record)
            f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")


def _dict_to_turn_record(data: dict) -> TurnRecord:
    """辞書からTurnRecordを復元（拡張された信念構造を安全に復元）"""
    return TurnRecord(
        turn=data["turn"],
        player=data["player"],
        my_pokemon=PokemonState(**data["my_pokemon"]),
        my_bench=[PokemonState(**p) for p in data["my_bench"]],
        opp_pokemon=PokemonState(**data["opp_pokemon"]),
        opp_bench=[PokemonState(**p) for p in data["opp_bench"]],
        field=FieldCondition(**data["field"]),
        item_beliefs=data["item_beliefs"],
        policy=data["policy"],
        value=data["value"],
        action=data["action"],
        action_id=data["action_id"],
        move_beliefs=data.get("move_beliefs", {}),       # 🌟 復元対応
        tera_beliefs=data.get("tera_beliefs", {}),       # 🌟 復元対応
        ability_beliefs=data.get("ability_beliefs", {}), # 🌟 復元対応
    )


def load_records_from_jsonl(input_path: str | Path) -> list[GameRecord]:
    """JSONL形式からGameRecordを読み込み"""
    input_path = Path(input_path)
    records = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            turns = [_dict_to_turn_record(t) for t in data.pop("turns")]
            record = GameRecord(**data, turns=turns)
            records.append(record)

    return records