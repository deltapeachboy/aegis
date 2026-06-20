"""
Self-Play データ生成モジュール

仮説ベースMCTS同士を対戦させ、各ターンの盤面・Policy・Valueを記録する。
生成されたデータはPolicy-Value Networkの学習に使用される。
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon

from .hypothesis_mcts import HypothesisMCTS, PolicyValue, _calculate_battle_score
from .item_belief_state import ItemBeliefState
from .item_prior_database import ItemPriorDatabase


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

    # 状態異常の詳細情報（オプション、後方互換性のためデフォルト値を設定）
    bad_poison_counter: int = 0  # もうどくカウンター（1から開始、毎ターン+1）
    sleep_counter: int = 0  # ねむりの残りターン数
    # PP情報（オプション）
    pp: list[int] = field(default_factory=list)  # 各技のPP残量（[pp1, pp2, pp3, pp4]）


@dataclass
class FieldCondition:
    """場の状態"""

    # 天候（残りターン数、0なら無し）
    sunny: int  # はれ
    rainy: int  # あめ
    snow: int  # ゆき
    sandstorm: int  # すなあらし

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

    # 場の状態
    field: FieldCondition

    # 持ち物信念状態
    item_beliefs: dict[str, dict[str, float]]  # {pokemon_name: {item: prob}}

    # MCTSの出力
    policy: dict[str, float]  # {action_str: probability}
    value: float  # 勝率予測

    # 実際に選択した行動
    action: str  # 行動の文字列表現
    action_id: int  # コマンドID


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
    turns: list[TurnRecord] = field(default_factory=list)


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
    Self-Playデータ生成器

    仮説ベースMCTS同士を対戦させ、学習用データを生成する。
    """

    def __init__(
        self,
        prior_db: ItemPriorDatabase,
        n_hypotheses: int = 20,
        mcts_iterations: int = 150,
    ):
        self.prior_db = prior_db
        self.n_hypotheses = n_hypotheses
        self.mcts_iterations = mcts_iterations

        # 各プレイヤー用のMCTSエージェント
        self.mcts_agents = [
            HypothesisMCTS(prior_db, n_hypotheses, mcts_iterations),
            HypothesisMCTS(prior_db, n_hypotheses, mcts_iterations),
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

        Args:
            trainer0_pokemons: Player0のポケモンデータ（3体分）
            trainer1_pokemons: Player1のポケモンデータ（3体分）
            trainer0_name: Player0の名前
            trainer1_name: Player1の名前
            game_id: ゲームID
            max_turns: 最大ターン数
            record_every_n_turns: 何ターンごとに記録するか

        Returns:
            GameRecord: 試合の記録
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

        # 信念状態の初期化
        belief_states = [
            ItemBeliefState(
                [p.name for p in battle.selected[1]], self.prior_db
            ),
            ItemBeliefState(
                [p.name for p in battle.selected[0]], self.prior_db
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

            # 各プレイヤーの行動を決定
            commands = [Battle.SKIP, Battle.SKIP]
            policies = [{}, {}]
            values = [0.5, 0.5]

            for player in range(2):
                available = battle.available_commands(player)
                if not available:
                    continue

                # MCTSで探索
                pv = self.mcts_agents[player].search(
                    battle, player, belief_states[player], phase="battle"
                )
                policies[player] = pv.policy
                values[player] = pv.value

                # 最も確率の高い行動を選択
                if pv.policy:
                    commands[player] = max(pv.policy.items(), key=lambda x: x[1])[0]
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

        # 最終結果で各ターンのValueを補正（実際の勝敗を反映）
        if game_record.winner is not None:
            self._adjust_values_by_outcome(game_record)

        return game_record

    def _create_pokemon_state(
        self, pokemon: Pokemon, is_opponent: bool = False
    ) -> PokemonState:
        """ポケモンの状態を記録"""
        if pokemon is None:
            return PokemonState(
                name="",
                hp=0,
                max_hp=0,
                hp_ratio=0.0,
                ailment="",
                rank=[0] * 8,
                types=[],
                ability="",
                item="",
                moves=[],
                terastallized=False,
                tera_type="",
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
            # 天候
            sunny=cond.get("sunny", 0),
            rainy=cond.get("rainy", 0),
            snow=cond.get("snow", 0),
            sandstorm=cond.get("sandstorm", 0),
            # フィールド
            electric_field=cond.get("elecfield", 0),
            grass_field=cond.get("glassfield", 0),
            psychic_field=cond.get("psycofield", 0),
            mist_field=cond.get("mistfield", 0),
            # その他
            gravity=cond.get("gravity", 0),
            trick_room=cond.get("trickroom", 0),
            # プレイヤー別効果
            reflector=list(cond.get("reflector", [0, 0])),
            light_screen=list(cond.get("lightwall", [0, 0])),
            tailwind=list(cond.get("oikaze", [0, 0])),
            safeguard=list(cond.get("safeguard", [0, 0])),
            mist=list(cond.get("whitemist", [0, 0])),
            # 設置技
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
        belief_state: ItemBeliefState,
    ) -> TurnRecord:
        """ターン記録を作成"""
        opp = 1 - player

        # 自分の場のポケモン
        my_pokemon_state = self._create_pokemon_state(battle.pokemon[player])

        # 自分の控え（場のポケモン以外）
        my_bench = []
        for p in battle.selected[player]:
            if p != battle.pokemon[player]:
                my_bench.append(self._create_pokemon_state(p))

        # 相手の場のポケモン
        opp_pokemon_state = self._create_pokemon_state(
            battle.pokemon[opp], is_opponent=True
        )

        # 相手の控え（場のポケモン以外）
        opp_bench = []
        for p in battle.selected[opp]:
            if p != battle.pokemon[opp]:
                opp_bench.append(self._create_pokemon_state(p, is_opponent=True))

        # 場の状態
        field_condition = self._create_field_condition(battle)

        # 信念状態をシリアライズ
        opp_team = [p.name for p in battle.selected[opp]]
        item_beliefs = {}
        for name in opp_team:
            item_beliefs[name] = belief_state.get_belief(name)

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
        )

    def _adjust_values_by_outcome(self, game_record: GameRecord) -> None:
        """
        試合結果に基づいてValueを補正

        MCTSのValueは予測値なので、実際の勝敗を反映して補正する。
        補正式: adjusted_value = alpha * mcts_value + (1 - alpha) * outcome
        """
        if game_record.winner is None:
            return

        alpha = 0.7  # MCTS予測の重み（0.7 = 70%は予測、30%は実際の結果）

        for turn_record in game_record.turns:
            # 実際の勝敗
            outcome = 1.0 if turn_record.player == game_record.winner else 0.0

            # 補正
            turn_record.value = alpha * turn_record.value + (1 - alpha) * outcome


def save_records_to_jsonl(records: list[GameRecord], output_path: str | Path) -> None:
    """GameRecordをJSONL形式で保存"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            # dataclassをdictに変換
            record_dict = asdict(record)
            f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")


def _dict_to_turn_record(data: dict) -> TurnRecord:
    """辞書からTurnRecordを復元（ネストしたdataclassも含む）"""
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
    )


def load_records_from_jsonl(input_path: str | Path) -> list[GameRecord]:
    """JSONL形式からGameRecordを読み込み"""
    input_path = Path(input_path)
    records = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # TurnRecordを復元（ネストしたdataclassも含む）
            turns = [_dict_to_turn_record(t) for t in data.pop("turns")]
            record = GameRecord(**data, turns=turns)
            records.append(record)

    return records
