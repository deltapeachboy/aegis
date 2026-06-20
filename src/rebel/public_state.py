"""
公開ゲーム状態 (Public Game State) の管理

ReBeL では、両プレイヤーが観測可能な「公開情報」と
各プレイヤーの「信念状態」を組み合わせた状態を扱う。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from src.hypothesis.selfplay import FieldCondition, PokemonState
from src.pokemon_battle_sim.battle import Battle

from .belief_state import PokemonBeliefState, PokemonTypeHypothesis

if TYPE_CHECKING:
    from src.pokemon_battle_sim.pokemon import Pokemon


def _extract_field_condition(battle: Battle) -> FieldCondition:
    """Battle から FieldCondition を抽出

    Note: Battle.condition の変数名と FieldCondition の変数名は異なる場合がある
    - elecfield -> electric_field
    - glassfield -> grass_field
    - psycofield -> psychic_field
    - mistfield -> mist_field
    - trickroom -> trick_room
    - lightwall -> light_screen
    - oikaze -> tailwind
    - whitemist -> mist
    - makibishi -> spikes
    - dokubishi -> toxic_spikes
    - stealthrock -> stealth_rock
    - nebanet -> sticky_web
    """
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
        # 壁
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


def _pokemon_to_state(pokemon: "Pokemon", include_hidden: bool = True) -> PokemonState:
    """Pokemon オブジェクトを PokemonState に変換"""
    max_hp = pokemon.status[0] if pokemon.status else 1
    hp_ratio = pokemon.hp / max_hp if max_hp > 0 else 0.0

    # 状態異常の詳細情報を抽出
    bad_poison_counter = 0
    sleep_counter = 0
    if hasattr(pokemon, "condition") and pokemon.condition:
        bad_poison_counter = pokemon.condition.get("badpoison", 0)
    if hasattr(pokemon, "sleep_count"):
        sleep_counter = pokemon.sleep_count

    # PP情報を抽出（include_hiddenがTrueの場合のみ）
    pp_list = []
    if include_hidden and hasattr(pokemon, "pp") and pokemon.pp:
        pp_list = list(pokemon.pp)

    return PokemonState(
        name=pokemon.name,
        hp=pokemon.hp,
        max_hp=max_hp,
        hp_ratio=hp_ratio,
        ailment=pokemon.ailment or "",
        rank=list(pokemon.rank) if pokemon.rank else [0] * 8,
        types=list(pokemon.types) if pokemon.types else [],
        ability=pokemon.ability if include_hidden else "",
        item=pokemon.item if include_hidden else "",
        moves=list(pokemon.moves) if include_hidden else [],
        terastallized=pokemon.terastal if hasattr(pokemon, "terastal") else False,
        tera_type=pokemon.Ttype if hasattr(pokemon, "Ttype") else "",
        bad_poison_counter=bad_poison_counter,
        sleep_counter=sleep_counter,
        pp=pp_list,
    )


@dataclass
class PublicPokemonState:
    """
    公開情報のみのポケモン状態

    相手のポケモンについて観測可能な情報のみを保持。
    """

    name: str
    hp_ratio: float  # HP比率（相手は正確なHPは見えないが比率は見える）
    ailment: str  # 状態異常（見える）
    rank: list[int]  # ランク変化（見える）
    types: list[str]  # 現在のタイプ（見える、テラスタル後は変化）
    terastallized: bool  # テラスタル済みか（見える）
    tera_type: str  # 使用後のテラスタイプ（使用後のみ見える）

    # 状態異常の詳細（観測可能）
    bad_poison_counter: int = 0  # もうどくカウンター（毎ターン増加）
    sleep_counter: int = 0  # ねむりの残りターン数

    # 観測により判明した情報
    revealed_moves: set[str] = field(default_factory=set)
    revealed_item: Optional[str] = None
    revealed_ability: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """シリアライズ可能な辞書に変換"""
        return {
            "name": self.name,
            "hp_ratio": self.hp_ratio,
            "ailment": self.ailment,
            "rank": self.rank,
            "types": self.types,
            "terastallized": self.terastallized,
            "tera_type": self.tera_type,
            "bad_poison_counter": self.bad_poison_counter,
            "sleep_counter": self.sleep_counter,
            "revealed_moves": list(self.revealed_moves),
            "revealed_item": self.revealed_item,
            "revealed_ability": self.revealed_ability,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PublicPokemonState":
        """辞書から復元"""
        return cls(
            name=data["name"],
            hp_ratio=data["hp_ratio"],
            ailment=data["ailment"],
            rank=data["rank"],
            types=data["types"],
            terastallized=data["terastallized"],
            tera_type=data["tera_type"],
            bad_poison_counter=data.get("bad_poison_counter", 0),
            sleep_counter=data.get("sleep_counter", 0),
            revealed_moves=set(data.get("revealed_moves", [])),
            revealed_item=data.get("revealed_item"),
            revealed_ability=data.get("revealed_ability"),
        )


@dataclass
class PublicGameState:
    """
    公開ゲーム状態

    両プレイヤーが観測可能な情報のみで構成される。
    自分の情報は完全、相手の情報は観測された部分のみ。
    """

    # 視点（どちらのプレイヤーから見た状態か）
    perspective: int  # 0 or 1

    # 自分の情報（完全）
    my_pokemon: PokemonState  # 場に出ているポケモン
    my_bench: list[PokemonState]  # 控えポケモン
    my_tera_available: bool  # テラスタル可能か

    # 相手の情報（観測された部分のみ）
    opp_pokemon: PublicPokemonState  # 場に出ているポケモン
    opp_bench: list[PublicPokemonState]  # 控えポケモン（HP比率のみ）
    opp_tera_available: bool  # 相手がテラスタル可能か

    # 場の状態（完全に観測可能）
    field: FieldCondition

    # ターン数
    turn: int

    @classmethod
    def from_battle(
        cls,
        battle: Battle,
        perspective: int,
        revealed_info: Optional[dict] = None,
    ) -> "PublicGameState":
        """
        Battle オブジェクトから PublicGameState を作成

        Args:
            battle: 現在の対戦状態
            perspective: 視点となるプレイヤー (0 or 1)
            revealed_info: 観測により判明した情報
                {
                    "moves": {pokemon_name: set[str]},
                    "items": {pokemon_name: str},
                    "abilities": {pokemon_name: str},
                }
        """
        revealed_info = revealed_info or {"moves": {}, "items": {}, "abilities": {}}
        opponent = 1 - perspective

        # 自分の情報（完全）
        my_pokemon = _pokemon_to_state(battle.pokemon[perspective], include_hidden=True)
        my_bench = [
            _pokemon_to_state(p, include_hidden=True)
            for p in battle.selected[perspective]
            if p != battle.pokemon[perspective] and p.hp > 0
        ]

        # テラスタル可能か（既に使用済みでないか）
        my_tera_available = not any(
            p.terastal if hasattr(p, "terastal") else False
            for p in battle.selected[perspective]
        )

        # 相手の情報（観測可能な部分のみ）
        opp_active = battle.pokemon[opponent]

        # 状態異常詳細を抽出
        opp_bad_poison = 0
        opp_sleep = 0
        if hasattr(opp_active, "condition") and opp_active.condition:
            opp_bad_poison = opp_active.condition.get("badpoison", 0)
        if hasattr(opp_active, "sleep_count"):
            opp_sleep = opp_active.sleep_count

        opp_pokemon = PublicPokemonState(
            name=opp_active.name,
            hp_ratio=opp_active.hp / opp_active.status[0] if opp_active.status[0] > 0 else 0,
            ailment=opp_active.ailment or "",
            rank=list(opp_active.rank) if opp_active.rank else [0] * 8,
            types=list(opp_active.types) if opp_active.types else [],
            terastallized=opp_active.terastal if hasattr(opp_active, "terastal") else False,
            tera_type=opp_active.Ttype if hasattr(opp_active, "Ttype") and opp_active.terastal else "",
            bad_poison_counter=opp_bad_poison,
            sleep_counter=opp_sleep,
            revealed_moves=set(revealed_info.get("moves", {}).get(opp_active.name, set())),
            revealed_item=revealed_info.get("items", {}).get(opp_active.name),
            revealed_ability=revealed_info.get("abilities", {}).get(opp_active.name),
        )

        opp_bench = []
        for p in battle.selected[opponent]:
            if p != opp_active and p.hp > 0:
                # 控えの状態異常詳細
                bench_bad_poison = 0
                bench_sleep = 0
                if hasattr(p, "condition") and p.condition:
                    bench_bad_poison = p.condition.get("badpoison", 0)
                if hasattr(p, "sleep_count"):
                    bench_sleep = p.sleep_count

                opp_bench.append(
                    PublicPokemonState(
                        name=p.name,
                        hp_ratio=p.hp / p.status[0] if p.status[0] > 0 else 0,
                        ailment=p.ailment or "",
                        rank=[0] * 8,  # 控えはランク変化がリセットされる
                        types=list(p.types) if p.types else [],
                        terastallized=p.terastal if hasattr(p, "terastal") else False,
                        tera_type=p.Ttype if hasattr(p, "Ttype") and p.terastal else "",
                        bad_poison_counter=bench_bad_poison,
                        sleep_counter=bench_sleep,
                        revealed_moves=set(revealed_info.get("moves", {}).get(p.name, set())),
                        revealed_item=revealed_info.get("items", {}).get(p.name),
                        revealed_ability=revealed_info.get("abilities", {}).get(p.name),
                    )
                )

        opp_tera_available = not any(
            p.terastal if hasattr(p, "terastal") else False
            for p in battle.selected[opponent]
        )

        return cls(
            perspective=perspective,
            my_pokemon=my_pokemon,
            my_bench=my_bench,
            my_tera_available=my_tera_available,
            opp_pokemon=opp_pokemon,
            opp_bench=opp_bench,
            opp_tera_available=opp_tera_available,
            field=_extract_field_condition(battle),
            turn=battle.turn if hasattr(battle, "turn") else 0,
        )

    def copy(self) -> "PublicGameState":
        """ディープコピー"""
        return deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        """シリアライズ可能な辞書に変換"""
        return {
            "perspective": self.perspective,
            "my_pokemon": asdict(self.my_pokemon),
            "my_bench": [asdict(p) for p in self.my_bench],
            "my_tera_available": self.my_tera_available,
            "opp_pokemon": self.opp_pokemon.to_dict(),
            "opp_bench": [p.to_dict() for p in self.opp_bench],
            "opp_tera_available": self.opp_tera_available,
            "field": asdict(self.field),
            "turn": self.turn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PublicGameState":
        """辞書から復元"""
        return cls(
            perspective=data["perspective"],
            my_pokemon=PokemonState(**data["my_pokemon"]),
            my_bench=[PokemonState(**p) for p in data["my_bench"]],
            my_tera_available=data["my_tera_available"],
            opp_pokemon=PublicPokemonState.from_dict(data["opp_pokemon"]),
            opp_bench=[PublicPokemonState.from_dict(p) for p in data["opp_bench"]],
            opp_tera_available=data["opp_tera_available"],
            field=FieldCondition(**data["field"]),
            turn=data["turn"],
        )


@dataclass
class PublicBeliefState:
    """
    Public Belief State (PBS)

    ReBeL の中核となる状態表現。
    公開状態と信念状態を組み合わせ、さらに両プレイヤーの戦略を含む。
    """

    # 公開ゲーム状態
    public_state: PublicGameState

    # 相手の型に対する信念
    belief: PokemonBeliefState

    # 両プレイヤーの現在の戦略（CFR で更新）
    # {action_id: probability}
    my_strategy: dict[int, float] = field(default_factory=dict)
    opp_strategy: dict[int, float] = field(default_factory=dict)

    # 両プレイヤーの期待値 (my, opp)
    # CFR で計算された値、または Value Network の予測
    values: tuple[float, float] = (0.5, 0.5)

    @classmethod
    def from_battle(
        cls,
        battle: Battle,
        perspective: int,
        belief: PokemonBeliefState,
    ) -> "PublicBeliefState":
        """Battle と信念状態から PBS を作成"""
        # 判明した情報を収集
        revealed_info = {
            "moves": belief.revealed_moves,
            "items": {k: v for k, v in belief.revealed_items.items() if v is not None},
            "abilities": {},  # 特性は現在未追跡
        }

        public_state = PublicGameState.from_battle(
            battle, perspective, revealed_info
        )

        return cls(
            public_state=public_state,
            belief=belief,
        )

    def get_available_actions(self, battle: Battle) -> list[int]:
        """
        利用可能なアクションを取得

        Args:
            battle: 現在の Battle 状態

        Returns:
            利用可能なコマンドIDのリスト
        """
        return battle.available_commands(self.public_state.perspective)

    def copy(self) -> "PublicBeliefState":
        """ディープコピー"""
        return PublicBeliefState(
            public_state=self.public_state.copy(),
            belief=self.belief.copy(),
            my_strategy=dict(self.my_strategy),
            opp_strategy=dict(self.opp_strategy),
            values=self.values,
        )

    def summary(self) -> str:
        """PBS の概要"""
        lines = ["PublicBeliefState Summary", "=" * 50]

        ps = self.public_state
        lines.append(f"\n視点: Player {ps.perspective}")
        lines.append(f"ターン: {ps.turn}")

        # 自分の状態
        lines.append(f"\n自分の場: {ps.my_pokemon.name} (HP: {ps.my_pokemon.hp_ratio:.1%})")
        bench_names = [p.name for p in ps.my_bench]
        lines.append(f"自分の控え: {', '.join(bench_names) or 'なし'}")
        lines.append(f"テラスタル: {'可' if ps.my_tera_available else '不可'}")

        # 相手の状態
        lines.append(f"\n相手の場: {ps.opp_pokemon.name} (HP: {ps.opp_pokemon.hp_ratio:.1%})")
        if ps.opp_pokemon.revealed_moves:
            lines.append(f"  判明した技: {', '.join(ps.opp_pokemon.revealed_moves)}")
        if ps.opp_pokemon.revealed_item:
            lines.append(f"  持ち物: {ps.opp_pokemon.revealed_item}")
        opp_bench_names = [p.name for p in ps.opp_bench]
        lines.append(f"相手の控え: {', '.join(opp_bench_names) or 'なし'}")
        lines.append(f"相手テラスタル: {'可' if ps.opp_tera_available else '不可'}")

        # 戦略
        if self.my_strategy:
            top_actions = sorted(self.my_strategy.items(), key=lambda x: -x[1])[:3]
            strategy_str = ", ".join(f"{a}:{p:.2f}" for a, p in top_actions)
            lines.append(f"\n自分の戦略: [{strategy_str}, ...]")

        # 期待値
        lines.append(f"期待値: (自分={self.values[0]:.3f}, 相手={self.values[1]:.3f})")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PBS(turn={self.public_state.turn}, my_value={self.values[0]:.3f})"

    def to_dict(self) -> dict[str, Any]:
        """シリアライズ可能な辞書に変換"""
        return {
            "public_state": self.public_state.to_dict(),
            "belief": self.belief.to_dict(),
            "my_strategy": {str(k): v for k, v in self.my_strategy.items()},
            "opp_strategy": {str(k): v for k, v in self.opp_strategy.items()},
            "values": list(self.values),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], usage_db: "PokemonUsageDatabase"
    ) -> "PublicBeliefState":
        """辞書から復元

        Args:
            data: シリアライズされた辞書
            usage_db: 信念状態の復元に必要な使用率データベース
        """
        from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase

        return cls(
            public_state=PublicGameState.from_dict(data["public_state"]),
            belief=PokemonBeliefState.from_dict(data["belief"], usage_db),
            my_strategy={int(k): v for k, v in data.get("my_strategy", {}).items()},
            opp_strategy={int(k): v for k, v in data.get("opp_strategy", {}).items()},
            values=tuple(data.get("values", [0.5, 0.5])),
        )


def _apply_hypothesis_to_pokemon(
    pokemon: "Pokemon",
    hypothesis: PokemonTypeHypothesis,
) -> None:
    """
    仮説をポケモンに適用（EV/IV含む）

    Args:
        pokemon: 対象ポケモン
        hypothesis: 適用する仮説
    """
    # 持ち物
    pokemon.item = hypothesis.item

    # テラスタイプ（未使用の場合のみ）
    if hasattr(pokemon, "terastal") and not pokemon.terastal:
        pokemon.Ttype = hypothesis.tera_type

    # 技（4つ設定）
    if hypothesis.moves:
        pokemon.moves = list(hypothesis.moves)

    # 特性
    if hypothesis.ability:
        pokemon.ability = hypothesis.ability

    # 性格
    if hypothesis.nature:
        pokemon.nature = hypothesis.nature

    # EV（努力値）を適用
    # effort setter が update_status() を呼ぶのでステータスも再計算される
    evs = hypothesis.get_evs()
    pokemon.effort = evs

    # IV（個体値）を適用
    ivs = hypothesis.get_ivs()
    pokemon.indiv = ivs


def instantiate_battle_from_hypothesis(
    pbs: PublicBeliefState,
    world: dict[str, PokemonTypeHypothesis],
    original_battle: Battle,
) -> Battle:
    """
    PBS と仮説から具体的な Battle 状態を構築

    Args:
        pbs: 公開信念状態
        world: 仮説（各ポケモンの型）
        original_battle: 元の Battle オブジェクト（クローンのベース）

    Returns:
        仮説が適用された新しい Battle
    """
    battle = deepcopy(original_battle)
    opponent = 1 - pbs.public_state.perspective

    # 相手のポケモンに仮説を適用
    for pokemon in battle.selected[opponent]:
        name = pokemon.name
        if name in world:
            _apply_hypothesis_to_pokemon(pokemon, world[name])

    # 場に出ているポケモンにも適用
    if battle.pokemon[opponent] is not None:
        name = battle.pokemon[opponent].name
        if name in world:
            _apply_hypothesis_to_pokemon(battle.pokemon[opponent], world[name])

    return battle
