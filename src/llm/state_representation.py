from __future__ import annotations

"""
ポケモン対戦用の LLM 入出力フォーマット定義

- Battle / Pokemon オブジェクトから盤面状態をテキスト化する
- 合法行動を列挙し、LLM に渡しやすい ID / 表示名を付与する
- LLM 出力（MOVE/ SWITCH）を内部コマンドにマッピングする

既存の `src.pokemon_battle_sim.battle.Battle` / `Pokemon` に強く依存するため、
一部のメソッドはプロジェクト固有の実装で補完していく想定。
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon


ActionType = Literal["move", "switch"]


@dataclass
class LLMAction:
    """LLM に提示・出力させる 1 つの行動"""

    id: str
    type: ActionType
    move_name: Optional[str] = None
    switch_name: Optional[str] = None

    def to_command_string(self) -> str:
        """
        LLM がそのまま出力できる簡潔な表現

        例:
            MOVE: シャドーボール
            SWITCH: ハバタクカミ
        """
        if self.type == "move" and self.move_name:
            return f"MOVE: {self.move_name}"
        if self.type == "switch" and self.switch_name:
            return f"SWITCH: {self.switch_name}"
        # フォールバック
        return f"{self.id}"


@dataclass
class LLMState:
    """LLM に渡す 1 ターン分の状態表現"""

    player: int
    state_text: str
    actions: List[LLMAction]


def _format_pokemon_short(p: Pokemon) -> str:
    """場に出ているポケモンの概要を 1 行で表現"""
    types = "/".join(p.types)
    hp_ratio = f"{int(p.hp_ratio * 100)}%"
    boosts = []
    for label, rank in zip(Pokemon.status_label_hiragana, p.rank):
        if rank > 0:
            boosts.append(f"{label}+{rank}")
        elif rank < 0:
            boosts.append(f"{label}{rank}")
    boosts_text = ", ".join(boosts) if boosts else "なし"
    ailment = p.ailment or "なし"

    return (
        f"{p.display_name} ({types}) "
        f"HP:{hp_ratio} "
        f"性格:{p.nature} 特性:{p.ability} 持ち物:{p.item or 'なし'} "
        f"状態異常:{ailment} ランク変化:[{boosts_text}]"
    )


def _format_party_short(party: List[Pokemon]) -> List[str]:
    """控えポケモンも含めたパーティ概要"""
    lines: List[str] = []
    for i, p in enumerate(party):
        types = "/".join(p.types)
        hp_ratio = f"{int(p.hp_ratio * 100)}%"
        ailment = p.ailment or "なし"
        lines.append(
            f"- {i}: {p.display_name} ({types}) HP:{hp_ratio} "
            f"特性:{p.ability} 持ち物:{p.item or 'なし'} 状態異常:{ailment}"
        )
    return lines


def battle_to_llm_state(battle: Battle, player: int) -> LLMState:
    """
    Battle オブジェクトから、LLM 用の状態テキストと合法行動一覧を構築する。

    player: 0 or 1
    """
    assert player in (0, 1)

    me = battle.pokemon[player]
    opp = battle.pokemon[1 - player]
    my_party = battle.selected[player]
    opp_party = battle.selected[1 - player]

    # 場・天候など
    weather = battle.weather(player)
    field = battle.field()

    # 盤面テキスト本体（YAML 風の構造化テキスト）
    lines: List[str] = []
    lines.append(f"turn: {battle.turn}")
    lines.append(f"player: {player}")
    lines.append("field:")
    lines.append(f"  weather: {weather or 'none'}")
    lines.append(f"  terrain: {field or 'none'}")
    lines.append(f"  trick_room: {bool(battle.condition.get('trickroom', 0))}")
    lines.append("")

    # 自分側
    lines.append("self:")
    lines.append(f"  active: {_format_pokemon_short(me)}")
    lines.append("  party:")
    for party_line in _format_party_short(my_party):
        lines.append("    " + party_line)
    lines.append("")

    # 相手側（観測情報ベース。ここでは簡略に selected をそのまま使う）
    lines.append("opponent:")
    lines.append(f"  active: {_format_pokemon_short(opp)}")
    lines.append("  party:")
    for party_line in _format_party_short(opp_party):
        lines.append("    " + party_line)
    lines.append("")

    # 選択可能な行動一覧
    actions = enumerate_legal_actions(battle, player)

    lines.append("legal_actions:")
    for a in actions:
        lines.append(f"  - id: {a.id}  text: {a.to_command_string()}")

    prompt = "\n".join(lines)

    return LLMState(player=player, state_text=prompt, actions=actions)


def enumerate_legal_actions(battle: Battle, player: int) -> List[LLMAction]:
    """
    現在ターンで player が選択可能な行動を列挙する。

    注意:
        Battle クラスは多機能なため、ここでは素直に
        - 目の前のポケモンの技（PP>0 のもの）
        - 控えポケモンへの交代（ひんしでないもの）
        を「合法手」とみなすシンプルな実装から始める。
    """
    me: Pokemon = battle.pokemon[player]
    party: List[Pokemon] = battle.selected[player]

    actions: List[LLMAction] = []

    # 技行動
    for i, move_name in enumerate(me.moves):
        if not move_name:
            continue
        # PP が 0 の技は選択不可
        if i < len(me.pp) and me.pp[i] <= 0:
            continue
        action_id = f"MOVE_{i}"
        actions.append(
            LLMAction(
                id=action_id,
                type="move",
                move_name=move_name,
            )
        )

    # 交代行動
    current_index = battle.current_index(player)
    for i, p in enumerate(party):
        if i == current_index:
            continue
        if p.hp <= 0:
            continue
        action_id = f"SWITCH_{i}"
        actions.append(
            LLMAction(
                id=action_id,
                type="switch",
                switch_name=p.display_name,
            )
        )

    # 行動が 1 つもない場合は SKIP を追加（ストラグルなどのフォールバック用）
    if not actions:
        actions.append(LLMAction(id="SKIP", type="move", move_name="ストラグル"))

    return actions


def parse_llm_action_output(
    text: str, actions: List[LLMAction]
) -> Tuple[LLMAction, Dict[str, float]]:
    """
    LLM の生出力文字列から、もっともそれらしい LLMAction を 1 つ選ぶ。

    返り値:
        (選択された行動, マッチスコア情報)

    実装はシンプルな文字列一致ベース。
    将来的に embedding マッチングなどに差し替え可能。
    """
    normalized = text.strip()

    # 完全一致 or 前方一致で優先的にマッチ
    for a in actions:
        cmd = a.to_command_string()
        if normalized == cmd or normalized.startswith(cmd):
            return a, {"match_type": "exact", "score": 1.0}

    # MOVE: / SWITCH: 形式をパース
    if normalized.upper().startswith("MOVE:"):
        name = normalized.split(":", 1)[1].strip()
        best, best_score = None, 0.0
        for a in actions:
            if a.type != "move" or not a.move_name:
                continue
            # 部分一致スコア（かなり雑だが軽量）
            if name in a.move_name:
                score = len(name) / len(a.move_name)
            else:
                score = 0.0
            if score > best_score:
                best, best_score = a, score
        if best is not None:
            return best, {"match_type": "move_partial", "score": best_score}

    if normalized.upper().startswith("SWITCH:"):
        name = normalized.split(":", 1)[1].strip()
        best, best_score = None, 0.0
        for a in actions:
            if a.type != "switch" or not a.switch_name:
                continue
            if name in a.switch_name:
                score = len(name) / len(a.switch_name)
            else:
                score = 0.0
            if score > best_score:
                best, best_score = a, score
        if best is not None:
            return best, {"match_type": "switch_partial", "score": best_score}

    # いずれにもマッチしない場合は、最初の行動にフォールバック
    fallback = actions[0]
    return fallback, {"match_type": "fallback", "score": 0.0}


