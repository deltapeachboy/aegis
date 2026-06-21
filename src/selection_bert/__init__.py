"""
Pokemon Selection BERT

ポケモンをトークンとして扱い、チーム構成から選出を予測する。

Phase 1: MLM事前学習でポケモンの埋め込みを学習
Phase 2: Token Classificationで選出予測（自分・相手両方）
"""

from .dataset import (
    FORM_TO_ZUKAN,
    POKEMON_NAME_ALIASES,
    PokemonMLMDataset,
    PokemonSelectionDataset,
    PokemonVocab,
    load_team_data,
    normalize_pokemon_name,
)
from .model import (
    PokemonBertConfig,
    PokemonBertForMLM,
    PokemonBertForTokenClassification,
    PokemonBertModel,
)

from .selection_belief import SelectionBeliefPredictor, SelectionPrediction

__all__ = [
    # 語彙・データセット
    "PokemonVocab",
    "PokemonMLMDataset",
    "PokemonSelectionDataset",
    "load_team_data",
    "normalize_pokemon_name",
    "POKEMON_NAME_ALIASES",
    "FORM_TO_ZUKAN",
    # モデル
    "PokemonBertConfig",
    "PokemonBertModel",
    "PokemonBertForMLM",
    "PokemonBertForTokenClassification",
    # 選出信念
    "SelectionBeliefPredictor",
    "SelectionPrediction",
]
