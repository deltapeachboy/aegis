"""
ReBeL AI サービス - チェックポイントからモデルをロードしてAI行動を選択
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase
from src.pokemon_battle_sim.battle import Battle
from src.rebel.belief_state import Observation, ObservationType, PokemonBeliefState
from src.rebel.cfr_solver import CFRConfig, ReBeLSolver, check_hopeless_situation
from src.rebel.public_state import PublicBeliefState
from src.rebel.value_network import PBSEncoder, ReBeLValueNetwork

# Selection BERT (optional import)
try:
    from src.selection_bert import (
        PokemonBertConfig,
        PokemonBertForTokenClassification,
        PokemonVocab,
        SelectionBeliefPredictor,
    )

    SELECTION_BERT_AVAILABLE = True
except ImportError:
    SELECTION_BERT_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class RebelAIConfig:
    """ReBeL AI設定"""

    checkpoint_path: str
    usage_db_path: str = "data/pokedb_usage/season_37_top150.json"
    cfr_iterations: int = 30
    cfr_world_samples: int = 10
    use_lightweight_cfr: bool = True
    device: str = "cpu"
    # Selection BERT settings
    selection_bert_hidden_size: int = 256
    selection_bert_num_layers: int = 4
    selection_bert_num_heads: int = 4
    # Value Network settings
    use_move_effectiveness: Optional[bool] = (
        None  # None = チェックポイント名から自動判定
    )


class RebelAI:
    """ReBeL AIインスタンス"""

    def __init__(self, config: RebelAIConfig):
        self.config = config
        self.device = config.device
        self.checkpoint_path = Path(config.checkpoint_path)

        # Usage database
        logger.info(f"Loading usage database from {config.usage_db_path}")
        self.usage_db = PokemonUsageDatabase.from_json(config.usage_db_path)

        # Value Network
        logger.info(f"Loading value network from {self.checkpoint_path}")
        self.value_network = self._load_value_network()

        # CFR Solver
        cfr_config = CFRConfig(
            num_iterations=config.cfr_iterations,
            num_world_samples=config.cfr_world_samples,
        )
        self.solver = ReBeLSolver(
            value_network=self.value_network,
            cfr_config=cfr_config,
            use_simplified=config.use_lightweight_cfr,
        )

        # Selection BERT
        self.selection_bert: Optional[PokemonBertForTokenClassification] = None
        self.selection_bert_vocab: Optional[PokemonVocab] = None
        self.selection_predictor: Optional[SelectionBeliefPredictor] = None
        self._load_selection_bert()

        # 信念状態（プレイヤーごと）
        self.beliefs: Dict[int, PokemonBeliefState] = {}

        # 観測済み情報の追跡（重複観測を防ぐ）
        self.observed_items: Dict[str, str] = {}  # pokemon_name -> item
        self.observed_abilities: Dict[str, str] = {}  # pokemon_name -> ability
        self.observed_tera: Dict[str, str] = {}  # pokemon_name -> tera_type

        logger.info("ReBeL AI initialized successfully")

    def _load_value_network(self) -> ReBeLValueNetwork:
        """Value Networkをロード"""
        # エンコーダーの辞書を先に読み込み（モデル初期化前に必要）
        encoder_path = self.checkpoint_path / "encoder_vocab.json"
        encoder_config = {}
        if encoder_path.exists():
            with open(encoder_path, "r", encoding="utf-8") as f:
                encoder_state = json.load(f)
            encoder_config = {
                "pokemon_to_id": encoder_state.get("pokemon_to_id", {}),
                "move_to_id": encoder_state.get("move_to_id", {}),
                "item_to_id": encoder_state.get("item_to_id", {}),
            }
            logger.info(f"Loaded encoder vocab from {encoder_path}")

        # use_move_effectiveness の判定
        use_move_effectiveness = self.config.use_move_effectiveness
        if use_move_effectiveness is None:
            # チェックポイント名から自動判定
            checkpoint_name = str(self.checkpoint_path).lower()
            use_move_effectiveness = "move_effective" in checkpoint_name
            logger.info(
                f"Auto-detected use_move_effectiveness={use_move_effectiveness} "
                f"from checkpoint path"
            )

        value_network = ReBeLValueNetwork(
            hidden_dim=256,
            num_res_blocks=4,
            encoder_config=encoder_config,
            use_move_effectiveness=use_move_effectiveness,
        )

        model_path = self.checkpoint_path / "value_network.pt"
        if model_path.exists():
            value_network.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            logger.info(f"Loaded value network from {model_path}")

        value_network.to(self.device)
        value_network.eval()
        return value_network

    def _load_selection_bert(self) -> None:
        """Selection BERTをロード"""
        if not SELECTION_BERT_AVAILABLE:
            logger.warning("Selection BERT not available")
            return

        vocab_path = self.checkpoint_path / "selection_bert_vocab.json"
        model_path = self.checkpoint_path / "selection_bert.pt"

        if not vocab_path.exists() or not model_path.exists():
            logger.info("Selection BERT not found in checkpoint")
            return

        try:
            self.selection_bert_vocab = PokemonVocab.load(vocab_path)

            config = PokemonBertConfig(
                vocab_size=len(self.selection_bert_vocab),
                hidden_size=self.config.selection_bert_hidden_size,
                num_hidden_layers=self.config.selection_bert_num_layers,
                num_attention_heads=self.config.selection_bert_num_heads,
                intermediate_size=self.config.selection_bert_hidden_size * 2,
            )

            self.selection_bert = PokemonBertForTokenClassification(config)
            self.selection_bert.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.selection_bert.to(self.device)
            self.selection_bert.eval()

            self.selection_predictor = SelectionBeliefPredictor(
                model=self.selection_bert,
                vocab=self.selection_bert_vocab,
                device=self.device,
            )

            logger.info(f"Loaded Selection BERT from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load Selection BERT: {e}")
            self.selection_bert = None
            self.selection_predictor = None

    def init_belief_state(
        self, player: int, opponent_pokemon_names: List[str]
    ) -> PokemonBeliefState:
        """信念状態を初期化"""
        belief = PokemonBeliefState(
            opponent_pokemon_names=opponent_pokemon_names,
            usage_db=self.usage_db,
        )
        self.beliefs[player] = belief
        return belief

    def get_selection(
        self,
        my_team_names: List[str],
        opponent_team_names: List[str],
        deterministic: bool = True,
    ) -> List[int]:
        """
        Selection BERTを使って選出を決定

        Returns:
            選出順序のリスト（先頭が先発）
        """
        if self.selection_predictor is None:
            # Selection BERTが使えない場合はランダム
            indices = list(range(len(my_team_names)))
            random.shuffle(indices)
            return indices[:3]

        try:
            selected, lead_idx = self.selection_predictor.select_team(
                my_team=my_team_names,
                opp_team=opponent_team_names,
                deterministic=deterministic,
            )
            return selected
        except Exception as e:
            logger.error(f"Selection BERT failed: {e}")
            indices = list(range(len(my_team_names)))
            random.shuffle(indices)
            return indices[:3]

    def should_surrender(self, battle: Battle, player: int) -> bool:
        """
        AIが降参すべきか判定

        Args:
            battle: Battleインスタンス
            player: AIのプレイヤー番号

        Returns:
            True: 降参すべき, False: 続行
        """
        try:
            return check_hopeless_situation(battle, player)
        except Exception as e:
            logger.error(f"Hopeless check failed: {e}")
            return False

    def get_battle_command(self, battle: Battle, player: int) -> int:
        """
        バトルフェーズで行動を選択

        Args:
            battle: Battleインスタンス
            player: プレイヤー番号（0 or 1）

        Returns:
            選択したコマンド
        """
        # 利用可能なコマンドを取得
        available = battle.available_commands(player, phase="battle")

        if not available or available == [Battle.NO_COMMAND]:
            return Battle.SKIP

        if len(available) == 1:
            return available[0]

        try:
            # 信念状態の初期化（必要なら）
            if player not in self.beliefs:
                opponent = 1 - player
                opponent_names = [
                    p.name for p in battle.selected[opponent] if p is not None
                ]
                self.init_belief_state(player, opponent_names)

            # PBS 構築
            pbs = PublicBeliefState.from_battle(battle, player, self.beliefs[player])

            # CFR で行動選択
            return self.solver.get_action(pbs, battle, explore=False)
        except Exception as e:
            logger.error(f"ReBeL action selection failed: {e}")
            # フォールバック: ランダム
            return random.choice(available)

    def get_change_command(self, battle: Battle, player: int) -> int:
        """
        交代フェーズで行動を選択

        Args:
            battle: Battleインスタンス
            player: プレイヤー番号

        Returns:
            選択したコマンド
        """
        available = battle.available_commands(player, phase="change")

        if not available or available == [Battle.NO_COMMAND]:
            return Battle.SKIP

        if len(available) == 1:
            return available[0]

        # HP比率が高いポケモンを優先
        best_switch = available[0]
        best_hp = 0.0

        for cmd in available:
            if cmd >= 20:
                idx = cmd - 20
                if idx < len(battle.selected[player]):
                    pokemon = battle.selected[player][idx]
                    if pokemon and pokemon.status[0] > 0:
                        hp_ratio = pokemon.hp / pokemon.status[0]
                        if hp_ratio > best_hp:
                            best_hp = hp_ratio
                            best_switch = cmd

        return best_switch

    def update_belief(self, player: int, observation: Observation) -> None:
        """信念状態を更新"""
        if player in self.beliefs:
            self.beliefs[player].update(observation)

    def observe_move(self, player: int, pokemon_name: str, move_name: str) -> None:
        """技使用を観測"""
        obs = Observation(
            type=ObservationType.MOVE_USED,
            pokemon_name=pokemon_name,
            details={"move": move_name},
        )
        self.update_belief(player, obs)

    def observe_item(self, player: int, pokemon_name: str, item_name: str) -> None:
        """持ち物を観測（重複観測を防ぐ）"""
        # 既に同じ持ち物を観測済みならスキップ
        if self.observed_items.get(pokemon_name) == item_name:
            return

        obs = Observation(
            type=ObservationType.ITEM_REVEALED,
            pokemon_name=pokemon_name,
            details={"item": item_name},
        )
        self.update_belief(player, obs)
        self.observed_items[pokemon_name] = item_name
        logger.info(f"AI observed item: {pokemon_name} has {item_name}")

    def observe_ability(
        self, player: int, pokemon_name: str, ability_name: str
    ) -> None:
        """特性を観測（重複観測を防ぐ）"""
        if self.observed_abilities.get(pokemon_name) == ability_name:
            return

        obs = Observation(
            type=ObservationType.ABILITY_REVEALED,
            pokemon_name=pokemon_name,
            details={"ability": ability_name},
        )
        self.update_belief(player, obs)
        self.observed_abilities[pokemon_name] = ability_name
        logger.info(f"AI observed ability: {pokemon_name} has {ability_name}")

    def observe_tera(self, player: int, pokemon_name: str, tera_type: str) -> None:
        """テラスタイプを観測（重複観測を防ぐ）"""
        if self.observed_tera.get(pokemon_name) == tera_type:
            return

        obs = Observation(
            type=ObservationType.TERASTALLIZED,
            pokemon_name=pokemon_name,
            details={"tera_type": tera_type},
        )
        self.update_belief(player, obs)
        self.observed_tera[pokemon_name] = tera_type
        logger.info(f"AI observed tera: {pokemon_name} terastallized to {tera_type}")

    def observe_battle_state(self, battle: Battle, ai_player: int) -> None:
        """
        バトル状態から観測可能な情報をAIに伝える

        Args:
            battle: Battleインスタンス
            ai_player: AIのプレイヤー番号（観測する側）
        """
        opponent = 1 - ai_player

        # 相手の場に出ているポケモンを観測
        opponent_pokemon = battle.pokemon[opponent]
        if opponent_pokemon is None:
            return

        pokemon_name = opponent_pokemon.name

        # 持ち物を観測（場に出ているポケモンの持ち物は公開情報）
        # ふうせん、ブーストエナジーなど場に出た時にアナウンスされるもの
        if opponent_pokemon.item:
            # 特定の持ち物は場に出た時点で公開される
            visible_items = {
                "ふうせん",  # 「〇〇はふうせんで浮いている！」
                "ブーストエナジー",  # 発動時にアナウンス
                # こだわり系は技を使うまで分からないので含めない
            }
            if opponent_pokemon.item in visible_items:
                self.observe_item(ai_player, pokemon_name, opponent_pokemon.item)

        # テラスタル状態を観測
        if opponent_pokemon.terastal and opponent_pokemon.Ttype:
            self.observe_tera(ai_player, pokemon_name, opponent_pokemon.Ttype)

    def reset(self) -> None:
        """状態をリセット"""
        self.beliefs.clear()
        self.observed_items.clear()
        self.observed_abilities.clear()
        self.observed_tera.clear()

    def get_analysis(self, battle: Battle, player: int) -> Dict[str, Any]:
        """
        現在の戦況分析を取得

        Args:
            battle: Battleインスタンス
            player: プレイヤー番号（0 or 1）

        Returns:
            分析結果（戦略分布、推定勝率など）
        """
        from src.rebel.cfr_solver import default_value_estimator

        analysis: Dict[str, Any] = {
            "available": False,
            "policy": {},
            "value": 0.5,
            "action_names": {},
        }

        try:
            # 利用可能なコマンドを取得
            available = battle.available_commands(player, phase="battle")
            if not available or available == [Battle.NO_COMMAND]:
                return analysis

            # 信念状態の初期化（必要なら）
            if player not in self.beliefs:
                opponent = 1 - player
                opponent_names = [
                    p.name for p in battle.selected[opponent] if p is not None
                ]
                self.init_belief_state(player, opponent_names)

            # PBS 構築
            pbs = PublicBeliefState.from_battle(battle, player, self.beliefs[player])

            # CFR で戦略計算
            my_strategy, _ = self.solver.solve(pbs, battle)

            # 価値推定
            value = default_value_estimator(battle, player)

            # 行動名のマッピング
            action_names = {}
            pokemon = battle.pokemon[player]
            if pokemon:
                for cmd in my_strategy.keys():
                    if 0 <= cmd <= 3:
                        # 通常技
                        if cmd < len(pokemon.moves):
                            action_names[cmd] = pokemon.moves[cmd]
                        else:
                            action_names[cmd] = f"技{cmd + 1}"
                    elif 10 <= cmd <= 13:
                        # テラスタル技
                        move_idx = cmd - 10
                        if move_idx < len(pokemon.moves):
                            action_names[cmd] = f"テラス+{pokemon.moves[move_idx]}"
                        else:
                            action_names[cmd] = f"テラス+技{move_idx + 1}"
                    elif 20 <= cmd <= 25:
                        # 交代
                        bench_idx = cmd - 20
                        if bench_idx < len(battle.selected[player]):
                            bench_pokemon = battle.selected[player][bench_idx]
                            if bench_pokemon:
                                action_names[cmd] = f"交代→{bench_pokemon.name}"
                            else:
                                action_names[cmd] = f"交代→{bench_idx + 1}番"
                        else:
                            action_names[cmd] = f"交代{cmd}"
                    elif cmd == Battle.STRUGGLE:
                        action_names[cmd] = "わるあがき"
                    else:
                        action_names[cmd] = f"行動{cmd}"

            analysis = {
                "available": True,
                "policy": my_strategy,
                "value": value,
                "action_names": action_names,
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")

        return analysis


# グローバルキャッシュ（同じチェックポイントの再ロードを防ぐ）
_rebel_ai_cache: Dict[str, RebelAI] = {}


def load_rebel_ai(checkpoint_path: str, usage_db_path: Optional[str] = None) -> RebelAI:
    """
    ReBeL AIをロード（キャッシュ付き）

    Args:
        checkpoint_path: チェックポイントのパス
        usage_db_path: 使用率データベースのパス（省略時はデフォルト）

    Returns:
        RebelAI インスタンス
    """
    cache_key = checkpoint_path

    if cache_key not in _rebel_ai_cache:
        config = RebelAIConfig(
            checkpoint_path=checkpoint_path,
            usage_db_path=usage_db_path or "data/pokedb_usage/season_37_top150.json",
        )
        _rebel_ai_cache[cache_key] = RebelAI(config)

    return _rebel_ai_cache[cache_key]


def clear_rebel_ai_cache() -> None:
    """キャッシュをクリア"""
    _rebel_ai_cache.clear()
