"""
Pokemon BERT用データセット

パーティデータからMLM事前学習用のデータセットを作成する。
語彙は data/zukan.txt を正とする。
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


# フォーム違いの名前マッピング（battle_data → zukan）
# battle_dataの短い名前を図鑑の正式名に変換
POKEMON_NAME_ALIASES: dict[str, str] = {
    "イエッサン": "イエッサン(オス)",
    "イダイトウ": "イダイトウ(オス)",
    "ウーラオス": "ウーラオス(いちげき)",  # デフォルト（formがない場合）
    "ギラティナ": "ギラティナ(アナザー)",
    "ザシアン": "ザシアン(けんのおう)",
    "ザマゼンタ": "ザマゼンタ(たてのおう)",
    "ストリンダー": "ストリンダー(ハイ)",
    "テラパゴス": "テラパゴス(ノーマル)",
    "デオキシス": "デオキシス(ノーマル)",
    "フーパ": "フーパ(いましめられし)",
    "ケンタロス": "ケンタロス(パルデア単)",
    "バドレックス(こくば)": "バドレックス(こくばじょう)",
    "バドレックス(はくば)": "バドレックス(はくばじょう)",
    "ランドロス": "ランドロス(れいじゅう)",
    "メロエッタ": "メロエッタ(ボイス)",
    "モルペコ": "モルペコ(まんぷく)",
}

# battle_dataのform値から図鑑名への変換
# battle_dataの {"pokemon": "ウーラオス", "form": "れんげきのかた"} を処理
FORM_TO_ZUKAN: dict[tuple[str, str], str] = {
    ("ウーラオス", "れんげきのかた"): "ウーラオス(れんげき)",
    ("ウーラオス", "いちげきのかた"): "ウーラオス(いちげき)",
    ("イエッサン", "オスのすがた"): "イエッサン(オス)",
    ("イエッサン", "メスのすがた"): "イエッサン(メス)",
    ("イダイトウ", "オスのすがた"): "イダイトウ(オス)",
    ("イダイトウ", "メスのすがた"): "イダイトウ(メス)",
    ("ギラティナ", "アナザーフォルム"): "ギラティナ(アナザー)",
    ("ギラティナ", "オリジンフォルム"): "ギラティナ(オリジン)",
    ("ストリンダー", "ハイなすがた"): "ストリンダー(ハイ)",
    ("ストリンダー", "ローなすがた"): "ストリンダー(ロー)",
    ("ランドロス", "けしんフォルム"): "ランドロス(けしん)",
    ("ランドロス", "れいじゅうフォルム"): "ランドロス(れいじゅう)",
}


@dataclass
class PokemonVocab:
    """ポケモン名の語彙管理"""

    pokemon_to_id: dict[str, int]
    id_to_pokemon: dict[int, str]

    # 特殊トークン
    PAD_TOKEN = "[PAD]"
    MASK_TOKEN = "[MASK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    UNK_TOKEN = "[UNK]"

    SPECIAL_TOKENS = [PAD_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN, UNK_TOKEN]

    @classmethod
    def from_zukan(cls, zukan_path: Path) -> "PokemonVocab":
        """
        data/zukan.txt から語彙を構築（推奨）

        zukan.txt形式:
        Num	Name	Type1	Type2	...
        906	ニャオハ	くさ	-	...
        """
        pokemon_names = []
        with open(zukan_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Num"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    name = parts[1]
                    pokemon_names.append(name)

        # 特殊トークン + ポケモン名（出現順を維持）
        all_tokens = cls.SPECIAL_TOKENS + pokemon_names

        pokemon_to_id = {name: idx for idx, name in enumerate(all_tokens)}
        id_to_pokemon = {idx: name for idx, name in enumerate(all_tokens)}

        return cls(pokemon_to_id=pokemon_to_id, id_to_pokemon=id_to_pokemon)

    @classmethod
    def from_teams(cls, teams: list[list[str]]) -> "PokemonVocab":
        """パーティデータから語彙を構築（非推奨: from_zukanを使用）"""
        pokemon_set = set()
        for team in teams:
            for pokemon in team:
                pokemon_set.add(pokemon)

        # 特殊トークン + ポケモン名
        all_tokens = cls.SPECIAL_TOKENS + sorted(pokemon_set)

        pokemon_to_id = {name: idx for idx, name in enumerate(all_tokens)}
        id_to_pokemon = {idx: name for idx, name in enumerate(all_tokens)}

        return cls(pokemon_to_id=pokemon_to_id, id_to_pokemon=id_to_pokemon)

    def __len__(self) -> int:
        return len(self.pokemon_to_id)

    @property
    def pad_id(self) -> int:
        return self.pokemon_to_id[self.PAD_TOKEN]

    @property
    def mask_id(self) -> int:
        return self.pokemon_to_id[self.MASK_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.pokemon_to_id[self.CLS_TOKEN]

    @property
    def sep_id(self) -> int:
        return self.pokemon_to_id[self.SEP_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.pokemon_to_id[self.UNK_TOKEN]

    def encode(self, pokemon: str) -> int:
        # エイリアスをチェック
        pokemon = POKEMON_NAME_ALIASES.get(pokemon, pokemon)
        return self.pokemon_to_id.get(pokemon, self.unk_id)

    def normalize_name(self, pokemon: str) -> str:
        """ポケモン名を正規化（エイリアス適用）"""
        return POKEMON_NAME_ALIASES.get(pokemon, pokemon)

    def decode(self, idx: int) -> str:
        return self.id_to_pokemon.get(idx, self.UNK_TOKEN)

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pokemon_to_id": self.pokemon_to_id,
                    "id_to_pokemon": {
                        str(k): v for k, v in self.id_to_pokemon.items()
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "PokemonVocab":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            pokemon_to_id=data["pokemon_to_id"],
            id_to_pokemon={int(k): v for k, v in data["id_to_pokemon"].items()},
        )


def normalize_pokemon_name(pokemon_data: dict) -> str:
    """
    battle_dataのポケモン情報から正規化された名前を取得

    Args:
        pokemon_data: {"pokemon": "ウーラオス", "form": "れんげきのかた", ...}

    Returns:
        図鑑に対応する正式名（例: "ウーラオス(れんげき)"）
    """
    name = pokemon_data["pokemon"]
    form = pokemon_data.get("form", "")

    # formがある場合はFORM_TO_ZUKANを優先
    if form:
        key = (name, form)
        if key in FORM_TO_ZUKAN:
            return FORM_TO_ZUKAN[key]

    # formがない場合やマッピングがない場合はPOKEMON_NAME_ALIASESを使用
    return POKEMON_NAME_ALIASES.get(name, name)


def load_team_data(data_dir: Path, seasons: list[int]) -> list[list[str]]:
    """
    指定シーズンのパーティデータを読み込む

    Returns:
        list[list[str]]: パーティのリスト（各パーティは6体のポケモン名、図鑑名に正規化済み）
    """
    teams = []
    for season in seasons:
        path = data_dir / f"s{season}_single_ranked_teams.json"
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for team_data in data["teams"]:
            team = [normalize_pokemon_name(p) for p in team_data["team"]]
            if len(team) == 6:  # 6体揃っているパーティのみ
                teams.append(team)

    return teams


class PokemonMLMDataset(Dataset):
    """
    Pokemon BERT MLM用データセット

    パーティ内の1体をマスクして予測するタスク。
    データ拡張として、同じパーティからシャッフルして複数サンプルを生成。
    """

    def __init__(
        self,
        teams: list[list[str]],
        vocab: PokemonVocab,
        mask_prob: float = 0.15,
        augment_factor: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Args:
            teams: パーティのリスト
            vocab: 語彙
            mask_prob: マスク確率（各ポケモンがマスクされる確率）
            augment_factor: データ拡張倍率（シャッフルで増やす）
            seed: ランダムシード
        """
        self.teams = teams
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.augment_factor = augment_factor

        if seed is not None:
            random.seed(seed)

        # データ拡張: 各パーティを複数回シャッフル
        self.samples = []
        for team in teams:
            for _ in range(augment_factor):
                shuffled = team.copy()
                random.shuffle(shuffled)
                self.samples.append(shuffled)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        team = self.samples[idx]

        # 入力系列: [CLS] pokemon1 pokemon2 ... pokemon6
        input_ids = [self.vocab.cls_id] + [self.vocab.encode(p) for p in team]
        labels = [-100] * len(input_ids)  # -100 はCrossEntropyで無視される

        # マスキング（[CLS]以外の位置）
        for i in range(1, len(input_ids)):
            if random.random() < self.mask_prob:
                labels[i] = input_ids[i]  # 正解ラベル

                # 80% MASK, 10% ランダム, 10% そのまま (BERT style)
                r = random.random()
                if r < 0.8:
                    input_ids[i] = self.vocab.mask_id
                elif r < 0.9:
                    # ランダムなポケモンに置換（特殊トークン以外）
                    random_id = random.randint(
                        len(PokemonVocab.SPECIAL_TOKENS), len(self.vocab) - 1
                    )
                    input_ids[i] = random_id
                # else: そのまま

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


class PokemonSelectionDataset(Dataset):
    """
    選出予測用データセット（Phase 2で使用）

    入力: [CLS] 自1 自2 自3 自4 自5 自6 [SEP] 相1 相2 相3 相4 相5 相6 [SEP]
    ラベル: 各トークンに NOT_SELECTED(0) / SELECTED(1) / LEAD(2)
    """

    # ラベル定義
    LABEL_NOT_SELECTED = 0
    LABEL_SELECTED = 1
    LABEL_LEAD = 2
    LABEL_IGNORE = -100  # CLS, SEP などは無視

    def __init__(
        self,
        matchups: list[dict],
        vocab: PokemonVocab,
    ):
        """
        Args:
            matchups: 対戦データのリスト
                {
                    "my_team": [6体のポケモン名],
                    "opp_team": [6体のポケモン名],
                    "my_selection": [選出した3体のインデックス],  # 先発が最初
                    "opp_selection": [相手の選出3体のインデックス],  # 先発が最初
                }
            vocab: 語彙
        """
        self.matchups = matchups
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.matchups)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        matchup = self.matchups[idx]

        my_team = matchup["my_team"]
        opp_team = matchup["opp_team"]
        my_selection = matchup["my_selection"]
        opp_selection = matchup["opp_selection"]

        # 入力系列: [CLS] my1-6 [SEP] opp1-6 [SEP]
        input_ids = (
            [self.vocab.cls_id]
            + [self.vocab.encode(p) for p in my_team]
            + [self.vocab.sep_id]
            + [self.vocab.encode(p) for p in opp_team]
            + [self.vocab.sep_id]
        )

        # ラベル作成
        labels = [self.LABEL_IGNORE]  # [CLS]

        # 自分チームのラベル
        for i in range(6):
            if i == my_selection[0]:  # 先発
                labels.append(self.LABEL_LEAD)
            elif i in my_selection:  # 選出（控え）
                labels.append(self.LABEL_SELECTED)
            else:
                labels.append(self.LABEL_NOT_SELECTED)

        labels.append(self.LABEL_IGNORE)  # [SEP]

        # 相手チームのラベル
        for i in range(6):
            if i == opp_selection[0]:  # 相手先発
                labels.append(self.LABEL_LEAD)
            elif i in opp_selection:  # 相手選出
                labels.append(self.LABEL_SELECTED)
            else:
                labels.append(self.LABEL_NOT_SELECTED)

        labels.append(self.LABEL_IGNORE)  # [SEP]

        # token_type_ids: 0=自分チーム, 1=相手チーム
        token_type_ids = (
            [0] * (1 + 6 + 1)  # [CLS] + my_team + [SEP]
            + [1] * (6 + 1)  # opp_team + [SEP]
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }
