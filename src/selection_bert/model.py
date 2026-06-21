"""
Pokemon BERT モデル

ポケモンをトークンとして扱い、パーティ構成からポケモンの埋め込みを学習する。
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PokemonBertConfig:
    """Pokemon BERT設定"""

    vocab_size: int = 834  # 特殊トークン5 + ポケモン829体
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 16  # [CLS] + 6体 + [SEP] + 6体 + [SEP] = 15
    type_vocab_size: int = 2  # 自チーム / 相手チーム
    layer_norm_eps: float = 1e-12

    # 特殊トークンID
    pad_token_id: int = 0
    mask_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3

    # 選出予測用
    num_selection_labels: int = 3  # NOT_SELECTED, SELECTED, LEAD


class PokemonEmbeddings(nn.Module):
    """ポケモン埋め込み層"""

    def __init__(self, config: PokemonBertConfig):
        super().__init__()
        self.pokemon_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids を登録
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        pokemon_embeds = self.pokemon_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = pokemon_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """マルチヘッド自己注意機構"""

    def __init__(self, config: PokemonBertConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask) * -10000.0
            attention_scores = attention_scores + extended_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)

        return context_layer


class TransformerLayer(nn.Module):
    """Transformerレイヤー"""

    def __init__(self, config: PokemonBertConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-Attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.attention_norm(hidden_states + attention_output)

        # Feed-Forward
        intermediate_output = F.gelu(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        hidden_states = self.output_norm(hidden_states + layer_output)

        return hidden_states


class PokemonBertEncoder(nn.Module):
    """Pokemon BERTエンコーダー"""

    def __init__(self, config: PokemonBertConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class PokemonBertModel(nn.Module):
    """Pokemon BERTベースモデル"""

    def __init__(self, config: PokemonBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = PokemonEmbeddings(config)
        self.encoder = PokemonBertEncoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embeddings = self.embeddings(input_ids, token_type_ids)
        hidden_states = self.encoder(embeddings, attention_mask)

        return hidden_states

    def get_pokemon_embeddings(self) -> torch.Tensor:
        """全ポケモンの埋め込みベクトルを取得"""
        return self.embeddings.pokemon_embeddings.weight.data


class PokemonBertForMLM(nn.Module):
    """MLM事前学習用Pokemon BERT"""

    def __init__(self, config: PokemonBertConfig):
        super().__init__()
        self.config = config
        self.bert = PokemonBertModel(config)

        # MLM Head
        self.mlm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 埋め込み重みを共有
        self.mlm_decoder.weight = self.bert.embeddings.pokemon_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len] マスクされた位置の正解ラベル（-100は無視）

        Returns:
            loss: MLM損失（labelsが与えられた場合）
            logits: [batch, seq_len, vocab_size]
        """
        hidden_states = self.bert(input_ids, attention_mask, token_type_ids)

        # MLM prediction
        mlm_hidden = F.gelu(self.mlm_dense(hidden_states))
        mlm_hidden = self.mlm_norm(mlm_hidden)
        logits = self.mlm_decoder(mlm_hidden) + self.mlm_bias

        output = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
            output["loss"] = loss

        return output


class PokemonBertForTokenClassification(nn.Module):
    """選出予測用Pokemon BERT (Token Classification)"""

    def __init__(self, config: PokemonBertConfig):
        super().__init__()
        self.config = config
        self.bert = PokemonBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_selection_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len] = [batch, 15]
                       [CLS] my1-6 [SEP] opp1-6 [SEP]
            labels: [batch, seq_len]
                    NOT_SELECTED=0, SELECTED=1, LEAD=2, IGNORE=-100

        Returns:
            loss: 分類損失
            logits: [batch, seq_len, num_labels]
        """
        hidden_states = self.bert(input_ids, attention_mask, token_type_ids)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        output = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, self.config.num_selection_labels), labels.view(-1)
            )
            output["loss"] = loss

        return output

    def predict_selection(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        選出を予測する

        Returns:
            my_selection_probs: [batch, 6, 3] 自チーム各ポケモンの選出確率
            opp_selection_probs: [batch, 6, 3] 相手チーム各ポケモンの選出確率
            my_lead_probs: [batch, 6] 自チーム先発確率
            opp_lead_probs: [batch, 6] 相手先発確率
        """
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask, token_type_ids)
            logits = output["logits"]  # [batch, 15, 3]

            # 位置: [CLS]=0, my1-6=1-6, [SEP]=7, opp1-6=8-13, [SEP]=14
            my_logits = logits[:, 1:7, :]  # [batch, 6, 3]
            opp_logits = logits[:, 8:14, :]  # [batch, 6, 3]

            my_probs = F.softmax(my_logits, dim=-1)
            opp_probs = F.softmax(opp_logits, dim=-1)

            # 先発確率 = LEAD確率 / (LEAD + SELECTED の合計)で正規化
            my_selected_mask = my_probs[:, :, 1] + my_probs[:, :, 2]  # SELECTED + LEAD
            my_lead_probs = my_probs[:, :, 2] / (my_selected_mask + 1e-8)

            opp_selected_mask = opp_probs[:, :, 1] + opp_probs[:, :, 2]
            opp_lead_probs = opp_probs[:, :, 2] / (opp_selected_mask + 1e-8)

        return {
            "my_selection_probs": my_probs,
            "opp_selection_probs": opp_probs,
            "my_lead_probs": my_lead_probs,
            "opp_lead_probs": opp_lead_probs,
        }

    @classmethod
    def from_pretrained_mlm(
        cls, mlm_model: PokemonBertForMLM, config: Optional[PokemonBertConfig] = None
    ) -> "PokemonBertForTokenClassification":
        """MLM事前学習済みモデルから初期化"""
        if config is None:
            config = mlm_model.config

        model = cls(config)
        # BERTエンコーダーの重みをコピー
        model.bert.load_state_dict(mlm_model.bert.state_dict())
        return model
