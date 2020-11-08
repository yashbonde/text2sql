"""model for text2sql
03.11.2020 - @yashbonde"""

from tabulate import tabulate


from transformers import DistilBertModel
from transformers.modeling_gpt2 import Attention, MLP

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelConfig:
    vocab_size = 30522
    n_positions = 1024
    n_ctx = 1024
    dim = 768
    n_embd = 768
    hidden_dim = 3072
    n_layers = 6
    n_head = 12
    n_inner = None
    activation_function = "gelu_new"
    activation = 'gelu'
    dropout = 0.1
    resid_pdrop = 0.1
    attention_dropout = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    qa_dropout = 0.1
    seq_classif_dropout = 0.2
    layer_norm_epsilon = 0.00001
    initializer_range = 0.02
    pad_token_id = 0
    max_position_embeddings = 512
    sinusoidal_pos_embds = False

    def __init__(self, **kwargs):
        """This is config file that is a mixture of GPTConfig, and Distill-BERT Config"""
        self.attrs = [
            "vocab_size",
            "n_positions",
            "n_ctx",
            "dim",
            "n_embd",
            "hidden_dim",
            "n_layers",
            "n_head",
            "n_inner",
            "activation_function",
            "activation",
            "dropout",
            "resid_pdrop",
            "attention_dropout",
            "embd_pdrop",
            "attn_pdrop",
            "qa_dropout",
            "seq_classif_dropout",
            "layer_norm_epsilon",
            "initializer_range",
            "pad_token_id",
            "max_position_embeddings",
            "sinusoidal_pos_embds",
        ]
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)
        self.attrs = list(set(self.attrs))

    def __repr__(self):
        return "---- DATASET CONFIGURATION ----\n" + \
            tabulate(
                [(k, getattr(self, k)) for k in list(set(self.attrs))],
                headers=["key", "value"],
                tablefmt="psql"
            )


class QuestionsEncoder(nn.Module):
    def __init__(self, config):
        self.l = None
    
        # make the bert model and freeze the layer
        model = DistilBertModel.from_pretrained('distilbert-base-cased', return_dict=True)
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False
        self.bert_model = model

        # incase a head layer is to be added
        if self.config.n_embd != 768:
            self.l = nn.Linear(768, config.n_embd)

    def forward(self, x):
        bert_out = self.bert_model.forward(x).last_hidden_variables
        if self.l:
            bert_out = self.l(bert_out)
        return bert_out


class DBEncoder(nn.Module):
    def __init__(self, config, embedding):
        self.config = config
        self.embedding = embedding

    def _convert_torch_geometric_to_normal(self, x):
        return None

    def forward(self, x):
        embeds = self.embedding(x)
        out = self._convert_torch_geometric_to_normal(out)
        return out

class TransformerDecoderBlock(nn.Module):
    def __init__(self, config):
        self.config = config
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd

        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.casual_attention = Attention(config.n_embd, config.n_ctx, config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2_mlp = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp2 = MLP(config.n_embd, inner_dim)
        self.question_attention = Attention(config.n_embd, config.n_ctx, config, is_cross_attention=True)
        self.ln3 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln3_mlp = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp3 = MLP(config.n_embd, inner_dim)
        self.db_attention = Attention(config.n_embd, config.n_ctx, config, is_cross_attention=True)
    
    def forward(self,
                sql_hidden,
                graph_embed,
                graph_mask,
                question_embed,
                question_mask,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                use_cache=False,
                output_attentions=False):
        # layer 1 of self attention
        out = self.casual_attention(
            self.ln1(sql_hidden),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        out = out[0] + sql_hidden

        # layer 2 of cross attention with questions
        question_cross_attention = self.question_attention(
            self.ln2(out),
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=question_embed,
            encoder_attention_mask=question_mask,
            output_attentions=output_attentions,
        )
        out = question_cross_attention[0] + out # residual connection
        out = self.mlp2(self.ln2_mlp(out)) # layer norm and MLP

        # layer 3 of cross attention with DB
        db_cross_attention = self.db_attention(
            self.ln3(out),
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=graph_embed,
            encoder_attention_mask=graph_mask,
            output_attentions=output_attentions,
        )
        out = db_cross_attention[0] + out # residual connection
        out = self.mlp3(self.ln3_mlp(out)) # layer norm and MLP
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, config, embedding):
        self.config = config
        self.embedding = embedding
        self.h = nn.ModuleList([TransformerDecoderBlock(config) for i in range(config.n_layers)])

    def forward(self, sql_query, graph_embed, sequence_embed):
        out = self.embedding(sql_query)
        for layer in self.h:
            out = layer(out, graph_embed, sequence_embed)
        return out


class Text2SQLModel(nn.Module):
    """This is the model for text2sql. It has a graph_encoder and sequence_encoder for encoding
    the given data and uses a modified conventional transformer decoder for decoding."""
    def __init__(self, config):
        self.graph_encoder = DBEncoder(config)
        self.sequence_encoder = QuestionsEncoder(config)
        self.np_bert_embedding = self.sequence_encoder.bert_model.get_input_embeddings().weights.numpy()
        self.sql_decoder = TransformerDecoder(config)

    def forward(self, db_input, question, sql_query, y = None):
        graph_embed = self.graph_encoder(db_input)
        sequence_embed = self.sequence_encoder(question)
        out = self.sql_decoder(sql_query, graph_embed, sequence_embed)
        output = [out]
        if y is not None:
            loss_ft = torch.nn.CrossEntropyLoss()
            loss = loss_ft(out, y)
            output.append(loss)
        return output
