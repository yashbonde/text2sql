"""model for text2sql
03.11.2020 - @yashbonde"""

import numpy as np
from tabulate import tabulate
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.activations import ACT2FN

# the code below is only a slighlty modified version from huggingface.
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)

        # print("((((", w.size(), attention_mask.size())

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # print(w.size(), v.size())

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            assert hasattr(
                self, "q_attn"
            ), "If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        # print(query.size(), key.size(), value.size())

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, config, n_ctx, add_cross_attention = False, scale=False):
        super().__init__()
        hidden_size= config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

    def forward(self, x):
        # this was not taking key word arguments in Sequential so need to pass around a tuple
        # so now I understood why huggingface coded by passing around lists, stupid!
        type_ =x[0]
        # print("^^^^", type_, len(x))
        if type_ in ["encoder", "self"]:
            (hidden_states, attention_mask) = x[1:]
        else:
            (hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask) = x[1:]

        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        hidden_states = attn_output + hidden_states # residual connection

        if type_ == "decoder":
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        outputs = [hidden_states] + outputs
        if type_ in ["encoder", "self"]:
            out = (type_, hidden_states, attention_mask)
        else:
            out = (type_, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        blocks = []
        for i in range(config.n_decoder_layers):
            blocks.append(Block(config, n_ctx = config.maxlen,  add_cross_attention = False)) # casual
            blocks.append(Block(config, n_ctx = config.maxlen, add_cross_attention = True)) # sent
            blocks.append(Block(config, n_ctx = config.maxlen, add_cross_attention = True)) # db
        self.blocks = nn.Sequential(*blocks)
        self.ln = nn.LayerNorm(config.n_embd, eps = config.layer_norm_epsilon)

    def forward(self, x):
        # this was not taking key word arguments in Sequential so need to pass around a tuple
        (hidden_states_sql, attention_mask_sql, hidden_states_sent, attention_mask_sent, hidden_states_db, attention_mask_db) = x
        
        hidden_states = hidden_states_sql
        l = 0
        for i, block in enumerate(self.blocks):
            if l == 0: # casual attention
                outputs = block(("self", hidden_states, attention_mask_sql))
                l = 1
            elif l == 1: # sentence attention
                outputs = block(("decoder", hidden_states, attention_mask_sql, hidden_states_sent, attention_mask_sent))
                l = 2
            else: # db attention
                outputs = block(("decoder", hidden_states, attention_mask_sql, hidden_states_db, attention_mask_db))
                l = 0
            hidden_states = outputs[1]
        hidden_states_sql = hidden_states
        return (hidden_states_sql, attention_mask_sql,
                hidden_states_sent, attention_mask_sent,
                hidden_states_db, attention_mask_db)


class Text2SQLModel(nn.Module):
    def __init__(self, config):
        super(Text2SQLModel, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.wte_sent = nn.Embedding(config.maxlen, config.n_embd)
        self.wte_db = nn.Embedding(config.maxlen, config.n_embd)
        self.wte_sql = nn.Embedding(config.maxlen, config.n_embd)

        self.sentence_encoder = nn.Sequential(*[
            Block(config, n_ctx=config.maxlen, add_cross_attention=False)
            for _ in range(config.n_sent_layers)
        ])
        self.db_encoder = nn.Sequential(*[
            Block(config, n_ctx=config.maxlen, add_cross_attention=False)
            for _ in range(config.n_db_layers)
        ])
        self.decoder = Decoder(config)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        self.apply(self._init_weights)
        print("number of parameters:", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # # separate out all parameters to those that will and won't experience regularizing weight decay
        # decay = set()
        # no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        # blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        # for mn, m in self.named_modules():
        #     for pn, p in m.named_parameters():
        #         # print(mn, "--", pn)
        #         fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
        #         print(fpn, type(m))
        #         if fpn.endswith('bias'):
        #             # all biases will not be decayed
        #             no_decay.add(fpn)
        #             print(fpn, "--", 1)
        #         elif fpn.endswith('weight') and isinstance(m, whitelist_weight_modules):
        #             # weights of whitelist modules will be weight decayed
        #             decay.add(fpn)
        #             print(fpn, "--", 2)
        #         elif fpn.endswith('weight') and isinstance(m, blacklist_weight_modules):
        #             # weights of blacklist modules will NOT be weight decayed
        #             no_decay.add(fpn)
        #             print(fpn, "--", 3)
        #         print()

        # # validate that we considered every parameter
        # param_dict = {pn: p for pn, p in self.named_parameters()}
        # inter_params = decay & no_decay
        # union_params = decay | no_decay
        # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )

        # # create the pytorch optimizer object
        # optim_groups = [
        #     {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
        #     {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        # ]
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.lr, betas=train_config.betas)
        return optimizer

    def get_position_ids(self, input, past_length, device):
        input_shape = input.size()
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        return position_ids

    def encoder_fn(self, sent, db, sent_attn, db_attn, device):
        sent = self.embedding(sent) + self.wte_sent(self.get_position_ids(sent, past_length = 0, device = device))
        db = self.embedding(db) + self.wte_db(self.get_position_ids(db, past_length = 0, device = device))
        sent_hidden_states = self.sentence_encoder(("encoder", sent, sent_attn),)[1]
        db_hidden_states = self.db_encoder(("encoder", db, db_attn),)[1]
        return SimpleNamespace(
            sent_hidden_states=sent_hidden_states,
            db_hidden_states=db_hidden_states,
            sent_attn=sent_attn,
            db_attn=db_attn
        )

    def decoder_fn(self, enc_out, sql, sql_attn, device):
        sql = self.embedding(sql) + self.wte_sql(self.get_position_ids(sql, past_length = 0, device = device))
        sql_output = self.decoder((sql, sql_attn, enc_out.sent_hidden_states,
                                   enc_out.sent_attn, enc_out.db_hidden_states, enc_out.db_attn),)[0]
        sql_output = self.lm_head(sql_output)
        return sql_output

    def forward(self, sql_ids, sent, db, sql_attn, sent_attn, db_attn, labels=None, past_length=0, device="cpu"):
        # make the embeddings
        sql = self.embedding(sql_ids) + self.wte_sql(self.get_position_ids(sql_ids, past_length = past_length, device = device))
        sent = self.embedding(sent) + self.wte_sent(self.get_position_ids(sent, past_length = 0, device = device))
        db = self.embedding(db) + self.wte_db(self.get_position_ids(db, past_length = 0, device = device))

        # get hidden_states for sentence_encoder
        sent_hidden_states = self.sentence_encoder(("encoder", sent, sent_attn),)[1]
        db_hidden_states = self.db_encoder(("encoder", db, db_attn),)[1]
        sql_output = self.decoder((sql, sql_attn, sent_hidden_states, sent_attn, db_hidden_states, db_attn),)[0]
        sql_output = self.lm_head(sql_output)
        output = [sql_output]

        if labels is not None:
            labels = labels.contiguous()
            logits = sql_output.contiguous()
            
            # loss_fct = nn.CrossEntropyLoss(reduction="none")
            # loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss_fct = nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # # get the indexes where the labels are not [PAD]
            # non_pad_mask = sql_attn[:, :, 0, 1:].contiguous().view(-1) == 0
            # print(loss.size(), non_pad_mask.size())
            # non_pad_loss = loss[non_pad_mask]
            # print("non_pad_loss", non_pad_loss)
            # loss = non_pad_loss.mean()
            output = [loss] + output

        return output


class Text2SQLModelConfig():
    vocab_size = 5012
    n_embd = 256
    maxlen = 128
    n_decoder_layers = 2
    n_sent_layers = 3
    n_db_layers = 3
    n_head = 8
    n_inner = None
    activation_function = "gelu_new"
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 0.00001
    initializer_range = 0.02

    def __init__(self, **kwargs):
        self.attrs = [
            "vocab_size",
            "n_embd",
            "maxlen",
            "n_decoder_layers",
            "n_sent_layers",
            "n_db_layers",
            "n_head",
            "n_inner",
            "activation_function",
            "resid_pdrop",
            "embd_pdrop",
            "attn_pdrop",
            "layer_norm_epsilon",
            "initializer_range",
        ]

        for k, v in kwargs.items():
            self.attrs.append(k)
            setattr(self, k, v)

    def __repr__(self):
        kvs = [(k, f"{getattr(self, k)}") for k in sorted(list(set(self.attrs)))]
        return tabulate(kvs, ["argument", "value"], tablefmt="psql")

# ====== Sampling Utils ====== #
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -1e6
    return out


@torch.no_grad()
def sample(model, sent, sent_attn, db, db_attn, t, sql_str = None, device="cpu", steps=50, temperature=50, top_k=None):
    model.eval()
    sent = sent.view(-1, sent.size(0))
    db = db.view(-1, db.size(0))
    sent_attn = sent_attn.view(-1, *sent_attn.size())
    db_attn = db_attn.view(-1, *db_attn.size())
    # print(sent.size(), db.size(), sent_attn.size(), db_attn.size())
    enc_out = model.encoder_fn(sent, db, sent_attn, db_attn, device=device)

    # convert string to sql_tokens
    if sql_str is not None:
        sql = torch.from_numpy(np.asarray([t.encode(sql_str)])).long()
    else:
        sql = torch.from_numpy(np.asarray([t.bos_id()])).view(-1, 1).long()
    sql = sql.to(device)

    # final sequence
    out = []

    for k in range(steps):
        x = sql if sql.size(1) < model.config.maxlen else sql[:, -model.config.maxlen:]

        sql_attn = np.ones((len(sql[0]), len(sql[0])))
        sql_attn = sql_attn - np.triu(sql_attn, k = 1)
        sql_attn = 1 - sql_attn
        sql_attn = sql_attn * -1e6
        sql_attn = torch.from_numpy(sql_attn.astype(np.float32)).to(device).view(1, 1, *sql_attn.shape)

        logits = model.decoder_fn(enc_out, x, sql_attn, device=device)

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

        out.append(t.decode_ids(ix[0].tolist()))

    return " ".join(out)
