# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
file to finetune / train from scratch the GPT2 model for Text2SQL dataset
@yashbonde - 07.08.2020 (I love holidays)
"""
import os
from argparse import ArgumentParser
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

if __name__ == "__main__":
    # load user args
    args = ArgumentParser(description="Train GPT2 model on t2sql corpus")
    args.add_argument("--tf", type=str, choices=["t", "f"],
                      help="Either to train the model from scratch (t) or finetune (f)")

    # data args
    args.add_argument("--pairs", type=str, default="t2sql_pairs.tsv",
                      help="path to pairs dump")
    args.add_argument("--tables", type=str, default="t2sql_tables.tsv",
                      help="path to tables lm dump")
    args.add_argument("--tokenizer", type=str, default=None,
                      help="If training from scratch provide sentencepeice tokenizer path")

    # train args
    args.add_argument("--num_epochs", type=int, default=20,
                      help="Number of epochs to train / finetune")
    args.add_argument("--save_folder", type=str,
                      default="models", help="Folder to save model to")
    args.add_argument("--model", type=str, default="t2sql_gpt",
                      help="Saved model to have filepath `<model>.pt`")
    args.add_argument("--save_every", type=int, default=1,
                      help="Save model every this epoch")
    args.add_argument("--tensorboard", nargs="?", default=True,
                      help="If passed, prepares tensorbaord summaries")

    # model config defaults set to one from GPT2Config
    args.add_argument("--gptmodel", default="gpt2", type=str,
                      help="GPT model to use for finetuning")
    args.add_argument("--maxlen", default=1024, type=int,
                      help="Maximum sequence length")
    args.add_argument("--n_embd", default=768, type=int,
                      help="Embedding Dimension of model")
    args.add_argument("--n_layer", default=12, type=int,
                      help="Number of layers in the model")
    args.add_argument("--n_head", default=12, type=int,
                      help="Number of attention heads in the model")
    args.add_argument("--activation_function", default="gelu_new",
                      type=str, help="Activation function to use in between the layers")
    args.add_argument("--resid_pdrop", default=0.1, type=float,
                      help="Residual connection dropout probability")
    args.add_argument("--embd_pdrop", default=0.1, type=float,
                      help="Embedding connection dropout probability")
    args.add_argument("--attn_pdrop", default=0.1, type=float,
                      help="Attention connection dropout probability")
    args.add_argument("--layer_norm_epsilon", default=0.00001,
                      type=float, help="Layer norm epsilon value")
    args.add_argument("--initializer_range", default=0.02,
                      type=float, help="Range for initializer")
    args.add_argument("--summary_type", default="cls_index",
                      type=str, help="summary_type in GPTConfig")
    args.add_argument("--summary_use_proj", default=True,
                      type=bool, help="summary_use_proj in GPTConfig")
    args.add_argument("--summary_activation", default=None,
                      type=bool, help="summary_activation in GPTConfig")
    args.add_argument("--summary_proj_to_labels", default=True,
                      type=bool, help="summary_proj_to_labels in GPTConfig")
    args.add_argument("--summary_first_dropout", default=0.1,
                      type=float, help="summary_first_dropout in GPTConfig")
    args = args.parse_args()

    # make dir if present
    os.makedirs(args.save_folder, exist_ok=True)

    NEW_TRAIN = {"t": True, "f": False}[args.tf]
    if NEW_TRAIN:
        assert (args.tokenizer is not None, "When training from scratch need "
            "to provide sentencepiece tokenizer path")
        print("🔋 Training the model from scratch")
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor()
        # here we assume that you have already added the special token from the dataset
        with open(args.tokenizer.replace(".model", ".vocab"), "r") as f:
            vocab_size = len(f.readlines())

        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=args.maxlen,
            n_ctx=args.maxlen,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            activation_function=args.activation_function,
            resid_pdrop=args.resid_pdrop,
            embd_pdrop=args.embd_pdrop,
            attn_pdrop=args.attn_pdrop,
            layer_norm_epsilon=args.layer_norm_epsilon,
            initializer_range=args.initializer_range,
            summary_type=args.summary_type,
            summary_use_proj=args.summary_use_proj,
            summary_activation=args.summary_activation,
            summary_proj_to_labels=args.summary_proj_to_labels,
            summary_first_dropout=args.summary_first_dropout,
            bos_token_id=tokenizer.bos_id(),
            eos_token_id=tokenizer.eos_id(),
        )
        
        model = GPT2LMHeadModel(config)
    else:
        print("🔋 Finetuning model from huggingface's transformers")
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["[table]", "[col]", "[question]", "[query]"]
        })
        model = GPT2LMHeadModel.from_pretrained(args.gptmodel)
        model.resize_token_embeddings(len(tokenizer))

    # now we load the dataset and proceed to training
    model.
    
