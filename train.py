"""going by o2f format and using huggingface library
15.09.2020 - @yashbonde"""

import os
from types import SimpleNamespace
from argparse import ArgumentParser

import sentencepiece as sp

# from trainer import *
# from data import StdzDataset3, StdzDataset2, StdzDataset, DatasetConfig, load_tokenizer
# from transformers import GPT2Config, GPT2LMHeadModel
# from model import GPT2Config, GPT2LMHeadModel

from text2sql.data import T2SDataset, T2SDatasetConfig
from text2sql.model import Text2SQLModel, Text2SQLModelConfig
from text2sql.trainer import *


# --- arguments
args = ArgumentParser(description="GPT based standardisation methods")

# --- paths
args.add_argument("--save_folder", default = "models", type = str, help = "folder to save all models")
args.add_argument("--name", type = str, help = "name of this particular model")
args.add_argument("--schema_file", type = str, help = "path to schema file",
                  default="/Users/yashbonde/Desktop/AI/text2sql/fdata/all_schema.json")
args.add_argument("--questions_tsv", type = str, help = "path to text/sql tsv",
                  default="/Users/yashbonde/Desktop/AI/text2sql/fdata/all_questions.tsv")
args.add_argument("--tokenizer_path", type = str, help = "path to sentencepiece model file",
                  default="/Users/yashbonde/Desktop/AI/text2sql/data/model.model")
args.add_argument("--seed", default = None, type = int, help = "seed value for training")

# --- arch
args.add_argument("--n_embd", default = 144, type = int, help = "Embedding Dim")
args.add_argument("--n_decoder_layers", default = 3, type = int, help = "Num Decoder layers")
args.add_argument("--n_sent_layers", default = 3, type = int, help = "Num layers for sentence encoder")
args.add_argument("--n_db_layers", default = 3, type = int, help = "Num layers for DB encoder")
args.add_argument("--n_head", default = 6, type = int, help = "Num Heads")
args.add_argument("--maxlen", default = 200, type = int, help = "Maximum length of decoder")

# --- data
args.add_argument("--mult", default = 3, type = int, help = "Size of dataset")
args.add_argument("--pf", default = 0.6, type = float, help = "Probability of using fields in training sequence")
args.add_argument("--fmax", default = 0.8, type = float, help = "Max fields probability")
args.add_argument("--fmin", default = 0.1, type = float, help = "Min fields probability")

# --- trainer
args.add_argument("--n_epochs", default = 100, type = int, help = "Number of epochs to train")
args.add_argument("--batch_size", default = 200, type = int, help = "Mini-Batch Size")
args.add_argument("--lr", default = 1e-3, type = float, help = "Learning Rate")
args.add_argument("--sample_every", default = 5, type = int, help = "After t")
args.add_argument("--train_ratio", default = 0.9, type = float, help = "Ratio of train data, rest is testing")
args.add_argument("--beta1", default = 0.9, type = float, help = "Adam.beta1")
args.add_argument("--beta2", default = 0.95, type = float, help = "Adam.beta2")
args.add_argument("--grad_norm_clip", default = 1.0, type = float, help = "Adam.beta2")

args.add_argument("--patience", default = 6, type = int, help = "training stops after patience runs out")

# --- parse and add more
args = args.parse_args()
tb_path = os.path.join(args.save_folder, args.name)
ckpt_path = os.path.join(tb_path, f"{args.name}.pt")
args = SimpleNamespace(**vars(args), ckpt_path = ckpt_path, tb_path = tb_path)

# make folders
os.makedirs(args.save_folder, exist_ok=True)
os.makedirs(args.tb_path, exist_ok=True)

# DataSet
datasetConf = T2SDatasetConfig(
    schema_file=args.schema_file,
    questions_file=args.questions_tsv,
    maxlen=args.maxlen,
    tokenizer_path=args.tokenizer_path
)
dtrain = T2SDataset(config=datasetConf, mode="train")
dtest = T2SDataset(config=datasetConf, mode="test")

# Model
modelConfig = Text2SQLModelConfig(
    vocab_size=datasetConf.tokenizer.vocab_size(),
    n_embd=args.n_embd,
    maxlen=args.maxlen,
    n_decoder_layers=args.n_decoder_layers,
    n_sent_layers=args.n_sent_layers,
    n_db_layers=args.n_db_layers,
    n_head=args.n_head,
)
model = Text2SQLModel(modelConfig)

# Trainer
trainConfig = TrainerConfig(
    lr=args.lr,
    max_epochs=args.n_epochs,
    batch_size=args.batch_size,
    betas=(args.beta1, args.beta2),
    grad_norm_clip=args.grad_norm_clip,
    sample_every=args.sample_every,
    num_batch=(len(dtrain) // args.batch_size) + int(len(dtrain) % args.batch_size != 0),
    patience=args.patience,
    tb_path=args.tb_path,
)

print(modelConfig)
print(datasetConf)
print(trainConfig)
trainer = Trainer(model, dtrain, dtest, trainConfig)
trainer.train()
