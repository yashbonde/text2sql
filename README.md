<img src="assets/header.png">

# Text2SQL

How many times have you pulled your hair apart writing a SQL query, now use natural language to convert to appropriate SQL and save your precious hair.

It's a bitch to train large networks, I am thinking of porting the model to tensorflow and running it on TPUs, any help would be welcome!

Install all the dependencies before running, see [requirements](requirements.txt).

## Parsing

In order to convert JSON objects into langauge models use the [`parse_to_lm.py`](./parse_to_lm.py), usage:
```
➜  text2sql git:(master) ✗ python3 parse_to_lm.py --help

This file converts the dataset sentences to my format to be used for langauge
modelling and use GPT insted of BERT models.

optional arguments:
  -h, --help            show this help message and exit
  --pairs PAIRS         path to pairs dump
  --tables TABLES       path to tables lm dump
  --dev-pairs DEV_PAIRS
                        path to dev pairs dump
  --fresh-tokenizer [FRESH_TOKENIZER]
                        if passed create a new sentencepiece tokenizer model.
                        Change args from file.
  --corpus CORPUS       what will be the file to feed to tokenizer.
```

## Trainer File

Once the parsing is completed we can then start to train / finetune GPT2 Model. If `finetune` option is chosen we use huggingface's `transformers.GPT2LMHeadModel`, else a new model is created based on your arguments. For arguments refer below:
```
➜  text2sql git:(master) ✗ python3 train_gpt2.py --help
Train GPT2 model on t2sql corpus

optional arguments:
  -h, --help            show this help message and exit
  --tf {t,f}            Either to train the model from scratch (t) or finetune
                        (f)
  --pairs PAIRS         path to pairs dump
  --tables TABLES       path to tables lm dump
  --num_epochs NUM_EPOCHS
                        Number of epochs to train / finetune
  --save_folder SAVE_FOLDER
                        Folder to save model to
  --model MODEL         Saved model to have filepath `<model>.pt`
  --save_every SAVE_EVERY
                        Save model every this epoch
  --tensorboard [TENSORBOARD]
                        If passed, prepares tensorbaord summaries
  --maxlen MAXLEN       Maximum sequence length
  --n_embd N_EMBD       Embedding Dimension of model
  --n_layer N_LAYER     Number of layers in the model
  --n_head N_HEAD       Number of attention heads in the model
  --activation_function ACTIVATION_FUNCTION
                        Activation function to use in between the layers
  --resid_pdrop RESID_PDROP
                        Residual connection dropout probability
  --embd_pdrop EMBD_PDROP
                        Embedding connection dropout probability
  --attn_pdrop ATTN_PDROP
                        Attention connection dropout probability
  --layer_norm_epsilon LAYER_NORM_EPSILON
                        Layer norm epsilon value
  --initializer_range INITIALIZER_RANGE
                        Range for initializer
  --summary_type SUMMARY_TYPE
                        summary_type in GPTConfig
  --summary_use_proj SUMMARY_USE_PROJ
                        summary_use_proj in GPTConfig
  --summary_activation SUMMARY_ACTIVATION
                        summary_activation in GPTConfig
  --summary_proj_to_labels SUMMARY_PROJ_TO_LABELS
                        summary_proj_to_labels in GPTConfig
  --summary_first_dropout SUMMARY_FIRST_DROPOUT
                        summary_first_dropout in GPTConfig
  --bos_token_id BOS_TOKEN_ID
                        beggining of statement ID in vocabulary
  --eos_token_id EOS_TOKEN_ID
                        end of statement ID in vocabulary
```

## Datasets

Using [CoSQL](https://yale-lily.github.io/cosql), [Spider](https://yale-lily.github.io/spider), [Sparc](https://yale-lily.github.io/sparc) datasets, credit to the authors. There are a couple of things to note, we have in total 178 tables, but only 166 tables in training date and dev set has 20 tables.


## License

The code I am using is under MIT License and `transformers` code is under Apache License, Version 2.0
