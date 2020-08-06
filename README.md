# Text2SQL using pretrained models

## Parsing

In order to convert JSON objects into langauge models use the [`parse_to_lm.py`](./parse_to_lm.py), usage:
```
➜  text2sql python3 parse_to_lm.py --help
usage: parse_to_lm.py [-h] [--pairs PAIRS] [--tables TABLES]
                      [--fresh-tokenizer [FRESH_TOKENIZER]]

This file converts the dataset sentences to my format to be used for langauge
modelling and use GPT insted of BERT models.

optional arguments:
  -h, --help            show this help message and exit
  --pairs PAIRS         path to pairs dump
  --tables TABLES       path to tables lm dump
  --fresh-tokenizer [FRESH_TOKENIZER]
                        if passed create a new sentencepiece tokenizer model
```

## Datasets

Using [CoSQL](https://yale-lily.github.io/cosql), [Spider](https://yale-lily.github.io/spider), [Sparc](https://yale-lily.github.io/sparc) datasets. Credit to the authors.