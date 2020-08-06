"""
This file converts the dataset sentences to my format to be used for 
langauge modelling and use GPT insted of BERT models.


# SParC: Cross-Domain Semantic Parsing in Context
=================================================
Each file in train.json and dev.json contains the following fields:
```
question: the natural language question
    question_toks: the natural language question tokens
    database_id: the database id to which this interaction is addressed.
    interaction: the query interaction including multiple DB query questions.
        For each question in the interaction, it includes:
    utterance: the natural language question
    utterance_toks: the natural language question tokens
    query: the SQL query corresponding to the question.
    sql: parsed results of this SQL query using process_sql.py. Please refer to
        the Spider Github page for the detailed documentation.
final: the final interaction query goal
    utterance: the natural language question of the final interaction goal
    query: the SQL query corresponding to the final interaction goal.
```

# Spider: A Large-Scale Human-Labeled Dataset for Complex and
Cross-Domain Semantic Parsing and Text-to-SQL Task
==================================================
Each file in train.json and dev.json contains the following fields:
```
question: the natural language question
question_toks: the natural language question tokens
db_id: the database id to which this question is addressed.
query: the SQL query corresponding to the question.
query_toks: the SQL query tokens corresponding to the question.
sql: parsed results of this SQL query using process_sql.py. Please refer to
    parsed_sql_examples.sql in thepreprocess directory for the detailed documentation.
```


# Tables
========
tables.json contains the following information for each database:
```
db_id: database id
table_names_original: original table names stored in the database.
table_names: cleaned and normalized table names. We make sure the
    table names are meaningful. [to be changed]
column_names_original: original column names stored in the database.
    Each column looks like: [0, "id"]. 0 is the index of table names in
    table_names, which is city in this case. "id" is the column name.
column_names: cleaned and normalized column names. We make sure the column
    names are meaningful. [to be changed]
column_types: data type of each column
foreign_keys: foreign keys in the database. [3, 8] means column indices
    in the column_names. These two columns are foreign keys of two different tables.
primary_keys: primary keys in the database. Each number is the index of column_names.
```


# CoSQL: A Conversational Text-to-SQL Challenge Towards
Cross-Domain Natural Language Interfaces to Databases
=====================================================

NO INFORMATION GIVEN ABOUT THIS ONE, BUT WE CAN STILL GET [table], [NL], [QUERY] triplets
"""


import json
from argparse import ArgumentParser

args = ArgumentParser(description= "This file converts the dataset"
                    " sentences to my format to be used for "
                    "langauge modelling and use GPT insted of BERT models.")
args.add_argument("--pairs", type = str, default = "t2sql_pairs.tsv",
    help = "path to pairs dump")
args.add_argument("--tables", type=str, default="t2sql_tables.tsv",
    help = "path to tables lm dump")
args.add_argument("--fresh-tokenizer", nargs='?', type = bool, default = True,
    help = "if passed create a new sentencepiece tokenizer model")
args = args.parse_args()

# paths to main files
OTHER_FILE = "spider/train_others.json"
SPIDER_FILE = "spider/train_spider.json"
SPARC_FILE = "sparc/train.json"
COSQL_FILE = "cosql_dataset/cosql_all_info_dialogs.json"

# files containing tables info
SPIDER_TABLES = "spider/tables.json"
SPARC_TABLES = "sparc/tables.json"
COSQL_TABLES = "cosql_dataset/tables.json"

# spider dataset already has sql files that we can read from to tokenize
SPIDER_SQL_TRAIN = "spider/train_gold.sql"
SPIDER_SQL_DEV = "spider/dev_gold.sql"


