"""data management piece
22.09.2020 - @yashbonde"""

import re
import json
import networkx as nx
from tabulate import tabulate
from torch.utils.data import Dataset

def parse_db_to_networkx(db):
    """convert the db to a networkx graph with proper attributes
    
    nodes have features:
      - id: {table}.{name}
      - primary: True/False
      - type: type
    edges have features:
      - 
    """
    columns = db["column_names"][1:] # ignore the attritbute
    table_names = db["table_names"]
    column_types = db["column_types"]
    foreign_keys = db["foreign_keys"]
    primary_keys = db["primary_keys"]
    
    if len(set([x[0] for x in columns])) != len(table_names):
        raise ValueError("More tables given in ")
        
    # make graph
    g = nx.Graph()
    
    # add nodes and data
    for i, c in enumerate(columns):
        name = c[1].replace(" ", "_")
        table = table_names[c[0]]
        g.add_node(i, id = f"{table}.{name}", name = name, table = table, primary = True if (i+1) in primary_keys else False, type = column_types[i])

    # for edges first foriegn keys because simpler
    for (s,t) in foreign_keys:
        g.add_edge(s-1, t-1, foreign = True)
    
    # then those within the table
    for i in range(len(table_names)):
        cols = list(filter(
            lambda c: c[1][0] == i, enumerate(columns)
        ))
        cols = [x[0] for x in cols]
        for i,c in enumerate(cols):
            for cc in cols[i+1:]:
                g.add_edge(c, cc, foreign = False)
    return g


def format_sql(in_str):
    for p in re.findall(r"[a-z_]+\sAS\st\d", in_str):
        table, id = [x.strip() for x in p.split("AS")]
        in_str = in_str.replace(p, table)
        in_str = in_str.replace(id, table)
    in_str = re.sub(r"\s+", " ", in_str)
    return in_str


# class T2SDataset(Dataset):
#     def __init__(self, config):
        
#         data = []
#         with open(config.OTHER_FILE) as f1, \
#             open(config.SPIDER_FILE) as f2, \
#             open(config.SPARC_FILE) as f3, \
#             open(config.COSQL_FILE) as f4:
#             # train_others.json
#             for x in json.load(f1):
#                 data.append((x["question"], x["query"], x["db_id"]))

#             # train_spider.json
#             for x in json.load(f2):
#                 data.append((x["question"], x["query"], x["db_id"]))

#             # sparc/train.json
#             for x in json.load(f3):
#                 data.append((x["final"]["utterance"], x["final"]["query"], x["database_id"]))

#             # cosql_all_info_dialogs.json
#             for x, y in json.load(f4).items():
#                 data.append((y["query_goal"], y["sql"], y["db_id"]))

#         self.qsd = data

#         tables = []
#         with open(config.SPIDER_TABLES) as f1, open(config.SPARC_TABLES) as f2, open(config.COSQL_TABLES) as f3:
#             # spider/tables.json
#             tables.extend(json.load(f1))

#             # sparc/tables.json
#             tables.extend(json.load(f2))

#             # cosql_dataset/tables.json
#             tables.extend(json.load(f3))

#         self.db = [parse_db_to_networkx(x) for x in tables]


#     def __len__(self):
#         return self.

class T2SDatasetConfig:

    # paths to main files
    OTHER_FILE = "/Users/yashbonde/Desktop/AI/text2sql/data/spider/train_others.json"
    SPIDER_FILE = "/Users/yashbonde/Desktop/AI/text2sql/data/spider/train_spider.json"
    SPARC_FILE = "/Users/yashbonde/Desktop/AI/text2sql/data/sparc/train.json"
    COSQL_FILE = "/Users/yashbonde/Desktop/AI/text2sql/data/cosql_dataset/cosql_all_info_dialogs.json"

    # files containing tables info
    SPIDER_TABLES = "/Users/yashbonde/Desktop/AI/text2sql/data/spider/tables.json"
    SPARC_TABLES = "/Users/yashbonde/Desktop/AI/text2sql/data/sparc/tables.json"
    COSQL_TABLES = "/Users/yashbonde/Desktop/AI/text2sql/data/cosql_dataset/tables.json"

    # spider dataset already has sql files that we can read from to tokenize
    SPIDER_SQL_TRAIN = "/Users/yashbonde/Desktop/AI/text2sql/data/spider/train_gold.sql"
    SPIDER_SQL_DEV = "/Users/yashbonde/Desktop/AI/text2sql/data/spider/dev_gold.sql"

    # dev set
    SPIDER_DEV = "/Users/yashbonde/Desktop/AI/text2sql/data/spider/dev.json"
    SPARC_DEV = "/Users/yashbonde/Desktop/AI/text2sql/data/sparc/dev.json"

    def __init__(self, **kwargs):
        self.attrs = []
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        t = "--------- DatasetConfig ---------\n"
        kvs = [(k, f"{getattr(self, k)}") for k in sorted(list(set([
            "OTHER_FILE",
            "SPIDER_FILE",
            "SPARC_FILE",
            "COSQL_FILE",
            "SPIDER_TABLES",
            "SPARC_TABLES",
            "COSQL_TABLES",
            "SPIDER_SQL_TRAIN",
            "SPIDER_SQL_DEV",
            "SPIDER_DEV",
            "SPARC_DEV",
        ] + self.attrs
        )))]
        t = t + tabulate(kvs, ["argument", "value"], tablefmt="psql")
        return t
