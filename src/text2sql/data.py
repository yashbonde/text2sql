"""data management piece
22.09.2020 - @yashbonde"""

import re
import json
import numpy as np
import pandas as pd
import networkx as nx
from tabulate import tabulate

import torch
from torch.utils.data import Dataset

from transformers import DistilBertTokenizer

# ====== Helper functions ======= #
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
        g.add_node(
            i, id = f"{table}.{name}", name = name, table = table,
            primary = True if (i+1) in primary_keys else False,
            type = column_types[i]
        )

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


def get_db_attention_mask(g, size, device = "cpu", inf = 1e6):
    A = nx.adjacency_matrix(g).todense()
    A = 1 - (A + np.eye(len(A)))  # add self loops
    if size == -1:
        m = A
    else:
        m = np.zeros((size, size))
        m[:len(A), :len(A)] = A  # add to big loop
    m = m * inf
    return torch.from_numpy(m).long().to(device), len(A)

def get_tokenised_attention_mask(g, t, size,inf = 1e6):
    """In method get_db_attention_mask() we do not consider that the tokens
    will have a subword splitting and so the final attention mask will look
    a bit different. This takes care of that by creating mask on subwords
    as well.
    :param g: graph
    :param t: tokenizer
    :param size: dimension of the output attention_mask
    :param inf: what will be the negative infinity value
    """
    # att = get_db_attention_mask(g, size = -1)
    ts = []
    sizes = []
    for x in g.nodes().data():
        # we directly call the internal wordpiece_tokenizer, helps in debugging
        tokens = tokenizer.wordpiece_tokenizer.tokenize(" ".join([x[1].get("table"), x[1].get("name")]))
        sizes.append(len(tokens))
        ts.extend(tokens)

    # get adjacency matrix 
    mat = nx.adjacency_matrix(g).todense()
    mat = mat + np.eye(len(mat)) # add self loops
    mat = mat.tolist()

    # now the code to expand the matrix in place
    tmat = np.zeros((sum(sizes),sum(sizes)))
    tid = 0
    for i in range(len(mat)):
        idx = np.arange(len(mat))[np.asarray(mat[i]) == 1]
        for s in range(sizes[i]):
            for j in idx:
                start = sum(sizes[:j])
                end = sum(sizes[:j+1])
                tmat[tid, start:end] = 1
            tid += 1
    tmat = tmat + tmat.T
    tmat[tmat > 1] = 1
    tmat = tmat.astype(int)
    
    # convert to required shapes and put in masking values
    fmat = np.zeros((size, size)).astype(int)
    fmat[:tmat.shape[0], :tmat.shape[0]] = tmat
    fmat = 1 - fmat 
    fmat = fmat * -inf

    return fmat, sum(sizes)


def format_sql(in_str):
    in_str = in_str.lower()
    for p in re.findall(r"\w+\s+as\s+t\d+", in_str):
        # print(p)
        try:
            table, id = [x.strip() for x in p.split("as")]
        except:
            table, id = [x.strip() for x in p.split(" as ")]
        # replace phrase that contains "<table_name> AS <id>"
        in_str = in_str.replace(p, table)

        # replace the table
        in_str = in_str.replace(id, table)
        in_str = in_str.replace(id.lower(), table)
        in_str = in_str.replace(id.upper(), table)

    # basic cleaning
    in_str = re.sub(r"\s+", " ", in_str)
    in_str = re.sub(r"\"+", '"', in_str)
    return in_str

# ====== Main Class Object ====== #

class T2SDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config

        self._tokenizer = config.tokenizer

        with open(config.schema_file) as f:
            self.schemas = {k:parse_db_to_networkx(v) for k,v in json.load(f).items()}

        with open(config.questions_file) as f:
            df = pd.read_csv(config.questions_file, sep="\t")
            mode = 1 if mode == "train" else 0
            df = df[df.train == mode]

        self.questions = df.question.values
        self.queries = df.query.values
        self.db_ids = df.db_id.values

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        config. = self.config
        t = self.config.tokenizer

        question = self.questions[index]
        query = self.queries[index]
        g = self.db_ids[index]

        # get the DB attention matrix
        db_attn_mat, len = get_tokenised_attention_mask(g, t, size = config.maxlen_db_att)
        question = t(question)
        sql = t(query)
        
        # 

        

        



class T2SDatasetConfig:
    schema_file = None # json file with schema dump
    questions_file = None # TSV file with questions-sql dump
    maxlen_sequence = None # maximum length sequence of SQL
    maxlen_db_att = None # maximum length of db attention sequence

    def __init__(self, **kwargs):
        self.attrs = ["schema_file", "questions_file"]
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __repr__(self):
        kvs = [(k, f"{getattr(self, k)}") for k in sorted(list(set(self.attrs)))]
        return tabulate(kvs, ["argument", "value"], tablefmt="psql")
