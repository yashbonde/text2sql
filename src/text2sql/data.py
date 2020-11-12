"""data management piece
22.09.2020 - @yashbonde"""

import re
import json
import numpy as np
import pandas as pd
import networkx as nx
from tabulate import tabulate
import sentencepiece as spm

import torch
from torch.utils.data import Dataset

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


def get_tokenised_attention_mask(g, t, size = None, inf = 1e6):
    """In method get_db_attention_mask() we do not consider that the tokens
    will have a subword splitting and so the final attention mask will look
    a bit different. This takes care of that by creating mask on subwords
    as well.
    :param g: graph
    :param t: sentencepiece tokenizer
    :param size: dimension of the output attention_mask
    :param inf: what will be the negative infinity value

    NOTE: 11th November, 2020 @yashbonde you ass don't do anymore complicated
    engineering, what you need is more compute not more engineering you ass.
    """
    # att = get_db_attention_mask(g, size = -1)
    ts = []
    sizes = []
    for x in g.nodes().data():
        # we directly call the internal wordpiece_tokenizer, helps in debugging
        tokens = t.encode(" ".join([x[1].get("table"), x[1].get("name")]))
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
    
    if size is not None:
        # convert to required shapes and put in masking values
        fmat = np.zeros((size, size)).astype(int)
        if size < len(tmat):
            tmat = tmat[:size, :size]
        fmat[:tmat.shape[0], :tmat.shape[0]] = tmat
        fmat = 1 - fmat 
        fmat = fmat * -inf
    else:
        fmat = tmat

    return fmat, ts, sum(sizes)


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
        self.queries = df["query"].values # I didn't know .query was a function
        self.db_ids = df.db_id.values

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        config = self.config
        t = self.config.tokenizer

        # prepare the questions
        question = self.questions[index]
        question = [t.bos_id()] + t.encode(question) + [t.eos_id()]
        sent_len = len(question)
        if config.maxlen > len(question):
            question = question + [t.pad_id() for _ in range(config.maxlen - len(question))]
        else:
            question = question[:config.maxlen]
        sent_attn = np.zeros((config.maxlen, config.maxlen)).astype(np.int32)
        sent_attn[:sent_len, :sent_len] = 1
        sent_attn = 1 - sent_attn
        sent_attn = sent_attn * -1e6

        # prepare the DB sequence
        g = self.schemas[self.db_ids[index]]
        db_attn_mat, db_tokens, len_db = get_tokenised_attention_mask(
            g, t, size=config.maxlen)
        db_tokens = [t.bos_id()] + db_tokens + [t.eos_id()]
        if config.maxlen > len(db_tokens):
            db_tokens = db_tokens + [t.pad_id() for _ in range(config.maxlen - len(db_tokens))]
        else:
            db_tokens = db_tokens[:config.maxlen]

        # prepare the sql query
        sql = self.queries[index]
        sql = [t.bos_id()] + t.encode(sql) + [t.bos_id()]
        sql_len = len(sql)
        if config.maxlen > len(sql) + 1:
            sql = sql + [t.pad_id() for _ in range(config.maxlen - len(sql) + 1)]
        else:
            sql = sql[:config.maxlen + 1]
        sql_attn = np.zeros((config.maxlen, config.maxlen)).astype(np.int32)
        sql_attn[:sql_len, :sql_len] = 1
        sql_attn = sql_attn - np.triu(sql_attn, k = 1) # casual masking
        sql_attn = 1 - sql_attn
        sql_attn = sql_attn * -1e6
    
        # create labels
        labels = torch.from_numpy(np.asarray(sql[1:])).long()
        labels[sql_len-1:] = -100 # minus 1 because already shifted

        # create input ids
        sql_ids = torch.from_numpy(np.asarray(sql[:-1])).long()
        sql_ids[sql_len:] = -100

        # return the output dictionary
        return {
            "sql_ids": sql_ids,
            "labels": labels,
            "sent": torch.from_numpy(np.asarray(question)).long(),
            "db": torch.from_numpy(np.asarray(db_tokens)).long(),
            "sql_attn": torch.from_numpy(np.asarray([sql_attn]).astype(np.float32)),
            "sent_attn": torch.from_numpy(np.asarray([sent_attn]).astype(np.float32)),
            "db_attn": torch.from_numpy(np.asarray([db_attn_mat]).astype(np.float32))
        }


class T2SDatasetConfig:
    schema_file = None # json file with schema dump
    questions_file = None # TSV file with questions-sql dump
    maxlen = 400 # maximum length for all is same for simplicity
    # also same size helps fit in the encoder mask as well as the
    # cross attention mask
    tokenizer_path = None

    def __init__(self, **kwargs):
        self.attrs = ["schema_file", "questions_file", "maxlen"]
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(self.tokenizer_path)

    def __repr__(self):
        kvs = [(k, f"{getattr(self, k)}") for k in sorted(list(set(self.attrs)))]
        return tabulate(kvs, ["argument", "value"], tablefmt="psql")
