"""trying streamlit as usage tool
22.09.2020 - @yashbonde"""

import os
import json
import random
from PIL import Image
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from text2sql.data import parse_db_to_networkx

import streamlit as st

st.write("""
# Text2SQL Converter
Convert your natural language question to SQL queries.

* Check code at my [Github](https://github.com/yashbonde/text2sql)
""")


# paths to main files
OTHER_FILE = "data/spider/train_others.json"
SPIDER_FILE = "data/spider/train_spider.json"
SPARC_FILE = "data/sparc/train.json"
COSQL_FILE = "data/cosql_dataset/cosql_all_info_dialogs.json"

# files containing tables info
SPIDER_TABLES = "data/spider/tables.json"
SPARC_TABLES = "data/sparc/tables.json"
COSQL_TABLES = "data/cosql_dataset/tables.json"

# spider dataset already has sql files that we can read from to tokenize
SPIDER_SQL_TRAIN = "data/spider/train_gold.sql"
SPIDER_SQL_DEV = "data/spider/dev_gold.sql"

# dev set
SPIDER_DEV = "data/spider/dev.json"
SPARC_DEV = "data/sparc/dev.json"

DB_ID = None

# load different dbs
tables = []
with open(SPIDER_TABLES) as f1, open(SPARC_TABLES) as f2, open(COSQL_TABLES) as f3:
    # spider/tables.json
    tables.extend(json.load(f1))
    
    # sparc/tables.json
    tables.extend(json.load(f2))
    
    # cosql_dataset/tables.json
    tables.extend(json.load(f3))

# load questions and corresponding outputs
data = []
with open(OTHER_FILE) as f1, open(SPIDER_FILE) as f2, open(SPARC_FILE) as f3, open(COSQL_FILE) as f4:
    # train_others.json
    for x in json.load(f1):
        data.append((x["question"], x["query"], x["db_id"]))
        
    # train_spider.json
    for x in json.load(f2):
        data.append((x["question"], x["query"], x["db_id"]))

    # sparc/train.json
    for x in json.load(f3):
        data.append((x["final"]["utterance"], x["final"]["query"], x["database_id"]))

    # cosql_all_info_dialogs.json
    for x,y in json.load(f4).items():
        data.append((y["query_goal"], y["sql"], y["db_id"]))

def get_random_db():
    random_table = random.choice(tables)
    global DB_ID
    DB_ID = random_table["db_id"]
    g = parse_db_to_networkx(random_table)
    eattr = nx.get_edge_attributes(g, 'foreign')
    pos = nx.spring_layout(g)

    plt.figure(figsize = (6, 4))
    nx.draw_networkx_nodes(g, pos, )
    nx.draw_networkx_labels(g, pos, nx.get_node_attributes(g, 'id'), font_size="x-small")
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=[k for k,v in eattr.items() if v],
        edge_color="r",
        
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=[k for k,v in eattr.items() if not v],
        edge_color="b",
        
    )
    plt.savefig("_temp.png")

st.write("""
## Problem Statement

**Given a database schema and a question in natural language get the appropriate
schema to get the information needed.**

For this we first go over the data.

### Sample Data

Below is a quick sample of how the DB looks like, `blue` edges represent
edges in the same table while `red` edges represent foreign keys. Click the
`Randomise` button to see another graphs. I am using [CoSQL](https://yale-lily.github.io/cosql),
[Spider](https://yale-lily.github.io/spider), [Sparc](https://yale-lily.github.io/sparc)
datasets.
""")
db_cntr = 0
if st.button('Randomise') or db_cntr == 0:
    get_random_db()

    # load the graph image
    x = Image.open("_temp.png")
    st.image(x, caption = f"Look Ma! Database is a graph. ({DB_ID})", clamp = True)

    # update samples
    data_this_db = list(filter(
        lambda x: x[2] == DB_ID, data
    ))
    st.write(f"from `{DB_ID}` we get following questions:\n\n" +
        "- " + "\n\n- ".join([f"{x[0]} ➡️ `{x[1]}`" for x in data_this_db][:3])
    )
    db_cntr += 1

st.write("""
### Database Schema
Any DB is converted to the graph, it is a combination of nodes and edges where each have a certain property:
```
nodes:
    - table: name of the table it belongs to
    - name: column name
    - type: one of ['boolean', 'time', 'others', 'text', 'number']
    - primary: boolean that tells if this is a primary key

edges:
    - foreign: boolean that tells if this is a foreign edge
```

### Natural Language Questions
We use the 
""")

st.write("""
## Algorithm

We are given database schema defined by $D$, and natural language question $N$.
We first obtain an embedding for database $d = \phi(D)$ and question $t = \\theta(N)$.
Thus we get the input state $s = [d;t]$, where $;$ denotes concatenation.
""")


