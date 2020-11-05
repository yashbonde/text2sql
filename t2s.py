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

* **Author: Yash Bonde**
* Website: [link](https://yashbonde.github.io)
* LinkedIn: [link](https://www.linkedin.com/in/yash-bonde/)
* Twitter: [@bondebhai](https://twitter.com/bondebhai)
* Check code at my [Github](https://github.com/yashbonde/text2sql)
""")


# # with st.spinner('Loading model ...'):
# from transformers import AutoTokenizer, AutoModel
# TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# THETA = AutoModel.from_pretrained("distilbert-base-uncased")

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
We use the [distillbert](https://huggingface.co/distilbert-base-uncased) and you can pass
it any text and see the output logits size.

## Algorithm

To pose any problem as RL you need to have the following setup:

```
<s,a,r,s'> format of data tuple
s: current state
a: action taken
r: reward obtained for taking that action
s': new state model reaches
```

We are given database schema defined by $D$, and natural language question $N$.
We first obtain an embedding for database $d = \phi(D)$ and question $t = \\theta(N)$.
Thus we get the input state $s = [d;t]$, where $;$ denotes concatenation. Now we denote
a function $\pi$ which is the policy network, which predicts the appropriate SQL.
$q = \pi(s)$ 

The main challenge is policy network, is is going to be a traditional Language modeling
LSTM or Transformer. So let us consider the network outputs:

* $\phi(D) \\rightarrow ( [N_{nodes}, E_{graph}], [1, E_{graph}] )$
* $\\theta(N) \\rightarrow [N_{tokens}, 768]$, we can possibly reduce this further by
max-pooling over the sequence as $[N_{tokens}, 768] \\rightarrow [1, 768]$

**23rd September, 2020**: Okay So I think I have a solution, the primary challenge has
been the definition of action space. The action space has all the vocabulary of SQL
commands + two special tags `<col>` and `<table>`. `<col>` tells that model will
select a column from node embeddings (dot product + softmax) and `table` will tell
to select table from node embeddings (dot product + sigmoid).

For this to work hwever we will have to modify the equations given in the dataset as
```sql
SELECT
    T2.name FROM Friend AS T1
    JOIN Highschooler AS T2 ON T1.student_id = T2.id
    WHERE T2.grade > 5
    GROUP BY T1.student_id
    HAVING count(*) >= 2
```
to something like the one below
```sql
SELECT 
    Highschooler.name FROM Friend
    JOIN Highschooler ON Friend.student_id = Highschooler.id
    WHERE Highschooler.grade > 5
    GROUP BY Friend.student_id
    HAVING count(*) >= 2
```

The idea with initial model was a complicated graph based approach but now I
am considering a much simpler model. Model is a simple Transformer where we have
two different encoder structures:
* BERT as question encoder
* Message-passing GNN as DB encoder

These two combined will be fed into a conventional transformer decoder.
""")

# # My assumption is that the dataset was created with langauge models in mind, however in practice
# # direclty pointing out the column is a better solution design.
# # pass the database through a graph encoder to get node and graph embeddings
# DB --> [GNN] ---> (node embedding) [N_1, E_1]        ... A
#               \-> (graph embedding) [1, E_1]         ... B

# # pass the natural language question, through any LM like BERT
# Q ---> [BERT] --> (token level embedding) [N_2, E_2] ... C

# # --- undecided --- 
# # concatenate the graph embedding and natural language embedding
# [B+C] --> [N_2, E_1 + E_2] ... D

# # --- policy ---
# For policy we can either use a GPT transformer or an LSTM

# ! TODO: add question parsing in real time here
# question = st.text_input(f"question for DB: {DB_ID} (do not press enter)", value = data_this_db[0][0], max_chars=100)
# st.button('Process')
# st.write(question)
# tokenised = TOKENIZER(question)["input_ids"]
# decoded = TOKENIZER.decode(tokenised)
# st.write(f"""IDs: `{tokenised}` ➡️ Decoded: `{decoded}`""")
