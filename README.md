<img src="assets/header.png">

# Text2SQL

How many times have you pulled your hair apart writing a SQL query, now use natural language to convert to appropriate SQL and save your precious hair.

Though this can be used as a standalone package, I highly recommend that you use `streamlit` to play with the model interactively, to run it interactively
```
streamlit run t2s.py
```

## Installation

Run
```
pip install text2sql
```

## Datasets

Using [CoSQL](https://yale-lily.github.io/cosql), [Spider](https://yale-lily.github.io/spider), [Sparc](https://yale-lily.github.io/sparc) datasets, credit to the authors. There are a couple of things to note, we have in total 178 tables, but only 166 tables in training date and dev set has 20 tables.

We convert the dateset into graphs using `text2sql.data.parse_db_to_networkx()` function. 

## Parsing

New method of parsing to convert each DB to a graph network, red denotes foreign keys.
<img src="assets/dbvis.png">

According to the initial idea I was going to pass a GNN on top of this, but it's too complicated, so instead I replicate the message passing using attention matrix in a standard transformer. Due to size constraints however I have not parsed the following tables: `'baseball_1', 'soccer_1', 'cre_Drama_Workshop_Groups'`.

## Model

Simple model with two transformer encoder (one for DB parsing and another for question) and a transformer decoder for sql generation. Similar to vanilla seq-2-seq transformer with one extra encoder and extra decoder attention matrix in decoder.

#### Tricks

There are couple of tricks I have used that can be improved:
* filtering message passing using attention masks
* fixed the sequence size in all blocks to 400

For generation I am using the code from my other [repo](https://github.com/yashbonde/o2f), which is trimmed down functional version of huggingface generation code.

## Training

To train the model first need to parse and create the datasets, download the data from above mentioned links, extract and place them all in the same folder (or use pre-parsed in `/fdata`). Then run the command
```
python parse_to_lm.py
```

To train the model run this command
```
python train.py
```

## License

`text2sql` is released under the MIT license. Some parts of the software are released under other licenses as specified.

