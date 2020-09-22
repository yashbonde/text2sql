"""data management piece
22.09.2020 - @yashbonde"""

import networkx as nx


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