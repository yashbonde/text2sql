"""So the complicated model with encoder and decoder networks is not working properly,
need to come up with something better.
12.11.2020 - @yashbonde"""

from text2sql.model import *

# https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html
from torch_scatter import scatter_add

class GPTModel(nn.Module):
    def __init__(self, config):
        super(GPTModel, self).__init__()

        self.config = config

    def merge_embeddings(self, expanded_embeddings, merging_index):
        """This function merges multiple embedding vectors into smaller embeddings. This is what
        is happening in this function. Challenge is that each node has information in this format
        <table_name>.<column_name> and we need to merge this information. Another challenge was
        that the older method created matrices that were 400x400 while the questions were 50x50.

        So continuing with the above mentioned case:

        <table_name> is tokenized into W1,W2,W3 and <column_name> is tokenized into W4,W5
        So the effective tokenizing becomes W1,W2,W3.W4,W5. Now in order to feed into the model
        we need to merge the expanded_embeddings is (5,256) and merging_index = [0,0,0,1,1]
        this function returns a matrix (2,256)

        Now in practice this takes a large matrix like (500, 500) and it returns a highly reduced
        version of this. The output in this case is going to be (num_nodes, num_nodes) and thus
        we do not need to use the complicated cell expanded attention matrix as done in
        `text2sql.data.get_tokenised_attention_mask`


        """"
        out = scatter_add(expanded_embeddings, merging_index)
        return out

    def forward():

        torch_scatter

