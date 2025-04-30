# %%
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
import random
device = 'cpu'
# %% Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
print(dataset)
print(dataset[2].num_nodes)

# %% 
# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)


class baseline:
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        nr_edges = 0
        nr_of_possible_edges = 0
        for i in range(len(train_dataset)):
            data = train_dataset[i]
            nr_edges += data.edge_index.shape[1]
            nr_of_possible_edges += data.num_nodes * (data.num_nodes - 1) / 2
        self.r = nr_edges / nr_of_possible_edges
    def sample(self):
        # Sample a batch of data from the dataset
        N = self.train_dataset[random.randint(0, len(self.train_dataset))].num_nodes
        sample = nx.erdos_renyi_graph(N,self.r)
        return sample

# %%

sampler = baseline(train_dataset)
sample = sampler.sample()
print(sample)

