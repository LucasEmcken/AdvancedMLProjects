# %%
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
import random
device = 'cpu'
# %% Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
print(dataset)
print(type(dataset[2]))
print(to_networkx(dataset[2],to_undirected=True))

# %% 
# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

def convert_to_nx(data):
    """Convert a PyTorch Geometric data object to a NetworkX graph."""
    output = []
    for i in range(len(data)):
        G = to_networkx(data[i], to_undirected=True)
        output.append(G)
    return output

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
        N = self.train_dataset[random.randint(0, len(self.train_dataset))-1].num_nodes
        sample = nx.erdos_renyi_graph(N,self.r)
        return sample
    def sample_batch(self, batch_size):
        # Sample a batch of data from the dataset
        samples = []
        for i in range(batch_size):
            samples.append(self.sample())
        return samples

# %%

train_dataset_nx = convert_to_nx(train_dataset)
baseline_samples = baseline(train_dataset).sample_batch(10000)

# %%
print(baseline_samples)
print("Percentage of unique samples:")
print(len(set(baseline_samples))/len(baseline_samples))
print("Percentage of novel samples:")
print(len([i for i in baseline_samples if i not in train_dataset_nx])/len(baseline_samples))
print("Percentage of novel and unique samples:")
print(len(set(baseline_samples) - set(train_dataset_nx))/len(baseline_samples))

