# %%
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from networkx import weisfeiler_lehman_graph_hash
import random
import create_graph_statistics as gs
import numpy as np

device = 'cpu'
# %% Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)

# %% 
# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

def batch_hash(graphs):
    """Compute the Weisfeiler-Lehman graph hash for a batch of graphs."""
    hashes = []
    for graph in graphs:
        # Compute the WL hash for each graph
        wl_hash = weisfeiler_lehman_graph_hash(graph)
        hashes.append(wl_hash)
    return hashes

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
        train_graph_sizes = []
        for data in train_dataset:
            train_graph_sizes.append(data.num_nodes)

        self.train_graph_sizes = train_graph_sizes

        self.num_unique = len(set(self.train_graph_sizes))
        self.unique_elements = list(set(self.train_graph_sizes))
        
        self.r = np.zeros(self.num_unique)
        
        k = 0
        for n in self.unique_elements:
            target_num_nodes = n  # Replace with the desired number
            filtered_graphs = [graph for graph in train_dataset if graph.num_nodes == target_num_nodes]

            self.r[k] = self.average_link_probability(filtered_graphs)
            k+=1
    
    def get_N(self):
        return random.choice(self.train_graph_sizes)

    def compute_possible_edges(self, num_nodes):
        return num_nodes * (num_nodes - 1) // 2

    def compute_link_probability(self, num_edges, num_nodes):
        r = num_edges/self.compute_possible_edges(num_nodes)
        return r
    
    def average_link_probability(self, graphs):
        link_probs = []
        for graph in graphs:
            num_nodes = graph.num_nodes
            num_edges = graph.num_edges

            if num_nodes < 2:
                continue  # Skip graphs that can't have edges

            lp = self.compute_link_probability(num_edges, num_nodes)
            link_probs.append(lp)

        return sum(link_probs) / len(link_probs) if link_probs else 0
    
    def sample(self):
        # Finds a random graph from dataset and gets its number of nodes. 
        N = self.train_dataset[random.randint(0, len(self.train_dataset))-1].num_nodes
        index = self.unique_elements.index(N)
        # Creates the erdos_renyi_graph with N nodes and edge probability self.r between each node pair 
        sample = nx.erdos_renyi_graph(N,self.r[index])
        return sample
    def sample_batch(self, batch_size):
        # Sample a batch of data from the dataset
        samples = []
        for i in range(batch_size):
            samples.append(self.sample())
        return samples
# %%
def calc_novel_and_uniques_samples(train_dataset, baseline_samples):
    """Calculate the number of novel and unique samples."""
    train_dataset_hash = batch_hash(convert_to_nx(train_dataset))
    baseline_samples_hash = batch_hash(baseline_samples)
    unique = len(set(baseline_samples_hash))/len(baseline_samples_hash)
    novel = len([i for i in baseline_samples_hash if i not in train_dataset_hash])/len(baseline_samples_hash)
    novel_and_unique = len(set(baseline_samples_hash) - set(train_dataset_hash))/len(baseline_samples_hash)
    print("Unique samples          :", unique)
    print("Novel samples           :", novel)
    print("Novel and unique samples:", novel_and_unique)
    return unique, novel, novel_and_unique



# %%

# train_dataset_nx = convert_to_nx(train_dataset)
# baseline_samples = baseline(train_dataset).sample_batch(100000)

# %%
# calc_novel_and_uniques_samples(train_dataset, baseline_samples)

########## replace input with (train_dataset_nx, baseline_nx, vae_nx) #################
# gs.create_histogram_grid(train_dataset_nx,baseline_samples,train_dataset_nx)


