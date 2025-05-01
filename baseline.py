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
def calc_novel_and_uniques_samples(train_dataset, baseline_samples):
    """Calculate the number of novel and unique samples."""
    train_dataset_hash = batch_hash(convert_to_nx(train_dataset))
    baseline_samples_hash = batch_hash(baseline_samples)
    unique = len(set(baseline_samples_hash))/len(baseline_samples_hash)
    novel = len([i for i in baseline_samples_hash if i not in train_dataset_hash])/len(baseline_samples_hash)
    novel_and_unique = len(set(baseline_samples_hash) - set(train_dataset_hash))/len(baseline_samples_hash)
    print("Percentage of unique samples:")
    print(unique)
    print("Percentage of novel samples:")
    print(novel)
    print("Percentage of novel and unique samples:")
    print(novel_and_unique)
    return unique, novel, novel_and_unique



# %%

train_dataset_nx = convert_to_nx(train_dataset)
baseline_samples = baseline(train_dataset).sample_batch(100000)

# %%
calc_novel_and_uniques_samples(train_dataset_nx, baseline_samples)


# %%
def calc_node_degrees(graphs):
    """Calculate the node degrees for a batch of graphs."""
    node_degrees = []
    for graph in graphs:
        node_degrees.append(len(graph.nodes()))
    graph.degree()
    return node_degrees

def calc_clustering(graphs):
    """Calculate the clustering coefficient for a batch of graphs."""
    clustering_coeffs = []
    for graph in graphs:
        # Calculate the clustering coefficient for each graph
        clustering_coeff = nx.clustering(graph)
        clustering_coeffs.append(clustering_coeff)
    return clustering_coeffs

def calc_eigenvector_centrality(graphs):
    """Calculate the eigenvector centrality for a batch of graphs."""
    eigenvector_centralities = []
    for graph in graphs:
        # Calculate the eigenvector centrality for each graph
        eigenvector_centrality = nx.eigenvector_centrality(graph)
        eigenvector_centralities.append(eigenvector_centrality)
    return eigenvector_centralities

# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(data, title="", xlabel="", ylabel=""):
    """Plot a histogram of the data."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# %%
# Calculate node degrees for the baseline samples
node_degrees_baseline = calc_node_degrees(baseline_samples)
node_degrees_training = calc_node_degrees(train_dataset_nx)
plot_histogram(node_degrees_baseline, title="Node Degrees in Baseline Samples", xlabel="Degree", ylabel="Frequency")
plot_histogram(node_degrees_training, title="Node Degrees in Training Samples", xlabel="Degree", ylabel="Frequency")
plot_histogram(calc_clustering(baseline_samples))
plot_histogram(calc_clustering(train_dataset_nx))
plot_histogram(calc_eigenvector_centrality(baseline_samples))
plot_histogram(calc_eigenvector_centrality(train_dataset_nx))

