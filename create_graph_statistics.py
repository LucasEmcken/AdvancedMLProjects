import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from networkx import weisfeiler_lehman_graph_hash
import random
import numpy as np
device = 'cpu'


dataset = TUDataset(root='./data/', name='MUTAG').to(device)

rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (188, 0, 0), generator=rng)

def convert_to_nx(data):
    """Convert a PyTorch Geometric data object to a NetworkX graph."""
    output = []
    for i in range(len(data)):
        G = to_networkx(data[i], to_undirected=True)
        output.append(G)
    return output

def calc_node_degrees(graphs):
    """Calculate the node degrees for a batch of graphs."""
    node_degrees = []
    for graph in graphs:
        degrees = [deg for node, deg in graph.degree()]
        node_degrees += [np.mean(degrees)]
        
    return node_degrees

def calc_clustering(graphs):
    """Calculate the clustering coefficient for a batch of graphs."""
    
    clustering_coeffs = []
    for graph in graphs:
        clustering = nx.average_clustering(graph)
        clustering_coeffs.append(clustering)
        
    return clustering_coeffs

def calc_eigenvector_centrality(graphs):
    """Calculate the eigenvector centrality for a batch of graphs."""
    eigenvector_centralities = []
    for graph in graphs:
        try:
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-06)
            eigenvector_values = list(eigenvector_centrality.values())
            eigenvector_centralities += [np.mean(eigenvector_values)]
        except nx.PowerIterationFailedConvergence:
            print("Warning: Power iteration failed to converge on a graph. Skipping.")
            continue  # or append NaNs/zeros depending on your use case
    return eigenvector_centralities

train_dataset_nx = convert_to_nx(train_dataset)

def max_unique_count(list1, list2, list3):
    """Return the highest number of unique values among the three input lists."""
    count1 = len(set(list1))
    count2 = len(set(list2))
    count3 = len(set(list3))
    
    return max(count1, count2, count3)


def create_histogram(nd_data, cc_data, ec_data,
                          nd_base, cc_base, ec_base,
                          nd_gen, cc_gen, ec_gen):
    
    bins_nd = max_unique_count(nd_data, nd_base, nd_gen)
    bins_cc = max_unique_count(cc_data, cc_base, cc_gen)
    #bins_ec = max_unique_count(ec_data, ec_base, ec_gen)
    bins_ec = 30
    print(bins_nd)
    print(bins_cc)
    print(bins_ec)
    
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    fig.suptitle('Graph Statistics', fontsize=16)

    # Row 1: Node Degrees
    axes[0, 0].hist(nd_data, color='blue', edgecolor='black')
    axes[0, 1].hist(nd_base, color='green', edgecolor='black')
    axes[0, 2].hist(nd_gen, color='red', edgecolor='black')

    # Row 2: Clustering Coefficients
    axes[1, 0].hist(cc_data, color='blue', edgecolor='black')
    axes[1, 1].hist(cc_base, color='green', edgecolor='black')
    axes[1, 2].hist(cc_gen, color='red', edgecolor='black')
    
    # Row 3: Eigenvector Centrality
    axes[2, 0].hist(ec_data, color='blue', edgecolor='black')
    axes[2, 1].hist(ec_base, color='green', edgecolor='black')
    axes[2, 2].hist(ec_gen, color='red', edgecolor='black')


    #Set titles on histograms
    axes[0, 0].set_title('Empirical Distribution')
    axes[0, 1].set_title('Baseline')
    axes[0, 2].set_title('Deep Generative')

    axes[0, 0].set_ylabel('Node degree')
    axes[1, 0].set_ylabel('Clustering coefficient')
    axes[2, 0].set_ylabel('Eigenvector centrality')

    plt.tight_layout()
    plt.savefig("Graph_Statistics.png")


def create_histogram_grid(data, base, gen):
    #___________ Training data _______________
    nd_data = calc_node_degrees(data)
    cc_data = calc_clustering(data)
    ec_data = calc_eigenvector_centrality(data)
    
    #___________ Training data _______________
    nd_base = calc_node_degrees(base)
    cc_base = calc_clustering(base)
    ec_base = calc_eigenvector_centrality(base)

    #___________ Training data _______________
    nd_gen = calc_node_degrees(gen)
    cc_gen = calc_clustering(gen)
    ec_gen = calc_eigenvector_centrality(gen)


    #_______________Create histograms grid _____________________

    create_histogram(nd_data, cc_data, ec_data,
                          nd_base, cc_base, ec_base,
                          nd_gen, cc_gen, ec_gen)



create_histogram_grid(train_dataset_nx,train_dataset_nx,train_dataset_nx)


