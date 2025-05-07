import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
import networkx as nx
from torch.distributions import Normal
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import os

os.system('cls' if os.name == 'nt' else 'clear')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data

class GraphVAE(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, latent_dim, max_num_nodes):
        super(GraphVAE, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_num_nodes = max_num_nodes

        # Encoder GCN
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)

        # MLP to mean and logvar of latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc_dec1 = nn.Linear(latent_dim, 256)
        self.fc_adj = nn.Linear(256, max_num_nodes * max_num_nodes)
        self.fc_node = nn.Linear(256, max_num_nodes * node_feature_dim)

    def encode(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = global_add_pool(x, batch)  # Get graph-level embedding
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_dec1(z))
        adj_flat = torch.sigmoid(self.fc_adj(x))
        node_feat_flat = self.fc_node(x)

        adj_recon = adj_flat.view(-1, self.max_num_nodes, self.max_num_nodes)
        node_features = node_feat_flat.view(-1, self.max_num_nodes, self.node_feature_dim)

        return adj_recon, node_features

    def forward(self, data):
        # Ensure node features exist
        x = data.x
        if x is None:
            x = torch.ones((data.num_nodes, 1), device=data.edge_index.device)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        mu, logvar = self.encode(x, data.edge_index, batch)
        z = self.reparameterize(mu, logvar)
        adj_recon, node_features = self.decode(z)
        return adj_recon, node_features, mu, logvar

    def sample(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            adj_recon, node_features = self.decode(z)

            graphs = []
            for i in range(num_samples):
                adj = adj_recon[i]
                adj = (adj > 0.5).float()
                edge_index, _ = dense_to_sparse(adj)
                x = node_features[i]

                # Take argmax if categorical features
                x = x.argmax(dim=1) if self.node_feature_dim > 1 else x
                data = Data(x=x, edge_index=edge_index)
                G = to_networkx(data, to_undirected=True)
                graphs.append(G)
            return graphs


def pad_adjacency_matrix(data_list, max_num_nodes):
    """Convert PyG graphs to padded adjacency matrices."""
    adj_matrices = []
    for data in data_list:
        adj = to_dense_adj(data.edge_index, max_num_nodes=max_num_nodes)[0]
        adj_matrices.append(adj)
    return torch.stack(adj_matrices)

def train_vae(model, train_dataset, epochs=100, batch_size=32, lr=0.001, device='cpu'):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        total_loss = 0
        for data in loader:
            data = data.to(device)

            # Handle missing node features
            if not hasattr(data, 'x') or data.x is None:
                data.x = torch.ones((data.num_nodes, 1), device=device)

            # Forward pass
            adj_recon, node_features, mu, logvar = model(data)

            # Create dense adjacency matrix (target)
            adj_target = to_dense_adj(data.edge_index, batch=data.batch, max_num_nodes=model.max_num_nodes).squeeze(1)

            # Pad node features if needed
            x_target = torch.zeros((data.num_graphs, model.max_num_nodes, model.node_feature_dim), device=device)
            for i in range(data.num_graphs):
                n = (data.batch == i).sum()
                x_target[i, :n] = data.x[data.batch == i]

            # Losses
            recon_loss = F.binary_cross_entropy(adj_recon, adj_target, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return model, avg_loss

if __name__ == "__main__":
    from baseline import baseline, calc_novel_and_uniques_samples
    import matplotlib.pyplot as plt
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
    import itertools


    device = 'cpu'
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)
    rng = torch.Generator().manual_seed(0)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (186, 1, 1), generator=rng)

    # Determine maximum number of nodes in the dataset
    max_num_nodes = max([data.num_nodes for data in train_dataset])
    node_feature_dim = train_dataset[0].x.shape[1] if hasattr(train_dataset[0], 'x') else 1
    
    print(f"Training VAE with max_num_nodes={max_num_nodes}, node_feature_dim={node_feature_dim}")
    
    
    epochs=500
    lr = 0.015
    batch_size = 16
    lat_dim = 128

    # Initialize and train VAE
    vae = GraphVAE(max_num_nodes=max_num_nodes, node_feature_dim=node_feature_dim, latent_dim=lat_dim, hidden_dim=128)

    vae, _ = train_vae(vae, train_dataset, epochs=epochs, batch_size=batch_size, lr=lr)
    """

    # Define search space
    param_grid = {
        'epochs': [100, 200, 300],
        'lr': [0.005, 0.001, 0.0005],
        'batch_size': [16, 32, 64],
        'latent_dim': [8, 16, 32],
    }

    # Generate all combinations
    all_combinations = list(itertools.product(
        param_grid['epochs'],
        param_grid['lr'],
        param_grid['batch_size'],
        param_grid['latent_dim']
    ))

    results = []

    for i, (epochs, lr, batch_size, lat_dim) in enumerate(all_combinations):
        print(f"\nTrial {i+1}/{len(all_combinations)} - epochs={epochs}, lr={lr}, batch_size={batch_size}, latent_dim={lat_dim}")
        
        # Initialize model
        vae = GraphVAE(
            max_num_nodes=max_num_nodes,
            node_feature_dim=node_feature_dim,
            latent_dim=lat_dim
        )
        
        # Train and get loss
        vae, avg_loss = train_vae(vae, train_dataset, epochs=epochs, batch_size=batch_size, lr=lr)
        
        # Log results
        results.append({
            'trial': i+1,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'latent_dim': lat_dim,
            'avg_loss': avg_loss
        })

    # Find and print best configuration
    best = min(results, key=lambda x: x['avg_loss'])
    print("\nBest configuration:")
    for key, value in best.items():
        print(f"{key}: {value}")
    """
    # Generate samples
    generated_graphs = vae.sample(num_samples=1000)
    baseline_samples = baseline(train_dataset).sample_batch(1000)

    print("________Number of Sampled graphs:______________ ")
    print("Baseline: ", len(baseline_samples))
    print("VAE     : ", len(generated_graphs))


    print("________Compute Novel and Unique:______________ ")
    print("\n Baseline:")
    calc_novel_and_uniques_samples(train_dataset, baseline_samples)
    print("\n VAE:")
    calc_novel_and_uniques_samples(train_dataset, generated_graphs)

    isolated_graphs = []
    
    for i in generated_graphs:
        #largest_cc = max(nx.connected_components(i), key=len)
        #i = i.subgraph(largest_cc)
        i.remove_nodes_from(list(nx.isolates(i)))
    
    # Evaluate samples using the same metrics as baseline
    # from baseline import calc_novel_and_uniques_samples, calc_node_degrees, calc_clustering, calc_eigenvector_centrality, plot_histogram
    import create_graph_statistics as gs
    # gs.calc_novel_and_uniques_samples(train_dataset, generated_graphs)
    print("________Creating Statistics Graph______________ ")

    gs.create_histogram_grid(gs.convert_to_nx(train_dataset),
                             baseline_samples,
                             generated_graphs)

    print("________Done______________ ")
    