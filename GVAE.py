import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
import networkx as nx
from torch.distributions import Normal
from torch_geometric.utils.convert import to_networkx

class GraphVAE(nn.Module):
    def __init__(self, max_num_nodes, node_feature_dim=7, latent_dim=16):
        super(GraphVAE, self).__init__()
        self.max_num_nodes = max_num_nodes
        self.node_feature_dim = node_feature_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * max_num_nodes * max_num_nodes, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder layers
        self.fc2 = nn.Linear(latent_dim, 256)
        self.fc3 = nn.Linear(256, max_num_nodes * max_num_nodes)
        
        # Node feature predictor (modified to handle batch processing)
        self.node_feature_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_num_nodes * node_feature_dim)
        )
    
    def encode(self, adj_matrix):
        # adj_matrix shape: (batch_size, max_num_nodes, max_num_nodes)
        x = adj_matrix.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        return self.fc_mu(x), self.fc_logvar(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # Reconstruct adjacency matrix
        x = F.relu(self.fc2(z))
        x = torch.sigmoid(self.fc3(x))
        adj_recon = x.view(-1, self.max_num_nodes, self.max_num_nodes)
        
        # Predict node features (modified to handle batch processing)
        node_features = self.node_feature_predictor(z)
        node_features = node_features.view(-1, self.max_num_nodes, self.node_feature_dim)
        
        return adj_recon, node_features
    
    def forward(self, adj_matrix):
        mu, logvar = self.encode(adj_matrix)
        z = self.reparameterize(mu, logvar)
        adj_recon, node_features = self.decode(z)
        return adj_recon, node_features, mu, logvar
    
    def sample(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            adj_recon, node_features = self.decode(z)
            
            # Convert to networkx graphs
            graphs = []
            for i in range(num_samples):
                # Threshold adjacency matrix
                adj = adj_recon[i]
                adj = (adj > 0.5).float()
                
                # Convert to sparse format
                edge_index, edge_attr = dense_to_sparse(adj)
                
                # Get node features (take argmax for categorical features)
                x = node_features[i].argmax(dim=1) if self.node_feature_dim > 1 else node_features[i]
                
                # Create PyG Data object
                data = Data(x=x, edge_index=edge_index)
                
                # Convert to NetworkX
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

def train_vae(model, train_dataset, epochs=100, batch_size=32, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Prepare data loader
    adj_matrices = pad_adjacency_matrix(train_dataset, model.max_num_nodes)
    dataset = torch.utils.data.TensorDataset(adj_matrices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            adj = batch[0]
            
            # Forward pass
            adj_recon, node_features, mu, logvar = model(adj)
            
            # Reconstruction loss (binary cross entropy for adjacency)
            recon_loss = F.binary_cross_entropy(adj_recon, adj, reduction='sum')
            
            # KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Total loss
            loss = recon_loss + kl_div
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dataset)}")
    
    return model

if __name__ == "__main__":
    from baseline import train_dataset
    
    # Determine maximum number of nodes in the dataset
    max_num_nodes = max([data.num_nodes for data in train_dataset])
    node_feature_dim = train_dataset[0].x.shape[1] if hasattr(train_dataset[0], 'x') else 1
    
    print(f"Training VAE with max_num_nodes={max_num_nodes}, node_feature_dim={node_feature_dim}")
    
    # Initialize and train VAE
    vae = GraphVAE(max_num_nodes=max_num_nodes, node_feature_dim=node_feature_dim)
    vae = train_vae(vae, train_dataset, epochs=50)

    # Generate samples
    generated_graphs = vae.sample(num_samples=100)
    
    # Evaluate samples using the same metrics as baseline
    from baseline import calc_novel_and_uniques_samples, calc_node_degrees, calc_clustering, calc_eigenvector_centrality, plot_histogram
    
    # train_dataset_nx = [to_networkx(data, to_undirected=True) for data in train_dataset]
    
    # print(generated_graphs)
    # print(train_dataset_nx)
    # exit()

    print("\nEvaluating VAE-generated samples:")
    calc_novel_and_uniques_samples(train_dataset, generated_graphs)

    # Compare distributions
    print("\nNode degree distributions:")
    node_degrees_vae = calc_node_degrees(generated_graphs)
    node_degrees_train = calc_node_degrees(to_networkx(train_dataset_nx))
    plot_histogram(node_degrees_vae, title="VAE Generated - Node Degrees", xlabel="Degree", ylabel="Frequency")
    plot_histogram(node_degrees_train, title="Training Data - Node Degrees", xlabel="Degree", ylabel="Frequency")
    
    print("\nClustering coefficient distributions:")
    clustering_vae = calc_clustering(generated_graphs)
    clustering_train = calc_clustering(to_networkx(train_dataset_nx))
    plot_histogram(clustering_vae, title="VAE Generated - Clustering", xlabel="Coefficient", ylabel="Frequency")
    plot_histogram(clustering_train, title="Training Data - Clustering", xlabel="Coefficient", ylabel="Frequency")