import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch.distributions as td
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from networkx import weisfeiler_lehman_graph_hash
import random
import torch.nn as nn
from itertools import combinations 
# torch binary cross-entropy loss
from torch.nn import BCELoss

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_mu, encoder_sigma):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_mu = encoder_mu
        self.encoder_sigma = encoder_sigma

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean = self.encoder_mu(x)
        std = self.encoder_sigma(x)
        return [td.Independent(td.Normal(loc=mean[i], scale=torch.exp(std[i])), 1) for i in range(x.x.shape[0])]


class BinomDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BinomDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def probability(self, z):
        return self.decoder_net(z)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        self.prob = self.decoder_net(z)
        return td.Independent(td.Bernoulli(probs=self.prob), 1)

class GNN_VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder_mu, encoder_sigma, train_dataset=None):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(GNN_VAE, self).__init__()
        self.train_dataset = train_dataset
        self.prior = prior
        self.decoder = decoder
        self.encoder_mu = encoder_mu
        self.encoder_sigma = encoder_sigma
        self.encoder = GaussianEncoder(encoder_mu, encoder_sigma)
        self.loss = BCELoss()

    def forward(self, x): 
        
        nodes = [i for i in range(x.x.shape[0])]
        pairs = combinations(nodes, 2)
        pairs = list(pairs)

        q_latents = self.encoder(x)
        z_latents = [q_latents[i].rsample() for i in nodes]

 
        loss = 0
        for pair in pairs:
            prob = self.decoder.probability(torch.dot(z_latents[pair[0]], z_latents[pair[1]]).unsqueeze(0))
            loss += self.loss(prob, torch.tensor(1 if (pair in x.edge_index.T.tolist() or pair[::-1] in x.edge_index.T.tolist()) else 0,
                                                 dtype=torch.float32).unsqueeze(0))
        kl_loss = sum([td.kl_divergence(q_latents[i], self.prior()) for i in range(len(q_latents))])
        total_loss = loss + kl_loss * 0.001
        return total_loss

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        samples = []
        for _ in range(n_samples):
            samples.append(self.single_sample())
        return samples
    
    def single_sample(self):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        N = self.train_dataset[random.randint(0, len(self.train_dataset))-1].num_nodes
        adjacency_matrix = torch.zeros((N, N))
        node_embeddings = [self.prior().sample() for _ in range(N)]
        node_pairs = combinations(range(N), 2)
        for pair in node_pairs:
            connection = self.decoder(torch.dot(node_embeddings[pair[0]], node_embeddings[pair[1]]).unsqueeze(0)).sample()
            adjacency_matrix[pair[0], pair[1]] = connection
            adjacency_matrix[pair[1], pair[0]] = connection
        # Convert the adjacency matrix to a graph
        G = nx.from_numpy_array(adjacency_matrix.numpy())
        return G



class SimpleGNN(torch.nn.Module):
    """Simple graph neural network for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        state_dim : Dimension of the node states
        num_message_passing_rounds : Number of message passing rounds
    """

    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, out_dim):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.out_dim = out_dim

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.ReLU()
            )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

    def forward(self, x):
        edge_index = x.edge_index
        #print(x)
        x = x.x
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """
        # Extract number of nodes and graphs
        num_nodes = x.shape[0]

        # Initialize node state from node features
        state = self.input_net(x)
        # state = x.new_zeros([num_nodes, self.state_dim]) # Uncomment to disable the use of node features

        # Loop over message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            state = state + self.update_net[r](aggregated)

        # Aggretate: Sum node features
        graph_state = x.new_zeros(self.state_dim)
        graph_state = graph_state+state

        # print(graph_state)
        # print(graph_state.shape)

        return graph_state
    
from baseline import baseline
from baseline import baseline, calc_novel_and_uniques_samples

import networkx as nx
from torch_geometric.utils import to_networkx
from tqdm import tqdm

def preprocess_with_structural_features(dataset):
    """
    Add structural features (degree, clustering coefficient, etc.) to each node.

    Parameters:
    dataset: torch_geometric.data.Dataset
        The input graph dataset.

    Returns:
    dataset: torch_geometric.data.Dataset
        The modified dataset with structural features.
    """
    processed_data_list = []
    for data in tqdm(dataset):
        G = to_networkx(data, to_undirected=True)
        
        # Compute structural features
        degree = dict(G.degree())
        clustering = nx.clustering(G)
        betweenness = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        
        # Create feature matrix
        features = []
        for node in G.nodes():
            features.append([
                degree[node],
                clustering[node],
                betweenness[node]
            ])
        
        # Replace node features with structural features
        data.x = torch.tensor(features, dtype=torch.float32)
        processed_data_list.append(data)
    
    # Return the updated dataset
    return processed_data_list

if __name__ == "__main__":
    GNN_mu = SimpleGNN(node_feature_dim=3, state_dim=16, num_message_passing_rounds=3, out_dim=16)
    GNN_sigma = SimpleGNN(node_feature_dim=3, state_dim=16, num_message_passing_rounds=3, out_dim=16)
    prior = GaussianPrior(M=16)
    decoder_net = nn.Sequential(
        nn.Linear(1, 1),
        nn.Sigmoid()
    )
    decoder = BinomDecoder(decoder_net=decoder_net)

    device = "cpu"
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)
    rng = torch.Generator().manual_seed(0)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

    train_dataset = preprocess_with_structural_features(train_dataset)

    model = GNN_VAE(prior=prior, decoder=decoder, encoder_mu=GNN_mu, encoder_sigma=GNN_sigma, train_dataset=train_dataset)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(3):
        for x in train_dataset:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            print(loss.item())
    baseline_samples = baseline(train_dataset).sample_batch(1000)
    generated_graphs = model.sample(n_samples=1000)
    for i, graph in enumerate(generated_graphs):
        largest_cc = max(nx.connected_components(graph), key=len)
        generated_graphs[i] = graph.subgraph(largest_cc)
        #i.remove_nodes_from(list(nx.isolates(i)))
    print("________Number of Sampled graphs:______________ ")
    print("Baseline: ", len(baseline_samples))
    print("VAE     : ", len(generated_graphs))


    print("________Compute Novel and Unique:______________ ")
    print("\n Baseline:")
    calc_novel_and_uniques_samples(train_dataset, baseline_samples)
    print("\n Node level VAE:")
    calc_novel_and_uniques_samples(train_dataset, generated_graphs)
    import create_graph_statistics as gs
    # gs.calc_novel_and_uniques_samples(train_dataset, generated_graphs)
    print("________Creating Statistics Graph______________ ")

    gs.create_histogram_grid(gs.convert_to_nx(train_dataset),
                             baseline_samples,
                             generated_graphs)

    print("________Done______________ ")
