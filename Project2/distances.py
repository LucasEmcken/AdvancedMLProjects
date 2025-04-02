import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import random
import os
print(os.getcwd())
from vae_model import VAE, GaussianPrior, GaussianDecoder, GaussianEncoder, VAE_ensemble, train, new_encoder, new_decoder
import argparse
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import subsample

device = "cpu"
parser = argparse.ArgumentParser()


parser.add_argument(
    "--decoders",
    type=int,
    default=4
)

parser.add_argument(
    "--model_path",
    type=str,
    default="./models"
)


test_tensors = datasets.MNIST(
    "data/",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

args = parser.parse_args()
M=2
max_num_decoders = args.decoders
model_path = args.model_path

test_data = test_tensors.data[0:1000].float().to(device)

sample_pairs = [(test_data[random.randint(0,999)], test_data[random.randint(0,999)]) for i in range(10)]
print(len(sample_pairs))
print(sample_pairs[0][0].shape)
euclidian_cov = dict()
geodesic_cov = dict()

for i in range(1,max_num_decoders+1):
    euclidian_cov[i] = []
    geodesic_cov[i] = []
    for pair in sample_pairs:
        euclidian_distances = []
        geodesic_distances = []
        for j in range(1,11):
            torch.manual_seed(0)
            model = VAE_ensemble(
                GaussianPrior(M),
                [GaussianDecoder(new_decoder(M=M)) for _ in range(max_num_decoders)],  # Create an ensemble of decoders
                GaussianEncoder(new_encoder(M=M))
            ).to(device)
            model.load_state_dict(torch.load(model_path+"/model_{}_{}.pt".format(max_num_decoders,j)))
            model.num_decoder = i 
            model.decoders = model.decoders[:i]  # Use only the first i decoders
            model.eval()
            q = model.encoder(pair[0].unsqueeze(0).unsqueeze(0))
            z1 = q.rsample().detach().cpu()
            q = model.encoder(pair[1].unsqueeze(0).unsqueeze(0))
            z2 = q.rsample().detach().cpu()
            euclidian_distances.append(torch.norm(z1-z2).item())
            #calculate geodesic distance
            #
            geodesic_distance = 1
            geodesic_distances.append(geodesic_distance)

            
        #calculate coefficient of variation
        euclidian_cov[i].append(np.std(euclidian_distances)/np.mean(euclidian_distances))
        geodesic_cov[i].append(np.std(geodesic_distances) / np.mean(geodesic_distances))



for i in range(1, max_num_decoders+1):
    euclidian_cov[i] = np.mean(euclidian_cov[i])
    geodesic_cov[i] = np.mean(geodesic_cov[i])
euclidian_cov = [euclidian_cov[i] for i in range(1, max_num_decoders+1)]
geodesic_cov = [geodesic_cov[i] for i in range(1, max_num_decoders+1)]
print(euclidian_cov)

plt.plot(range(1,max_num_decoders+1), geodesic_cov, label="Euclidian Covariance")
plt.plot(range(1, max_num_decoders+1), euclidian_cov, label="Euclidian Covariance")
plt.xlabel("Number of Decoders")
plt.savefig("euclidian_covariance.png")
