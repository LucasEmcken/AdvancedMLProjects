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

sample_pairs = [(test_data[random.randint(0,1000)], test_data[random.randint(0,1000)]) for i in range(10)]
print(len(sample_pairs))
print(sample_pairs[0][0].shape)
euclidian_cov = dict()
geodesic_cov = dict()

for i in range(1, max_num_decoders+1):
    euclidian_cov[i] = []
    geodesic_cov[i] = []
    for pair in sample_pairs:
        euclidian_distances = []
        geodesic_distances = []
        for j in range(1, 11):
            model = VAE_ensemble(
                GaussianPrior(M),
                [GaussianDecoder(new_decoder(M=M)) for _ in range(i)],
                GaussianEncoder(new_encoder(M=M))
            ).to(device)
            model.load_state_dict(torch.load(model_path+"/model_{}_{}.pt".format(i, j)))
            model.eval()
            
            q = model.encoder(pair[0].unsqueeze(0).unsqueeze(0))
            z1 = q.rsample().detach().cpu()
            q = model.encoder(pair[1].unsqueeze(0).unsqueeze(0))
            z2 = q.rsample().detach().cpu()
            
            euclidian_distances.append(torch.norm(z1 - z2).item())
            
            #Compute Geodesic distance
            t_steps = 10 #Ponts on the geodesic
            t_vals = torch.linspace(0,1,t_steps)
            geodesic_energy=0
            
            for t in t_vals[:-1]:
                t1=t_vals[i] 
                t2=t_vals[i+1]
            
                interpolated_z1 = (1 - t1) * z1 + t1 * z2
                interpolated_z2 = (1 - t2) * z1 + t2 * z2
                    
                samples = 20
                ensemble_energy = 0
                pairs = 0
                
                for _ in range(samples):  #We are using Monte Carlo approximation, however im not sure that i implemented it correctly
                    decoder_list = [model.decoders[k] for k in range(i)]
                    if len(decoder_list)==1:
                        decoder1=decoder_list[0]
                        decoder2=decoder_list[0]
                    else:
                        decoder1, decoder2 = random.sample(decoder_list, 2)
                    mean1=decoder1(interpolated_z1.to(device)).mean.cpu()
                    mean2=decoder2(interpolated_z2.to(device)).mean.cpu()
                    
                    ensemble_energy+=torch.norm(mean1-mean2).item()**2 #Norm^2
                    pairs+=1

                geodesic_energy+=ensemble_energy/pairs #Average energy

            geodesic_distances.append(math.sqrt(geodesic_energy)) #Geodesic distance
        
        #CoV
        euclidian_cov[i].append(np.std(euclidian_distances) / np.mean(euclidian_distances))
        geodesic_cov[i].append(np.std(geodesic_distances) / np.mean(geodesic_distances))

#Average of CoVs
for i in range(1, max_num_decoders+1):
    euclidian_cov[i] = np.mean(euclidian_cov[i])
    geodesic_cov[i] = np.mean(geodesic_cov[i])

euclidian_cov = [euclidian_cov[i] for i in range(1, max_num_decoders+1)]
geodesic_cov = [geodesic_cov[i] for i in range(1, max_num_decoders+1)]

print("Euclidean Covariance: ", euclidian_cov)
print("Geodesic Covariance: ", geodesic_cov)

#CoV plot
plt.plot(range(1, max_num_decoders+1), euclidian_cov, label="Euclidean Covariance")
plt.plot(range(1, max_num_decoders+1), geodesic_cov, label="Geodesic Covariance")
plt.xlabel("Number of Decoders")
plt.ylabel("Coefficient of Variation")
plt.title("Covariance Comparison")
plt.legend()
plt.savefig("covariance_comparison.png")
