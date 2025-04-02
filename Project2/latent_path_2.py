from vae_model import *
from torchvision import datasets, transforms
import torch
import warnings
import numpy as np
warnings.filterwarnings("ignore")

M = 2

def subsample(data, targets, num_data, num_classes):
    idx = targets < num_classes
    new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
    new_targets = targets[idx][:num_data]

    return torch.utils.data.TensorDataset(new_data, new_targets)

train_tensors = datasets.MNIST(
    "data/",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)
test_tensors = datasets.MNIST(
    "data/",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)
num_train_data = 2048
num_classes = 3
train_data = subsample(
    train_tensors.data, train_tensors.targets, num_train_data, num_classes
)
test_data = subsample(
    test_tensors.data, test_tensors.targets, num_train_data, num_classes
)

mnist_train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=64, shuffle=True
)
mnist_test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=64, shuffle=False
)

def new_encoder():
    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2 * M),
    )
    return encoder_net

def new_decoder():
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.Unflatten(-1, (32, 4, 4)),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
    )
    return decoder_net

def calculate_energy(z_0, z_1, decoders, steps = 15, lr = 0.01, verbose=False, num_epochs=75, device = "cuda"):
    # Learnable parameters for the latent path
    device = torch.device(device)
    num_steps = 15  # Number of discrete points along the curve
    t_values = torch.linspace(0, 1, num_steps, device=device).unsqueeze(1)  # Time steps
    interior_points = nn.Parameter(torch.lerp(z_0, z_1, t_values[1:-1]))  # Interpolation with learnable adjustment

    # Define optimizer
    optimizer = torch.optim.Adam([interior_points], lr=0.01)

    num_decoders = len(decoders)

    # Define energy weights (random for now, but could be learned)
    E_lk = torch.rand(num_decoders, num_decoders, device=device)


    # Define energy function
    def compute_energy(latent_path, decoders, E_lk):
        energy_sum = 0
        for i in range(num_steps - 1):
            z_t = latent_path[i]
            z_t = z_t.unsqueeze(0)
            z_next = latent_path[i + 1]
            z_next = z_next.unsqueeze(0)

            dec_outputs_t = torch.stack([dec(z_t).mean for dec in decoders])
            dec_outputs_next = torch.stack([dec(z_next).mean for dec in decoders])

            # Compute energy functional
            for l in range(num_decoders):
                for k in range(num_decoders):
                    diff = dec_outputs_t[l] - dec_outputs_next[k]
                    energy_sum += E_lk[l, k] * (diff.norm(2) ** 2)
        
        return energy_sum

    def compute_arc_length(latent_path):
        """Compute cumulative arc length of a given path."""
        distances = torch.norm(latent_path[1:] - latent_path[:-1], dim=1)
        return torch.cat([torch.zeros(1, device=latent_path.device), distances.cumsum(dim=0)])
    
    def resample_latent_path(latent_path, num_steps):
        """Resample latent path to maintain approximately equal spacing."""
        arc_lengths = compute_arc_length(latent_path)
        total_length = arc_lengths[-1]
        
        # Generate evenly spaced arc-length values
        target_lengths = torch.linspace(0, total_length, num_steps, device=latent_path.device)

        # Interpolate new points at these arc lengths
        new_path = []
        for t in target_lengths:
            # Find indices where t is between arc_lengths[i] and arc_lengths[i+1]
            idx = torch.searchsorted(arc_lengths, t, right=True) - 1
            idx = torch.clamp(idx, 0, len(arc_lengths) - 2)  # Avoid out-of-bounds

            # Linear interpolation
            t1, t2 = arc_lengths[idx], arc_lengths[idx + 1]
            z1, z2 = latent_path[idx], latent_path[idx + 1]

            alpha = (t - t1) / (t2 - t1 + 1e-8)  # Avoid division by zero
            new_z = (1 - alpha) * z1 + alpha * z2
            new_path.append(new_z)

        return torch.stack(new_path)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Construct full latent path with fixed endpoints
        latent_path = torch.cat([z_0.unsqueeze(0), interior_points, z_1.unsqueeze(0)], dim=0)

        # Compute energy and optimize
        energy = compute_energy(latent_path, decoders, E_lk)
        energy.backward()
        optimizer.step()

        # Resample points to maintain equal spacing
        if epoch % 10 == 0:
            with torch.no_grad():
                latent_path = resample_latent_path(latent_path, num_steps)
                interior_points.copy_(latent_path[1:-1])  # Update learnable points

        if epoch % 50 == 0 and verbose:
            print(f"Epoch {epoch}: Energy = {energy.item():.6f}")
        
    return energy.item()