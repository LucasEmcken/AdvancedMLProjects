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
if __name__ == "__main__":

    device = torch.device("cuda")

    num_decoders=4
    
    model = VAE_ensemble(
        GaussianPrior(M),
        [GaussianDecoder(new_decoder()) for _ in range(num_decoders)],  # Create an ensemble of decoders
        GaussianEncoder(new_encoder())
    ).to(device)
    # from scipy.stats import multivariate_normal
    # import seaborn as sns
    # from scipy.stats import gaussian_kde
    # import numpy as np
    model.load_state_dict(torch.load("models/model_4_1.pt"))
    model.eval()
    
    
    # latent_points_np = torch.cat(latent_points, dim=0).numpy()
    latent_points = []
    with torch.no_grad():
        for x, _ in mnist_train_loader:  # Images only, ignore labels
            x = x.to(device)
            z_test = model.encoder(x).mean  # Extract mean latent representation
            latent_points.append(z_test.cpu())
    latent_points_np = torch.cat(latent_points, dim=0).numpy()

    
    # print(model.decoders)
    decoders = model.decoders


    # Define energy weights (random for now, but could be learned)
    E_lk = torch.rand(num_decoders, num_decoders, device=device)

    # Sample two latent points
    N_pairs = 100
    z_0 = torch.tensor([0.0, -1.0], device=device, requires_grad=False)
    z_1 = torch.tensor([1.0, -4.0], device=device, requires_grad=False)

    # Learnable parameters for the latent path
    num_steps = 15  # Number of discrete points along the curve
    t_values = torch.linspace(0, 1, num_steps, device=device).unsqueeze(1)  # Time steps
    interior_points = nn.Parameter(torch.lerp(z_0, z_1, t_values[1:-1]))  # Interpolation with learnable adjustment

    # Define optimizer
    optimizer = torch.optim.Adam([interior_points], lr=0.01)

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

    # Optimization loop
    num_epochs = 50

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

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Energy = {energy.item():.6f}")
    # Final optimized latent path
    # optimized_path = interior_points.detach()
    optimized_path = torch.cat([z_0.unsqueeze(0), interior_points.detach(), z_1.unsqueeze(0)], dim=0)

    optimized_path_np = optimized_path.detach().cpu().numpy()


    # Define grid over the latent space
    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.8, 1.7, 100),  # Adjust grid range as needed
        np.linspace(-6, -0.5, 100)
    )
    grid_points_np = np.c_[grid_x.ravel(), grid_y.ravel()]  # Flattened grid points

    grid_points = torch.tensor(grid_points_np, dtype=torch.float32, device=device)
    num_points = grid_points.shape[0]

    # Storage for uncertainty values
    uncertainty_values = np.zeros(num_points)

    # Compute uncertainty at each grid point
    with torch.no_grad():
        for i, grid_point in enumerate(grid_points):
            grid_point = grid_point.unsqueeze(0)  # Add batch dimension
            decoder_outputs = torch.stack([dec(grid_point).mean for dec in decoders])  # (num_decoders, 1, 28, 28)
            pixel_std = decoder_outputs.std(dim=0)  # Standard deviation across decoders (shape: 1, 28, 28)
            total_std = pixel_std.sum()  # Sum over all pixels to get a scalar value
            uncertainty_values[i] = total_std.item()  # Store result

    # Reshape uncertainty values for contour plot
    uncertainty = uncertainty_values.reshape(grid_x.shape)
    # print(uncertainty)
    # Plotting
    plt.figure(figsize=(7, 6))

    # Contour plot of uncertainty
    plt.contourf(grid_x, grid_y, uncertainty, levels=100, cmap="coolwarm", alpha=0.7)
    plt.colorbar(label="Uncertainty (Mean Variance)")

    # Plot the latent path
    # plt.figure(figsize=(6, 6))
    # plt.plot(optimized_path_np[:, 0], optimized_path_np[:, 1], marker="o", linestyle="-", color="b", label="Latent Path")
    plt.plot(optimized_path_np[:, 0], optimized_path_np[:, 1], linestyle="-", color="b", label="Latent Path")
    plt.scatter(latent_points_np[:, 0], latent_points_np[:, 1], color="gray", alpha=0.5, s=10, label="Test Data")

    # Mark start and end points
    # plt.scatter(optimized_path_np[0, 0], optimized_path_np[0, 1], color="green", s=100, label="Start (z_0)")
    # plt.scatter(optimized_path_np[-1, 0], optimized_path_np[-1, 1], color="red", s=100, label="End (z_1)")

    # Labels and title
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Optimized Latent Path")
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig("latent_path_arc_long.png", dpi=300)
    plt.show()