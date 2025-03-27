from vae_model import *
from torchvision import datasets, transforms
import torch
import warnings
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
    # print(model.decoders)
    decoders = model.decoders

    N_points = 100

    # Define energy weights (random for now, but could be learned)
    E_lk = torch.rand(num_decoders, num_decoders, device=device)

    # Sample two latent points
    z_0 = torch.randn(M, device=device, requires_grad=True)
    z_1 = torch.randn(M, device=device, requires_grad=True) + 1

    # Learnable parameters for the latent path
    num_steps = 10  # Number of discrete points along the curve
    t_values = torch.linspace(0, 1, num_steps, device=device).unsqueeze(1)  # Time steps
    latent_path = nn.Parameter(torch.lerp(z_0, z_1, t_values))  # Interpolation with learnable adjustment

    # Define optimizer
    optimizer = torch.optim.Adam([latent_path], lr=0.01)

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

    # Optimization loop
    num_epochs = 500

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        energy = compute_energy(latent_path, decoders, E_lk)
        energy.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Energy = {energy.item():.6f}")

    # Final optimized latent path
    optimized_path = latent_path.detach()
    
    optimized_path_np = optimized_path.detach().cpu().numpy()

    # latent_points_np = torch.cat(latent_points, dim=0).numpy()
    latent_points = []
    with torch.no_grad():
        for x, _ in mnist_test_loader:  # Images only, ignore labels
            x = x.to(device)
            z_test = model.encoder(x).mean  # Extract mean latent representation
            latent_points.append(z_test.cpu())
    latent_points_np = torch.cat(latent_points, dim=0).numpy()


    # Plot the latent path
    plt.figure(figsize=(6, 6))
    plt.plot(optimized_path_np[:, 0], optimized_path_np[:, 1], marker="o", linestyle="-", color="b", label="Latent Path")
    plt.scatter(latent_points_np[:, 0], latent_points_np[:, 1], color="gray", alpha=0.5, s=10, label="Test Data")

    # Mark start and end points
    plt.scatter(optimized_path_np[0, 0], optimized_path_np[0, 1], color="green", s=100, label="Start (z_0)")
    plt.scatter(optimized_path_np[-1, 0], optimized_path_np[-1, 1], color="red", s=100, label="End (z_1)")

    # Labels and title
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Optimized Latent Path")
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig("latent_path_2.png", dpi=300)
    plt.show()