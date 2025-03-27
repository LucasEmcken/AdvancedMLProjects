# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
import random
from Project2.vae_model import VAE, GaussianPrior, GaussianDecoder, GaussianEncoder


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse
    # python3 Project2/ensemble_vae.py train --device cpu --latent-dim 2 --epochs 50 --batch-size 64 --experiment-folder Project2/models

    # python3 Project2/ensemble_vae.py geodesics --device cpu --latent-dim 2 --epochs 50 --batch-size 64 --experiment-folder Project2/models

    # python3 Project2/ensemble_vae.py TestEnsamble --device cpu --latent-dim 2 --epochs 50 --batch-size 64 --experiment-folder Project2/models

    # python3 ensemble_vae.py testing --device cpu --latent-dim 2 --epochs 50 --batch-size 64 --experiment-folder models

    # python3 ensemble_vae.py sample --device cpu --latent-dim 2 --epochs 50 --batch-size 64 --experiment-folder models

    # python3 Project2/ensemble_vae.py trainEnsamble --device cpu --latent-dim 2 --epochs 50 --batch-size 64 --experiment-folder Project2/models

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics","testing", "trainEnsamble","TestEnsamble"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
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
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

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

    # Choose mode to run
    if args.mode == "TestEnsamble":
        num_decoders=3
        model = VAE_ensemble(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(num_decoders)],  # Create an ensemble of decoders
            GaussianEncoder(new_encoder())
        ).to(device)
        from scipy.stats import multivariate_normal
        import seaborn as sns
        from scipy.stats import gaussian_kde
        import numpy as np
        model.load_state_dict(torch.load(args.experiment_folder + "/model_3_1.pt"))
        model.eval()
        
        data_iter = iter(mnist_train_loader)
        
        point_list = []
        target = []
        for i in range(30):
            x, y = next(data_iter)
            x = x.to(device)
            q = model.encoder(x)
            
            z = q.rsample().detach().cpu()
            
            point_list.append(z)  # Keep as tensor
            target.append(y)
       
        # Concatenate across batches
        
        z_matrix = torch.cat(point_list, dim=0).numpy()  # Shape: [50*32, 20]
        target_tensor = torch.cat(target, dim=0)  # Shape: [50*32]
        
        #____Sample From Prior _________
        x = z_matrix[:, 0]  # First principal component
        y = z_matrix[:, 1]  # Second principal component

        scatter = plt.scatter(x, y, c=target_tensor.detach().numpy(), alpha=1, s=3)

        # Create a legend
        # Assuming target_tensor contains values 0, 1, 2 for the three classes
        classes = [0, 1, 2]
        class_labels = ['0', '1', '2']

        # Create a legend with color patches
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(class_value)), markersize=8, label=label)
            for class_value, label in zip(classes, class_labels)
        ]

        plt.legend(handles=legend_handles, title="Classes")
        # Labels and title
        plt.title("Ensamble")
        plt.savefig("plot_ensamble_test.png", dpi=300, bbox_inches="tight")
    
    if args.mode == "trainEnsamble":
        # Number of decoders in the ensemble
        num_decoders = 3  
        experiments_folder = args.experiment_folder
        # Initialize the model
        model = VAE_ensemble(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(num_decoders)],  # Create an ensemble of decoders
            GaussianEncoder(new_encoder())
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model_3_1.pt",
        )

    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )
    elif args.mode == "testing":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        from scipy.stats import multivariate_normal
        import seaborn as sns
        from scipy.stats import gaussian_kde
        import numpy as np
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()
        
        data_iter = iter(mnist_train_loader)
        
        point_list = []
        target = []
        for i in range(30):
            x, y = next(data_iter)
            x = x.to(device)
            q = model.encoder(x)
            
            z = q.rsample().detach().cpu()
            
            point_list.append(z)  # Keep as tensor
            target.append(y)
       
        # Concatenate across batches
        
        z_matrix = torch.cat(point_list, dim=0).numpy()  # Shape: [50*32, 20]
        target_tensor = torch.cat(target, dim=0)  # Shape: [50*32]
        
        #____Sample From Prior _________
        x = z_matrix[:, 0]  # First principal component
        y = z_matrix[:, 1]  # Second principal component

        scatter = plt.scatter(x, y, c=target_tensor.detach().numpy(), alpha=1, s=3)

        # Create a legend
        # Assuming target_tensor contains values 0, 1, 2 for the three classes
        classes = [0, 1, 2]
        class_labels = ['0', '1', '2']

        # Create a legend with color patches
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(class_value)), markersize=8, label=label)
            for class_value, label in zip(classes, class_labels)
        ]

        plt.legend(handles=legend_handles, title="Classes")
        # Labels and title
        plt.title("Gauss Prior")
        plt.savefig("plot_samples.png", dpi=300, bbox_inches="tight")

    elif args.mode == "PartB":

        print("Hello")


    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()
        
        def select_random_point_pairs(num_points, n):
            """
            Selects n random pairs of points from the given tensor without repetition.

            Args:
            - points (torch.Tensor): A tensor of shape (1920, 2) containing the x,y coordinates.
            - n (int): The number of pairs to select.

            Returns:
            - List[Tuple[torch.Tensor, torch.Tensor]]: A list of n pairs of points.
            """
            if n > num_points * (num_points - 1) // 2:
                raise ValueError("n is too large for the number of points available without repetition.")

            selected_pairs = set()

            while len(selected_pairs) < n:
                # Randomly select two different indices
                i, j = random.sample(range(num_points), 2)
                # Sort the indices to ensure consistent pair ordering
                pair = tuple(sorted([i, j]))
                selected_pairs.add(pair)

            return list(selected_pairs)

        def mean_func(z):
            return model.decoder(z).mean.view(-1)  # Flatten output to (784,)

        def compute_pull_back(z):
            #Find Jacobian
            J = torch.autograd.functional.jacobian(mean_func, z)
            J = J.squeeze(1)  # Remove the batch dimension

            #Compute Pull-back metric
            return torch.matmul(J.T, J)
        
        def compute_energy(c):
            delta_t = 1 / (S - 1)
            energy = 0
            for i in range(c.shape[0] - 1):
                v_i = (c[i+1,:] - c[i,:]) / delta_t
                
                # Recompute Jacobian for each point during optimization
                J = compute_pull_back(c[i,:].reshape(1, 2))  # Use current point for Jacobian
                energy += torch.matmul(v_i, torch.matmul(J, v_i))
            return energy
        
        #________Extract points____________
        data_iter = iter(mnist_train_loader)
        point_list = []
        target = []
        images = []
        for i in range(30):
            x, y = next(data_iter)
            images.append(x)
            x = x.to(device)
            q = model.encoder(x)
            
            z = q.rsample().detach().cpu()
            
            point_list.append(z)  # Keep as tensor
            target.append(y)
        # Concatenate across batches
        z_matrix = torch.cat(point_list, dim=0).numpy()  # Shape: [50*32, 20]
        target_tensor = torch.cat(target, dim=0)  # Shape: [50*32]

        x = z_matrix[:, 0]  # First principal component
        y = z_matrix[:, 1]  # Second principal component

        tensor_list = [tensor.squeeze(1) for tensor in images]

        # Concatenate the list of tensors along the first dimension
        images_tensor = torch.cat(tensor_list, dim=0)

        n = 10  # You can set this to any number of pairs you want to process

        point_pair = select_random_point_pairs(z_matrix.shape[0], n)

        # Track energy values and curves for plotting
        all_curves = []
        all_energy_values = []
        #x1, y1 = next(iter(mnist_test_loader))  # This gives you the first batch
        #x2, y2 = next(iter(mnist_test_loader))  # This gives you the second batch
        for pair in tqdm(range(n), desc="Optimizing points", unit="Point Pair"):
            r_1, r_2 = point_pair[pair]
            
            img_1 = images_tensor[r_1,:,:].unsqueeze(0).to(device)  # First image from the first batch (add batch dimension)

            # Access the second batch
            img_2 = images_tensor[r_2,:,:].unsqueeze(0).to(device)

            point_1 = torch.tensor(z_matrix[r_1])  # Assuming model.encoder gives the latent representation
            point_2 = torch.tensor(z_matrix[r_2])

            # Number of sample points along the curve
            S = 25

            # Initialize intermediate points as a straight line between start and end (excluding start and end)
            c_inner = torch.linspace(0, 1, S-2).unsqueeze(1) * (point_2 - point_1) + point_1  # Exclude start and end
            # Now pass only the intermediate points to nn.Parameter
            c_inner = torch.nn.Parameter(c_inner)  # Optimizable parameters
            
            # Track energy over iterations
            energy_values = []
            num_epoch = 100
            # Example optimization loop
            optimizer = torch.optim.Adam([c_inner], lr=0.01)  # Optimizing the intermediate points `c_inner`

            # Wrap the epoch loop with tqdm for progress tracking
            for epoch in range(num_epoch):
                optimizer.zero_grad()  # Zero the gradients
                energy = compute_energy(c_inner)  # Compute the current energy
                energy_values.append(energy.item())  # Store the energy value for this iteration

                energy.backward()  # Compute gradients
                optimizer.step()  # Update the curve points to minimize the energy

            # Store initial curve for plotting
            c_final = torch.vstack([point_1, c_inner.detach(), point_2]).numpy() 

            #plot curve:
            plt.plot(c_final[:, 0], c_final[:, 1])  # Example if it's 2D latent space
        

        scatter = plt.scatter(x, y, c=target_tensor.detach().numpy(), alpha=1, s=3)

        # Create a legend
        # Assuming target_tensor contains values 0, 1, 2 for the three classes
        classes = [0, 1, 2]
        class_labels = ['0', '1', '2']

        # Create a legend with color patches
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(class_value)), markersize=8, label=label)
            for class_value, label in zip(classes, class_labels)
        ]
        plt.legend(handles=legend_handles, title="Classes")

        plt.title('Optimized Geodesic Curve')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.savefig("Project2/curve_plot.png")
        plt.show()

        # After optimization, plot the energy values
        plt.plot(energy_values)
        plt.xlabel('Epochs')
        plt.ylabel('Energy')
        plt.title('Energy During Geodesic Optimization')
        plt.savefig("Project2/energy_optim.png")
        plt.show()

       





           



