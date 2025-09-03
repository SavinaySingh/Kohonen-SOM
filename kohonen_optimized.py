import torch


def train(
    input_data: torch.Tensor,
    n_max_iterations: int,
    width: int,
    height: int,
    batch_size: int = 32,
) -> torch.Tensor:
    """Train a Self-Organizing Map (SOM) using the Kohonen algorithm with batch updates (float32)."""

    # Select device
    device = (
        torch.device("mps") if (width >= 500 or height >= 500) else torch.device("cpu")
    )
    input_data = input_data.float().to(device)  # ensure float32

    n_neurons = width * height
    dim = input_data.shape[1]

    # Initialize flat weights as float32
    weights = torch.rand((n_neurons, dim), device=device, dtype=torch.float32)

    # Precompute neuron coordinates (flattened) as float32
    x_coords, y_coords = torch.meshgrid(
        torch.arange(width, device=device, dtype=torch.float32),
        torch.arange(height, device=device, dtype=torch.float32),
        indexing="ij",
    )
    neuron_coords = torch.stack([x_coords.ravel(), y_coords.ravel()], dim=1)

    # Precompute pairwise squared distances between neurons (float32)
    pairwise_dists_sq = (
        (neuron_coords[:, None, :] - neuron_coords[None, :, :]) ** 2
    ).sum(dim=-1)

    # SOM learning rate and neighborhood decay
    σ0 = max(width, height) / 2
    α0 = 0.1
    λ = (
        n_max_iterations
        / torch.log(torch.tensor(σ0, device=device, dtype=torch.float32))
        if σ0 > 1
        else 1.0
    )

    t_array = torch.arange(n_max_iterations, device=device, dtype=torch.float32)
    σt_array = σ0 * torch.exp(-t_array / λ)
    σt2_array = σt_array**2
    αt_array = α0 * torch.exp(-t_array / λ)

    n_samples = input_data.shape[0]

    # Training loop
    for t in range(n_max_iterations):
        σt2 = σt2_array[t]
        αt = αt_array[t]

        # Process in batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = input_data[start:end]  # shape: (batch_size, dim)

            # Compute distances from all neurons to all batch samples: shape (batch_size, n_neurons)
            dists = ((weights[None, :, :] - batch[:, None, :]) ** 2).sum(dim=2)

            # Find BMU indices for each sample in batch
            bmu_indices = torch.argmin(dists, dim=1)  # (batch,)

            # Compute neighborhood and update weights for each sample in batch
            for i, bmu_index in enumerate(bmu_indices):
                θt = torch.exp(-pairwise_dists_sq[bmu_index] / (2 * σt2))
                weights += αt * θt[:, None] * (batch[i] - weights)

    # Reshape back to 3D map and return
    return weights.view(width, height, dim).cpu()
