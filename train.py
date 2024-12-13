import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import create_wall_dataloader
from JEPA_model import JEPAModel
import numpy as np
from tqdm import tqdm


def off_diagonal(x):
    """Return off-diagonal elements of a square matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1, z2, sim_coef=25.0, std_coef=25.0, cov_coef=1.0):
    """VicReg loss computation"""
    # Invariance loss
    sim_loss = F.mse_loss(z1, z2)

    # Variance loss
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-04)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

    # Covariance loss
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
    cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
    cov_loss = (
        off_diagonal(cov_z1).pow_(2).sum() / z1.shape[1]
        + off_diagonal(cov_z2).pow_(2).sum() / z2.shape[1]
    )

    loss = sim_coef * sim_loss + std_coef * std_loss + cov_coef * cov_loss
    return loss


def train_jepa(model, train_loader, optimizer, device, epochs=100, log_interval=100):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            states = batch.states.to(device)
            actions = batch.actions.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(states, actions)
            targets = model.compute_target(states)

            # Compute VicReg loss
            loss = vicreg_loss(predictions, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Momentum update of target network
            model.momentum_update()

            # Logging
            total_loss += loss.item()
            if batch_idx % log_interval == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Avg Loss": f"{total_loss/(batch_idx+1):.4f}",
                    }
                )

        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}")


def main():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loader
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train", batch_size=BATCH_SIZE, train=True
    )

    # Initialize model
    model = JEPAModel(latent_dim=256, use_momentum=True).to(DEVICE)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    train_jepa(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
    )

    # Save model
    torch.save(model.state_dict(), "jepa_model.pth")


if __name__ == "__main__":
    main()
