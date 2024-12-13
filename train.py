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
    """VicReg loss computation per timestep"""
    B, T, D = z1.shape

    total_loss = 0
    for t in range(T):
        # Take each timestep: [B, D]
        z1_t = z1[:, t]
        z2_t = z2[:, t]

        # Invariance loss
        sim_loss = F.mse_loss(z1_t, z2_t)

        # Variance loss
        std_z1 = torch.sqrt(z1_t.var(dim=0) + 1e-04)
        std_z2 = torch.sqrt(z2_t.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # Covariance loss
        z1_t = z1_t - z1_t.mean(dim=0)
        z2_t = z2_t - z2_t.mean(dim=0)
        cov_z1 = (z1_t.T @ z1_t) / (z1_t.shape[0] - 1)
        cov_z2 = (z2_t.T @ z2_t) / (z2_t.shape[0] - 1)
        cov_loss = (
            off_diagonal(cov_z1).pow_(2).sum() / D
            + off_diagonal(cov_z2).pow_(2).sum() / D
        )

        loss = sim_coef * sim_loss + std_coef * std_loss + cov_coef * cov_loss
        total_loss += loss

    return total_loss / T  # Average over timesteps


def train_jepa(
    model,
    train_loader,
    optimizer,
    device,
    epochs=20,
    log_interval=100,
    patience=3,  # Number of epochs to wait for improvement
    min_delta=1e-4,  # Minimum change to qualify as an improvement
):
    model.train()

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

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

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            # Restore best model
            model.load_state_dict(best_model_state)
            break

    return model, best_loss


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

    # Train with early stopping
    model, best_loss = train_jepa(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        patience=10,  # Stop if no improvement for 10 epochs
        min_delta=1e-4,  # Minimum improvement threshold
    )

    print(f"Training finished with best loss: {best_loss:.4f}")
    torch.save(model.state_dict(), "jepa_model.pth")


if __name__ == "__main__":
    main()
