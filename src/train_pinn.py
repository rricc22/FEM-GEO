#!/usr/bin/env python3
"""Train Parametric PINN on FEM data"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb
import json
from pathlib import Path
from pinn_arch import PINN

def physics_loss(model, inputs, alpha=1e-5):
    """Compute heat equation residual: dT/dt - alpha * (d2T/dx2 + d2T/dy2 + d2T/dz2)"""
    inputs.requires_grad_(True)
    T = model(inputs)

    # First derivatives (only for spatial and time, not parameters)
    grad = torch.autograd.grad(T.sum(), inputs, create_graph=True)[0]
    dT_dx = grad[:, 0:1]
    dT_dy = grad[:, 1:2]
    dT_dz = grad[:, 2:3]
    dT_dt = grad[:, 3:4]

    # Second derivatives
    d2T_dx2 = torch.autograd.grad(dT_dx.sum(), inputs, create_graph=True)[0][:, 0:1]
    d2T_dy2 = torch.autograd.grad(dT_dy.sum(), inputs, create_graph=True)[0][:, 1:2]
    d2T_dz2 = torch.autograd.grad(dT_dz.sum(), inputs, create_graph=True)[0][:, 2:3]

    # Heat equation residual
    residual = dT_dt - alpha * (d2T_dx2 + d2T_dy2 + d2T_dz2)
    return torch.mean(residual ** 2)


# Training

def train():
    # Config
    config = {
        'batch_size': 256,
        'lr': 1e-3,
        'epochs': 100,
        'hidden_dim': 64,
        'alpha': 1e-5,
        'lambda_physics': 0.1,
    }

    wandb.init(project="fem-pinn-parametric", config=config)

    # Load manifest to get parameters
    print("Loading manifest...")
    project_root = Path(__file__).parent.parent
    manifest_path = project_root / 'scripts' / 'fem_cases' / 'manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Load NPZ data
    print("Loading NPZ data...")
    npz_path = project_root / 'saves' / 'fem_all_timesteps.npz'
    data = np.load(npz_path)
    coords = data['coords']
    temp = data['temperature'][:, None]
    time = data['time'][:, None]
    n_timesteps_per_case = data['n_timesteps']

    # Extract parameters for each case and filter training cases
    train_indices = []
    heat_flux_list = []
    T_initial_list = []

    n_points_per_timestep = len(coords) // (len(manifest['cases']) * n_timesteps_per_case)

    for i, case_info in enumerate(manifest['cases']):
        if case_info['split'] == 'train':
            params = case_info['params']
            start_idx = i * n_timesteps_per_case * n_points_per_timestep
            end_idx = start_idx + n_timesteps_per_case * n_points_per_timestep

            train_indices.extend(range(start_idx, end_idx))
            heat_flux_list.extend([params['heat_flux']] * (end_idx - start_idx))
            T_initial_list.extend([params['T_initial']] * (end_idx - start_idx))

    # Filter to training data only
    train_indices = np.array(train_indices)
    coords = coords[train_indices]
    temp = temp[train_indices]
    time = time[train_indices]
    heat_flux = np.array(heat_flux_list)[:, None]
    T_initial = np.array(T_initial_list)[:, None]

    print(f"Training data: {len(coords)} points from training cases only")

    # Normalize
    coords_mean, coords_std = coords.mean(0), coords.std(0)
    temp_mean, temp_std = temp.mean(), temp.std()
    time_mean, time_std = time.mean(), time.std()
    heat_flux_mean, heat_flux_std = heat_flux.mean(), heat_flux.std()
    T_initial_mean, T_initial_std = T_initial.mean(), T_initial.std()

    coords_norm = (coords - coords_mean) / coords_std
    temp_norm = (temp - temp_mean) / temp_std
    time_norm = (time - time_mean) / time_std
    heat_flux_norm = (heat_flux - heat_flux_mean) / heat_flux_std
    T_initial_norm = (T_initial - T_initial_mean) / T_initial_std

    # Create input (x, y, z, t, heat_flux, T_initial)
    X = np.hstack([coords_norm, time_norm, heat_flux_norm, T_initial_norm])
    Y = temp_norm

    print(f"Input shape: {X.shape} (x, y, z, t, heat_flux, T_initial)")

    # Save normalization params
    np.savez('normalization_params.npz',
             coords_mean=coords_mean, coords_std=coords_std,
             temp_mean=temp_mean, temp_std=temp_std,
             time_mean=time_mean, time_std=time_std,
             heat_flux_mean=heat_flux_mean, heat_flux_std=heat_flux_std,
             T_initial_mean=T_initial_mean, T_initial_std=T_initial_std)

    # Dataset
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, Y_tensor)

    # Split
    n_train = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'])

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PINN(hidden_dim=config['hidden_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    print(f"Training on {device}")

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        train_data_loss = 0
        train_phys_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward
            preds = model(inputs)
            loss_data = torch.mean((preds - targets) ** 2)
            loss_phys = physics_loss(model, inputs, alpha=config['alpha'])
            loss = loss_data + config['lambda_physics'] * loss_phys

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_data_loss += loss_data.item()
            train_phys_loss += loss_phys.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                val_loss += torch.mean((preds - targets) ** 2).item()

        train_data_loss /= len(train_loader)
        train_phys_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_data_loss': train_data_loss,
            'train_physics_loss': train_phys_loss,
            'train_total_loss': train_data_loss + config['lambda_physics'] * train_phys_loss,
            'val_loss': val_loss
        })

        print(f"[{epoch+1}/{config['epochs']}] Data: {train_data_loss:.4e} | Phys: {train_phys_loss:.4e} | Val: {val_loss:.4e}")

    # Save
    torch.save(model.state_dict(), 'pinn_model.pt')
    wandb.save('pinn_model.pt')
    print("Training complete!")

if __name__ == '__main__':
    train()
