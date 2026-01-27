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
    """Compute heat equation residual: dT/dt - alpha * (d2T/dx2 + d2T/dy2 + d2T/dz2)

    Note: inputs are [x, y, z, t, heat_flux, T_initial, L1, L2, H, thickness, volume, surface_area, S_V_ratio]
    We only compute derivatives w.r.t. spatial (0,1,2) and time (3)
    """
    inputs.requires_grad_(True)
    T = model(inputs)

    # First derivatives (only for spatial and time, not parameters/geometry)
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
        'batch_size': 512,
        'lr': 1e-3,
        'epochs': 100,
        'hidden_dim': 128,
        'alpha': 1e-5,
        'lambda_physics': 0.1,
    }

    wandb.init(project="fem-pinn-geometry-aware", config=config)

    # Load training data (already filtered)
    print("Loading training data...")
    project_root = Path(__file__).parent.parent
    npz_path = project_root / 'saves' / 'fem_data_train.npz'
    data = np.load(npz_path)

    # Extract all features
    coords = data['coords']
    time = data['time'][:, None]
    heat_flux = data['heat_flux'][:, None]
    T_initial = data['T_initial'][:, None]
    L1 = data['L1'][:, None]
    L2 = data['L2'][:, None]
    H = data['H'][:, None]
    thickness = data['thickness'][:, None]
    volume = data['volume'][:, None]
    surface_area = data['surface_area'][:, None]
    S_V_ratio = data['S_V_ratio'][:, None]
    temp = data['temperature'][:, None]

    print(f"Training data: {len(coords):,} points")
    print(f"  Coords: {coords.shape}")
    print(f"  Process params: heat_flux={heat_flux.shape}, T_initial={T_initial.shape}")
    print(f"  Geometry params: L1={L1.shape}, L2={L2.shape}, H={H.shape}, thickness={thickness.shape}")
    print(f"  Geometry features: volume={volume.shape}, surface_area={surface_area.shape}, S_V_ratio={S_V_ratio.shape}")

    # Normalize all inputs
    coords_mean, coords_std = coords.mean(0), coords.std(0)
    time_mean, time_std = time.mean(), time.std()
    heat_flux_mean, heat_flux_std = heat_flux.mean(), heat_flux.std()
    T_initial_mean, T_initial_std = T_initial.mean(), T_initial.std()
    L1_mean, L1_std = L1.mean(), L1.std()
    L2_mean, L2_std = L2.mean(), L2.std()
    H_mean, H_std = H.mean(), H.std()
    thickness_mean, thickness_std = thickness.mean(), thickness.std()
    volume_mean, volume_std = volume.mean(), volume.std()
    surface_area_mean, surface_area_std = surface_area.mean(), surface_area.std()
    S_V_ratio_mean, S_V_ratio_std = S_V_ratio.mean(), S_V_ratio.std()
    temp_mean, temp_std = temp.mean(), temp.std()

    coords_norm = (coords - coords_mean) / coords_std
    time_norm = (time - time_mean) / time_std
    heat_flux_norm = (heat_flux - heat_flux_mean) / heat_flux_std
    T_initial_norm = (T_initial - T_initial_mean) / T_initial_std
    L1_norm = (L1 - L1_mean) / L1_std
    L2_norm = (L2 - L2_mean) / L2_std
    H_norm = (H - H_mean) / H_std
    thickness_norm = (thickness - thickness_mean) / thickness_std
    volume_norm = (volume - volume_mean) / volume_std
    surface_area_norm = (surface_area - surface_area_mean) / surface_area_std
    S_V_ratio_norm = (S_V_ratio - S_V_ratio_mean) / S_V_ratio_std
    temp_norm = (temp - temp_mean) / temp_std

    # Create input: [x, y, z, t, heat_flux, T_initial, L1, L2, H, thickness, volume, surface_area, S_V_ratio]
    X = np.hstack([
        coords_norm,           # x, y, z (3)
        time_norm,             # t (1)
        heat_flux_norm,        # heat_flux (1)
        T_initial_norm,        # T_initial (1)
        L1_norm,               # L1 (1)
        L2_norm,               # L2 (1)
        H_norm,                # H (1)
        thickness_norm,        # thickness (1)
        volume_norm,           # volume (1)
        surface_area_norm,     # surface_area (1)
        S_V_ratio_norm         # S_V_ratio (1)
    ])
    Y = temp_norm

    print(f"\nInput shape: {X.shape} (13 features)")
    print(f"  Spatial: x, y, z (3)")
    print(f"  Temporal: t (1)")
    print(f"  Process: heat_flux, T_initial (2)")
    print(f"  Geometry: L1, L2, H, thickness, volume, surface_area, S_V_ratio (7)")

    # Save normalization params
    np.savez('normalization_params.npz',
             coords_mean=coords_mean, coords_std=coords_std,
             time_mean=time_mean, time_std=time_std,
             heat_flux_mean=heat_flux_mean, heat_flux_std=heat_flux_std,
             T_initial_mean=T_initial_mean, T_initial_std=T_initial_std,
             L1_mean=L1_mean, L1_std=L1_std,
             L2_mean=L2_mean, L2_std=L2_std,
             H_mean=H_mean, H_std=H_std,
             thickness_mean=thickness_mean, thickness_std=thickness_std,
             volume_mean=volume_mean, volume_std=volume_std,
             surface_area_mean=surface_area_mean, surface_area_std=surface_area_std,
             S_V_ratio_mean=S_V_ratio_mean, S_V_ratio_std=S_V_ratio_std,
             temp_mean=temp_mean, temp_std=temp_std)

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
