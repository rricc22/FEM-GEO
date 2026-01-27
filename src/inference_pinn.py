#!/usr/bin/env python3
"""Parametric PINN Inference"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path
from pinn_arch import PINN

def load_model(model_path='pinn_model.pt', hidden_dim=64):
    """Load trained PINN model"""
    model = PINN(hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict(model, coords, time, heat_flux, T_initial, norm_params):
    """Predict temperature with parameters"""
    # Normalize inputs
    coords_norm = (coords - norm_params['coords_mean']) / norm_params['coords_std']
    time_norm = (time - norm_params['time_mean']) / norm_params['time_std']
    heat_flux_norm = (heat_flux - norm_params['heat_flux_mean']) / norm_params['heat_flux_std']
    T_initial_norm = (T_initial - norm_params['T_initial_mean']) / norm_params['T_initial_std']

    # Combine (x, y, z, t, heat_flux, T_initial)
    inputs = np.hstack([coords_norm, time_norm, heat_flux_norm, T_initial_norm])
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        temp_norm = model(inputs_tensor).numpy()

    # Denormalize
    temp = temp_norm * norm_params['temp_std'] + norm_params['temp_mean']
    return temp

def load_normalization_params(npz_path='normalization_params.npz'):
    """Load normalization parameters"""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}

def visualize_prediction(coords, temp_pred, temp_true, case_id, timestep):
    """Visualize prediction vs ground truth"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Prediction
    scatter1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=temp_pred, cmap='hot', s=20)
    axes[0].set_title(f'PINN Prediction\n{case_id}, t={timestep}')
    axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
    plt.colorbar(scatter1, ax=axes[0], label='Temperature (K)')

    # Ground truth
    scatter2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=temp_true, cmap='hot', s=20)
    axes[1].set_title(f'FEM Ground Truth\n{case_id}, t={timestep}')
    axes[1].set_xlabel('X'); axes[1].set_ylabel('Y')
    plt.colorbar(scatter2, ax=axes[1], label='Temperature (K)')

    # Error
    error = np.abs(temp_pred - temp_true)
    scatter3 = axes[2].scatter(coords[:, 0], coords[:, 1], c=error, cmap='Reds', s=20)
    axes[2].set_title(f'Absolute Error\nMean: {error.mean():.2f} K')
    axes[2].set_xlabel('X'); axes[2].set_ylabel('Y')
    plt.colorbar(scatter3, ax=axes[2], label='Error (K)')

    plt.tight_layout()
    plt.savefig(f'prediction_{case_id}_t{timestep}.png', dpi=150)
    print(f"Saved: prediction_{case_id}_t{timestep}.png")
    plt.close()

if __name__ == '__main__':
    # Load model and normalization params
    print("Loading model...")
    model = load_model('pinn_model.pt')
    norm_params = load_normalization_params('normalization_params.npz')

    # Load manifest
    project_root = Path(__file__).parent.parent
    manifest_path = project_root / 'scripts' / 'fem_cases' / 'manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Load all data
    npz_path = project_root / 'saves' / 'fem_all_timesteps.npz'
    data = np.load(npz_path)
    coords = data['coords']
    temp_true = data['temperature']
    time = data['time']
    n_timesteps = data['n_timesteps']

    # Find an extrapolation test case
    test_cases = [c for c in manifest['cases'] if c['split'] == 'extrapolation_test']

    if not test_cases:
        print("No extrapolation test cases found!")
        exit(1)

    # Test on first extrapolation case
    case_info = test_cases[4]
    case_id = case_info['case_id']
    params = case_info['params']

    print(f"\nTesting on EXTRAPOLATION case: {case_id}")
    print(f"  heat_flux: {params['heat_flux']:.0f} W/m²")
    print(f"  T_initial: {params['T_initial']:.2f} K")
    print(f"  (This is OUTSIDE the training range!)")

    # Get data for this case
    case_idx = manifest['cases'].index(case_info)
    n_points_per_timestep = len(coords) // (len(manifest['cases']) * n_timesteps)
    start_idx = case_idx * n_timesteps * n_points_per_timestep

    # Test on timestep 10
    timestep_idx = 10
    point_start = start_idx + timestep_idx * n_points_per_timestep
    point_end = point_start + n_points_per_timestep

    coords_t = coords[point_start:point_end]
    time_t = time[point_start:point_end, None]
    temp_t = temp_true[point_start:point_end]

    # Create parameter arrays
    heat_flux_t = np.full((len(coords_t), 1), params['heat_flux'])
    T_initial_t = np.full((len(coords_t), 1), params['T_initial'])

    # Predict
    print(f"\nPredicting timestep {timestep_idx}...")
    temp_pred = predict(model, coords_t, time_t, heat_flux_t, T_initial_t, norm_params)

    # Calculate error
    error = np.abs(temp_pred.flatten() - temp_t)
    print(f"Mean absolute error: {error.mean():.4f} K")
    print(f"Max error: {error.max():.4f} K")
    print(f"RMSE: {np.sqrt((error**2).mean()):.4f} K")

    # Visualize
    visualize_prediction(coords_t, temp_pred.flatten(), temp_t, case_id, timestep_idx)

    print("\nExtrapolation test complete!")
