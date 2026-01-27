#!/usr/bin/env python3
"""Geometry-Aware Parametric PINN Inference"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path
from pinn_arch import PINN

def load_model(model_path='pinn_model.pt', hidden_dim=128):
    """Load trained PINN model"""
    model = PINN(hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict(model, coords, time, heat_flux, T_initial, L1, L2, H, thickness,
            volume, surface_area, S_V_ratio, norm_params):
    """Predict temperature with all parameters (geometry-aware)"""
    # Normalize inputs
    coords_norm = (coords - norm_params['coords_mean']) / norm_params['coords_std']
    time_norm = (time - norm_params['time_mean']) / norm_params['time_std']
    heat_flux_norm = (heat_flux - norm_params['heat_flux_mean']) / norm_params['heat_flux_std']
    T_initial_norm = (T_initial - norm_params['T_initial_mean']) / norm_params['T_initial_std']
    L1_norm = (L1 - norm_params['L1_mean']) / norm_params['L1_std']
    L2_norm = (L2 - norm_params['L2_mean']) / norm_params['L2_std']
    H_norm = (H - norm_params['H_mean']) / norm_params['H_std']
    thickness_norm = (thickness - norm_params['thickness_mean']) / norm_params['thickness_std']
    volume_norm = (volume - norm_params['volume_mean']) / norm_params['volume_std']
    surface_area_norm = (surface_area - norm_params['surface_area_mean']) / norm_params['surface_area_std']
    S_V_ratio_norm = (S_V_ratio - norm_params['S_V_ratio_mean']) / norm_params['S_V_ratio_std']

    # Combine all 13 inputs
    inputs = np.hstack([
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

def visualize_prediction(coords, temp_pred, temp_true, case_id, timestep, geom_info, process_info):
    """Visualize prediction vs ground truth with geometry and process info"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Prediction
    scatter1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=temp_pred, cmap='hot', s=20)
    title1 = f'PINN Prediction\n{case_id}, t={timestep}s'
    axes[0].set_title(title1)
    axes[0].set_xlabel('X (mm)'); axes[0].set_ylabel('Y (mm)')
    plt.colorbar(scatter1, ax=axes[0], label='Temperature (K)')

    # Ground truth
    scatter2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=temp_true, cmap='hot', s=20)
    title2 = f'FEM Ground Truth\n{geom_info["name"]}'
    axes[1].set_title(title2)
    axes[1].set_xlabel('X (mm)'); axes[1].set_ylabel('Y (mm)')
    plt.colorbar(scatter2, ax=axes[1], label='Temperature (K)')

    # Error
    error = np.abs(temp_pred - temp_true)
    scatter3 = axes[2].scatter(coords[:, 0], coords[:, 1], c=error, cmap='Reds', s=20)
    title3 = f'Absolute Error\nMean: {error.mean():.2f} K, Max: {error.max():.2f} K'
    axes[2].set_title(title3)
    axes[2].set_xlabel('X (mm)'); axes[2].set_ylabel('Y (mm)')
    plt.colorbar(scatter3, ax=axes[2], label='Error (K)')

    # Add text box with parameters
    info_text = (
        f"Geometry: L1={geom_info['L1']}mm, L2={geom_info['L2']}mm, "
        f"H={geom_info['H']}mm, t={geom_info['thickness']}mm\n"
        f"Volume: {geom_info['volume']:.0f} mm³, S/V: {geom_info['S_V_ratio']:.3f}\n"
        f"Process: heat_flux={process_info['heat_flux']:.0f} W/m², "
        f"T_initial={process_info['T_initial']:.2f} K"
    )
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(f'prediction_{case_id}_t{timestep}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: prediction_{case_id}_t{timestep}.png")
    plt.close()

if __name__ == '__main__':
    # Load model and normalization params
    print("Loading model...")
    model = load_model('pinn_model.pt', hidden_dim=128)
    norm_params = load_normalization_params('normalization_params.npz')

    # Load manifest
    project_root = Path(__file__).parent.parent
    manifest_path = project_root / 'fem_cases' / 'manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Load geometry features
    geom_path = project_root / 'CAD' / 'geometries' / 'all_geometries.json'
    with open(geom_path, 'r') as f:
        geometries = json.load(f)
    geom_dict = {g['geometry_id']: g for g in geometries}

    # Load test data (extrapolation cases)
    print("Loading test data...")
    test_data_path = project_root / 'saves' / 'fem_data_test.npz'
    test_data = np.load(test_data_path)

    coords = test_data['coords']
    time = test_data['time']
    heat_flux = test_data['heat_flux']
    T_initial = test_data['T_initial']
    L1 = test_data['L1']
    L2 = test_data['L2']
    H = test_data['H']
    thickness = test_data['thickness']
    volume = test_data['volume']
    surface_area = test_data['surface_area']
    S_V_ratio = test_data['S_V_ratio']
    temp_true = test_data['temperature']

    print(f"\nTest dataset: {len(coords):,} points")

    # Find unique test cases
    test_cases = [c for c in manifest['cases'] if c['split'] == 'extrapolation_test']

    if not test_cases:
        print("No extrapolation test cases found!")
        exit(1)

    print(f"Found {len(test_cases)} extrapolation test cases")

    # Test on first extrapolation case
    case_info = test_cases[4]
    case_id = case_info['case_id']
    geometry_id = case_info['geometry_id']
    geometry_name = case_info['geometry_name']
    params = case_info['params']
    geom = geom_dict[geometry_id]

    print(f"\n{'='*60}")
    print(f"Testing on EXTRAPOLATION case: {case_id}")
    print(f"{'='*60}")
    print(f"Geometry: {geometry_name}")
    print(f"  L1={geom['params']['L1']}mm, L2={geom['params']['L2']}mm, "
          f"H={geom['params']['H']}mm, thickness={geom['params']['thickness']}mm")
    print(f"  Volume: {geom['volume']:.0f} mm³, S/V: {geom['S_V_ratio']:.3f}")
    print(f"Process Parameters:")
    print(f"  heat_flux: {params['heat_flux']:.0f} W/m²")
    print(f"  T_initial: {params['T_initial']:.2f} K")
    print(f"  (OUTSIDE training range: heat_flux should be 50k-250k)")
    print(f"{'='*60}")

    # Sample some points for testing (e.g., timestep 25 - middle of simulation)
    # Get indices where parameters match this case
    mask = (np.abs(heat_flux - params['heat_flux']) < 1.0) & \
           (np.abs(T_initial - params['T_initial']) < 0.1) & \
           (np.abs(L1 - geom['params']['L1']) < 0.1)

    case_indices = np.where(mask)[0]

    if len(case_indices) == 0:
        print("ERROR: No points found for this case!")
        exit(1)

    # Get points at a specific timestep (middle of sequence)
    unique_times = np.unique(time[case_indices])
    mid_time_idx = len(unique_times) // 2
    target_time = unique_times[mid_time_idx]

    timestep_mask = case_indices[np.abs(time[case_indices] - target_time) < 0.1]

    print(f"\nPredicting timestep {int(target_time)} ({len(timestep_mask)} points)...")

    # Extract data for this timestep
    coords_t = coords[timestep_mask]
    time_t = time[timestep_mask][:, None]
    heat_flux_t = heat_flux[timestep_mask][:, None]
    T_initial_t = T_initial[timestep_mask][:, None]
    L1_t = L1[timestep_mask][:, None]
    L2_t = L2[timestep_mask][:, None]
    H_t = H[timestep_mask][:, None]
    thickness_t = thickness[timestep_mask][:, None]
    volume_t = volume[timestep_mask][:, None]
    surface_area_t = surface_area[timestep_mask][:, None]
    S_V_ratio_t = S_V_ratio[timestep_mask][:, None]
    temp_t = temp_true[timestep_mask]

    # Predict
    temp_pred = predict(model, coords_t, time_t, heat_flux_t, T_initial_t,
                       L1_t, L2_t, H_t, thickness_t, volume_t, surface_area_t, S_V_ratio_t,
                       norm_params)

    # Calculate error
    error = np.abs(temp_pred.flatten() - temp_t)
    print(f"\nResults:")
    print(f"  Mean absolute error: {error.mean():.4f} K")
    print(f"  Max error: {error.max():.4f} K")
    print(f"  RMSE: {np.sqrt((error**2).mean()):.4f} K")
    print(f"  Relative error: {(error.mean() / temp_t.mean() * 100):.2f}%")

    # Visualize
    geom_info = {
        'name': geometry_name,
        'L1': geom['params']['L1'],
        'L2': geom['params']['L2'],
        'H': geom['params']['H'],
        'thickness': geom['params']['thickness'],
        'volume': geom['volume'],
        'S_V_ratio': geom['S_V_ratio']
    }
    process_info = {
        'heat_flux': params['heat_flux'],
        'T_initial': params['T_initial']
    }

    visualize_prediction(coords_t, temp_pred.flatten(), temp_t,
                        case_id, int(target_time), geom_info, process_info)

    print(f"\n{'='*60}")
    print("Extrapolation test complete!")
    print(f"{'='*60}")
