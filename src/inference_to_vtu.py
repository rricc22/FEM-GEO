#!/usr/bin/env python3
"""Generate VTU files from PINN predictions for ParaView visualization"""

import numpy as np
import torch
import json
import meshio
from pathlib import Path
from pinn_arch import PINN
from tqdm import tqdm

def load_model(model_path='pinn_model.pt', hidden_dim=128):
    """Load trained PINN model"""
    model = PINN(hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def load_normalization_params(npz_path='normalization_params.npz'):
    """Load normalization parameters"""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}

def predict_batch(model, coords, time, heat_flux, T_initial, L1, L2, H, thickness,
                  volume, surface_area, S_V_ratio, norm_params):
    """Predict temperature for a batch of points"""
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

def generate_vtu_series(case_id, geometry_id, params, geom_features, model, norm_params,
                        reference_vtu_path, output_dir, num_timesteps=50, dt=1.0):
    """Generate VTU time series for a case using PINN predictions"""

    # Load mesh structure from reference VTU
    print(f"Loading mesh from {reference_vtu_path}")
    mesh = meshio.read(reference_vtu_path)
    points = mesh.points
    cells = mesh.cells

    n_points = len(points)
    print(f"Mesh has {n_points} points")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract parameters
    heat_flux = params['heat_flux']
    T_initial = params['T_initial']
    L1 = geom_features['params']['L1']
    L2 = geom_features['params']['L2']
    H = geom_features['params']['H']
    thickness = geom_features['params']['thickness']
    volume = geom_features['volume']
    surface_area = geom_features['surface_area']
    S_V_ratio = geom_features['S_V_ratio']

    # Generate predictions for each timestep
    print(f"\nGenerating {num_timesteps} timesteps...")
    for timestep in tqdm(range(num_timesteps), desc="Timesteps"):
        time_val = timestep * dt

        # Create input arrays (same parameters for all points)
        time_array = np.full((n_points, 1), time_val)
        heat_flux_array = np.full((n_points, 1), heat_flux)
        T_initial_array = np.full((n_points, 1), T_initial)
        L1_array = np.full((n_points, 1), L1)
        L2_array = np.full((n_points, 1), L2)
        H_array = np.full((n_points, 1), H)
        thickness_array = np.full((n_points, 1), thickness)
        volume_array = np.full((n_points, 1), volume)
        surface_area_array = np.full((n_points, 1), surface_area)
        S_V_ratio_array = np.full((n_points, 1), S_V_ratio)

        # Predict temperatures
        temp_pred = predict_batch(
            model, points, time_array, heat_flux_array, T_initial_array,
            L1_array, L2_array, H_array, thickness_array,
            volume_array, surface_area_array, S_V_ratio_array,
            norm_params
        )

        # Create new mesh with predictions
        mesh_out = meshio.Mesh(
            points=points,
            cells=cells,
            point_data={'temperature': temp_pred.flatten()}
        )

        # Write VTU file
        output_file = output_dir / f"{case_id}_pinn_t{timestep:04d}.vtu"
        mesh_out.write(output_file)

    print(f"\nGenerated {num_timesteps} VTU files in {output_dir}")
    print(f"Load in ParaView: {output_dir}/{case_id}_pinn_t*.vtu")

    return output_dir

def compare_with_fem(case_id, pinn_dir, fem_dir, timestep=25):
    """Load and compare PINN vs FEM for a specific timestep"""
    pinn_file = pinn_dir / f"{case_id}_pinn_t{timestep:04d}.vtu"
    fem_file = fem_dir / f"case_t{timestep:04d}.vtu"

    if not pinn_file.exists() or not fem_file.exists():
        print(f"Files not found for comparison at timestep {timestep}")
        return

    pinn_mesh = meshio.read(pinn_file)
    fem_mesh = meshio.read(fem_file)

    pinn_temp = pinn_mesh.point_data['temperature']
    fem_temp = fem_mesh.point_data['temperature']

    error = np.abs(pinn_temp - fem_temp)

    print(f"\n{'='*60}")
    print(f"Comparison at timestep {timestep}")
    print(f"{'='*60}")
    print(f"PINN temperature range: {pinn_temp.min():.2f} - {pinn_temp.max():.2f} K")
    print(f"FEM temperature range:  {fem_temp.min():.2f} - {fem_temp.max():.2f} K")
    print(f"\nError statistics:")
    print(f"  Mean absolute error: {error.mean():.4f} K")
    print(f"  Max error: {error.max():.4f} K")
    print(f"  RMSE: {np.sqrt((error**2).mean()):.4f} K")
    print(f"  Relative error: {(error.mean() / fem_temp.mean() * 100):.2f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    # Load model and normalization params
    print("Loading PINN model...")
    model = load_model('/home/riccardo/Documents/FEM-GEO/pinn_model.pt', hidden_dim=128)
    norm_params = load_normalization_params('/home/riccardo/Documents/FEM-GEO/normalization_params.npz')

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

    # Find extrapolation test cases
    test_cases = [c for c in manifest['cases'] if c['split'] == 'extrapolation_test']

    if not test_cases:
        print("No extrapolation test cases found!")
        exit(1)

    print(f"\nFound {len(test_cases)} extrapolation test cases")
    print("\nSelect a case to generate VTU files:")
    for i, case in enumerate(test_cases[:10]):  # Show first 10
        print(f"  [{i}] {case['case_id']}: {case['geometry_name']}, "
              f"heat_flux={case['params']['heat_flux']:.0f} W/m²")

    # Default: use first test case
    case_idx = 4
    case_info = test_cases[case_idx]

    case_id = case_info['case_id']
    geometry_id = case_info['geometry_id']
    geometry_name = case_info['geometry_name']
    params = case_info['params']
    geom = geom_dict[geometry_id]

    print(f"\n{'='*60}")
    print(f"Generating VTU series for: {case_id}")
    print(f"{'='*60}")
    print(f"Geometry: {geometry_name}")
    print(f"  L1={geom['params']['L1']}mm, L2={geom['params']['L2']}mm, "
          f"H={geom['params']['H']}mm, thickness={geom['params']['thickness']}mm")
    print(f"Process Parameters:")
    print(f"  heat_flux: {params['heat_flux']:.0f} W/m²")
    print(f"  T_initial: {params['T_initial']:.2f} K")
    print(f"{'='*60}\n")

    # Find reference VTU file (from FEM results)
    fem_case_dir = project_root / 'fem_cases' / case_id
    reference_vtu = list(fem_case_dir.glob('case_t*.vtu'))[0]

    # Output directory for PINN VTU files
    output_dir = project_root / 'pinn_predictions' / case_id

    # Generate VTU time series
    generate_vtu_series(
        case_id=case_id,
        geometry_id=geometry_id,
        params=params,
        geom_features=geom,
        model=model,
        norm_params=norm_params,
        reference_vtu_path=reference_vtu,
        output_dir=output_dir,
        num_timesteps=50,
        dt=1.0
    )

    # Compare with FEM at middle timestep
    compare_with_fem(case_id, output_dir, fem_case_dir, timestep=25)

