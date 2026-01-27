#!/usr/bin/env python3
"""Extract FEM results from all cases"""

import numpy as np
import xml.etree.ElementTree as ET
import base64
from pathlib import Path
import re
import json

def parse_data_array(data_array):
    """Parse DataArray from VTU"""
    format_type = data_array.get('format', 'ascii')
    data_type = data_array.get('type', 'Float32')

    if format_type == 'ascii':
        return np.fromstring(data_array.text, sep=' ')
    elif format_type == 'binary':
        raw = base64.b64decode(data_array.text)
        dtype = np.float64 if data_type == 'Float64' else np.float32
        return np.frombuffer(raw[4:], dtype=dtype)

def read_vtu(filepath):
    """Read VTU file and extract coords and temperature"""
    root = ET.parse(filepath).getroot()

    piece = root.find('.//Piece')
    n_points = int(piece.get('NumberOfPoints'))

    # Get coordinates
    points_data = root.find('.//Points/DataArray')
    n_comp = int(points_data.get('NumberOfComponents', 3))
    coords = parse_data_array(points_data).reshape(-1, n_comp)

    # Get temperature
    temp_array = root.find('.//PointData/DataArray[@Name="temperature"]')
    temperature = parse_data_array(temp_array)

    return coords, temperature

def extract_timestep(filename):
    """Extract timestep from filename like case_t0001.vtu"""
    match = re.search(r't(\d+)', filename)
    return int(match.group(1)) if match else 0

def process_all_cases(fem_cases_dir, geom_features_file, output_dir):
    """Process all VTU files from all cases with geometry and process parameters"""

    fem_cases_path = Path(fem_cases_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = fem_cases_path / 'manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Load geometry features
    with open(geom_features_file, 'r') as f:
        geometries = json.load(f)

    # Create geometry lookup dict
    geom_dict = {g['geometry_id']: g for g in geometries}

    print("="*60)
    print(f"Extracting VTU data from {len(manifest['cases'])} cases")
    print(f"  {manifest['n_geometries']} geometries × {manifest['n_process_params']} process params")
    print("="*60)

    all_coords = []
    all_temps = []
    all_times = []
    all_heat_flux = []
    all_T_initial = []
    all_L1 = []
    all_L2 = []
    all_H = []
    all_thickness = []
    all_volume = []
    all_surface_area = []
    all_S_V_ratio = []
    all_case_ids = []
    all_geom_ids = []
    all_splits = []

    for i, case_info in enumerate(manifest['cases'], 1):
        case_id = case_info['case_id']
        case_dir = Path(case_info['case_dir'])
        geometry_id = case_info['geometry_id']
        geometry_name = case_info['geometry_name']
        split = case_info.get('split', 'train')
        params = case_info['params']

        marker = "🧪" if split == 'extrapolation_test' else "✓"
        print(f"\n{marker} [{i}/{len(manifest['cases'])}] {case_id}: {geometry_name} ({split})")

        # Get geometry features
        geom = geom_dict[geometry_id]
        geom_params = geom['params']

        # Find all VTU files
        vtu_files = sorted(case_dir.glob('*.vtu'))

        if not vtu_files:
            print(f"  ⚠ No VTU files found, skipping...")
            continue

        print(f"  Found {len(vtu_files)} timesteps")

        # Process each timestep
        for vtu_file in vtu_files:
            coords, temp = read_vtu(str(vtu_file))
            timestep = extract_timestep(vtu_file.name)
            n_points = len(temp)

            all_coords.append(coords)
            all_temps.append(temp)
            all_times.append(np.full(n_points, timestep))

            # Process parameters (same for all points in this case)
            all_heat_flux.append(np.full(n_points, params['heat_flux']))
            all_T_initial.append(np.full(n_points, params['T_initial']))

            # Geometry parameters (same for all points in this case)
            all_L1.append(np.full(n_points, geom_params['L1']))
            all_L2.append(np.full(n_points, geom_params['L2']))
            all_H.append(np.full(n_points, geom_params['H']))
            all_thickness.append(np.full(n_points, geom_params['thickness']))
            all_volume.append(np.full(n_points, geom['volume']))
            all_surface_area.append(np.full(n_points, geom['surface_area']))
            all_S_V_ratio.append(np.full(n_points, geom['S_V_ratio']))

            # Case metadata
            all_case_ids.extend([case_id] * n_points)
            all_geom_ids.extend([geometry_id] * n_points)
            all_splits.extend([split] * n_points)

        print(f"  ✓ Extracted {len(vtu_files)} timesteps")

    # Combine all data
    print("\nCombining all data...")
    coords_combined = np.vstack(all_coords)
    temp_combined = np.concatenate(all_temps)
    time_combined = np.concatenate(all_times)
    heat_flux_combined = np.concatenate(all_heat_flux)
    T_initial_combined = np.concatenate(all_T_initial)
    L1_combined = np.concatenate(all_L1)
    L2_combined = np.concatenate(all_L2)
    H_combined = np.concatenate(all_H)
    thickness_combined = np.concatenate(all_thickness)
    volume_combined = np.concatenate(all_volume)
    surface_area_combined = np.concatenate(all_surface_area)
    S_V_ratio_combined = np.concatenate(all_S_V_ratio)

    # Save combined dataset
    output_file = output_path / 'fem_data_all.npz'
    np.savez(output_file,
             # Spatial coordinates
             coords=coords_combined,
             # Time
             time=time_combined,
             # Process parameters
             heat_flux=heat_flux_combined,
             T_initial=T_initial_combined,
             # Geometry parameters
             L1=L1_combined,
             L2=L2_combined,
             H=H_combined,
             thickness=thickness_combined,
             volume=volume_combined,
             surface_area=surface_area_combined,
             S_V_ratio=S_V_ratio_combined,
             # Target
             temperature=temp_combined,
             # Metadata
             case_ids=all_case_ids,
             geom_ids=all_geom_ids,
             splits=all_splits)

    # Also save train/test splits separately
    train_mask = np.array([s == 'train' for s in all_splits])
    test_mask = np.array([s == 'extrapolation_test' for s in all_splits])

    # Training data
    train_file = output_path / 'fem_data_train.npz'
    np.savez(train_file,
             coords=coords_combined[train_mask],
             time=time_combined[train_mask],
             heat_flux=heat_flux_combined[train_mask],
             T_initial=T_initial_combined[train_mask],
             L1=L1_combined[train_mask],
             L2=L2_combined[train_mask],
             H=H_combined[train_mask],
             thickness=thickness_combined[train_mask],
             volume=volume_combined[train_mask],
             surface_area=surface_area_combined[train_mask],
             S_V_ratio=S_V_ratio_combined[train_mask],
             temperature=temp_combined[train_mask])

    # Test data
    test_file = output_path / 'fem_data_test.npz'
    np.savez(test_file,
             coords=coords_combined[test_mask],
             time=time_combined[test_mask],
             heat_flux=heat_flux_combined[test_mask],
             T_initial=T_initial_combined[test_mask],
             L1=L1_combined[test_mask],
             L2=L2_combined[test_mask],
             H=H_combined[test_mask],
             thickness=thickness_combined[test_mask],
             volume=volume_combined[test_mask],
             surface_area=surface_area_combined[test_mask],
             S_V_ratio=S_V_ratio_combined[test_mask],
             temperature=temp_combined[test_mask])

    print("\n" + "="*60)
    print(f"✓ Saved complete dataset: {output_file}")
    print(f"✓ Saved training split: {train_file}")
    print(f"✓ Saved test split: {test_file}")
    print(f"\nDataset Summary:")
    print(f"  Total cases: {len(manifest['cases'])}")
    print(f"  Total points: {len(temp_combined):,}")
    print(f"  Training points: {train_mask.sum():,}")
    print(f"  Test points: {test_mask.sum():,}")
    print(f"\nData shapes:")
    print(f"  coords: {coords_combined.shape}")
    print(f"  time: {time_combined.shape}")
    print(f"  heat_flux: {heat_flux_combined.shape}")
    print(f"  T_initial: {T_initial_combined.shape}")
    print(f"  L1, L2, H, thickness: {L1_combined.shape}")
    print(f"  volume, surface_area, S_V_ratio: {volume_combined.shape}")
    print(f"  temperature (target): {temp_combined.shape}")
    print("="*60)

if __name__ == '__main__':
    # Use automatic paths
    project_root = Path(__file__).parent.parent

    process_all_cases(
        fem_cases_dir=project_root / 'fem_cases',
        geom_features_file=project_root / 'CAD' / 'geometries' / 'all_geometries.json',
        output_dir=project_root / 'saves'
    )
