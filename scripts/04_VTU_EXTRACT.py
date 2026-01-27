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

def process_all_cases(fem_cases_dir, output_dir):
    """Process all VTU files from all cases"""

    fem_cases_path = Path(fem_cases_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = fem_cases_path / 'manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print("="*60)
    print(f"Extracting VTU data from {len(manifest['cases'])} cases")
    print("="*60)

    all_coords = []
    all_temps = []
    all_times = []

    for i, case_info in enumerate(manifest['cases'], 1):
        case_id = case_info['case_id']
        case_dir = Path(case_info['case_dir'])
        split = case_info.get('split', 'train')

        print(f"\n[{i}/{len(manifest['cases'])}] {case_id} ({split})")

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

            all_coords.append(coords)
            all_temps.append(temp)
            all_times.append(np.full(len(temp), timestep))

        print(f"  ✓ Extracted {len(vtu_files)} timesteps")

    # Combine all data
    print("\nCombining all data...")
    coords_combined = np.vstack(all_coords)
    temp_combined = np.concatenate(all_temps)
    time_combined = np.concatenate(all_times)

    # Save
    output_file = output_path / 'fem_all_timesteps.npz'
    np.savez(output_file,
             coords=coords_combined,
             temperature=temp_combined,
             time=time_combined,
             n_timesteps=len(vtu_files))  # per case

    print("\n" + "="*60)
    print(f"Saved to: {output_file}")
    print(f"  Total cases: {len(manifest['cases'])}")
    print(f"  Total points: {len(temp_combined):,}")
    print(f"  coords shape: {coords_combined.shape}")
    print(f"  temperature shape: {temp_combined.shape}")
    print(f"  time shape: {time_combined.shape}")
    print("="*60)

if __name__ == '__main__':
    # Use automatic paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    process_all_cases(
        fem_cases_dir=script_dir / 'fem_cases',
        output_dir=project_root / 'saves'
    )
