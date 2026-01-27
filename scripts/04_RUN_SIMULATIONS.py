#!/usr/bin/env python3
"""Run ElmerSolver on all generated cases"""

import json
import shutil
import subprocess
from pathlib import Path

def run_all_simulations(base_dir='fem_cases', mesh_base_dir='elmer_mesh/geometries'):
    """Run ElmerSolver for all cases in manifest"""

    base_path = Path(base_dir)
    mesh_base = Path(mesh_base_dir)

    # Load manifest
    manifest_path = base_path / 'manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Mesh files to copy
    mesh_files = ['mesh.boundary', 'mesh.elements', 'mesh.header', 'mesh.nodes']

    print("="*60)
    print(f"Running ElmerSolver on {len(manifest['cases'])} cases")
    print(f"  {manifest['n_geometries']} geometries × {manifest['n_process_params']} process params")
    print("="*60)

    for i, case_info in enumerate(manifest['cases'], 1):
        case_id = case_info['case_id']
        case_dir = Path(case_info['case_dir'])
        geometry_id = case_info['geometry_id']
        geometry_name = case_info['geometry_name']
        split = case_info.get('split', 'train')

        marker = "🧪" if split == 'extrapolation_test' else "✓"
        print(f"\n{marker} [{i}/{len(manifest['cases'])}] {case_id}: {geometry_name} ({split})")

        # Find geometry mesh directory
        geom_mesh_dir = mesh_base / f"{geometry_id}_{geometry_name}"

        if not geom_mesh_dir.exists():
            print(f"  ✗ Mesh directory not found: {geom_mesh_dir}")
            continue

        # Copy mesh files
        print(f"  Copying mesh from {geometry_id}...")
        for mesh_file in mesh_files:
            src = geom_mesh_dir / mesh_file
            dst = case_dir / mesh_file
            if not src.exists():
                print(f"  ✗ Missing mesh file: {mesh_file}")
                continue
            shutil.copy2(src, dst)

        # Run ElmerSolver
        print("  Running ElmerSolver...")
        try:
            result = subprocess.run(
                ['ElmerSolver', 'case.sif'],
                cwd=case_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout per case
            )

            if result.returncode == 0:
                print("  ✓ Simulation completed successfully")
            else:
                print(f"  ✗ Simulation failed (exit code {result.returncode})")
                print(f"  Error: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            print("  ✗ Simulation timeout (5 min)")
        except FileNotFoundError:
            print("  ✗ ElmerSolver not found! Install Elmer or check PATH")
            return
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*60)
    print("All simulations complete!")
    print("="*60)

if __name__ == '__main__':
    # Use absolute paths
    project_root = Path(__file__).parent.parent
    run_all_simulations(
        base_dir=project_root / 'fem_cases',
        mesh_base_dir=project_root / 'elmer_mesh' / 'geometries'
    )
