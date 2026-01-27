#!/usr/bin/env python3
"""Run ElmerSolver on all generated cases"""

import json
import shutil
import subprocess
from pathlib import Path

def run_all_simulations(base_dir='fem_cases', mesh_dir='elmer_mesh'):
    """Run ElmerSolver for all cases in manifest"""

    base_path = Path(base_dir)
    mesh_path = Path(mesh_dir)

    # Load manifest
    manifest_path = base_path / 'manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Mesh files to copy
    mesh_files = ['mesh.boundary', 'mesh.elements', 'mesh.header', 'mesh.nodes']

    print("="*60)
    print(f"Running ElmerSolver on {len(manifest['cases'])} cases")
    print("="*60)

    for i, case_info in enumerate(manifest['cases'], 1):
        case_id = case_info['case_id']
        case_dir = Path(case_info['case_dir'])
        split = case_info.get('split', 'train')

        print(f"\n[{i}/{len(manifest['cases'])}] {case_id} ({split})")

        # Copy mesh files
        print("  Copying mesh files...")
        for mesh_file in mesh_files:
            src = mesh_path / mesh_file
            dst = case_dir / mesh_file
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
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    run_all_simulations(
        base_dir=script_dir / 'fem_cases',
        mesh_dir=project_root / 'elmer_mesh'
    )
