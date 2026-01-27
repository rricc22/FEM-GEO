#!/usr/bin/env python3
"""
Generate SIF files for multiple geometries × process parameters
"""

import numpy as np
from pathlib import Path
import json

class FEMCaseGenerator:
    """Generate Elmer FEM cases with geometry and parameter variations"""

    def __init__(self, base_dir='fem_cases'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def generate_elmer_sif(self, case_params, output_dir):
        """Generate Elmer SIF file"""

        sif_template = f"""
Header
  CHECK KEYWORDS Warn
  Mesh DB "." "."
  Include Path ""
  Results Directory ""
End

Simulation
  Max Output Level = 5
  Coordinate System = Cartesian
  Simulation Type = Transient
  Steady State Max Iterations = 1
  Output Intervals = 1
  Timestepping Method = BDF
  BDF Order = 2
  Timestep intervals = {case_params['n_timesteps']}
  Timestep Sizes = {case_params['dt']}
  Solver Input File = case.sif
  Post File = case.vtu
  Output File = case.result
End

Body 1
  Target Bodies(1) = 1
  Equation = 1
  Material = 1
  Initial condition = 1
End

Solver 1
  Equation = Heat Equation
  Procedure = "HeatSolve" "HeatSolver"
  Variable = Temperature
  Stabilize = True
  Steady State Convergence Tolerance = 1.0e-5
  Nonlinear System Convergence Tolerance = 1.0e-7
  Nonlinear System Max Iterations = 20
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 500
  Linear System Convergence Tolerance = 1.0e-10
  Linear System Preconditioning = ILU0
End

Solver 2
  Equation = Result Output
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = "case"
  Vtu format = True
  Binary Output = False
  Exec Solver = After Timestep
End

Equation 1
  Name = "Heat"
  Active Solvers(1) = 1
End

Material 1
  Name = "Steel"
  Heat Conductivity = {case_params['conductivity']}
  Heat Capacity = {case_params['heat_capacity']}
  Density = {case_params['density']}
End

Initial Condition 1
  Temperature = {case_params['T_initial']}
End

Boundary Condition 1
  Target Boundaries(1) = 1
  Heat Flux = {case_params['heat_flux']}
  Heat Transfer Coefficient = {case_params['h_conv']}
  External Temperature = {case_params['T_ambient']}
End
"""
        output_path = Path(output_dir) / 'case.sif'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(sif_template)

        return output_path

    def generate_cases(self, geometries, n_process_params=5):
        """Generate cases: geometries × process parameters"""

        cases = []

        # Base parameters
        base = {
            'conductivity': 50.0,
            'heat_capacity': 500.0,
            'density': 7850.0,
            'T_ambient': 293.15,
            'h_conv': 25.0,
            'dt': 1.0,
            'n_timesteps': 50,
        }

        # Process parameter ranges
        heat_flux_min = 50000.0
        heat_flux_max = 250000.0
        T_initial_min = 273.15
        T_initial_max = 363.15

        np.random.seed(42)

        case_id = 0

        # For each geometry
        for geom in geometries:
            # Training cases: varied process parameters
            for i in range(n_process_params):
                params = base.copy()
                params['heat_flux'] = np.random.uniform(heat_flux_min, heat_flux_max)
                params['T_initial'] = np.random.uniform(T_initial_min, T_initial_max)
                params['case_id'] = f'case_{case_id:04d}'
                params['geometry_id'] = geom['geometry_id']
                params['geometry_name'] = geom['name']
                params['split'] = 'train'
                cases.append(params)
                case_id += 1

        # Add extrapolation test cases (random geometries, extreme parameters)
        n_test = min(5, len(geometries))
        test_geoms = np.random.choice(geometries, n_test, replace=False)

        for geom in test_geoms:
            params = base.copy()
            params['heat_flux'] = 280000.0  # Outside training range
            params['T_initial'] = 303.15
            params['case_id'] = f'case_{case_id:04d}'
            params['geometry_id'] = geom['geometry_id']
            params['geometry_name'] = geom['name']
            params['split'] = 'extrapolation_test'
            cases.append(params)
            case_id += 1

        return cases

    def generate_all_cases(self, geometries, n_process_params=5):
        """Generate all SIF files and manifest"""

        cases = self.generate_cases(geometries, n_process_params)

        n_train = sum(1 for c in cases if c['split'] == 'train')
        n_test = sum(1 for c in cases if c['split'] == 'extrapolation_test')

        print(f"\n{'='*60}")
        print(f"Generating {len(cases)} Cases")
        print(f"  {len(geometries)} geometries × {n_process_params} process params")
        print(f"  {n_train} training + {n_test} extrapolation test")
        print(f"{'='*60}\n")

        manifest = {
            'n_cases': len(cases),
            'n_geometries': len(geometries),
            'n_process_params': n_process_params,
            'training_range': {
                'heat_flux': [50000.0, 250000.0],
                'T_initial': [273.15, 363.15]
            },
            'cases': []
        }

        for case_params in cases:
            case_id = case_params['case_id']
            case_dir = self.base_dir / case_id

            # Generate SIF file
            sif_path = self.generate_elmer_sif(case_params, case_dir)

            # Store case info
            manifest['cases'].append({
                'case_id': case_id,
                'geometry_id': case_params['geometry_id'],
                'geometry_name': case_params['geometry_name'],
                'params': case_params,
                'sif_file': str(sif_path),
                'case_dir': str(case_dir),
                'split': case_params['split']
            })

            marker = "🧪" if case_params['split'] == 'extrapolation_test' else "✓"
            print(f"{marker} {case_id}: {case_params['geometry_name']}, "
                  f"heat={case_params['heat_flux']:.0f}, T={case_params['T_initial']:.1f}K")

        # Save manifest
        manifest_path = self.base_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ Generated {len(cases)} cases")
        print(f"✓ Manifest: {manifest_path}")
        print(f"{'='*60}")

        return manifest

def main():
    project_root = Path(__file__).parent.parent

    # Load geometries
    geom_file = project_root / 'CAD' / 'geometries' / 'all_geometries.json'
    if not geom_file.exists():
        print(f"ERROR: {geom_file} not found!")
        print("Run 01_GENERATE_VARIED_CORNERS.py first")
        return

    with open(geom_file, 'r') as f:
        geometries = json.load(f)

    print(f"Loaded {len(geometries)} geometries")

    # Generate cases
    generator = FEMCaseGenerator(base_dir=project_root / 'fem_cases')
    generator.generate_all_cases(geometries, n_process_params=5)

if __name__ == '__main__':
    main()
