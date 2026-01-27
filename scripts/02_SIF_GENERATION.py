#!/usr/bin/env python3
"""
Generate multiple FEM simulation cases with parameter variations
Creates Elmer SIF files for batch simulation
"""

import os
import numpy as np
from pathlib import Path
import json

class FEMCaseGenerator:
    """Generate Elmer FEM cases with parameter variations"""

    def __init__(self, base_dir='fem_cases'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def generate_elmer_sif(self, case_params, output_dir):
        """Generate Elmer SIF file for L-shape 2D heat equation"""

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
  Coordinate Mapping(3) = 1 2 3
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

Constants
  Gravity(4) = 0 -1 0 9.82
  Stefan Boltzmann = 5.67e-08
  Permittivity of Vacuum = 8.8542e-12
  Boltzmann Constant = 1.3807e-23
  Unit Charge = 1.602e-19
End

Body 1
  Target Bodies(1) = 1
  Name = "Body 1"
  Equation = 1
  Material = 1
  Initial condition = 1
End

Solver 1
  Equation = Heat Equation
  Procedure = "HeatSolve" "HeatSolver"
  Variable = Temperature
  Exec Solver = Always
  Stabilize = True
  Bubbles = False
  Lumped Mass Matrix = False
  Optimize Bandwidth = True
  Steady State Convergence Tolerance = 1.0e-5
  Nonlinear System Convergence Tolerance = 1.0e-7
  Nonlinear System Max Iterations = 20
  Nonlinear System Newton After Iterations = 3
  Nonlinear System Newton After Tolerance = 1.0e-3
  Nonlinear System Relaxation Factor = 1
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 500
  Linear System Convergence Tolerance = 1.0e-10
  BiCGstabl polynomial degree = 2
  Linear System Preconditioning = ILU0
  Linear System ILUT Tolerance = 1.0e-3
  Linear System Abort Not Converged = False
  Linear System Residual Output = 10
  Linear System Precondition Recompute = 1
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
  Name = "InitialTemp"
  Temperature = {case_params['T_initial']}
End

Boundary Condition 1
  Target Boundaries(1) = 1
  Name = "AllBoundaries"
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

    def generate_parameter_sweep(self, n_cases=50):
        """Generate parameter variations: 45 training + 5 extrapolation test"""

        cases = []

        # Base parameters
        base = {
            'conductivity': 50.0,      # W/m·K (steel)
            'heat_capacity': 500.0,    # J/kg·K
            'density': 7850.0,         # kg/m³ (steel)
            'T_ambient': 293.15,       # K
            'h_conv': 25.0,            # W/m·K (air cooling)
            'dt': 1.0,                 # timestep size (1 second)
            'n_timesteps': 50,         # number of steps (50 seconds total)
        }

        # TRAINING RANGE
        train_heat_flux_min = 50000.0      # 50 kW/m²
        train_heat_flux_max = 250000.0     # 250 kW/m²
        train_T_initial_min = 273.15       # 0°C
        train_T_initial_max = 363.15       # 90°C

        # Generate 45 TRAINING cases with good parameter coverage
        n_train = min(45, n_cases)
        np.random.seed(42)  # Reproducible

        for i in range(n_train):
            params = base.copy()

            # Sample parameters uniformly in training range
            params['heat_flux'] = np.random.uniform(train_heat_flux_min, train_heat_flux_max)
            params['T_initial'] = np.random.uniform(train_T_initial_min, train_T_initial_max)
            params['case_id'] = f'case_{i:04d}'
            params['split'] = 'train'
            cases.append(params)

        # Generate 5 EXTRAPOLATION TEST cases (outside training range)
        if n_cases >= 46:
            # Test 1: Much higher heat flux
            params = base.copy()
            params['heat_flux'] = 280000.0
            params['T_initial'] = 303.15  # 30°C (mid-range)
            params['case_id'] = f'case_{n_train:04d}'
            params['split'] = 'extrapolation_test'
            cases.append(params)

        if n_cases >= 47:
            # Test 2: Even higher heat flux
            params = base.copy()
            params['heat_flux'] = 310000.0
            params['T_initial'] = 323.15  # 50°C
            params['case_id'] = f'case_{n_train+1:04d}'
            params['split'] = 'extrapolation_test'
            cases.append(params)

        if n_cases >= 48:
            # Test 3: Very high heat flux
            params = base.copy()
            params['heat_flux'] = 340000.0
            params['T_initial'] = 293.15  # 20°C
            params['case_id'] = f'case_{n_train+2:04d}'
            params['split'] = 'extrapolation_test'
            cases.append(params)

        if n_cases >= 49:
            # Test 4: Very cold initial temperature
            params = base.copy()
            params['heat_flux'] = 150000.0  # Mid-range heat flux
            params['T_initial'] = 250.15    # -23°C (very cold!)
            params['case_id'] = f'case_{n_train+3:04d}'
            params['split'] = 'extrapolation_test'
            cases.append(params)

        if n_cases >= 50:
            # Test 5: Very hot initial temperature
            params = base.copy()
            params['heat_flux'] = 150000.0  # Mid-range heat flux
            params['T_initial'] = 383.15    # 110°C (very hot!)
            params['case_id'] = f'case_{n_train+4:04d}'
            params['split'] = 'extrapolation_test'
            cases.append(params)

        return cases[:n_cases]

    def generate_all_cases(self, n_cases=50):
        """Generate all SIF files and parameter manifest"""

        cases = self.generate_parameter_sweep(n_cases)

        manifest = {
            'n_cases': len(cases),
            'training_range': {
                'heat_flux': [50000.0, 250000.0],
                'T_initial': [273.15, 363.15]
            },
            'cases': []
        }

        n_train = sum(1 for c in cases if c.get('split') == 'train')
        n_test = sum(1 for c in cases if c.get('split') == 'extrapolation_test')

        print(f"Generating {len(cases)} FEM cases ({n_train} training + {n_test} extrapolation test)...")

        for case_params in cases:
            case_id = case_params['case_id']
            case_dir = self.base_dir / case_id
            split = case_params.get('split', 'train')

            # Generate SIF file
            sif_path = self.generate_elmer_sif(case_params, case_dir)

            # Store case info
            manifest['cases'].append({
                'case_id': case_id,
                'params': case_params,
                'sif_file': str(sif_path),
                'case_dir': str(case_dir),
                'split': split
            })

            marker = "🧪" if split == 'extrapolation_test' else "✓"
            print(f"  {marker} {case_id}: heat_flux={case_params['heat_flux']:.2e} W/m², T_init={case_params['T_initial']:.1f}K [{split}]")

        # Save manifest
        manifest_path = self.base_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n✓ Generated {len(cases)} cases ({n_train} training + {n_test} extrapolation test)")
        print(f"✓ Training range:")
        print(f"    heat_flux: 50,000 - 250,000 W/m²")
        print(f"    T_initial: 273 - 363 K (0°C - 90°C)")
        print(f"✓ Extrapolation test cases go OUTSIDE these ranges")
        print(f"✓ Manifest saved: {manifest_path}")

        return manifest

def main():

    # Create generator
    generator = FEMCaseGenerator(base_dir='fem_cases')

    # Generate cases: 45 training + 5 extrapolation test
    n_cases = 50
    manifest = generator.generate_all_cases(n_cases=n_cases)

if __name__ == '__main__':
    main()
