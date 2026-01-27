# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a parametric Physics-Informed Neural Network (PINN) project for predicting heat transfer in varied L-shape geometries. The workflow generates geometric variations, meshes them, runs Elmer FEM simulations, extracts results, and trains a PINN to learn the physics across geometry and process parameter spaces.

## Key Technologies

- **FreeCAD**: CAD geometry generation and meshing (run via `freecadcmd`)
- **Elmer FEM**: Finite element solver for heat transfer simulations
- **PyTorch**: PINN training and inference
- **Gmsh**: Mesh generation (via FreeCAD FEM workbench)
- **Weights & Biases (wandb)**: Experiment tracking

## Sequential Pipeline Workflow

The numbered scripts must be run in order:

### 1. Generate Geometries
```bash
freecadcmd scripts/01_GENERATE_VARIED_CORNERS.py
```
Creates 18 L-shape geometries with varied dimensions (L1, L2, H, thickness) covering thin sheet metal to ultra-thick heat sinks. Outputs:
- `CAD/geometries/geometry_*.step` - STEP files for each geometry
- `CAD/geometries/all_geometries.json` - Geometric features (volume, surface area, S/V ratio, corners)

### 2. Mesh All Geometries
```bash
freecadcmd scripts/02_MESH_ALL_GEOMETRIES.py
```
Creates Elmer-compatible meshes for all geometries using Gmsh. Outputs:
- `elmer_mesh/geometries/geometry_XXX_<name>/` - One directory per geometry containing:
  - `mesh.nodes` - Node coordinates
  - `mesh.elements` - Volume elements (tetrahedra)
  - `mesh.boundary` - Boundary faces
  - `mesh.header` - Mesh metadata

### 3. Generate SIF Files
```bash
python scripts/03_SIF_GENERATION.py
```
Creates Elmer Solver Input Files for combinations of geometries × process parameters (heat flux, initial temperature). Generates training cases (within parameter range) and extrapolation test cases (outside range). Outputs:
- `fem_cases/case_XXXX/case.sif` - Elmer input files
- `fem_cases/manifest.json` - Complete case manifest with parameters and metadata

### 4. Run Simulations
```bash
python scripts/04_RUN_SIMULATIONS.py
```
Copies meshes to case directories and runs ElmerSolver for all cases. Each simulation produces transient heat transfer results. Outputs:
- `fem_cases/case_XXXX/mesh.*` - Copied mesh files
- `fem_cases/case_XXXX/case_t*.vtu` - VTU files for each timestep (50 per case)
- `fem_cases/case_XXXX/case.result` - Elmer result summary

### 5. Extract VTU Data
```bash
python scripts/05_VTU_EXTRACT.py
```
Parses all VTU files and combines data into training-ready format. Extracts coordinates, temperature, time, and all geometry/process parameters. Outputs:
- `saves/fem_data_all.npz` - Complete dataset with metadata
- `saves/fem_data_train.npz` - Training split
- `saves/fem_data_test.npz` - Extrapolation test split

### 6. Train PINN
```bash
cd src
python train_pinn.py
```
Trains a parametric PINN that learns the heat equation across geometry and process parameter spaces. The model takes 13 inputs: spatial coordinates (x, y, z), time (t), process parameters (heat_flux, T_initial), and geometry features (L1, L2, H, thickness, volume, surface_area, S_V_ratio). Outputs:
- `pinn_model.pt` - Trained model weights
- `normalization_params.npz` - Input/output normalization statistics
- wandb logs for training monitoring

### 7. Run Inference
```bash
cd src
python inference_pinn.py
```
Tests the trained PINN on extrapolation cases with parameters outside training range. Generates visualizations comparing predictions to FEM ground truth.

## Directory Structure

```
FEM-GEO/
├── scripts/               # Pipeline scripts (01-05) - run sequentially
├── src/                   # PINN implementation
│   ├── pinn_arch.py      # Neural network architecture
│   ├── train_pinn.py     # Training loop with physics loss
│   └── inference_pinn.py # Prediction and visualization
├── CAD/
│   └── geometries/       # Generated STEP files and features JSON
├── elmer_mesh/
│   └── geometries/       # Elmer mesh files per geometry
├── fem_cases/            # 90+ case directories (geometries × process params)
│   ├── case_XXXX/       # Each contains SIF, mesh, and VTU results
│   └── manifest.json    # Case metadata and parameters
├── saves/                # Extracted NPZ datasets for training
└── Archives/             # Old iterations and backups
```

## Data Flow

1. **Geometric parameters** (L1, L2, H, thickness) → L-shape geometry → STEP file
2. **STEP file** → Gmsh meshing → Elmer mesh files
3. **Mesh + process parameters** (heat_flux, T_initial) → SIF file
4. **SIF + mesh** → ElmerSolver → VTU time series
5. **VTU + geometry features** → NPZ datasets
6. **NPZ datasets** → PINN training → Trained model
7. **Trained model + new parameters** → Temperature predictions

## Important Constraints

- Scripts 01-05 MUST run sequentially - each depends on the previous step's output
- Scripts 01 and 02 require `freecadcmd` (FreeCAD command line)
- Script 04 requires ElmerSolver to be installed and in PATH
- The PINN expects exactly 13 input features in the order: x, y, z, t, heat_flux, T_initial, L1, L2, H, thickness, volume, surface_area, S_V_ratio
- Geometry IDs are 0-padded 3-digit integers (geometry_000 through geometry_017)
- Case IDs are 0-padded 4-digit integers (case_0000 onward)

## Key Files for Understanding

- `scripts/03_SIF_GENERATION.py` - Defines the FEM problem setup (material properties, boundary conditions, heat equation parameters)
- `src/train_pinn.py` - Implements physics loss (heat equation residual) and training loop
- `fem_cases/manifest.json` - Complete record of all cases with their parameters and splits
- `CAD/geometries/all_geometries.json` - Geometric features extracted from CAD models

## Physics Context

The project simulates transient heat conduction in steel L-shaped brackets:
- Heat flux applied to boundary (50-250 kW/m²)
- Convective cooling to ambient temperature (25 W/m²K)
- Material: Steel (k=50 W/mK, cp=500 J/kgK, ρ=7850 kg/m³)
- Thermal diffusivity α = k/(ρ·cp) ≈ 1.3e-5 m²/s
- Timesteps: 50 steps at 1 second intervals

The PINN learns to predict temperature evolution by minimizing both:
1. Data loss: MSE between predictions and FEM results
2. Physics loss: Heat equation residual dT/dt - α·∇²T
