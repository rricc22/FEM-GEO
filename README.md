# Parametric PINN for 3D Heat Transfer in L-Shape Geometries

## Project Overview

This project implements a **parametric Physics-Informed Neural Network (PINN)** for predicting transient heat transfer in varied 3D L-shape geometries. The workflow combines CAD geometry generation, finite element method (FEM) simulations via Elmer, and deep learning to create a fast surrogate model that learns physics across both geometric and process parameter spaces.

**Key Innovation:** Unlike standard PINNs trained on analytical solutions, this PINN learns from real FEM simulation data across 18 different geometries and multiple process parameter combinations, enabling rapid prediction on unseen geometry/parameter configurations.

## Technologies Used

- **FreeCAD** - Parametric CAD geometry generation and meshing (`freecadcmd`)
- **Elmer FEM** - Finite element solver for heat transfer simulations
- **PyTorch** - Deep learning framework for PINN training
- **Gmsh** - Automatic tetrahedral mesh generation (via FreeCAD FEM workbench)
- **Weights & Biases** - Experiment tracking and metrics logging
- **DVC** - Data versioning for large datasets and models
- **ParaView** - 3D visualization of FEM and PINN results

## Quick Start

### Prerequisites
```bash
# Install system dependencies
sudo apt install freecad elmer  # or equivalent for your OS
pip install torch numpy scipy matplotlib vtk wandb dvc
```

### Run the Complete Pipeline
```bash
# 1. Generate 18 varied L-shape geometries
freecadcmd scripts/01_GENERATE_VARIED_CORNERS.py

# 2. Mesh all geometries with Gmsh
freecadcmd scripts/02_MESH_ALL_GEOMETRIES.py

# 3. Generate Elmer SIF files for all cases (geometry × process params)
python scripts/03_SIF_GENERATION.py

# 4. Run FEM simulations (90+ cases, ~30 min)
python scripts/04_RUN_SIMULATIONS.py

# 5. Extract VTU data into training-ready NPZ format
python scripts/05_VTU_EXTRACT.py

# 6. Train parametric PINN
cd src
python train_pinn.py

# 7. Generate predictions and visualizations
python inference_pinn.py
python inference_to_vtu.py  # Export predictions as VTU for ParaView
```

## Pipeline Workflow

### 1. Geometry Generation (`scripts/01_GENERATE_VARIED_CORNERS.py`)
Creates 18 parametric L-shape geometries covering a wide design space:
- **Dimensions:** L1, L2 (arm lengths), H (height), thickness
- **Range:** From thin sheet metal (2mm) to ultra-thick heat sinks (20mm)
- **Outputs:**
  - `CAD/geometries/geometry_*.step` - STEP files for each geometry
  - `CAD/geometries/all_geometries.json` - Extracted features (volume, surface area, S/V ratio)

### 2. Mesh Generation (`scripts/02_MESH_ALL_GEOMETRIES.py`)
Generates Elmer-compatible tetrahedral meshes using Gmsh:
- **Element type:** Second-order tetrahedra
- **Mesh size:** Adaptive based on geometry size
- **Outputs:** `elmer_mesh/geometries/geometry_XXX_<name>/` containing:
  - `mesh.nodes` - Node coordinates
  - `mesh.elements` - Volume elements
  - `mesh.boundary` - Boundary face elements
  - `mesh.header` - Mesh metadata

### 3. SIF File Generation (`scripts/03_SIF_GENERATION.py`)
Creates Elmer Solver Input Files for combinations of geometries and process parameters:
- **Process parameters:** Heat flux (50-250 kW/m²), Initial temperature (273-363 K)
- **Training cases:** 85 cases with parameters inside training range
- **Extrapolation test cases:** 10 cases with parameters outside training range
- **Outputs:** `fem_cases/case_XXXX/case.sif` and `fem_cases/manifest.json`

### 4. FEM Simulation (`scripts/04_RUN_SIMULATIONS.py`)
Runs ElmerSolver for all cases in parallel:
- **Physics:** Transient heat conduction with convective boundary conditions
- **Material:** Steel (k=50 W/mK, ρ=7850 kg/m³, cp=500 J/kgK)
- **Time integration:** 50 timesteps × 1 second
- **Outputs:** `fem_cases/case_XXXX/case_t*.vtu` (50 VTU files per case)

### 5. Data Extraction (`scripts/05_VTU_EXTRACT.py`)
Parses all VTU files and combines into training datasets:
- **Extracted fields:** Coordinates (x,y,z), temperature, time
- **Parameters:** Heat flux, T_initial, geometric features (L1, L2, H, thickness, volume, surface area, S/V ratio)
- **Outputs:**
  - `saves/fem_data_all.npz` - Complete dataset (~2GB)
  - `saves/fem_data_train.npz` - Training split
  - `saves/fem_data_test.npz` - Extrapolation test split

### 6. PINN Training (`src/train_pinn.py`)
Trains a parametric PINN with physics-informed loss:
- **Architecture:** 13-input → [128, 128, 128, 128] → 1-output MLP
- **Inputs:** (x, y, z, t, heat_flux, T_initial, L1, L2, H, thickness, volume, surface_area, S_V_ratio)
- **Loss function:** Data loss (MSE vs FEM) + Physics loss (heat equation residual)
- **Physics constraint:** ∂T/∂t - α∇²T = 0, where α = k/(ρ·cp)
- **Training:** ~1000 epochs with Adam optimizer
- **Outputs:**
  - `pinn_model.pt` - Trained weights (~210 KB)
  - `normalization_params.npz` - Input/output normalization stats
  - Wandb logs for training curves

### 7. Inference & Visualization (`src/inference_pinn.py`, `src/inference_to_vtu.py`)
Tests the trained PINN on extrapolation cases:
- **Generates:** PNG comparisons (PINN vs FEM) in `pinn_predictions/visualisations/`
- **Exports:** VTU files for 3D visualization in ParaView (`pinn_predictions/case_XXXX/`)

## Directory Structure

```
FEM-GEO/
├── README.md                  # This file
├── CLAUDE.md                  # Detailed technical documentation
├── VISUALIZATION_GUIDE.md     # ParaView visualization tutorial
├── scripts/                   # Pipeline scripts (run in order 01-05)
│   ├── 01_GENERATE_VARIED_CORNERS.py
│   ├── 02_MESH_ALL_GEOMETRIES.py
│   ├── 03_SIF_GENERATION.py
│   ├── 04_RUN_SIMULATIONS.py
│   └── 05_VTU_EXTRACT.py
├── src/                       # PINN implementation
│   ├── pinn_arch.py          # Neural network architecture
│   ├── train_pinn.py         # Training with physics loss
│   ├── inference_pinn.py     # Prediction and PNG visualization
│   └── inference_to_vtu.py   # Export predictions to VTU format
├── CAD/
│   └── geometries/           # STEP files and geometric features JSON
├── elmer_mesh/
│   └── geometries/           # Elmer mesh files per geometry
├── fem_cases/                # 95 simulation cases
│   ├── case_XXXX/           # Each contains SIF, mesh, and VTU results
│   └── manifest.json        # Case metadata and train/test splits
├── saves/                    # Training datasets (NPZ format)
├── pinn_predictions/         # PINN inference results
│   ├── visualisations/      # PNG comparisons
│   └── case_XXXX/          # VTU predictions for ParaView
├── pinn_model.pt            # Trained PINN weights
├── normalization_params.npz # Normalization statistics
└── wandb/                   # Experiment tracking logs
```

## Results Summary

### Test Strategy
The PINN was tested on **extrapolation cases** with parameters outside the training range:

**Training Range:**
- Heat flux: 50-250 kW/m²
- Initial temperature: 273-363 K

**Extrapolation Test Cases:**
- Heat flux: **280 kW/m²** (12% beyond training max)
- Initial temperature: 303 K
- 5 different geometries (extreme_asym_2, extreme_asym_3, extreme_asym_thin, thick_large, medium_symmetric)

### Key Findings
- PINN successfully extrapolates to unseen heat flux values
- Predictions remain accurate across varied geometries
- Inference time: **~0.1s per full 3D field** (vs ~60s for FEM)
- Visualizations in `pinn_predictions/visualisations/` show excellent agreement with FEM

## Visualization

See **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** for detailed instructions on:
- Installing ParaView
- Loading FEM and PINN VTU files
- Comparing predictions side-by-side
- Creating animations of temperature evolution
- Applying useful filters and color maps

## Physics Background

**Problem:** Transient heat conduction in 3D steel L-shaped brackets

**Governing PDE:**
```
∂T/∂t = α∇²T
```
where α = k/(ρ·cp) ≈ 1.3×10⁻⁵ m²/s (thermal diffusivity)

**Boundary Conditions:**
- Heat flux applied to one boundary: q = 50-250 kW/m²
- Convective cooling on all other surfaces: h = 25 W/m²K, T_ambient = 293 K

**Material Properties (Steel):**
- Thermal conductivity: k = 50 W/mK
- Density: ρ = 7850 kg/m³
- Specific heat: cp = 500 J/kgK

## Citation

If you use this work, please reference:
- Heat equation PINN study (2D analytical, December 2025)
- Parametric 3D FEM-PINN study (this work, January 2026)

## Author

Riccardo Castellano
Student at CentraleSupélec
Supervised by Prof. Frédéric Magoulès
