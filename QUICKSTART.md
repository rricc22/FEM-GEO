# Quick Start Guide

Get up and running with the FEM-PINN project in minutes.

---

## Installation

### 1. Clone and Setup Python Environment
```bash
git clone <repository-url>
cd FEM-GEO

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt install freecad paraview
sudo apt-add-repository ppa:elmer-csc-ubuntu/elmer-csc-ppa
sudo apt update && sudo apt install elmerfem-csc
```

**macOS:**
```bash
brew install --cask freecad paraview
# Elmer: Download from https://www.csc.fi/web/elmer/binaries
```

**Windows:** Download installers from official websites:
- FreeCAD: https://www.freecad.org/downloads.php
- Elmer: https://www.csc.fi/web/elmer/binaries
- ParaView: https://www.paraview.org/download/

---

## Git Workflow

```bash
# Check status
git status

# Commit changes
git add scripts/ src/
git commit -m "Descriptive message"

# Push to remote
git push origin main

# Pull latest changes
git pull origin main
```

**Note:** Large data files are tracked by DVC, not Git!

---

## DVC Data Management

DVC tracks large files separately. You'll see `.dvc` pointer files in Git.

```bash

# add the credential :
dvc remote modify storage --local user <username>
dvc remote modify storage --local password <app-password>

# Pull data from remote storage
dvc pull

# After generating new data
dvc add saves/
git add saves.dvc
git commit -m "Update datasets"
dvc push
```

**DVC-tracked folders:** `CAD/`, `elmer_mesh/`, `fem_cases/`, `saves/`, `pinn_model.pt`

---

## Running the Pipeline

### Option A: Use Pre-Generated Data (Fastest )
```bash
# Pull existing data
dvc pull

# Run inference with trained model
cd src
python inference_pinn.py
python inference_to_vtu.py
```

### Option B: Train PINN from Existing FEM Data
```bash
# Pull FEM data and datasets
dvc pull

# Train PINN (~10-30 min)
cd src
python train_pinn.py

# Generate predictions
python inference_pinn.py
python inference_to_vtu.py
```

### Option C: Complete Pipeline from Scratch (~1-2 hours)
```bash
# 1. Generate geometries
freecadcmd scripts/01_GENERATE_VARIED_CORNERS.py

# 2. Mesh geometries
freecadcmd scripts/02_MESH_ALL_GEOMETRIES.py

# 3. Generate SIF files
python scripts/03_SIF_GENERATION.py

# 4. Run FEM simulations
python scripts/04_RUN_SIMULATIONS.py

# 5. Extract data
python scripts/05_VTU_EXTRACT.py

# 6. Train PINN
cd src
python train_pinn.py

# 7. Generate predictions
python inference_pinn.py
python inference_to_vtu.py
```

---

## Viewing Results

### View CAD Geometries
```bash
# FreeCAD GUI
freecad CAD/geometries/geometry_000_ultra_thin.step

# Online viewer (alternative)
# Upload STEP file to: https://3dviewer.net/
```

### View FEM Results (ParaView)
```bash
# Single timestep
paraview fem_cases/case_0000/case_t0001.vtu

# Time series animation
paraview fem_cases/case_0000/case_t*.vtu
```

**Full ParaView tutorial:** See [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)

### View PINN Predictions

**Quick PNG comparisons:**
```bash
# Open any image viewer
xdg-open pinn_predictions/visualisations/prediction_case_0090_t26.png
```

**Interactive 3D comparison:**
```bash
# Load both FEM and PINN in ParaView
paraview fem_cases/case_0090/case_t0026.vtu &
paraview pinn_predictions/case_0090/case_0090_pinn_t0026.vtu &
```

### Inspect Datasets
```python
import numpy as np

# Load training data
data = np.load('saves/fem_data_train.npz')
print(f"Data points: {len(data['x'])}")
print(f"Temperature range: {data['temperature'].min():.1f} - {data['temperature'].max():.1f} K")
print(f"Available fields: {list(data.keys())}")
```

---

## Quick Reference

```bash
# Setup
pip install -r requirements.txt
dvc pull

# Run inference (fastest)
cd src && python inference_pinn.py

# Visualize
paraview fem_cases/case_0000/case_t*.vtu
paraview pinn_predictions/case_0090/case_0090_pinn_t*.vtu

# Git
git add . && git commit -m "Description" && git push

# DVC
dvc status && dvc push
```

---
