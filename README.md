# FEM Workflow Test

## Clean Test from Scratch

### Step 1: Generate Cases (SIF files)
```bash
cd /home/riccardo/Documents/FEM
python generate_Elmer_SIF-file.py
```
**Expected output:**
- Creates `fem_cases/case_0000/` and `fem_cases/case_0001/`
- Each contains `case.sif` file
- Creates `fem_cases/manifest.json`

### Step 2: Generate Mesh
```bash
freecadcmd mesh_saving.py
```
**Expected output:**
- Creates `elmer_mesh/` directory with:
  - `mesh.nodes`
  - `mesh.elements`
  - `mesh.boundary`
  - `mesh.header`
- Copies mesh files to both case directories
- Creates `corner_scripted.FCStd`

### Step 3: Run Simulations
```bash
cd /home/riccardo/Documents/FEM/fem_cases/case_0000
ElmerSolver case.sif

cd /home/riccardo/Documents/FEM/fem_cases/case_0001
ElmerSolver case.sif
```
**Expected output:**
- 50 VTU files per case: `case_t0001.vtu` to `case_t0050.vtu`
- `case.result` file

### Step 4: Visualize in ParaView
```bash
paraview /home/riccardo/Documents/FEM/fem_cases/case_0000/case_t0001.vtu
```

## Verification Checklist

- [ ] fem_cases created with 2 case directories
- [ ] Each case has case.sif file
- [ ] elmer_mesh created with 4 mesh files
- [ ] Mesh files copied to both cases
- [ ] ElmerSolver runs without errors
- [ ] Temperature values evolve over time (not all zeros)
- [ ] VTU files open in ParaView
- [ ] Animation shows heat propagation

## Directory Structure After Test
```
/home/riccardo/Documents/FEM/
├── generate_Elmer_SIF-file.py
├── mesh_saving.py
├── corner_scripted.FCStd
├── elmer_mesh/
│   ├── mesh.nodes
│   ├── mesh.elements
│   ├── mesh.boundary
│   └── mesh.header
└── fem_cases/
    ├── manifest.json
    ├── case_0000/
    │   ├── case.sif
    │   ├── mesh.* (4 files)
    │   ├── case_t0001.vtu to case_t0050.vtu
    │   └── case.result
    └── case_0001/
        └── (same as case_0000)
```
