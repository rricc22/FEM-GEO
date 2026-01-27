#!/usr/bin/env python3
"""
Setup FreeCAD FEM Analysis from Elmer SIF file
Reads SIF parameters and creates FEM objects in FreeCAD for visualization
Run after mesh_saving.py

Usage:
    freecadcmd setup_fem_from_sif.py
"""

import FreeCAD
import ObjectsFem
import re
from pathlib import Path

def parse_sif_file(sif_path):
    """Parse Elmer SIF file and extract parameters"""

    params = {
        'simulation': {},
        'material': {},
        'initial_condition': {},
        'boundary_conditions': []
    }

    with open(sif_path, 'r') as f:
        content = f.read()

    # Extract Simulation parameters
    sim_match = re.search(r'Simulation\s*\n(.*?)End', content, re.DOTALL)
    if sim_match:
        sim_block = sim_match.group(1)

        # Simulation Type
        sim_type_match = re.search(r'Simulation Type\s*=\s*(\w+)', sim_block)
        if sim_type_match:
            params['simulation']['type'] = sim_type_match.group(1)

        # Timestepping Method
        timestep_method_match = re.search(r'Timestepping Method\s*=\s*(\w+)', sim_block)
        if timestep_method_match:
            params['simulation']['timestepping_method'] = timestep_method_match.group(1)

        # BDF Order
        bdf_order_match = re.search(r'BDF Order\s*=\s*(\d+)', sim_block)
        if bdf_order_match:
            params['simulation']['bdf_order'] = int(bdf_order_match.group(1))

        # Timestep intervals
        timestep_intervals_match = re.search(r'Timestep intervals\s*=\s*(\d+)', sim_block)
        if timestep_intervals_match:
            params['simulation']['timestep_intervals'] = int(timestep_intervals_match.group(1))

        # Timestep Sizes
        timestep_sizes_match = re.search(r'Timestep Sizes\s*=\s*([\d.]+)', sim_block)
        if timestep_sizes_match:
            params['simulation']['timestep_sizes'] = float(timestep_sizes_match.group(1))

    # Extract Material properties
    material_match = re.search(r'Material 1\s*\n(.*?)End', content, re.DOTALL)
    if material_match:
        material_block = material_match.group(1)

        # Heat Conductivity
        k_match = re.search(r'Heat Conductivity\s*=\s*([\d.]+)', material_block)
        if k_match:
            params['material']['conductivity'] = float(k_match.group(1))

        # Heat Capacity
        cp_match = re.search(r'Heat Capacity\s*=\s*([\d.]+)', material_block)
        if cp_match:
            params['material']['heat_capacity'] = float(cp_match.group(1))

        # Density
        rho_match = re.search(r'Density\s*=\s*([\d.]+)', material_block)
        if rho_match:
            params['material']['density'] = float(rho_match.group(1))

    # Extract Initial Condition
    ic_match = re.search(r'Initial Condition 1\s*\n(.*?)End', content, re.DOTALL)
    if ic_match:
        ic_block = ic_match.group(1)

        temp_match = re.search(r'Temperature\s*=\s*([\d.]+)', ic_block)
        if temp_match:
            params['initial_condition']['temperature'] = float(temp_match.group(1))

    # Extract Boundary Conditions
    bc_pattern = r'Boundary Condition \d+\s*\n(.*?)End'
    for bc_match in re.finditer(bc_pattern, content, re.DOTALL):
        bc_block = bc_match.group(1)
        bc = {}

        # Name
        name_match = re.search(r'Name\s*=\s*"([^"]+)"', bc_block)
        if name_match:
            bc['name'] = name_match.group(1)

        # Heat Flux
        flux_match = re.search(r'Heat Flux\s*=\s*([\d.eE+-]+)', bc_block)
        if flux_match:
            bc['heat_flux'] = float(flux_match.group(1))

        # Fixed Temperature (for Dirichlet BC)
        temp_match = re.search(r'^\s*Temperature\s*=\s*([\d.]+)', bc_block, re.MULTILINE)
        if temp_match:
            bc['temperature'] = float(temp_match.group(1))

        # Heat Transfer Coefficient
        htc_match = re.search(r'Heat Transfer Coefficient\s*=\s*([\d.]+)', bc_block)
        if htc_match:
            bc['h_conv'] = float(htc_match.group(1))

        # External Temperature (for convection)
        ext_temp_match = re.search(r'External Temperature\s*=\s*([\d.]+)', bc_block)
        if ext_temp_match:
            bc['external_temp'] = float(ext_temp_match.group(1))

        params['boundary_conditions'].append(bc)

    return params

def setup_fem_analysis(doc, sif_path):
    """Setup FEM analysis in FreeCAD document from SIF file"""

    print(f"\nReading SIF file: {sif_path}")
    params = parse_sif_file(sif_path)

    print("\nParsed parameters:")
    print(f"  Material: {params['material']}")
    print(f"  Initial Condition: {params['initial_condition']}")
    print(f"  Boundary Conditions: {params['boundary_conditions']}")

    # Get the solid object
    solide = doc.getObject('solide')
    if not solide:
        print("ERROR: 'solide' object not found in document")
        return None

    # Get or create Analysis
    analysis = None
    for obj in doc.Objects:
        if obj.TypeId == 'Fem::FemAnalysis':
            analysis = obj
            break

    if not analysis:
        analysis = ObjectsFem.makeAnalysis(doc, "ElmerAnalysis")
        print("✓ Created FEM Analysis")
    else:
        print("✓ Using existing Analysis")

    # Create Material
    mat_params = params['material']
    material = ObjectsFem.makeMaterialSolid(doc, "Steel")

    # FreeCAD material properties format
    material.Material = {
        'Name': 'Steel',
        'ThermalConductivity': f"{mat_params.get('conductivity', 50.0)} W/m/K",
        'SpecificHeat': f"{mat_params.get('heat_capacity', 500.0)} J/kg/K",
        'Density': f"{mat_params.get('density', 7850.0)} kg/m^3"
    }

    # Set material reference to solid
    material.References = [(solide, 'Solid1')]
    analysis.addObject(material)
    print(f"✓ Created Material: k={mat_params.get('conductivity')} W/m/K, "
          f"Cp={mat_params.get('heat_capacity')} J/kg/K, "
          f"ρ={mat_params.get('density')} kg/m³")

    # Create Initial Temperature
    initial_temp = params['initial_condition'].get('temperature', 293.15)
    init_temp_obj = ObjectsFem.makeConstraintInitialTemperature(doc, "InitialTemperature")
    init_temp_obj.initialTemperature = initial_temp
    init_temp_obj.References = [(solide, 'Solid1')]
    analysis.addObject(init_temp_obj)
    print(f"✓ Created Initial Temperature: {initial_temp} K ({initial_temp-273.15:.1f}°C)")

    # Get all faces of the solid for boundary conditions
    all_faces = []
    if hasattr(solide.Shape, 'Faces'):
        n_faces = len(solide.Shape.Faces)
        all_faces = [(solide, f"Face{i+1}") for i in range(n_faces)]
        print(f"\nℹ Solid has {n_faces} faces - will assign all to boundary conditions")

    # Create Boundary Conditions
    print("\n✓ Creating Boundary Conditions:")
    for idx, bc in enumerate(params['boundary_conditions']):
        bc_name = bc.get('name', f'BC_{idx}')

        # Create Fixed Temperature constraint (Dirichlet BC)
        if 'temperature' in bc:
            temp_bc = ObjectsFem.makeConstraintTemperature(doc, f"Temperature_{bc_name}")
            temp_bc.Temperature = bc['temperature']
            temp_bc.Scale = 1.0

            # Assign all faces if bc_name contains "AllBoundaries"
            if 'AllBoundaries' in bc_name and all_faces:
                temp_bc.References = all_faces
                print(f"  • Created Fixed Temperature BC: {bc['temperature']} K ({bc['temperature']-273.15:.1f}°C)")
                print(f"    → Assigned to all {len(all_faces)} faces")
            else:
                print(f"  • Created Fixed Temperature BC: {bc['temperature']} K ({bc['temperature']-273.15:.1f}°C)")
                print(f"    → No faces assigned (assign manually in GUI)")

            analysis.addObject(temp_bc)

        # Create Heat Flux constraint if present
        if 'heat_flux' in bc:
            flux_bc = ObjectsFem.makeConstraintHeatflux(doc, f"HeatFlux_{bc_name}")

            # Assign all faces if bc_name contains "AllBoundaries"
            if 'AllBoundaries' in bc_name and all_faces:
                flux_bc.References = all_faces

            # Set heat flux parameters
            flux_bc.ConstraintType = "DFlux"  # Direct heat flux
            flux_bc.DFlux = bc['heat_flux']  # Heat flux value in W/m²
            flux_bc.AmbientTemp = bc.get('external_temp', 293.15)
            flux_bc.FilmCoef = bc.get('h_conv', 0.0)

            if 'AllBoundaries' in bc_name and all_faces:
                print(f"  • Created Heat Flux BC: {bc['heat_flux']} W/m²")
                if 'h_conv' in bc:
                    print(f"    → Convection: h={bc['h_conv']} W/m²K")
                print(f"    → Assigned to all {len(all_faces)} faces")
            else:
                print(f"  • Created Heat Flux BC: {bc['heat_flux']} W/m²")
                if 'h_conv' in bc:
                    print(f"    → Convection: h={bc['h_conv']} W/m²K")
                print(f"    → No faces assigned (assign manually in GUI)")

            analysis.addObject(flux_bc)

    print("\n✓ Boundary condition objects created with face assignments.")

    # Create Elmer Solver
    print("\n✓ Creating Elmer Solver:")
    solver = ObjectsFem.makeSolverElmer(doc, "SolverElmer")
    analysis.addObject(solver)

    # Configure solver parameters from SIF
    sim_params = params['simulation']

    # Simulation type - access via the solver's properties
    if sim_params.get('type') == 'Transient':
        solver.Proxy.SteadyState = False
        print(f"  • Simulation Type: Transient")
    else:
        solver.Proxy.SteadyState = True
        print(f"  • Simulation Type: Steady State")

    # Timestepping parameters
    if 'timestep_intervals' in sim_params:
        solver.Proxy.TimeStepIntervals = sim_params['timestep_intervals']
        print(f"  • Timestep Intervals: {sim_params['timestep_intervals']}")

    if 'timestep_sizes' in sim_params:
        solver.Proxy.TimeStepSizes = sim_params['timestep_sizes']
        print(f"  • Timestep Size: {sim_params['timestep_sizes']} s")

    # BDF method
    if sim_params.get('timestepping_method') == 'BDF':
        if 'bdf_order' in sim_params:
            solver.Proxy.BDFOrder = sim_params['bdf_order']
            print(f"  • BDF Order: {sim_params['bdf_order']}")

    print("✓ Elmer Solver configured and added to analysis")

    doc.recompute()
    return analysis

def main():
    # Open the FreeCAD document
    fcstd_path = "/home/riccardo/Documents/FEM/corner_scripted.FCStd"

    print("="*70)
    print("Setup FEM Analysis from SIF File")
    print("="*70)

    if not Path(fcstd_path).exists():
        print(f"ERROR: {fcstd_path} not found!")
        print("Run mesh_saving.py first to create the document.")
        return

    print(f"\nOpening: {fcstd_path}")
    doc = FreeCAD.openDocument(fcstd_path)

    # Choose which case to visualize
    sif_files = [
        "/home/riccardo/Documents/FEM/fem_cases/case_0000/case.sif",
        "/home/riccardo/Documents/FEM/fem_cases/case_0001/case.sif"
    ]

    # Default to case_0000
    sif_path = sif_files[0]

    if not Path(sif_path).exists():
        print(f"ERROR: {sif_path} not found!")
        print("Run generate_Elmer_SIF-file.py first to create SIF files.")
        return

    # Setup FEM analysis
    analysis = setup_fem_analysis(doc, sif_path)

    if analysis:
        # Save the document
        output_path = "/home/riccardo/Documents/FEM/corner_scripted_fem.FCStd"
        doc.saveAs(output_path)
        print(f"\n✓ Saved as: {output_path}")
        print("\nOpen in FreeCAD GUI to review:")
        print("  freecad corner_scripted_fem.FCStd")
        print("\nIn FEM Workbench, you can see:")
        print("  • Material properties (Steel)")
        print("  • Initial Temperature condition")
        print("  • Mesh geometry")

    print("\n" + "="*70)
    print("Setup Complete!")
    print("="*70)

if __name__ == '__main__':
    main()
