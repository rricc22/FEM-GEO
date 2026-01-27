#!/usr/bin/env python3
"""Generate Elmer meshes for all geometries"""

import sys
from pathlib import Path
from collections import Counter

try:
    import FreeCAD
    import Part
    import ObjectsFem
    import femmesh.gmshtools as gmshtools
except ImportError:
    print("ERROR: Run with freecadcmd!")
    sys.exit(1)

def write_elmer_mesh(fem_mesh, output_dir):
    """Write Elmer mesh files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nodes = fem_mesh.Nodes
    volume_ids = fem_mesh.getIdByElementType('Volume')
    face_ids = fem_mesh.getIdByElementType('Face')

    n_nodes = len(nodes)
    n_elements = len(volume_ids)
    n_boundary = len(face_ids)

    # Count element types
    volume_types = Counter()
    for volume_id in volume_ids:
        elem_nodes = fem_mesh.getElementNodes(volume_id)
        n_nodes_elem = len(elem_nodes)
        if n_nodes_elem == 4:
            volume_types[504] += 1
        elif n_nodes_elem == 10:
            volume_types[510] += 1

    boundary_types = Counter()
    for face_id in face_ids:
        face_nodes = fem_mesh.getElementNodes(face_id)
        if len(face_nodes) == 3:
            boundary_types[303] += 1
        elif len(face_nodes) == 4:
            boundary_types[404] += 1

    # Write mesh.header
    with open(output_path / 'mesh.header', 'w') as f:
        all_types = list(volume_types.items()) + list(boundary_types.items())
        f.write(f"{n_nodes} {n_elements} {n_boundary}\n")
        f.write(f"{len(all_types)}\n")
        for elem_type, count in all_types:
            f.write(f"{elem_type} {count}\n")

    # Write mesh.nodes
    with open(output_path / 'mesh.nodes', 'w') as f:
        for node_id, coords in nodes.items():
            f.write(f"{node_id} -1 {coords[0]} {coords[1]} {coords[2]}\n")

    # Write mesh.elements
    with open(output_path / 'mesh.elements', 'w') as f:
        for idx, volume_id in enumerate(volume_ids, start=1):
            elem_nodes = fem_mesh.getElementNodes(volume_id)
            elem_type = 504 if len(elem_nodes) == 4 else 510
            nodes_str = ' '.join(str(n) for n in elem_nodes)
            f.write(f"{idx} 1 {elem_type} {nodes_str}\n")

    # Write mesh.boundary
    with open(output_path / 'mesh.boundary', 'w') as f:
        for idx, face_id in enumerate(face_ids, start=1):
            face_nodes = fem_mesh.getElementNodes(face_id)
            elem_type = 303 if len(face_nodes) == 3 else 404
            nodes_str = ' '.join(str(n) for n in face_nodes)
            f.write(f"{idx} 1 0 0 {elem_type} {nodes_str}\n")

    return n_nodes, n_elements

def mesh_geometry(step_file, output_dir, char_length_max=2.0):
    """Create mesh for a geometry"""

    # Create temporary document
    doc = FreeCAD.newDocument("temp")

    # Load STEP file
    shape = Part.Shape()
    shape.read(str(step_file))

    # Add to document
    part = doc.addObject("Part::Feature", "Part")
    part.Shape = shape

    # Create FEM mesh
    mesh_obj = ObjectsFem.makeMeshGmsh(doc, 'FEMmesh')
    mesh_obj.Shape = part
    mesh_obj.CharacteristicLengthMax = char_length_max
    mesh_obj.CharacteristicLengthMin = char_length_max / 4

    doc.recompute()

    # Generate mesh with Gmsh
    gmsh_mesh = gmshtools.GmshTools(mesh_obj)
    gmsh_mesh.create_mesh()

    # Write Elmer files
    n_nodes, n_elements = write_elmer_mesh(mesh_obj.FemMesh, output_dir)

    # Cleanup
    FreeCAD.closeDocument(doc.Name)

    return n_nodes, n_elements

def main():
    """Mesh all geometries"""

    project_root = Path(__file__).parent.parent
    geom_dir = project_root / 'CAD' / 'geometries'
    mesh_base_dir = project_root / 'elmer_mesh' / 'geometries'

    # Find all STEP files
    step_files = sorted(geom_dir.glob('geometry_*.step'))

    if not step_files:
        print(f"No STEP files found in {geom_dir}")
        sys.exit(1)

    print("="*60)
    print(f"Meshing {len(step_files)} Geometries")
    print("="*60)

    for i, step_file in enumerate(step_files, 1):
        geom_name = step_file.stem
        mesh_dir = mesh_base_dir / geom_name

        print(f"\n[{i}/{len(step_files)}] {geom_name}")

        try:
            n_nodes, n_elements = mesh_geometry(step_file, mesh_dir)
            print(f"  ✓ {n_nodes} nodes, {n_elements} elements")
            print(f"  ✓ Saved: {mesh_dir}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")

    print("\n" + "="*60)
    print(f"✓ Meshing complete!")
    print(f"✓ Meshes saved in: {mesh_base_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
