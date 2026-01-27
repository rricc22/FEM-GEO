import FreeCAD
import FreeCADGui
import Part
import ObjectsFem
import femmesh.gmshtools as gmshtools
import shutil
from pathlib import Path

def write_elmer_mesh(fem_mesh, output_dir):
    """Write Elmer mesh files directly from FreeCAD FemMesh object"""
    from collections import Counter

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
        elif n_nodes_elem == 8:
            volume_types[605] += 1
        elif n_nodes_elem == 10:
            volume_types[510] += 1
        elif n_nodes_elem == 6:
            volume_types[706] += 1
        else:
            volume_types[500] += 1

    boundary_types = Counter()
    for face_id in face_ids:
        face_nodes = fem_mesh.getElementNodes(face_id)
        n_nodes_face = len(face_nodes)
        if n_nodes_face == 3:
            boundary_types[303] += 1
        elif n_nodes_face == 4:
            boundary_types[404] += 1
        else:
            boundary_types[300] += 1

    # Write mesh.header
    with open(output_path / 'mesh.header', 'w') as f:
        f.write(f"{n_nodes} {n_elements} {n_boundary}\n")
        all_types = list(volume_types.items()) + list(boundary_types.items())
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
            n_nodes_elem = len(elem_nodes)

            if n_nodes_elem == 4:
                elem_type = 504  # Tet4
            elif n_nodes_elem == 8:
                elem_type = 605  # Hexa8
            elif n_nodes_elem == 10:
                elem_type = 510  # Tet10
            elif n_nodes_elem == 6:
                elem_type = 706  # Wedge6
            else:
                elem_type = 500

            nodes_str = ' '.join(str(n) for n in elem_nodes)
            f.write(f"{idx} 1 {elem_type} {nodes_str}\n")

    # Write mesh.boundary
    with open(output_path / 'mesh.boundary', 'w') as f:
        for idx, face_id in enumerate(face_ids, start=1):
            face_nodes = fem_mesh.getElementNodes(face_id)
            n_nodes_face = len(face_nodes)

            if n_nodes_face == 3:
                elem_type = 303
            elif n_nodes_face == 4:
                elem_type = 404
            else:
                elem_type = 300

            nodes_str = ' '.join(str(n) for n in face_nodes)
            f.write(f"{idx} 1 0 0 {elem_type} {nodes_str}\n")

    print(f"✓ Mesh written: {n_nodes} nodes, {n_elements} elements")

# Main script
doc_name = 'Shape_studied'
doc = FreeCAD.newDocument(doc_name)

w = 10
h = 10

points = [
    FreeCAD.Vector(0, 0, 0),
    FreeCAD.Vector(w, 0, 0),
    FreeCAD.Vector(w, h/2, 0),
    FreeCAD.Vector(w/2, h/2, 0),
    FreeCAD.Vector(w/2, h, 0),
    FreeCAD.Vector(0, h, 0),
    FreeCAD.Vector(0, 0, 0)
]

wire = Part.makePolygon(points)
face = Part.Face(wire)
extrusion = face.extrude(FreeCAD.Vector(0, 0, 10))

solide = doc.addObject("Part::Feature", "solide")
solide.Shape = extrusion

analysis = ObjectsFem.makeAnalysis(doc, "Anlysis")

mesh_obj = ObjectsFem.makeMeshGmsh(doc, 'FEMmeshGmsh')
mesh_obj.Shape = solide
mesh_obj.CharacteristicLengthMax = 2.0
mesh_obj.CharacteristicLengthMin = 0.5

analysis.addObject(mesh_obj)

doc.recompute()

gmsh_mesh = gmshtools.GmshTools(mesh_obj)
create_mesh = gmsh_mesh.create_mesh()

doc.saveAs("/home/riccardo/Documents/FEM/saves/corner_scripted.FCStd")

# Write and copy mesh
elmer_mesh_dir = Path("/home/riccardo/Documents/FEM/elmer_mesh")
write_elmer_mesh(mesh_obj.FemMesh, elmer_mesh_dir)

fem_cases_dir = Path("/home/riccardo/Documents/FEM/fem_cases")
mesh_files = ['mesh.nodes', 'mesh.elements', 'mesh.boundary', 'mesh.header']

if fem_cases_dir.exists():
    case_dirs = sorted([d for d in fem_cases_dir.iterdir()
                        if d.is_dir() and d.name.startswith('case_')])

    for case_dir in case_dirs:
        for mesh_file in mesh_files:
            shutil.copy2(elmer_mesh_dir / mesh_file, case_dir / mesh_file)

    print(f"✓ Copied to {len(case_dirs)} cases")
else:
    print("⚠ fem_cases directory not found")