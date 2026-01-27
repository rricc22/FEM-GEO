#!/usr/bin/env python3
"""
Generate varied corner/L-shape geometries with different dimensions
Creates CAD models, exports STEP files, and extracts geometric features
"""

import sys
import json
from pathlib import Path

try:
    import FreeCAD
    import Part
    import Mesh
except ImportError:
    print("ERROR: Run with freecadcmd!")
    sys.exit(1)

def create_L_shape(L1, L2, H, thickness):
    """
    Create L-shape corner with given dimensions
    L1: First leg length (mm)
    L2: Second leg length (mm)
    H: Height (mm)
    thickness: Wall thickness (mm)
    """

    # Create two boxes that form an L
    box1 = Part.makeBox(L1, thickness, H)
    box2 = Part.makeBox(thickness, L2, H)

    # Position box2 to form L-shape
    box2.translate(FreeCAD.Vector(0, thickness, 0))

    # Union to create L-shape
    L_shape = box1.fuse(box2)

    return L_shape

def generate_geometry_set():
    """Generate set of varied L-shape geometries with WIDE thickness variation"""

    geometries = []

    # 1. Ultra-thin (sheet metal - heats VERY fast)
    geometries.append({
        'name': 'ultra_thin',
        'L1': 30, 'L2': 30, 'H': 30, 'thickness': 1,
        'description': 'Ultra-thin sheet metal (1mm)'
    })

    geometries.append({
        'name': 'very_thin',
        'L1': 25, 'L2': 35, 'H': 30, 'thickness': 1.5,
        'description': 'Very thin asymmetric (1.5mm)'
    })

    # 2. Thin (typical sheet metal)
    geometries.append({
        'name': 'thin_small',
        'L1': 20, 'L2': 20, 'H': 25, 'thickness': 2,
        'description': 'Thin small bracket (2mm)'
    })

    geometries.append({
        'name': 'thin_medium',
        'L1': 30, 'L2': 30, 'H': 30, 'thickness': 2,
        'description': 'Thin medium bracket (2mm)'
    })

    # 3. Medium thickness (standard)
    geometries.append({
        'name': 'medium_symmetric',
        'L1': 30, 'L2': 30, 'H': 30, 'thickness': 3,
        'description': 'Medium symmetric baseline (3mm)'
    })

    geometries.append({
        'name': 'medium_asymmetric',
        'L1': 25, 'L2': 35, 'H': 30, 'thickness': 3.5,
        'description': 'Medium asymmetric (3.5mm)'
    })

    # 4. Thick (heavy duty)
    geometries.append({
        'name': 'thick_compact',
        'L1': 30, 'L2': 30, 'H': 30, 'thickness': 5,
        'description': 'Thick compact bracket (5mm)'
    })

    geometries.append({
        'name': 'thick_large',
        'L1': 40, 'L2': 40, 'H': 35, 'thickness': 6,
        'description': 'Thick large bracket (6mm)'
    })

    # 5. Very thick (extreme - massive heat sink)
    geometries.append({
        'name': 'very_thick',
        'L1': 30, 'L2': 30, 'H': 30, 'thickness': 7,
        'description': 'Very thick (7mm) - slow heating'
    })

    geometries.append({
        'name': 'ultra_thick',
        'L1': 35, 'L2': 35, 'H': 30, 'thickness': 8,
        'description': 'Ultra-thick (8mm) - extreme heat sink'
    })

    # 6. Edge case: tall thin
    geometries.append({
        'name': 'tall_thin',
        'L1': 20, 'L2': 20, 'H': 45, 'thickness': 2,
        'description': 'Tall and thin - high aspect ratio'
    })

    # 7. Edge case: short thick
    geometries.append({
        'name': 'short_thick',
        'L1': 25, 'L2': 25, 'H': 20, 'thickness': 6,
        'description': 'Short and thick - compact mass'
    })

    # 8. Extreme asymmetry - SHORT + LONG combinations
    geometries.append({
        'name': 'extreme_asym_1',
        'L1': 15, 'L2': 45, 'H': 30, 'thickness': 2.5,
        'description': 'Extreme asymmetric: very short + very long'
    })

    geometries.append({
        'name': 'extreme_asym_2',
        'L1': 18, 'L2': 42, 'H': 30, 'thickness': 3,
        'description': 'High asymmetry: short + long'
    })

    geometries.append({
        'name': 'extreme_asym_3',
        'L1': 20, 'L2': 40, 'H': 30, 'thickness': 3.5,
        'description': 'Strong asymmetry: medium + long'
    })

    geometries.append({
        'name': 'extreme_asym_thin',
        'L1': 15, 'L2': 35, 'H': 25, 'thickness': 2,
        'description': 'Asymmetric + thin: short leg heats fast'
    })

    geometries.append({
        'name': 'extreme_asym_thick',
        'L1': 18, 'L2': 38, 'H': 30, 'thickness': 5,
        'description': 'Asymmetric + thick: massive one leg'
    })

    # 9. Reversed asymmetry (swap L1 and L2)
    geometries.append({
        'name': 'reversed_asym',
        'L1': 42, 'L2': 18, 'H': 30, 'thickness': 3,
        'description': 'Reversed: long first leg + short second'
    })

    return geometries

def extract_features(shape, params):
    """Extract geometric features from shape"""

    volume = shape.Volume
    area = shape.Area
    bbox = shape.BoundBox

    # Corners
    corners = [[v.X, v.Y, v.Z] for v in shape.Vertexes]

    # Junction angles
    angles = []
    for edge in shape.Edges:
        faces = [f for f in shape.Faces if edge in f.Edges]
        if len(faces) == 2:
            try:
                n1 = faces[0].normalAt(0.5, 0.5)
                n2 = faces[1].normalAt(0.5, 0.5)
                angle = n1.getAngle(n2)
                angles.append(angle)
            except:
                pass

    return {
        'params': params,
        'volume': volume,
        'surface_area': area,
        'S_V_ratio': area / volume if volume > 0 else 0,
        'bbox_dims': [bbox.XLength, bbox.YLength, bbox.ZLength],
        'n_corners': len(corners),
        'corners': corners,
        'n_faces': len(shape.Faces),
        'n_edges': len(shape.Edges)
    }

def main():
    """Generate all geometries and export"""

    project_root = Path(__file__).parent.parent
    cad_dir = project_root / 'CAD' / 'geometries'
    cad_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Generating Varied Corner Geometries")
    print("="*60)

    geometries = generate_geometry_set()
    all_features = []

    for i, geom_spec in enumerate(geometries):
        name = geom_spec['name']
        L1 = geom_spec['L1']
        L2 = geom_spec['L2']
        H = geom_spec['H']
        thickness = geom_spec['thickness']

        print(f"\n[{i+1}/{len(geometries)}] {name}")
        print(f"  L1={L1}mm, L2={L2}mm, H={H}mm, t={thickness}mm")

        # Create shape
        shape = create_L_shape(L1, L2, H, thickness)

        # Extract features
        features = extract_features(shape, {
            'L1': L1, 'L2': L2, 'H': H, 'thickness': thickness
        })
        features['geometry_id'] = f'geometry_{i:03d}'
        features['name'] = name
        features['description'] = geom_spec['description']

        all_features.append(features)

        # Export STEP file
        step_file = cad_dir / f'geometry_{i:03d}_{name}.step'
        shape.exportStep(str(step_file))

        print(f"  Volume: {features['volume']:.2f} mm³")
        print(f"  Surface: {features['surface_area']:.2f} mm²")
        print(f"  S/V: {features['S_V_ratio']:.3f} m⁻¹")
        print(f"  Saved: {step_file.name}")

    # Save all features to JSON
    features_file = cad_dir / 'all_geometries.json'
    with open(features_file, 'w') as f:
        json.dump(all_features, f, indent=2)

    print("\n" + "="*60)
    print(f"✓ Generated {len(geometries)} geometries")
    print(f"✓ STEP files: {cad_dir}")
    print(f"✓ Features: {features_file}")
    print("="*60)

    # Print summary
    print("\nGeometry Summary:")
    print(f"  Volume range: {min(g['volume'] for g in all_features):.0f} - {max(g['volume'] for g in all_features):.0f} mm³")
    print(f"  S/V range: {min(g['S_V_ratio'] for g in all_features):.3f} - {max(g['S_V_ratio'] for g in all_features):.3f} m⁻¹")

if __name__ == '__main__':
    main()
