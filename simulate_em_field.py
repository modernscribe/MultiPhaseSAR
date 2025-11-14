bl_info = {
    "name": "Sinous Antenna Array Sim",

    
}

import bpy
import bmesh
import numpy as np
import math
from mathutils import Color, Vector
from bpy.props import (
    FloatProperty,
    IntProperty,
    BoolProperty,
    EnumProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup, Panel

# ------------- Physical Constants -------------
C = 299792458.0  # Speed of light in m/s
MU0 = 4.0 * np.pi * 1e-7  # Permeability of free space (H/m)
EPS0 = 8.854187817e-12  # Permittivity of free space (F/m)
ETA0 = np.sqrt(MU0 / EPS0)  # Impedance of free space (377 ohms)

# ------------- Wire gauge & colors -------------
WIRE_GAUGE_DIAMETERS = {
    "10": 2.588, "12": 2.053, "14": 1.628, "16": 1.291,
    "18": 1.024, "20": 0.812, "22": 0.644, "24": 0.511,
}

WIRE_RESISTANCE_PER_M = {  # Ohms per meter at 20°C
    "10": 0.00325, "12": 0.00516, "14": 0.00821, "16": 0.01305,
    "18": 0.02074, "20": 0.03297, "22": 0.05243, "24": 0.08333,
}

ANTENNA_COLORS = [
    (0.2, 0.5, 1.0, 1.0), (1.0, 0.2, 0.2, 1.0), (0.2, 1.0, 0.2, 1.0), (1.0, 1.0, 0.2, 1.0),
    (1.0, 0.2, 1.0, 1.0), (0.2, 1.0, 1.0, 1.0), (1.0, 0.5, 0.2, 1.0), (0.5, 0.2, 1.0, 1.0),
]

# ------------- Helper Functions -------------
def clear_antenna_objects():
    """Remove all antenna-related objects from the scene"""
    to_remove = [o for o in bpy.data.objects if o.name.startswith(("AntennaCone_", "AntennaCoil_", "EMField_", "RadiationPattern_"))]
    for o in to_remove:
        bpy.data.objects.remove(o, do_unlink=True)
    print(f"Cleared {len(to_remove)} antenna objects")

def get_wire_diameter_mm(gauge):
    return WIRE_GAUGE_DIAMETERS.get(gauge, 1.024)

def get_wire_resistance_per_m(gauge):
    return WIRE_RESISTANCE_PER_M.get(gauge, 0.02074)

def ensure_collection(name):
    """Ensure a collection exists and return it"""
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

def safe_divide(a, b, default=1e-9):
    return a / (b if abs(b) > 1e-12 else default)

# ------------- Simplified EM Field Mathematics -------------
def calculate_simple_field_pattern(antenna_segments, frequency_hz, current_amplitude, grid_points):
    """
    Simplified electromagnetic field calculation for performance
    Uses dipole approximation and simplified math for real-time visualization
    """
    wavelength = C / frequency_hz
    k = 2.0 * np.pi / wavelength
    
    # Initialize field array
    field_values = np.zeros(grid_points.shape[0])
    
    print(f"Computing simplified EM fields: {len(antenna_segments)} antennas, {grid_points.shape[0]} points")
    
    for ant_idx, segments in enumerate(antenna_segments):
        if len(segments) < 2:
            continue
            
        # Use antenna center as effective radiator
        center = np.mean(segments, axis=0)
        
        # Calculate effective antenna length (total wire length)
        total_length = 0
        for i in range(len(segments) - 1):
            total_length += np.linalg.norm(segments[i+1] - segments[i])
        
        # Simple dipole field calculation
        for i, obs_point in enumerate(grid_points):
            R = obs_point - center
            R_mag = np.linalg.norm(R)
            
            if R_mag < 1e-6:
                continue
                
            # Simplified far-field calculation
            if R_mag > wavelength / 10:  # Far field
                # Electric field strength (simplified Hertzian dipole)
                E_field = (60.0 * current_amplitude * total_length * k) / R_mag
                # Apply directional pattern (simplified)
                theta = np.arccos(np.abs(R[2]) / R_mag)  # Angle from Z-axis
                pattern_factor = np.sin(theta)**2  # Simple dipole pattern
                E_field *= pattern_factor
            else:  # Near field (inductive)
                # Magnetic field dominates
                E_field = (MU0 * current_amplitude * total_length) / (4 * np.pi * R_mag**2)
            
            # Apply frequency-dependent phase (simplified)
            phase_factor = 1.0 + 0.1 * np.sin(k * R_mag)
            field_values[i] += E_field * phase_factor
    
    return field_values

def calculate_antenna_stats_fast(antenna_segments, frequency_hz, wire_gauge, current_amplitude):
    """Fast antenna parameter calculation for UI display"""
    wavelength = C / frequency_hz
    
    results = {
        'frequency_mhz': frequency_hz / 1e6,
        'wavelength_m': wavelength,
        'num_antennas': len(antenna_segments),
    }
    
    # Calculate total wire length
    total_length = 0.0
    for segments in antenna_segments:
        seg_array = np.array(segments)
        for i in range(len(seg_array) - 1):
            dl = np.linalg.norm(seg_array[i+1] - seg_array[i])
            total_length += dl
    
    results['total_wire_length_m'] = total_length
    results['dc_resistance_ohms'] = total_length * get_wire_resistance_per_m(wire_gauge)
    
    # Approximate inductance (Wheeler's formula for helical coil)
    if antenna_segments:
        first_antenna = np.array(antenna_segments[0])
        coil_radius = np.mean([np.sqrt(p[0]**2 + p[1]**2) for p in first_antenna])
        coil_height = np.max(first_antenna[:, 2]) - np.min(first_antenna[:, 2])
        n_turns = len(first_antenna) / 50
        
        if coil_height > 0 and coil_radius > 0:
            L_uh = (coil_radius * 1000)**2 * n_turns**2 / (230 * coil_radius * 1000 + 254 * coil_height * 1000)
            results['inductance_uh'] = L_uh
        else:
            results['inductance_uh'] = 0.0
    
    # Basic power calculations
    omega = 2 * np.pi * frequency_hz
    L = results['inductance_uh'] * 1e-6
    R = results['dc_resistance_ohms']
    
    XL = omega * L
    Z_mag = np.sqrt(R**2 + XL**2)
    I_rms = current_amplitude / np.sqrt(2)
    
    results['impedance_ohms'] = Z_mag
    results['resistive_power_w'] = I_rms**2 * R
    results['q_factor'] = XL / R if R > 0 else 0
    
    # Approximate radiation resistance (very simplified)
    electrical_length = total_length / wavelength
    if electrical_length < 0.1:  # Small antenna
        Rr = 20 * (np.pi * electrical_length)**2  # Small loop/short dipole
    else:  # Larger antenna
        Rr = 73.0 * (electrical_length / 0.25)**2  # Scaled from quarter-wave dipole
    
    results['radiation_resistance_ohms'] = min(Rr, 1000)  # Cap at reasonable value
    results['radiated_power_w'] = I_rms**2 * results['radiation_resistance_ohms']
    results['efficiency_percent'] = (results['radiated_power_w'] / 
                                   (results['radiated_power_w'] + results['resistive_power_w'])) * 100
    
    return results

# ------------- Cone Creation & Management -------------
def create_or_update_cone(props):
    """Create or update the main antenna cone structure"""
    obj = bpy.data.objects.get("AntennaCone_Main")
    base_radius = props.cone_base_diameter / 2000.0
    tip_radius = max(props.cone_tip_h / 2000.0, 0.001) if props.cone_tip_h > 0 else 0.001
    height = props.cone_height / 1000.0

    if obj is None:
        bpy.ops.mesh.primitive_cone_add(
            vertices=props.cone_segments,
            radius1=base_radius,
            radius2=tip_radius,
            depth=height,
            location=(0, 0, height / 2.0)
        )
        obj = bpy.context.active_object
        obj.name = "AntennaCone_Main"
        
        # Create material
        if "CopperMaterial" not in bpy.data.materials:
            mat = bpy.data.materials.new("CopperMaterial")
            mat.use_nodes = True
            mat.node_tree.nodes.clear()
            bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.inputs['Base Color'].default_value = (0.8, 0.4, 0.2, 1.0)
            bsdf.inputs['Metallic'].default_value = 1.0
            bsdf.inputs['Roughness'].default_value = 0.2
            output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
            mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        obj.data.materials.append(bpy.data.materials["CopperMaterial"])
    else:
        # Update existing cone
        mesh = obj.data
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=props.cone_segments,
                            diameter1=base_radius * 2.0, diameter2=tip_radius * 2.0, depth=height)
        bm.to_mesh(mesh)
        bm.free()
        obj.location = (0, 0, height / 2.0)
    
    print(f"Cone updated: {base_radius:.3f}m base, {height:.3f}m height")
    return obj

# ------------- Coil Path Generation -------------
def generate_coil_path_points(props, antenna_index):
    """Generate 3D path points for a coil antenna in meters"""
    base_radius = props.cone_base_diameter / 2000.0
    tip_radius = max(props.cone_tip_h / 2000.0, 0.001) if props.cone_tip_h > 0 else 0.001
    height = props.cone_height / 1000.0
    num_wraps = props.coil_wrap_count
    points_per_wrap = min(props.coil_resolution, 200)  # Limit for performance
    total_points = int(num_wraps * points_per_wrap)
    
    # Array positioning
    array_twist_rad = math.radians(props.coil_array_twist)
    antenna_base_rotation = (antenna_index * 2.0 * math.pi) / max(1, props.num_antennas)

    pts = []
    for i in range(total_points + 1):
        t = i / total_points if total_points > 0 else 0
        z = t * height
        
        # Linear interpolation between base and tip radius
        r = base_radius * (1 - t) + tip_radius * t
        
        # Helical winding
        theta = num_wraps * 2.0 * math.pi * t
        theta += antenna_base_rotation + array_twist_rad * t
        
        # Convert to Cartesian coordinates
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        pts.append((x, y, z))
    
    return np.array(pts, dtype=np.float64)

def create_or_update_coil_curve(props, antenna_index):
    """Create or update a coil curve object"""
    name = f"AntennaCoil_{antenna_index}"
    obj = bpy.data.objects.get(name)
    points = generate_coil_path_points(props, antenna_index)
    
    if obj is None:
        # Create new curve
        curve_data = bpy.data.curves.new(name, type='CURVE')
        curve_data.dimensions = '3D'
        spline = curve_data.splines.new('POLY')
        spline.points.add(len(points) - 1)
        
        for i, p in enumerate(points):
            spline.points[i].co = (p[0], p[1], p[2], 1.0)
        
        curve_data.bevel_depth = get_wire_diameter_mm(props.coil_gauge) / 2000.0
        curve_data.bevel_resolution = 3  # Lower for performance
        curve_data.fill_mode = 'FULL'
        
        obj = bpy.data.objects.new(name, curve_data)
        bpy.context.collection.objects.link(obj)
        
        # Create material
        color = ANTENNA_COLORS[antenna_index % len(ANTENNA_COLORS)]
        mat_name = f"CoilMaterial_{antenna_index}"
        
        if mat_name not in bpy.data.materials:
            mat = bpy.data.materials.new(mat_name)
            mat.use_nodes = True
            mat.node_tree.nodes.clear()
            bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Metallic'].default_value = 0.8
            bsdf.inputs['Roughness'].default_value = 0.3
            output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
            mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        obj.data.materials.append(bpy.data.materials[mat_name])
    else:
        # Update existing curve
        spline = obj.data.splines[0]
        if len(spline.points) != len(points):
            obj.data.splines.remove(spline)
            spline = obj.data.splines.new('POLY')
            spline.points.add(len(points) - 1)
        
        for i, p in enumerate(points):
            spline.points[i].co = (p[0], p[1], p[2], 1.0)
        
        obj.data.bevel_depth = get_wire_diameter_mm(props.coil_gauge) / 2000.0
    
    return obj

# ------------- Simplified Field Visualization -------------
def create_simple_field_visualization(props):
    """Create simplified field visualization for performance"""
    if not props.show_em_fields:
        return
        
    print("Creating simplified field visualization...")
    
    # Get antenna segments
    antenna_segments = []
    for a in range(props.num_antennas):
        if props.signal_amplitude_per_antenna[a] > 0:
            segments = generate_coil_path_points(props, a)
            antenna_segments.append(segments)
    
    if not antenna_segments:
        print("No active antennas")
        return
    
    # Generate simplified grid (much smaller for performance)
    nx = min(props.slice_resolution_x, 32)
    ny = min(props.slice_resolution_y, 32) 
    nz = min(props.slice_resolution_z, 4)
    
    # Grid dimensions
    w = props.slice_width / 1000.0
    d = props.slice_depth / 1000.0
    h = props.slice_height / 1000.0
    z0 = props.slice_z_offset / 1000.0
    
    xs = np.linspace(-w / 2.0, w / 2.0, nx)
    ys = np.linspace(-d / 2.0, d / 2.0, ny)
    zs = np.linspace(z0, z0 + h, nz)
    
    # Calculate fields for Z-planes
    frequency_hz = props.frequency_ghz * 1e9
    current_amp = props.current_amplitude
    
    col = ensure_collection("EMFields")
    
    # Remove existing field objects
    for obj in list(bpy.data.objects):
        if obj.name.startswith("EMField_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Create field visualization for Z-planes
    for plane_idx, z_plane in enumerate(zs):
        X_plane, Y_plane = np.meshgrid(xs, ys, indexing='ij')
        Z_plane = np.full_like(X_plane, z_plane)
        
        obs_points = np.stack([X_plane.ravel(), Y_plane.ravel(), Z_plane.ravel()], axis=1)
        
        # Calculate simplified field
        field_values = calculate_simple_field_pattern(antenna_segments, frequency_hz, current_amp, obs_points)
        
        if np.max(field_values) <= 0:
            continue
            
        # Create mesh plane
        create_simple_field_plane(X_plane, Y_plane, Z_plane, field_values.reshape(X_plane.shape), 
                                plane_idx, col, props)

def create_simple_field_plane(X, Y, Z, field_values, plane_idx, collection, props):
    """Create a simplified mesh plane for field visualization"""
    name = f"EMField_Plane_{plane_idx}"
    
    # Build vertices
    verts = []
    nx, ny = X.shape
    for i in range(nx):
        for j in range(ny):
            verts.append((float(X[i, j]), float(Y[i, j]), float(Z[i, j])))
    
    # Build faces
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = a + 1
            c = a + ny + 1
            d = a + ny
            faces.append((a, b, c, d))
    
    # Create mesh
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    
    # Simple coloring based on field strength
    max_val = np.max(field_values) + 1e-30
    
    # Set up vertex colors
    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name="FieldIntensity")
    vcol = mesh.vertex_colors.active
    
    # Color polygons
    for poly in mesh.polygons:
        # Get average field value for this polygon
        avg_val = 0
        count = 0
        for vertex_idx in poly.vertices:
            vi = vertex_idx // ny
            vj = vertex_idx % ny
            if vi < field_values.shape[0] and vj < field_values.shape[1]:
                avg_val += field_values[vi, vj]
                count += 1
        
        if count > 0:
            avg_val /= count
            norm = avg_val / max_val
            
            # Simple color mapping: blue -> red
            if norm < 0.5:
                color = (0, 0, 1 - norm, 0.7)  # Blue to purple
            else:
                color = (norm, 0, 0, 0.7)      # Purple to red
            
            for loop_idx in poly.loop_indices:
                vcol.data[loop_idx].color = color
    
    # Simple material
    mat_name = "SimpleFieldMaterial"
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        output = nodes.new("ShaderNodeOutputMaterial")
        vcol_node = nodes.new("ShaderNodeVertexColor")
        vcol_node.layer_name = "FieldIntensity"
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        
        mat.node_tree.links.new(vcol_node.outputs['Color'], bsdf.inputs['Base Color'])
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        mat.blend_method = 'BLEND'
        bsdf.inputs['Alpha'].default_value = 0.7
    
    mesh.materials.append(bpy.data.materials[mat_name])
    obj.show_transparent = True

# ------------- Statistics Output -------------
def print_antenna_statistics(props):
    """Print simplified antenna statistics"""
    print("\n" + "="*60)
    print("ANTENNA PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Get active antenna segments
    antenna_segments = []
    for a in range(props.num_antennas):
        if props.signal_amplitude_per_antenna[a] > 0:
            segments = generate_coil_path_points(props, a)
            antenna_segments.append(segments)
    
    if not antenna_segments:
        print("No active antennas configured")
        return
    
    # Calculate parameters
    frequency_hz = props.frequency_ghz * 1e9
    params = calculate_antenna_stats_fast(antenna_segments, frequency_hz, props.coil_gauge, props.current_amplitude)
    
    # Display results
    print(f"OPERATING PARAMETERS:")
    print(f"  Frequency: {params['frequency_mhz']:.3f} MHz")
    print(f"  Wavelength: {params['wavelength_m']:.4f} m")
    print(f"  Active antennas: {params['num_antennas']}")
    
    print(f"\nPHYSICAL CHARACTERISTICS:")
    print(f"  Wire gauge: {props.coil_gauge} AWG")
    print(f"  Total wire length: {params['total_wire_length_m']:.3f} m")
    print(f"  DC resistance: {params['dc_resistance_ohms']:.4f} Ω")
    print(f"  Estimated inductance: {params['inductance_uh']:.2f} μH")
    
    print(f"\nELECTRICAL PARAMETERS:")
    print(f"  Impedance: {params['impedance_ohms']:.2f} Ω")
    print(f"  Q-factor: {params['q_factor']:.1f}")
    print(f"  Current: {props.current_amplitude:.2f} A")
    
    print(f"\nPOWER ANALYSIS:")
    print(f"  Resistive loss: {params['resistive_power_w']:.3f} W")
    print(f"  Estimated radiated: {params['radiated_power_w']:.3f} W")
    print(f"  Efficiency: {params['efficiency_percent']:.1f}%")
    
    print("="*60)

# ------------- Update Functions -------------
def update_antenna_geometry(self, context):
    """Update antenna geometry only"""
    props = context.scene.antenna_props
    
    try:
        print("Updating antenna geometry...")
        
        # Update cone and coils
        create_or_update_cone(props)
        
        for i in range(props.num_antennas):
            create_or_update_coil_curve(props, i)
        
        # Remove excess coils
        for obj in list(bpy.data.objects):
            if obj.name.startswith("AntennaCoil_"):
                try:
                    idx = int(obj.name.split("_")[-1])
                    if idx >= props.num_antennas:
                        bpy.data.objects.remove(obj, do_unlink=True)
                except (ValueError, IndexError):
                    pass
        
        # Update viewport
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        print("Geometry update complete")
        
    except Exception as e:
        print(f"Error updating geometry: {e}")

def update_fields_only(self, context):
    """Update electromagnetic fields only"""
    props = context.scene.antenna_props
    
    try:
        print("Updating electromagnetic fields...")
        create_simple_field_visualization(props)
        
        # Update viewport
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        print("Field update complete")
        
    except Exception as e:
        print(f"Error updating fields: {e}")

# ------------- Property Definitions -------------
class ANTENNA_PG_Properties(PropertyGroup):
    """Property group for antenna designer settings"""
    
    # Cone geometry
    cone_base_diameter: FloatProperty(
        name="Base Diameter (mm)", default=250.0, min=1.0, max=2000.0, 
        description="Diameter of cone base", update=update_antenna_geometry
    )
    cone_height: FloatProperty(
        name="Cone Height (mm)", default=500.0, min=1.0, max=2000.0,
        description="Total height of cone", update=update_antenna_geometry
    )
    cone_tip_h: FloatProperty(
        name="Tip Height (mm)", default=20.0, min=0.0, max=1000.0,
        description="Height of cone tip section", update=update_antenna_geometry
    )
    cone_segments: IntProperty(
        name="Cone Segments", default=64, min=8, max=256,
        description="Number of segments in cone mesh", update=update_antenna_geometry
    )
    
    # Coil geometry
    num_antennas: IntProperty(
        name="Number of Antennas", default=4, min=1, max=8,
        description="Number of antennas in array", update=update_antenna_geometry
    )
    coil_wrap_count: IntProperty(
        name="Coil Wraps", default=20, min=1, max=100,
        description="Number of coil wraps around cone", update=update_antenna_geometry
    )
    coil_resolution: IntProperty(
        name="Coil Resolution", default=50, min=10, max=200,
        description="Points per coil wrap", update=update_antenna_geometry
    )
    coil_gauge: EnumProperty(
        name="Wire Gauge (AWG)", 
        items=[(g, g, g) for g in WIRE_GAUGE_DIAMETERS.keys()],
        default='12', description="Wire gauge", update=update_antenna_geometry
    )
    coil_array_twist: FloatProperty(
        name="Array Twist (deg)", default=90.0, min=-360.0, max=360.0,
        description="Twist angle between antennas", update=update_antenna_geometry
    )
    
    # Signal parameters
    signal_amplitude_per_antenna: bpy.props.FloatVectorProperty(
        name="Amplitudes", size=8, default=(1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0),
        description="Signal amplitude per antenna", update=update_fields_only
    )
    signal_phase: bpy.props.FloatVectorProperty(
        name="Phase (rad)", size=8, default=(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0),
        description="Signal phase per antenna", update=update_fields_only
    )
    
    # RF parameters
    frequency_ghz: FloatProperty(
        name="Frequency (GHz)", default=2.4, min=0.001, max=100.0,
        description="Operating frequency", update=update_fields_only
    )
    current_amplitude: FloatProperty(
        name="Current (A)", default=10.0, min=0.0, max=1000.0,
        description="Current amplitude", update=update_fields_only
    )
    
    # Field visualization
    show_em_fields: BoolProperty(
        name="Show EM Fields", default=False,
        description="Enable field visualization", update=update_fields_only
    )
    
    # Simplified field parameters
    slice_width: FloatProperty(
        name="Field Width (mm)", default=400.0, min=10.0, max=2000.0,
        description="Field sampling width", update=update_fields_only
    )
    slice_depth: FloatProperty(
        name="Field Depth (mm)", default=400.0, min=10.0, max=2000.0,
        description="Field sampling depth", update=update_fields_only
    )
    slice_height: FloatProperty(
        name="Field Height (mm)", default=600.0, min=10.0, max=2000.0,
        description="Field sampling height", update=update_fields_only
    )
    slice_z_offset: FloatProperty(
        name="Field Z Offset (mm)", default=-100.0, min=-2000.0, max=2000.0,
        description="Z offset for field sampling", update=update_fields_only
    )
    
    # Performance settings
    slice_resolution_x: IntProperty(
        name="Field Res X", default=16, min=4, max=64,
        description="Field resolution X", update=update_fields_only
    )
    slice_resolution_y: IntProperty(
        name="Field Res Y", default=16, min=4, max=64,
        description="Field resolution Y", update=update_fields_only
    )
    slice_resolution_z: IntProperty(
        name="Field Planes", default=3, min=1, max=8,
        description="Number of field planes", update=update_fields_only
    )

# ------------- UI Panel -------------
class ANTENNA_PT_main_panel(Panel):
    """Main UI panel for antenna designer"""
    bl_label = "Antenna Designer"
    bl_idname = "ANTENNA_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Antenna"

    def draw(self, context):
        layout = self.layout
        props = context.scene.antenna_props

        # Header
        box = layout.box()
        box.label(text="Professional Antenna Designer", icon='OUTLINER_OB_MESH')
        
        col = box.column(align=True)
        
        # Cone geometry section
        col.label(text="Cone Geometry:", icon='MESH_CONE')
        col.prop(props, "cone_base_diameter")
        col.prop(props, "cone_height")
        col.prop(props, "cone_tip_h")
        col.prop(props, "cone_segments")
        
        col.separator()
        
        # Coil geometry section
        col.label(text="Coil Array:", icon='CURVE_BEZCURVE')
        col.prop(props, "num_antennas")
        col.prop(props, "coil_wrap_count")
        col.prop(props, "coil_resolution")
        col.prop(props, "coil_gauge")
        col.prop(props, "coil_array_twist")
        
        col.separator()
        
        # Signal parameters section
        col.label(text="Signal Parameters:", icon='DRIVER')
        col.prop(props, "frequency_ghz")
        col.prop(props, "current_amplitude")
        
        # Per-antenna parameters (simplified)
        if props.num_antennas > 0:
            col.label(text="Per-Antenna Control:")
            for i in range(min(props.num_antennas, 4)):  # Limit to 4 for UI space
                if i < props.num_antennas:
                    row = col.row(align=True)
                    row.prop(props, "signal_amplitude_per_antenna", index=i, text=f"Amp {i+1}")
                    row.prop(props, "signal_phase", index=i, text=f"φ{i+1}")
        
        col.separator()
        
        # Electromagnetic field visualization
        col.label(text="Field Visualization:", icon='FORCE_MAGNETIC')
        col.prop(props, "show_em_fields")
        
        if props.show_em_fields:
            sub_box = col.box()
            sub_col = sub_box.column(align=True)
            sub_col.label(text="Field Volume:")
            
            row = sub_col.row(align=True)
            row.prop(props, "slice_width", text="W")
            row.prop(props, "slice_depth", text="D")
            sub_col.prop(props, "slice_height", text="Height")
            sub_col.prop(props, "slice_z_offset", text="Z Offset")
            
            sub_col.separator()
            sub_col.label(text="Resolution (Lower = Faster):")
            row = sub_col.row(align=True)
            row.prop(props, "slice_resolution_x", text="X")
            row.prop(props, "slice_resolution_y", text="Y")
            sub_col.prop(props, "slice_resolution_z", text="Planes")
        
        col.separator()
        
        # Action buttons
        action_col = col.column(align=True)
        action_col.scale_y = 1.3
        
        row = action_col.row(align=True)
        row.operator("antenna.refresh", text="Update Design", icon='FILE_REFRESH')
        
        row = action_col.row(align=True)
        row.operator("antenna.show_statistics", text="Show Stats", icon='TEXT')
        
        row = action_col.row(align=True)
        row.operator("antenna.clear_objects", text="Clear All", icon='TRASH')
        
        # Performance info
        info_col = col.column(align=True)
        info_col.scale_y = 0.8
        info_col.label(text="Performance Optimized Version", icon='INFO')
        if props.show_em_fields:
            total_samples = props.slice_resolution_x * props.slice_resolution_y * props.slice_resolution_z
            info_col.label(text=f"Field samples: {total_samples:,}")

# ------------- Operators -------------
class ANTENNA_OT_clear_objects(bpy.types.Operator):
    """Clear all antenna objects from the scene"""
    bl_idname = "antenna.clear_objects"
    bl_label = "Clear Antenna Objects"
    bl_description = "Remove all antenna-related objects"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        clear_antenna_objects()
        self.report({'INFO'}, "Cleared antenna objects")
        return {'FINISHED'}

class ANTENNA_OT_refresh(bpy.types.Operator):
    """Refresh antenna design"""
    bl_idname = "antenna.refresh"
    bl_label = "Refresh Antenna"
    bl_description = "Update antenna geometry and fields"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.antenna_props
        try:
            update_antenna_geometry(None, context)
            if props.show_em_fields:
                update_fields_only(None, context)
            self.report({'INFO'}, "Antenna design updated")
        except Exception as e:
            self.report({'ERROR'}, f"Update failed: {str(e)}")
            return {'CANCELLED'}
        return {'FINISHED'}

class ANTENNA_OT_show_statistics(bpy.types.Operator):
    """Show antenna statistics"""
    bl_idname = "antenna.show_statistics"
    bl_label = "Show Statistics"
    bl_description = "Display antenna performance in console"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.antenna_props
        try:
            print_antenna_statistics(props)
            self.report({'INFO'}, "Statistics shown in console (Window > Toggle System Console)")
        except Exception as e:
            self.report({'ERROR'}, f"Statistics failed: {str(e)}")
            return {'CANCELLED'}
        return {'FINISHED'}

# ------------- Utility Functions -------------
def initialize_scene_properties():
    """Initialize default scene properties"""
    try:
        if hasattr(bpy.context.scene, 'antenna_props'):
            props = bpy.context.scene.antenna_props
            
            # Set reasonable defaults
            default_amps = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            for i in range(len(default_amps)):
                if i < len(props.signal_amplitude_per_antenna):
                    if props.signal_amplitude_per_antenna[i] == 0 and i < 4:
                        props.signal_amplitude_per_antenna[i] = default_amps[i]
            
            print("Scene properties initialized")
    except Exception as e:
        print(f"Error initializing properties: {e}")

def validate_properties(props):
    """Simple property validation"""
    issues = []
    
    if props.cone_base_diameter <= 0:
        issues.append("Cone diameter must be positive")
    if props.frequency_ghz <= 0:
        issues.append("Frequency must be positive")
    if props.num_antennas < 1:
        issues.append("Need at least one antenna")
    
    # Check for reasonable field resolution
    if props.show_em_fields:
        total_samples = props.slice_resolution_x * props.slice_resolution_y * props.slice_resolution_z
        if total_samples > 8192:  # 32x32x8 = 8192 max
            issues.append(f"Field resolution too high ({total_samples} samples)")
    
    return issues

# ------------- Registration -------------
classes = [
    ANTENNA_PG_Properties,
    ANTENNA_PT_main_panel,
    ANTENNA_OT_clear_objects,
    ANTENNA_OT_refresh,
    ANTENNA_OT_show_statistics,
]

def register():
    """Register addon classes and properties"""
    print("Registering Professional Antenna Designer (Performance Optimized)...")
    
    # Register classes
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as e:
            print(f"Failed to register {cls.__name__}: {e}")
    
    # Add property group to scene
    bpy.types.Scene.antenna_props = PointerProperty(type=ANTENNA_PG_Properties)
    
    # Initialize properties
    initialize_scene_properties()
    
    # Schedule initial setup (delayed to avoid blocking startup)
    def initial_create():
        try:
            if hasattr(bpy.context.scene, 'antenna_props'):
                update_antenna_geometry(None, bpy.context)
                print("Initial antenna created")
        except Exception as e:
            print(f"Initial setup failed: {e}")
        return None
    
    # Delay initial creation by 0.5 seconds to ensure Blender is ready
    bpy.app.timers.register(initial_create, first_interval=0.5)
    
    print("Registration complete! Look for 'Antenna' tab in 3D Viewport N-panel")

def unregister():
    """Unregister addon"""
    print("Unregistering Professional Antenna Designer...")
    
    # Clear objects
    try:
        clear_antenna_objects()
    except:
        pass
    
    # Remove property
    try:
        del bpy.types.Scene.antenna_props
    except:
        pass
    
    # Unregister classes
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            print(f"Failed to unregister {cls.__name__}: {e}")
    
    print("Unregistration complete")

# ------------- Main Execution -------------
if __name__ == "__main__":
    # Development registration
    try:
        unregister()
    except:
        pass
    
    register()
    
    print("\n" + "="*60)
    print("PROFESSIONAL ANTENNA DESIGNER - PERFORMANCE OPTIMIZED")
    print("="*60)
    print("FEATURES:")
    print("• Fast antenna geometry creation and visualization")
    print("• Simplified electromagnetic field calculations")
    print("• Real-time parameter analysis and statistics")
    print("• Performance-optimized for smooth Blender operation")
    print("• Wire gauge selection and electrical calculations")
    print("• Multi-antenna array support with phase control")
    print()
    print("USAGE:")
    print("1. Enable add-on in Preferences > Add-ons")
    print("2. Open 3D Viewport > press N > click 'Antenna' tab")
    print("3. Adjust geometry parameters")
    print("4. Enable 'Show EM Fields' for field visualization")
    print("5. Use 'Show Stats' for performance analysis")
    print()
    print("PERFORMANCE TIPS:")
    print("• Start with low field resolution settings")
    print("• Increase resolution gradually as needed")
    print("• Use 'Update Design' button after changes")
    print("• Check console (Window > Toggle System Console) for stats")
    print("="*60)
