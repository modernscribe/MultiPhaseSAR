intended to by run in blender under scripts

import bpy
import bmesh
import numpy as np
import math
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === Parameters ===
scale = 10.0
cone_height = 0.06 * scale
base_radius = 0.015 * scale
top_radius = 0.004 * scale
num_loops = 5
steps_per_loop = 200
ribbon_width = 0.0015 * scale
ribbon_thickness = 0.0008 * scale
num_arms = 4
arm_phase_offsets = [i * (np.pi / 2) for i in range(num_arms)]
sections_cone = 256
wire_segment_length = 0.0005 * scale
output_dir = bpy.path.abspath("//")
stl_out_name = "solid_cone_with_outcrop_ribbons.stl"
csv_pattern_name = "radiation_pattern.csv"
png_pattern_name = "radiation_pattern.png"
frequency_hz = 2.4e9
current_amplitude = 1.0
arm_feed_phases = [0.0 for _ in range(num_arms)]
c = 299792458.0
wavelength = c / frequency_hz
k = 2.0 * np.pi / wavelength
beta_prop = 2.0 * np.pi / wavelength
observation_theta_samples = 181
observation_phi_samples = 361

# === Cleanup Scene ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# === Create Cone ===
bpy.ops.mesh.primitive_cone_add(vertices=sections_cone, radius1=base_radius, radius2=top_radius, depth=cone_height, location=(0,0,cone_height/2))
main_cone = bpy.context.active_object
main_cone.name = "MainCone"

# === Generate Arm Centerline ===
def generate_arm_centerline(phase_offset):
    total_steps = num_loops * steps_per_loop
    points = []
    for i in range(total_steps + 1):
        t = i / total_steps
        z = t * cone_height
        r = base_radius * (1 - t) + top_radius * t
        theta = 2 * np.pi * num_loops * t + phase_offset
        sin_mod = np.sin(4 * np.pi * t + phase_offset) * (0.3 + 0.7 * t)
        r_mod = r + sin_mod * ribbon_width
        x = r_mod * np.cos(theta)
        y = r_mod * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

# === Local Frames for Sweep ===
def compute_frames(points):
    n = len(points)
    tangents = np.zeros((n, 3))
    normals = np.zeros((n, 3))
    binormals = np.zeros((n, 3))
    for i in range(n):
        if i == 0:
            t = points[1] - points[0]
        elif i == n-1:
            t = points[-1] - points[-2]
        else:
            t = points[i+1] - points[i-1]
        t = t / (np.linalg.norm(t) + 1e-12)
        tangents[i] = t
    up = np.array([0.0, 0.0, 1.0])
    normals[0] = np.cross(up, tangents[0])
    if np.linalg.norm(normals[0]) < 1e-6:
        normals[0] = np.array([1.0, 0.0, 0.0])
    normals[0] /= np.linalg.norm(normals[0])
    binormals[0] = np.cross(tangents[0], normals[0])
    for i in range(1, n):
        v = np.cross(tangents[i-1], tangents[i])
        if np.linalg.norm(v) < 1e-6:
            normals[i] = normals[i-1]
        else:
            normals[i] = v / np.linalg.norm(v)
            normals[i] = np.cross(normals[i], tangents[i])
            if np.linalg.norm(normals[i]) < 1e-6:
                normals[i] = normals[i-1]
            else:
                normals[i] /= np.linalg.norm(normals[i])
        binormals[i] = np.cross(tangents[i], normals[i])
    return tangents, normals, binormals

# === Build Ribbon Mesh ===
def build_ribbon_mesh(centerline, width, thickness):
    tangents, normals, binormals = compute_frames(centerline)
    mesh = bpy.data.meshes.new("Ribbon")
    bm = bmesh.new()
    verts = []
    for i in range(len(centerline)):
        p = centerline[i]
        tvec = tangents[i]
        nvec = normals[i]
        half_w = width / 2.0
        top_center = p + (thickness / 2.0) * tvec
        bottom_center = p - (thickness / 2.0) * tvec
        v0 = bm.verts.new(top_center + nvec * half_w)
        v1 = bm.verts.new(top_center - nvec * half_w)
        v2 = bm.verts.new(bottom_center - nvec * half_w)
        v3 = bm.verts.new(bottom_center + nvec * half_w)
        verts.append((v0, v1, v2, v3))
        if i > 0:
            pv0, pv1, pv2, pv3 = verts[i-1]
            cv0, cv1, cv2, cv3 = verts[i]
            bm.faces.new([pv0, pv1, cv1, cv0])
            bm.faces.new([pv1, pv2, cv2, cv1])
            bm.faces.new([pv2, pv3, cv3, cv2])
            bm.faces.new([pv3, pv0, cv0, cv3])
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("RibbonObj", mesh)
    bpy.context.collection.objects.link(obj)
    return obj

# === Build All Arms ===
arm_centerlines = [generate_arm_centerline(phase) for phase in arm_phase_offsets]
for cl in arm_centerlines:
    build_ribbon_mesh(cl, ribbon_width, ribbon_thickness)

# === Radiation Pattern ===
def sample_wire_segments(centerline):
    segments = []
    for i in range(len(centerline)-1):
        p0 = centerline[i]
        p1 = centerline[i+1]
        dl = p1 - p0
        length = np.linalg.norm(dl)
        if length == 0: continue
        num = max(1, int(math.ceil(length / wire_segment_length)))
        for j in range(num):
            a = j/num
            b = (j+1)/num
            rp0 = p0 + dl * a
            rp1 = p0 + dl * b
            rcenter = 0.5 * (rp0 + rp1)
            dvec = rp1 - rp0
            segments.append((rcenter, dvec, np.linalg.norm(dvec)))
    return segments

def compute_radiation_pattern(arms_centerlines, freq_hz, I_amps, feed_phases):
    k_local = 2.0 * np.pi * freq_hz / c
    thetas = np.linspace(0, np.pi, observation_theta_samples)
    phis = np.linspace(0, 2*np.pi, observation_phi_samples)
    P = np.zeros((len(thetas), len(phis)))
    segments_all = []
    arm_index = 0
    for centerline in arms_centerlines:
        segs = sample_wire_segments(centerline)
        phase_feed = feed_phases[arm_index] if arm_index < len(feed_phases) else 0.0
        for (rcenter, dvec, seglen) in segs:
            segments_all.append((rcenter, dvec, seglen, phase_feed))
        arm_index += 1
    for ti, theta in enumerate(thetas):
        for pi, phi in enumerate(phis):
            rhat = np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])
            V = np.zeros(3, dtype=complex)
            for (rcenter, dvec, seglen, phase_feed) in segments_all:
                I_seg = I_amps * np.exp(1j*(phase_feed + beta_prop * np.linalg.norm(rcenter)))
                expo = np.exp(1j * k_local * np.dot(rhat, rcenter))
                V += I_seg * dvec * expo
            cross = np.cross(rhat, np.cross(rhat, V))
            E = cross
            P[ti, pi] = np.real(np.vdot(E, E))
    P_norm = P / (np.max(P) + 1e-30)
    return thetas, phis, P_norm

thetas, phis, P = compute_radiation_pattern(arm_centerlines, frequency_hz, current_amplitude, arm_feed_phases)

# === Save Outputs ===
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

stl_path = os.path.join(output_dir, stl_out_name)
bpy.ops.export_mesh.stl(filepath=stl_path, use_selection=False)

csv_path = os.path.join(output_dir, csv_pattern_name)
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['theta_deg','phi_deg','power_norm'])
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            writer.writerow([np.degrees(theta), np.degrees(phi), float(P[i,j])])

plt.figure(figsize=(8,6))
phi0_index = 0
plt.polar(thetas, P[:, phi0_index])
plt.title('Radiation cut (phi=0) normalized')
png_path = os.path.join(output_dir, png_pattern_name)
plt.savefig(png_path, dpi=200, bbox_inches='tight')
plt.close()

print("STL saved to:", stl_path)
print("CSV saved to:", csv_path)
print("Pattern image saved to:", png_path)
