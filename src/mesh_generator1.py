import os
import io
import json
import struct
import numpy as np

# ── Patch STAR config at runtime ──────────────────────────────────────────────
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "star_weights")

def _get_apose():
    """
    A-pose: arms angled ~45° down from shoulders.
    STAR uses 72 pose parameters (24 joints × 3 axis-angle each).
    Joint indices: 16=left shoulder, 17=right shoulder
    """
    poses = torch.zeros(1, 72)

    # Left shoulder — rotate down (negative Z axis = lower arm)
    poses[0, 16*3 + 2] = -1.0   # left shoulder Z rotation

    # Right shoulder — rotate down (positive Z = lower arm)
    poses[0, 17*3 + 2] =  1.0   # right shoulder Z rotation

    return poses



try:
    from star import config as star_config
    star_config.path_male_star    = os.path.join(WEIGHTS_DIR, "male",   "model.npz")
    star_config.path_female_star  = os.path.join(WEIGHTS_DIR, "female", "model.npz")
    star_config.path_neutral_star = os.path.join(WEIGHTS_DIR, "female", "model.npz")
    star_config.cfg.path_male_star    = star_config.path_male_star
    star_config.cfg.path_female_star  = star_config.path_female_star
    star_config.cfg.path_neutral_star = star_config.path_neutral_star

    import torch
    from star.pytorch.star import STAR as StarModel
    STAR_AVAILABLE = True
    print("✅ STAR model ready")
except Exception as e:
    STAR_AVAILABLE = False
    print(f"⚠️  STAR unavailable, using procedural fallback: {e}")


# ── Size label lookup tables ───────────────────────────────────────────────────
CHEST_CM = {"S": 86, "M": 96, "L": 106, None: 96}
WAIST_CM = {"S": 70, "M": 80, "L": 90,  None: 80}


# ── Beta estimation from measurements ─────────────────────────────────────────
def _measurements_to_betas(height_cm, weight_kg, gender,
                            chest_label=None, waist_label=None):

    h_m      = height_cm / 100.0
    bmi      = weight_kg / (h_m ** 2)

    # ✅ Clamp BMI to realistic trained range (17–38)
    bmi_clamped = max(17.0, min(bmi, 38.0))
    bmi_norm    = (bmi_clamped - 22) / 6.0  # range: -0.83 to +2.67

    h_norm     = (height_cm - 170) / 10.0

    chest_cm   = CHEST_CM.get(chest_label, 96)
    waist_cm   = WAIST_CM.get(waist_label, 80)
    chest_norm = (chest_cm - 96) / 10.0
    waist_norm = (waist_cm - 80) / 10.0
    taper      = chest_norm - waist_norm

    betas = torch.zeros(1, 10)
    betas[0, 0] = h_norm
    betas[0, 1] = bmi_norm * 1.5
    betas[0, 2] = chest_norm * 1.2
    betas[0, 3] = waist_norm + bmi_norm * 0.5
    betas[0, 4] = taper * 0.8
    betas[0, 5] = max(0.0, float(bmi_norm)) * 0.8

    if gender.lower() == "female":
        betas[0, 3] = betas[0, 3] * 0.85
        betas[0, 6] = 0.5 + waist_norm * 0.3
        betas[0, 7] = chest_norm * 0.4

    return betas
# ── GLB export helper ──────────────────────────────────────────────────────────
def _export_glb(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Pack vertices + faces into a valid GLTFv2 .glb binary."""
    verts = vertices.astype(np.float32)
    tris  = faces.astype(np.uint32)

    vert_bytes = verts.tobytes()
    idx_bytes  = tris.tobytes()

    padding = (4 - len(vert_bytes) % 4) % 4
    bin_data = vert_bytes + b'\x00' * padding + idx_bytes

    v_min = verts.min(axis=0).tolist()
    v_max = verts.max(axis=0).tolist()

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {
                "bufferView": 0, "componentType": 5126,
                "count": len(verts), "type": "VEC3",
                "min": v_min, "max": v_max
            },
            {
                "bufferView": 1, "componentType": 5125,
                "count": len(tris) * 3, "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0,
             "byteLength": len(vert_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(vert_bytes) + padding,
             "byteLength": len(idx_bytes), "target": 34963}
        ],
        "buffers": [{"byteLength": len(bin_data)}]
    }

    json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    json_pad   = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * json_pad

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)

    buf = io.BytesIO()
    buf.write(b'glTF')
    buf.write(struct.pack('<I', 2))
    buf.write(struct.pack('<I', total_len))
    buf.write(struct.pack('<I', len(json_bytes)))
    buf.write(b'JSON')
    buf.write(json_bytes)
    buf.write(struct.pack('<I', len(bin_data)))
    buf.write(b'BIN\x00')
    buf.write(bin_data)

    return buf.getvalue()


# ── STAR mesh ──────────────────────────────────────────────────────────────────
def _generate_star_mesh(height_cm: float, weight_kg: float,
                        gender: str,
                        chest_size: str = None,
                        waist_size: str = None) -> bytes:
    gender_key = "female" if gender.lower() == "female" else "male"

    model = StarModel(gender=gender_key, num_betas=10)
    model.eval()

    betas = _measurements_to_betas(
        height_cm, weight_kg, gender,
        chest_label=chest_size,
        waist_label=waist_size
    )

    # poses = torch.zeros(1, 72)
    poses = _get_apose()
    trans = torch.zeros(1, 3)

    with torch.no_grad():
        out = model(poses, betas, trans)

    verts = out[0].cpu().numpy()   # (6890, 3)
    faces = model.f                # numpy array directly

    return _export_glb(verts, faces)


# ── Procedural fallback ────────────────────────────────────────────────────────
def _generate_procedural_mesh(height_cm: float, weight_kg: float,
                               gender: str) -> bytes:
    import trimesh

    h    = height_cm / 100.0
    h_m  = h
    bmi  = weight_kg / (h_m ** 2)

    # BMI-based scaling: 22 = neutral, below = thinner, above = wider
    bmi_scale  = 1.0 + (bmi - 22) * 0.025
    bmi_scale  = max(0.6, min(bmi_scale, 1.8))   # clamp extremes
    is_female  = gender.lower() == "female"

    parts = []

    # Head
    head = trimesh.creation.icosphere(subdivisions=3, radius=h * 0.075)
    head.apply_translation([0, h * 0.915, 0])
    parts.append(head)

    # Neck
    neck = trimesh.creation.cylinder(radius=h * 0.030 * bmi_scale,
                                     height=h * 0.06, sections=20)
    neck.apply_translation([0, h * 0.855, 0])
    parts.append(neck)

    # Torso — elliptical (wider than deep)
    torso_w = h * 0.14 * bmi_scale
    torso_d = h * 0.10 * bmi_scale
    torso = trimesh.creation.cylinder(radius=torso_w, height=h * 0.32, sections=28)
    torso.vertices[:, 2] *= (torso_d / torso_w)
    torso.apply_translation([0, h * 0.655, 0])
    parts.append(torso)

    # Hips
    hip_w = h * 0.13 * bmi_scale * (1.1 if is_female else 1.0)
    hip_d = h * 0.10 * bmi_scale
    hips = trimesh.creation.cylinder(radius=hip_w, height=h * 0.14, sections=28)
    hips.vertices[:, 2] *= (hip_d / hip_w)
    hips.apply_translation([0, h * 0.44, 0])
    parts.append(hips)

    # Upper arms
    upper_arm_r = h * 0.035 * bmi_scale
    for side in [-1, 1]:
        arm = trimesh.creation.cylinder(radius=upper_arm_r,
                                        height=h * 0.18, sections=16)
        arm.apply_translation([side * (torso_w + upper_arm_r * 0.8), h * 0.70, 0])
        parts.append(arm)

    # Forearms
    forearm_r = h * 0.028 * bmi_scale
    for side in [-1, 1]:
        forearm = trimesh.creation.cylinder(radius=forearm_r,
                                            height=h * 0.16, sections=16)
        forearm.apply_translation([side * (torso_w + forearm_r * 0.8), h * 0.50, 0])
        parts.append(forearm)

    # Thighs
    thigh_r = h * 0.065 * bmi_scale
    for side in [-1, 1]:
        thigh = trimesh.creation.cylinder(radius=thigh_r,
                                          height=h * 0.22, sections=20)
        thigh.apply_translation([side * h * 0.07, h * 0.285, 0])
        parts.append(thigh)

    # Calves
    calf_r = h * 0.042 * bmi_scale
    for side in [-1, 1]:
        calf = trimesh.creation.cylinder(radius=calf_r,
                                         height=h * 0.22, sections=20)
        calf.apply_translation([side * h * 0.07, h * 0.055, 0])
        parts.append(calf)

    # Feet
    for side in [-1, 1]:
        foot = trimesh.creation.box(extents=[h * 0.04, h * 0.03, h * 0.13])
        foot.apply_translation([side * h * 0.07, h * -0.01, h * 0.04])
        parts.append(foot)

    mesh = trimesh.util.concatenate(parts)
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    return scene.export(file_type="glb")


# ── Public API ─────────────────────────────────────────────────────────────────
def generate_mesh_glb(height_cm: float,
                      weight_kg: float,
                      gender: str = "male",
                      chest_size: str = None,
                      waist_size: str = None) -> bytes:
    """
    Generate a human body mesh as GLB bytes.
    Uses STAR model if available, otherwise procedural fallback.

    Args:
        height_cm:  Height in centimeters (140–200)
        weight_kg:  Weight in kilograms (20–150)
        gender:     "male" or "female"
        chest_size: "S", "M", "L", or None
        waist_size: "S", "M", "L", or None

    Returns:
        bytes: GLB binary data
    """
    if STAR_AVAILABLE:
        try:
            return _generate_star_mesh(
                height_cm, weight_kg, gender,
                chest_size=chest_size,
                waist_size=waist_size
            )
        except Exception as e:
            print(f"STAR inference failed, falling back to procedural: {e}")

    return _generate_procedural_mesh(height_cm, weight_kg, gender)