import os
import io
import numpy as np

# ── Patch STAR config at runtime ──────────────────────────────────────────────
WEIGHTS_DIR = r"C:\Users\DELL\Desktop\Cloth-Size-Recommendation-System-cm\src\star_weights"

try:
    from star import config as star_config
    star_config.path_male_star    = os.path.join(WEIGHTS_DIR, "male",   "model.npz")
    star_config.path_female_star  = os.path.join(WEIGHTS_DIR, "female", "model.npz")
    star_config.path_neutral_star = os.path.join(WEIGHTS_DIR, "female", "model.npz")

    from star.pytorch.star import STAR as StarModel
    import torch
    STAR_AVAILABLE = True
    print("✅ STAR model ready")
except Exception as e:
    STAR_AVAILABLE = False
    print(f"⚠️  STAR unavailable, using procedural fallback: {e}")


# ── GLB export helper ──────────────────────────────────────────────────────────
def _export_glb(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Convert vertices + faces to a .glb binary (GLTFv2)."""
    import struct, json

    verts = vertices.astype(np.float32)
    tris  = faces.astype(np.uint32)

    vert_bytes = verts.tobytes()
    idx_bytes  = tris.tobytes()

    # Align index buffer to 4 bytes
    padding = (4 - len(vert_bytes) % 4) % 4
    bin_data = vert_bytes + b'\x00' * padding + idx_bytes

    vert_count = len(verts)
    idx_count  = len(tris) * 3

    v_min = verts.min(axis=0).tolist()
    v_max = verts.max(axis=0).tolist()

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1
            }]
        }],
        "accessors": [
            {
                "bufferView": 0, "componentType": 5126,
                "count": vert_count, "type": "VEC3",
                "min": v_min, "max": v_max
            },
            {
                "bufferView": 1, "componentType": 5125,
                "count": idx_count, "type": "SCALAR"
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
    buf.write(b'glTF')                          # magic
    buf.write(struct.pack('<I', 2))             # version
    buf.write(struct.pack('<I', total_len))     # total length
    buf.write(struct.pack('<I', len(json_bytes)))
    buf.write(b'JSON')
    buf.write(json_bytes)
    buf.write(struct.pack('<I', len(bin_data)))
    buf.write(b'BIN\x00')
    buf.write(bin_data)

    return buf.getvalue()


# ── STAR mesh ──────────────────────────────────────────────────────────────────
# def _generate_star_mesh(height_cm: float, weight_kg: float, gender: str) -> bytes:
#     gender_key = 'female' if gender.lower() == 'female' else 'male'
#     model = StarModel(gender=gender_key, num_betas=10)
#     model.eval()

#     height_norm = (height_cm - 170) / 10.0
#     weight_norm = (weight_kg  -  70) / 15.0

#     betas = torch.zeros(1, 10)
#     betas[0, 0] = float(height_norm)
#     betas[0, 1] = float(weight_norm)

#     poses = torch.zeros(1, 72)
#     trans = torch.zeros(1, 3)

#     with torch.no_grad():
#         out = model(poses, betas, trans)

#     verts = out[0].cpu().numpy()   # (6890, 3)
#     faces = model.f                # already numpy, no .cpu()

#     return _export_glb(verts, faces)


def _generate_star_mesh(height_cm: float, weight_kg: float, gender: str) -> bytes:
    gender_key = 'female' if gender.lower() == 'female' else 'male'
    model = StarModel(gender=gender_key, num_betas=10)
    model.eval()

    h = height_cm / 100.0
    bmi = weight_kg / (h ** 2)

    # Normalize from typical ranges
    height_norm = (height_cm - 170) / 10.0   # 0 at 170cm
    weight_norm = (weight_kg - 70)  / 15.0   # 0 at 70kg

    # BMI excess above healthy (22) — drives fat distribution
    bmi_excess = max(0.0, (bmi - 22) / 12.0)

    # Gender adjustments — females carry more fat in hips/thighs
    is_female = gender.lower() == 'female'
    hip_factor  = 1.4 if is_female else 0.8
    bust_factor = 1.2 if is_female else 0.6
    waist_factor = 0.9 if is_female else 1.2   # males carry more belly fat

    betas = torch.zeros(1, 10)
    betas[0, 0] = height_norm                         # overall height/size
    betas[0, 1] = weight_norm                         # overall body volume
    betas[0, 2] = bmi_excess * waist_factor           # abdominal/belly width
    betas[0, 3] = bmi_excess * hip_factor             # hip & thigh volume
    betas[0, 4] = bmi_excess * bust_factor            # chest/bust area
    betas[0, 5] = bmi_excess * 0.5                    # arm thickness
    betas[0, 6] = bmi_excess * 0.4                    # calf/lower leg volume
    betas[0, 7] = height_norm * 0.3                   # limb proportions
    betas[0, 8] = -bmi_excess * 0.2                   # shoulder narrowing at high BMI
    betas[0, 9] = weight_norm * 0.2                   # general torso depth

    # Clamp betas to safe range to avoid mesh artifacts
    betas = torch.clamp(betas, -3.0, 3.0)

    poses = torch.zeros(1, 72)
    trans = torch.zeros(1, 3)

    with torch.no_grad():
        out = model(poses, betas, trans)

    verts = out[0].cpu().numpy()   # (6890, 3)
    faces = model.f                # already numpy

    return _export_glb(verts, faces)
# 

# ── Size label → approximate circumference in cm ──────────────────────────────
CHEST_CM = {"S": 86, "M": 96, "L": 106, None: 96}
WAIST_CM = {"S": 70, "M": 80, "L": 90,  None: 80}


def _measurements_to_betas(height_cm, weight_kg, gender,
                            chest_label=None, waist_label=None):
    """Convert user measurements to STAR shape betas."""
    chest_cm = CHEST_CM.get(chest_label, 96)
    waist_cm = WAIST_CM.get(waist_label, 80)

    # Normalize each measurement relative to average body
    h_norm     = (height_cm - 170) / 10.0   # 0 = 170cm avg height
    w_norm     = (weight_kg  - 70)  / 15.0   # 0 = 70kg avg weight
    chest_norm = (chest_cm   - 96)  / 10.0   # 0 = M chest, +1 = L, -1 = S
    waist_norm = (waist_cm   - 80)  / 10.0   # 0 = M waist, +1 = L, -1 = S

    # Torso taper: big chest + small waist = athletic; equal = rectangular
    taper = chest_norm - waist_norm

    betas = torch.zeros(1, 10)
    betas[0, 0] = h_norm                     # overall height / body scale
    betas[0, 1] = w_norm * 0.7               # general body volume from weight
    betas[0, 2] = chest_norm * 1.2           # chest / upper torso width
    betas[0, 3] = waist_norm * 1.0           # waist / mid-torso width
    betas[0, 4] = taper * 0.8                # torso taper shape
    betas[0, 5] = w_norm * 0.4               # fat distribution (thighs, arms)

    if gender.lower() == "female":
        betas[0, 3] *= 0.85                  # women tend toward smaller waist
        betas[0, 6]  = 0.5 + waist_norm * 0.3  # hip width (female shape)
        betas[0, 7]  = chest_norm * 0.4      # bust shaping

    return betas


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

    poses = torch.zeros(1, 72)
    trans = torch.zeros(1, 3)

    with torch.no_grad():
        out = model(poses, betas, trans)

    verts = out[0].cpu().numpy()   # (6890, 3)
    faces = model.f                # numpy array, no .cpu() needed

    return _export_glb(verts, faces)



# ── Procedural fallback ────────────────────────────────────────────────────────
def _generate_procedural_mesh(height_cm: float, weight_kg: float, gender: str) -> bytes:
    import trimesh

    h = height_cm / 100.0  # convert to meters
    bmi = weight_kg / (h ** 2)
    bmi_scale = 1.0 + (bmi - 22) * 0.02  # scale body width with BMI
    is_female = gender.lower() == "female"

    parts = []

    # ── Head ──────────────────────────────────────────
    head_r = h * 0.075
    head = trimesh.creation.icosphere(subdivisions=3, radius=head_r)
    head.apply_translation([0, h * 0.915, 0])
    parts.append(head)

    # ── Neck ──────────────────────────────────────────
    neck_r = h * 0.030
    neck_h = h * 0.06
    neck = trimesh.creation.cylinder(radius=neck_r, height=neck_h, sections=20)
    neck.apply_translation([0, h * 0.855, 0])
    parts.append(neck)

    # ── Torso ─────────────────────────────────────────
    torso_w = h * 0.14 * bmi_scale
    torso_d = h * 0.10 * bmi_scale
    torso_h = h * 0.32
    torso = trimesh.creation.cylinder(radius=torso_w, height=torso_h, sections=28)
    # Squish to make it elliptical (wider than deep)
    torso.vertices[:, 2] *= (torso_d / torso_w)
    torso.apply_translation([0, h * 0.655, 0])
    parts.append(torso)

    # ── Hips / Pelvis ─────────────────────────────────
    hip_w = h * 0.13 * bmi_scale * (1.1 if is_female else 1.0)
    hip_d = h * 0.10 * bmi_scale
    hip_h = h * 0.14
    hips = trimesh.creation.cylinder(radius=hip_w, height=hip_h, sections=28)
    hips.vertices[:, 2] *= (hip_d / hip_w)
    hips.apply_translation([0, h * 0.44, 0])
    parts.append(hips)

    # ── Upper Arms ────────────────────────────────────
    upper_arm_r = h * 0.035 * bmi_scale
    upper_arm_h = h * 0.18
    for side in [-1, 1]:
        arm = trimesh.creation.cylinder(radius=upper_arm_r, height=upper_arm_h, sections=16)
        # Tilt arms slightly outward
        arm.apply_translation([side * (torso_w + upper_arm_r * 0.8), h * 0.70, 0])
        parts.append(arm)

    # ── Forearms ──────────────────────────────────────
    forearm_r = h * 0.028 * bmi_scale
    forearm_h = h * 0.16
    for side in [-1, 1]:
        forearm = trimesh.creation.cylinder(radius=forearm_r, height=forearm_h, sections=16)
        forearm.apply_translation([side * (torso_w + forearm_r * 0.8), h * 0.50, 0])
        parts.append(forearm)

    # ── Thighs ────────────────────────────────────────
    thigh_r = h * 0.065 * bmi_scale
    thigh_h = h * 0.22
    for side in [-1, 1]:
        thigh = trimesh.creation.cylinder(radius=thigh_r, height=thigh_h, sections=20)
        thigh.apply_translation([side * h * 0.07, h * 0.285, 0])
        parts.append(thigh)

    # ── Calves ────────────────────────────────────────
    calf_r = h * 0.042 * bmi_scale
    calf_h = h * 0.22
    for side in [-1, 1]:
        calf = trimesh.creation.cylinder(radius=calf_r, height=calf_h, sections=20)
        calf.apply_translation([side * h * 0.07, h * 0.055, 0])
        parts.append(calf)

    # ── Feet ──────────────────────────────────────────
    for side in [-1, 1]:
        foot = trimesh.creation.box(extents=[h * 0.04, h * 0.03, h * 0.13])
        foot.apply_translation([side * h * 0.07, h * -0.01, h * 0.04])
        parts.append(foot)

    # Combine all parts
    mesh = trimesh.util.concatenate(parts)

    # Export to GLB
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    glb_bytes = scene.export(file_type="glb")
    return glb_bytes

# ── Public API ─────────────────────────────────────────────────────────────────
# def generate_mesh_glb(height_cm: float, weight_kg: float, gender: str = "male") -> bytes:
#     if STAR_AVAILABLE:
#         try:
#             return _generate_star_mesh(height_cm, weight_kg, gender)
#         except Exception as e:
#             print(f"STAR inference failed, using fallback: {e}")
#     return _generate_procedural_mesh(height_cm, weight_kg, gender)
#     if STAR_AVAILABLE:
#         try:
#             return _generate_star_mesh(height_cm, weight_kg, gender)
#         except Exception as e:
#             print(f"STAR inference failed, using fallback: {e}")
#     return _generate_procedural_mesh(height_cm, weight_kg, gender)


def generate_mesh_glb(height_cm: float, weight_kg: float,
                      gender: str = "male",
                      chest_size: str = None,
                      waist_size: str = None) -> bytes:
    if STAR_AVAILABLE:
        try:
            return _generate_star_mesh(
                height_cm, weight_kg, gender,
                chest_size=chest_size,
                waist_size=waist_size
            )
        except Exception as e:
            print(f"STAR inference failed, using fallback: {e}")
    return _generate_procedural_mesh(height_cm, weight_kg, gender)