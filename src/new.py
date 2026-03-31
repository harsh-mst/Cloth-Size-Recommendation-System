import os

from star import config as star_config
star_config.path_male_star    = r"C:\Users\DELL\Desktop\Cloth-Size-Recommendation-System-cm\src\star_weights\male\model.npz"
star_config.path_female_star  = r"C:\Users\DELL\Desktop\Cloth-Size-Recommendation-System-cm\src\star_weights\female\model.npz"
star_config.path_neutral_star = r"C:\Users\DELL\Desktop\Cloth-Size-Recommendation-System-cm\src\star_weights\female\model.npz"
star_config.cfg.path_male_star    = star_config.path_male_star
star_config.cfg.path_female_star  = star_config.path_female_star
star_config.cfg.path_neutral_star = star_config.path_neutral_star

import torch
from star.pytorch.star import STAR

model = STAR(gender='male', num_betas=10)
model.eval()

betas = torch.zeros(1, 10)
betas[0, 0] = 0.5
betas[0, 1] = 0.3

poses = torch.zeros(1, 72)
trans = torch.zeros(1, 3)

with torch.no_grad():
    out = model(poses, betas, trans)

verts = out[0].cpu().numpy()   # (6890, 3)
faces = model.f                # already numpy — no .cpu() needed

print("Vertices shape:", verts.shape)
print("Faces shape:", faces.shape)
print("✅ STAR working correctly!")