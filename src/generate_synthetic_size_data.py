import pandas as pd
import numpy as np

np.random.seed(42)

# ------------------------------------------------------------------
# 1. Define size profiles (clean, separable distributions)
# ------------------------------------------------------------------
# These ranges are constructed to create clearer boundaries between sizes
size_profiles = {
    "S": {
        "weight": (45, 60),
        "height": (150, 167),
        "chest":  ["S"] * 7 + ["M"] * 2 + ["L"] * 1,   # mostly S
        "waist":  ["S"] * 7 + ["M"] * 2 + ["L"] * 1,
    },
    "M": {
        "weight": (58, 72),
        "height": (160, 175),
        "chest":  ["S"] * 1 + ["M"] * 7 + ["L"] * 2,
        "waist":  ["S"] * 1 + ["M"] * 7 + ["L"] * 2,
    },
    "L": {
        "weight": (70, 85),
        "height": (165, 182),
        "chest":  ["M"] * 3 + ["L"] * 7,
        "waist":  ["M"] * 3 + ["L"] * 7,
    },
    "XL": {
        "weight": (82, 100),
        "height": (168, 188),
        "chest":  ["L"] * 9 + ["M"] * 1,
        "waist":  ["L"] * 9 + ["M"] * 1,
    },
    "XXL": {
        "weight": (95, 125),
        "height": (170, 195),
        "chest":  ["L"] * 10,   # very large upper body
        "waist":  ["L"] * 10,
    },
}

# number of samples per size (balanced but you can tune)
samples_per_size = {
    "S": 8000,
    "M": 8000,
    "L": 8000,
    "XL": 8000,
    "XXL": 6000,   # slight fewer but enough
}

data = []

# ------------------------------------------------------------------
# 2. Generate synthetic samples with Gaussian noise around ranges
# ------------------------------------------------------------------
for size, count in samples_per_size.items():
    profile = size_profiles[size]
    w_min, w_max = profile["weight"]
    h_min, h_max = profile["height"]

    # mean and std for normal distribution
    w_mean = (w_min + w_max) / 2
    h_mean = (h_min + h_max) / 2
    w_std = (w_max - w_min) / 6    # 99% within range
    h_std = (h_max - h_min) / 6

    for _ in range(count):
        weight = np.random.normal(w_mean, w_std)
        height = np.random.normal(h_mean, h_std)

        # clamp to extended range to allow some overlap but still structured
        weight = np.clip(weight, w_min - 3, w_max + 3)
        height = np.clip(height, h_min - 2, h_max + 2)

        chest = np.random.choice(profile["chest"])
        waist = np.random.choice(profile["waist"])

        # Optional: randomly drop chest/waist to simulate missing fields
        if np.random.rand() < 0.15:   # 15% missing chest
            chest_out = None
        else:
            chest_out = chest

        if np.random.rand() < 0.15:   # 15% missing waist
            waist_out = None
        else:
            waist_out = waist

        data.append(
            {
                "height": round(float(height), 2),
                "weight": round(float(weight), 1),
                "chest_size": chest_out,
                "waist_size": waist_out,
                "size": size,
            }
        )

df = pd.DataFrame(data)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total samples: {len(df):,}")
print("\nSize distribution:")
print(df["size"].value_counts())

print("\nHeight summary by size:")
print(df.groupby("size")["height"].describe().round(2))

print("\nWeight summary by size:")
print(df.groupby("size")["weight"].describe().round(2))

print("\nChest distribution:")
print(df["chest_size"].value_counts(dropna=False))

print("\nWaist distribution:")
print(df["waist_size"].value_counts(dropna=False))

# Save to CSV
output_file = "synthetic_size_training_v1.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Saved synthetic dataset to: {output_file}")
