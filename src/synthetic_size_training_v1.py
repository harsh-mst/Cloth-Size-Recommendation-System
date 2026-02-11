import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1) Load synthetic data
df = pd.read_csv("synthetic_size_training_v1.csv")

# 2) Clean / fill missing
df["chest_size"].fillna("Unknown", inplace=True)
df["waist_size"].fillna("Unknown", inplace=True)

# 3) Encode chest/waist
chest_encoder = LabelEncoder()
waist_encoder = LabelEncoder()
df["chest_encoded"] = chest_encoder.fit_transform(df["chest_size"])
df["waist_encoded"] = waist_encoder.fit_transform(df["waist_size"])

print("Chest classes:", list(chest_encoder.classes_))
print("Waist classes:", list(waist_encoder.classes_))

# 4) Features & target
X = df[["height", "weight", "chest_encoded", "waist_encoded"]]
y = df["size"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# 6) Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
print(classification_report(y_test, y_pred))

# 7) Save
with open("size_recommender_model_synth.pkl", "wb") as f:
    pickle.dump(model, f)
with open("chest_encoder_synth.pkl", "wb") as f:
    pickle.dump(chest_encoder, f)
with open("waist_encoder_synth.pkl", "wb") as f:
    pickle.dump(waist_encoder, f)

print("\n✓ Saved: size_recommender_model_synth.pkl, chest_encoder_synth.pkl, waist_encoder_synth.pkl")
