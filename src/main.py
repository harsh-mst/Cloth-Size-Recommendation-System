from .mesh_generator import generate_mesh_glb
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Clothing Size Recommender API v2",
    description="Predict clothing size and generate a 3D human body mesh (.glb)",
    version="2.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Load ML artefacts ─────────────────────────────────────────────────────────
with open("size_recommender_model_synth.pkl", "rb") as f:
    model = pickle.load(f)

with open("chest_encoder_synth.pkl", "rb") as f:
    chest_encoder = pickle.load(f)

with open("waist_encoder_synth.pkl", "rb") as f:
    waist_encoder = pickle.load(f)


# ── Schemas ───────────────────────────────────────────────────────────────────
class UserMeasurements(BaseModel):
    height: float = Field(..., description="Height in centimeters", ge=140, le=200)
    weight: float = Field(..., description="Weight in kilograms", ge=20, le=150)
    gender: Literal["male", "female"] = Field(
        "male", description="Gender — used for mesh shape and size tuning"
    )
    chest_size: Optional[Literal["S", "M", "L"]] = Field(
        None, description="Chest size (optional)"
    )
    waist_size: Optional[Literal["S", "M", "L"]] = Field(
        None, description="Waist size (optional)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "height": 170,
                    "weight": 65,
                    "gender": "female",
                    "chest_size": "M",
                    "waist_size": "M"
                },
                {
                    "height": 180,
                    "weight": 85,
                    "gender": "male",
                    "chest_size": None,
                    "waist_size": None
                }
            ]
        }


class SizePrediction(BaseModel):
    predicted_size: str
    confidence: float
    input_data: dict
    message: str
    all_probabilities: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    supported_sizes: list[str]
    features_used: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────
def encode_size(size: Optional[str], encoder) -> int:
    if size is None:
        return encoder.transform(["M"])[0]
    return encoder.transform([size])[0]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        supported_sizes=["S", "M", "L", "XL", "XXL"],
        features_used=["height", "weight", "gender", "chest_size (optional)", "waist_size (optional)"]
    )


@app.post("/predict", response_model=SizePrediction)
def predict_size(measurements: UserMeasurements):
    try:
        chest_encoded = encode_size(measurements.chest_size, chest_encoder)
        waist_encoded = encode_size(measurements.waist_size, waist_encoder)

        features = np.array([[
            measurements.height,
            measurements.weight,
            chest_encoded,
            waist_encoded
        ]])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities)) * 100

        all_probs = {
            size: round(float(prob) * 100, 2)
            for size, prob in zip(model.classes_, probabilities)
        }

        return SizePrediction(
            predicted_size=prediction,
            confidence=round(confidence, 2),
            input_data={
                "height": measurements.height,
                "weight": measurements.weight,
                "gender": measurements.gender,
                "chest_size": measurements.chest_size,
                "waist_size": measurements.waist_size,
            },
            message=f"Based on your measurements, we recommend size {prediction}",
            all_probabilities=all_probs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/generate-mesh")
def generate_mesh(measurements: UserMeasurements):
    try:
        glb_bytes = generate_mesh_glb(
            height_cm=measurements.height,
            weight_kg=measurements.weight,
            gender=measurements.gender,
            chest_size=measurements.chest_size,   # "S", "M", "L" or None
            waist_size=measurements.waist_size
        )
        return Response(
            content=glb_bytes,
            media_type="model/gltf-binary",
            headers={"Content-Disposition": "attachment; filename=body_mesh.glb"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-and-mesh", response_model=SizePrediction)
def predict_and_mesh(measurements: UserMeasurements):
    """
    Convenience endpoint — returns the size prediction JSON.
    Frontend should call /generate-mesh separately for the GLB binary.
    """
    return predict_size(measurements)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)