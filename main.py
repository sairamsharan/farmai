# To run this app : "uvicorn main:app --reload"

import os
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pickle
import numpy as np
from inference_sdk import InferenceHTTPClient
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from dict import plant_diseases_to_fertilizers
from mappings import PLANT_MODEL_MAPPING, PREDICTED_LABEL_TO_DISEASE_MAPPING

with open("models/crop_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open("models/label_encoder.pkl", "rb") as encoder_file:
    loaded_label_encoder = pickle.load(encoder_file)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all HTTP headers
)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key="yjGS9C041cEmorXMHwZI"
)


def get_supported_plant_types():
    return list(PLANT_MODEL_MAPPING.keys())


class CropPredictionInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.post("/predict-crop")
def predict_crop(data: CropPredictionInput):
    try:
        X_testing = np.array(
            [
                data.nitrogen,
                data.phosphorus,
                data.potassium,
                data.temperature,
                data.humidity,
                data.ph,
                data.rainfall,
            ]
        ).reshape(1, -1)

        X_testing_scaled = loaded_scaler.transform(X_testing)

        y_testing = model.predict(X_testing_scaled)[0]

        label = loaded_label_encoder.inverse_transform([y_testing])[0]

        return {"predicted_crop": label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-disease/{plant_type}")
async def detect_disease(
    plant_type: str,
    file: UploadFile = File(...),
):
    plant_type = plant_type.lower()

    if plant_type not in PLANT_MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported plant type. Supported types are: {', '.join(PLANT_MODEL_MAPPING.keys())}",
        )

    try:
        temp_file_path = f"temp_{plant_type}_{file.filename}"

        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        model_id = PLANT_MODEL_MAPPING[plant_type]
        result = CLIENT.infer(temp_file_path, model_id=model_id)

        os.remove(temp_file_path)

        predicted_class = result["predictions"][0]["class"]
        confidence = result["predictions"][0]["confidence"]
        if predicted_class in PREDICTED_LABEL_TO_DISEASE_MAPPING:
            label_final = PREDICTED_LABEL_TO_DISEASE_MAPPING[predicted_class]
            disease_info = plant_diseases_to_fertilizers.get(label_final, None)
        else:
            disease_info = plant_diseases_to_fertilizers.get(predicted_class, None)

        if disease_info:
            return {
                "plant_type": plant_type,
                "disease": predicted_class,
                "confidence": f"{confidence * 100:.2f}%",
                "fertilizer_recommendation": disease_info["fertilizer"],
                "treatment_recommendation": disease_info["treatment"],
            }
        else:
            return {
                "plant_type": plant_type,
                "disease": predicted_class,
                "confidence": f"{confidence * 100:.2f}%",
                "fertilizer_recommendation": "No specific fertilizer recommendation for this disease",
                "treatment_recommendation": "No specific treatment recommendation for this disease",
            }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing {plant_type} image: {str(e)}"
        )


@app.get("/supported-plants")
async def get_supported_plants():
    supported_plants = get_supported_plant_types()
    return {
        "supported_plants": supported_plants,
        "total_count": len(supported_plants),
    }
