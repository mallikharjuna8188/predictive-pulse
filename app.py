from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

label_map = {
    0: "Low Hypertension Risk",
    1: "High Hypertension Risk",
}


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    try:
        # order must match the training columns (saved inside the scaler)
        values = {
            "Age": float(data["age"]),
            "BMI": float(data["bmi"]),
            "Systolic_BP": float(data["systolic"]),
            "Diastolic_BP": float(data["diastolic"]),
            "Stress_Level": float(data["stress_score"]),
            "Salt_Intake": float(data["salt_intake"]),
            "Sleep_Duration": float(data["sleep_duration"]),
            "Gender": int(data["gender"]),
            "Smoking_Status": int(data["smoking_status"]),
            "Physical_Activity_Level": int(data["activity_level"]),
            "Family_History": int(data["family_history"]),
            "Diabetes": int(data["diabetes"]),
        }
    except (KeyError, ValueError, TypeError) as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid input: {e}"})

    input_data = pd.DataFrame([values])[list(scaler.feature_names_in_)]
    input_scaled = scaler.transform(input_data)

    prediction = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0][prediction])

    return {
        "stage": label_map.get(prediction, "Unknown"),
        "confidence": round(probability * 100, 1),
    }
