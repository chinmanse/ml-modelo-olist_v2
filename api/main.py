from fastapi import FastAPI, HTTPException
from api.request_schema import DelayRequest
from api.model_loader import MODEL, FEATURE_LIST
import pandas as pd

app = FastAPI(
    title="OLIST Delivery Time Prediction API",
    version="1.0",
    description="API para predecir días de entrega usando el mejor modelo ML entrenado."
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API running"}

@app.post("/predict")
def predict(input_data: DelayRequest):
    try:
        # Convert Pydantic object to DataFrame en el orden correcto
        row = [getattr(input_data, f) for f in FEATURE_LIST]
        df = pd.DataFrame([row], columns=FEATURE_LIST)

        # Predicción
        prediction = MODEL.predict(df)[0]

        return {"delivery_time_days_prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")