import pickle
import pandas as pd
from pydantic import BaseModel

from fastapi import FastAPI
app = FastAPI()

with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

class DiamondInput(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

@app.get("/")
def home():
    return {"message": "Diamond Price Prediction API is running"}

@app.post("/predict")
def predict(data: DiamondInput):
    try:
        df = pd.DataFrame([data.model_dump()])

        transformed_data = preprocessor.transform(df)

        prediction = model.predict(transformed_data)

        return {
            "prediction": float(prediction[0])
        }
    except Exception as e:
        return {"error": str(e)}
