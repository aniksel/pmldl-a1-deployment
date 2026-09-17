# model API: takes district values and returns the predicted price

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from features import CATEGORICAL, MODEL_NUMERIC, add_features

model = joblib.load("model.pkl")

app = FastAPI(title="California Housing API")


class District(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(district: District):
    df = add_features(pd.DataFrame([district.model_dump()]))
    price = model.predict(df[MODEL_NUMERIC + CATEGORICAL])[0]
    return {"predicted_price": round(float(price), 2)}
