from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np


class Prediction(BaseModel):
    feature1: float
    feature2: float
    feature3: float

app = FastAPI()

model = pickle.load("model.pkl")


@app.post('/predict')

def predict(request: Prediction):
    features = np.array([[request.feature1, request.feature2, request.feature3]])

    prediction = model.predict(features)
    
    return {"prediction": prediction}