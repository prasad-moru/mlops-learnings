# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

model = mlflow.pyfunc.load_model("models:/iris_classifier/Production")

@app.post("/predict")
def predict(features: IrisFeatures):
    data = [[features.sepal_length, features.sepal_width, 
             features.petal_length, features.petal_width]]
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}