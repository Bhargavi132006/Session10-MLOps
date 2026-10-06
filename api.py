from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X, y = load_iris(return_X_y=True)

TARGET_NAMES = load_iris().target_names.tolist()

model = RandomForestClassifier(
    n_estimators=50,
    random_state=0
)

model.fit(X, y)

app = FastAPI()


class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):

    features = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])

    pred = model.predict(features)[0]

    return {
        "predicted_class": TARGET_NAMES[pred]
    }