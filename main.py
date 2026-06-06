# main.py
import os
from urllib.parse import urlparse

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


def load_model():
    """Load the model from S3 if MODEL_S3_URI is set, else from the local pkl.

    The S3 path is used in-cluster (the FastAPI deployment sets MODEL_S3_URI
    to the model produced by the Kubeflow pipeline). The local path is the
    fallback for `docker run` and local `uvicorn` development.
    """
    s3_uri = os.getenv("MODEL_S3_URI")
    if not s3_uri:
        return joblib.load("diabetes_model.pkl")

    import boto3

    parsed = urlparse(s3_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    local_path = "/tmp/diabetes_model.pkl"

    boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    ).download_file(bucket, key, local_path)

    return joblib.load(local_path)


app = FastAPI()
model = load_model()


class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int


@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is live"}


@app.post("/predict")
def predict(data: DiabetesInput):
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    prediction = model.predict(input_data)[0]
    return {"diabetic": bool(prediction)}
