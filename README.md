# 🩺 Diabetes Prediction Model – MLOps Project (FastAPI + Docker + K8s)

An end-to-end MLOps project I built to learn the workflow of training, serving, containerizing, and deploying a machine learning model. It predicts whether a person is diabetic based on health metrics, and walks through:

- ✅ Model Training
- ✅ Building the Model locally
- ✅ API Deployment with FastAPI
- ✅ Dockerization
- ✅ Kubernetes Deployment

---

## 📊 Problem Statement

Predict if a person is diabetic based on:
- Pregnancies
- Glucose
- Blood Pressure
- BMI
- Age

We use a Random Forest Classifier trained on the **Pima Indians Diabetes Dataset**.

---

## 🚀 Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/yashyaadav/first-mlops-project.git
cd first-mlops-project
```

### 2. Create Virtual Environment

```
python3 -m venv .mlops
source .mlops/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

## Train the Model

```
python train.py
```

## Run the API Locally

```
uvicorn main:app --reload
```

### Sample Input for /predict

```
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "BMI": 28.5,
  "Age": 45
}
```

## Dockerize the API

### Build the Docker Image

```
docker build -t diabetes-prediction-model .
```

### Run the Container

```
docker run -p 8000:8000 diabetes-prediction-model
```

## Deploy to Kubernetes

For a real cluster (image pushed to a registry, LoadBalancer service):

```
kubectl apply -f k8s-deploy.yml
```

### Try it locally on a kind cluster

`k8s-deploy-kind.yml` uses the locally-built image and a NodePort service so you can run it without a registry or cloud LB. Spin up a cluster, load the image into it, and apply:

```
kind create cluster
kind load docker-image diabetes-prediction-model:latest
kubectl apply -f k8s-deploy-kind.yml
```

Forward the service to your host and hit it at http://localhost:8000:

```
kubectl port-forward svc/diabetes-api-service 8000:80
```

Tear down when you're done:

```
kind delete cluster
```

---

## 🙌 Credits

This project is based on the **"Build Your First MLOps Project"** tutorial by Abhishek Veeramalla. All credit for the original walkthrough goes to him — check out his YouTube channel `Abhishek.Veeramalla` for great DevOps + MLOps content.

I worked through it as my first hands-on MLOps project to learn the model → API → container → Kubernetes pipeline end-to-end.

