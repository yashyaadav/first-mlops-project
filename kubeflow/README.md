# Kubeflow Pipeline

A Kubeflow Pipelines (KFP v2) version of the diabetes training workflow. Mirrors `train.py` as two tracked components:

1. **train_op** — pulls the CSV, splits 80/20, trains a `RandomForestClassifier`, outputs a Model artifact and the held-out test set as a Dataset artifact.
2. **evaluate_op** — loads the model and test set, logs `accuracy`, `precision`, `recall`, and `f1` as KFP Metrics.

Serving (FastAPI + Docker + the k8s manifests in the project root) is unchanged — this folder only replaces the *training* step.

## Prerequisites

- A Kubernetes cluster with Kubeflow Pipelines installed (the project's kind setup works).
- KFP UI reachable (`kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`).

## Compile

```
pip install -r requirements.txt
python pipeline.py
```

Produces `diabetes_pipeline.yaml` (gitignored).

## Run

In the KFP UI:

1. **Pipelines → Upload Pipeline**, choose `diabetes_pipeline.yaml`.
2. **Create Run**, leave `data_url` at its default (or override).
3. Open the run to inspect the DAG, artifacts, and metrics.

## Notes

- Each component pins `scikit-learn==1.9.0` to match the version used elsewhere in the project.
- `base_image` is `python:3.10-slim`; the component installs its deps at runtime via `packages_to_install`. For faster iteration in production you'd bake those into a custom base image.
