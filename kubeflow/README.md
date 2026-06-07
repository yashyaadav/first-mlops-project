# Kubeflow Pipeline

A Kubeflow Pipelines (KFP v2) version of the diabetes training workflow. Three tracked components in series:

1. **train_op** — pulls the CSV, splits 80/20, trains a `RandomForestClassifier`, outputs the Model + test set as KFP artifacts, **and** uploads the model to seaweedfs at `s3://mlpipeline/models/diabetes/latest.pkl`.
2. **evaluate_op** — loads the model + test set, logs `accuracy`, `precision`, `recall`, `f1` as KFP Metrics.
3. **deploy_op** — patches the `diabetes-api` Deployment's `restartedAt` annotation via the in-cluster Kubernetes API, triggering a rolling restart so fresh pods pull the new model on startup. Requires `k8s-pipeline-rbac.yml` to be applied (grants the `pipeline-runner` SA `patch` on the Deployment).

## Prerequisites

- A Kubernetes cluster with Kubeflow Pipelines installed (the project's kind setup works).
- KFP UI reachable (`kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`).

## Compile

Use a **separate** virtual environment from the serving/FastAPI one — the `kfp` SDK pulls in many transitive deps (kubernetes, protobuf, grpc, ...) that you don't want polluting `.mlops`.

From the project root:

```
python3.12 -m venv .kfp
source .kfp/bin/activate
pip install --upgrade pip
pip install -r kubeflow/requirements.txt

python kubeflow/pipeline.py     # produces diabetes_pipeline.yaml

deactivate                       # when done
```

The compiled `diabetes_pipeline.yaml` is gitignored.

## Run

In the KFP UI:

1. **Pipelines → Upload Pipeline**, choose `diabetes_pipeline.yaml`.
2. **Create Run**, leave `data_url` at its default (or override).
3. Open the run to inspect the DAG, artifacts, and metrics.

## Notes

- Each component pins `scikit-learn==1.9.0` to match the version used elsewhere in the project.
- `base_image` is `python:3.12-slim`; components install their deps at runtime via `packages_to_install`. For faster iteration in production you'd bake those into a custom base image.
- `deploy_op` uses the in-cluster Kubernetes client (no `kubectl` binary needed). The pipeline pod inherits its credentials from the `pipeline-runner` ServiceAccount in `kubeflow`; cross-namespace patch permission on the target Deployment comes from the RoleBinding in `k8s-pipeline-rbac.yml`.
