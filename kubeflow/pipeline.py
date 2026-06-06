"""Kubeflow Pipelines version of the diabetes training workflow.

Mirrors train.py as a KFP v2 pipeline so each run is tracked in the KFP
UI with versioned inputs, outputs, and metrics. After training, the
model is also uploaded to seaweedfs (S3-compatible) under a known key
so the FastAPI serving pods can pull it on rollout — closing the
train -> serve loop.

Compile:
    python pipeline.py
Then upload diabetes_pipeline.yaml in the KFP UI and create a run.
"""
from kfp import dsl, compiler


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "scikit-learn==1.9.0",
        "pandas==2.2.3",
        "joblib==1.4.2",
        "boto3==1.35.0",
    ],
)
def train_op(
    data_url: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_key: str,
    s3_access_key: str,
    s3_secret_key: str,
    model: dsl.Output[dsl.Model],
    test_set: dsl.Output[dsl.Dataset],
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import boto3

    df = pd.read_csv(data_url)
    features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
    X, y = df[features], df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    joblib.dump(clf, model.path)
    pd.concat([X_test, y_test], axis=1).to_csv(test_set.path, index=False)

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )
    s3.upload_file(model.path, s3_bucket, s3_key)
    print(f"✅ Uploaded model to s3://{s3_bucket}/{s3_key}")


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "scikit-learn==1.9.0",
        "pandas==2.2.3",
        "joblib==1.4.2",
    ],
)
def evaluate_op(
    model: dsl.Input[dsl.Model],
    test_set: dsl.Input[dsl.Dataset],
    metrics: dsl.Output[dsl.Metrics],
):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    clf = joblib.load(model.path)
    df = pd.read_csv(test_set.path)
    X_test = df.drop(columns=["Outcome"])
    y_test = df["Outcome"]
    y_pred = clf.predict(X_test)

    metrics.log_metric("accuracy", float(accuracy_score(y_test, y_pred)))
    metrics.log_metric("precision", float(precision_score(y_test, y_pred)))
    metrics.log_metric("recall", float(recall_score(y_test, y_pred)))
    metrics.log_metric("f1", float(f1_score(y_test, y_pred)))


@dsl.pipeline(
    name="diabetes-prediction-pipeline",
    description="Train, evaluate, and publish a RandomForest diabetes classifier to seaweedfs.",
)
def diabetes_pipeline(
    data_url: str = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    s3_endpoint: str = "http://seaweedfs.kubeflow:9000",
    s3_bucket: str = "mlpipeline",
    s3_key: str = "models/diabetes/latest.pkl",
    s3_access_key: str = "minio",
    s3_secret_key: str = "minio123",
):
    train_task = train_op(
        data_url=data_url,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_key=s3_key,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    evaluate_op(
        model=train_task.outputs["model"],
        test_set=train_task.outputs["test_set"],
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=diabetes_pipeline,
        package_path="diabetes_pipeline.yaml",
    )
    print("✅ Compiled to diabetes_pipeline.yaml")
