import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5001")

exp = mlflow.get_experiment_by_name("movielens_knn")
runs = mlflow.search_runs(exp.experiment_id)

runs.to_csv("evaluations/mlflow_results.csv", index=False)

print("Saved results!")