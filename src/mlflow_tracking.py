import mlflow

def init_experiment():
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("movielens_knn")

def log_params(k):
    mlflow.log_param("k", k)

def log_metrics(rmse, mae):
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)