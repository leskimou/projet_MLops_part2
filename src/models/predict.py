import mlflow
logged_model = 'runs:/aa78c59a72cc444fa2a65eb3c8cdc9de/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)