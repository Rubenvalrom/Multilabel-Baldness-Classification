import mlflow
import mlflow.pytorch

def load_model(device="cpu"):
    mlflow.set_tracking_uri("http://localhost:5000")

    model_name = "Alopecia Classifier"

    # Versión específica
    model_version = 1

    # Cargar el modelo desde el registro
    model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")
    model = model.to(device)
    return model
