import mlflow
import mlflow.pytorch
import torch

def load_model(model_name, model_version):
    mlflow.set_tracking_uri("http://localhost:5000")

    # Cargar el modelo desde el registro
    model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")
    model = model
    return model

model_name = "Alopecia Classifier"
model_version = 1

model = load_model(model_name, model_version)
print(f"Model loaded: {type(model)}")

torch.save(model, "alopecia_classifier.pt")
print("Model saved")