import torch
import os

def load_model(device="cpu"):
    # Ruta relativa al archivo .pt
    model_path = os.path.join(os.path.dirname(__file__), "model", "alopecia_classifier.pt")
    
    # Cargar el modelo
    model = torch.load(model_path, weights_only=False, map_location=device)
    return model
