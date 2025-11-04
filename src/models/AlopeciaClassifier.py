import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import maxvit_t
import kornia.augmentation as K
import numpy as np
from pathlib import Path

class AlopeciaClassifier(nn.Module):
    def __init__(self):
        super(AlopeciaClassifier, self).__init__()

        # Load the pre-trained MaxVit-T model
        weights = torch.load(f"{Path(__file__).parent}/state_dicts/maxvit_t_model_state_dict.pth")
        self.model = maxvit_t()
        self.model.classifier[-1] = nn.Sequential(
            nn.Linear(self.model.classifier[-1].in_features, 1),
            nn.Sigmoid()
        )

        self.model.load_state_dict(weights)

        self.transform = nn.Sequential(
            K.Resize((224, 224)),
            K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        )
        
    def forward(self, x):
        x = self.transform(x)
        x = self.model(x)
        x = x * 6
        x = torch.round(x).squeeze().long()
        return x.cpu().numpy()
    
