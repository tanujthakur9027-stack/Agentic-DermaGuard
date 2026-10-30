import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
import json

class HAM10000Dataset(Dataset):
    """
    HAM10000 Dataset wrapper
    Loads images and applies transformations
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

class DiagnosisAgent(nn.Module):
    """
    Multi-class skin lesion classifier
    Based on EfficientNet-B0 pretrained on ImageNet
    Fine-tuned for 7 HAM10000 lesion classes
    """
    
    def __init__(self, num_classes: int = 7):
        super(DiagnosisAgent, self).__init__()
        self.num_classes = num_classes
        self.lesion_classes = {
            0: 'Melanoma',
            1: 'Melanocytic nevus',
            2: 'Basal cell carcinoma',
            3: 'Actinic keratoses',
            4: 'Benign keratosis',
            5: 'Dermatofibroma',
            6: 'Vascular lesion'
        }
        
        # Load pretrained EfficientNet-B0
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def predict(self, image_tensor: torch.Tensor, device: str = 'cpu') -> Dict:
        """
        Make prediction on single image
        Returns class probabilities and confidence
        """
        self.to(device)
        self.eval()
        
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            logits = self.forward(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()
        
        # Build prediction dictionary
        predictions = {}
        for idx, class_name in self.lesion_classes.items():
            predictions[class_name] = round(probabilities[0, idx].item(), 4)
        
        return {
            'predicted_class': self.lesion_classes[pred_class],
            'predicted_idx': pred_class,
            'confidence': round(confidence, 4),
            'all_probabilities': predictions,
            'top_3': sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def get_triage_recommendation(self, prediction: Dict) -> str:
        """
        Convert diagnostic prediction to triage action
        """
        confidence = prediction['confidence']
        predicted_class = prediction['predicted_class']
        
        # High risk classes
        high_risk = ['Melanoma', 'Basal cell carcinoma', 'Actinic keratoses']
        
        # Benign classes
        benign = ['Dermatofibroma', 'Vascular lesion']
        
        if confidence < 0.6:
            return "REFER"  # High uncertainty
        elif predicted_class in high_risk:
            return "REFER"  # Suspected malignancy
        elif predicted_class in benign and confidence > 0.75:
            return "AUTO_TRIAGE"  # Confident benign
        else:
            return "REFER"  # Uncertain

# Get device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Image preprocessing
diagnosis_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load or initialize model
def load_diagnosis_agent(model_path: str = 'models/diagnosis_agent.pt') -> DiagnosisAgent:
    """
    Load trained model or initialize new one
    """
    model = DiagnosisAgent(num_classes=7)
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[OK] Loaded diagnosis agent from {model_path}")
    else:
        print("[INFO] No trained model found. Using pretrained EfficientNet-B0")
    
    return model.to(DEVICE)

# Singleton instance
diagnosis_agent = load_diagnosis_agent()
