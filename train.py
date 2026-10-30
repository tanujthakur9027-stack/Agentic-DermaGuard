import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from agents.diagnosis_agent import DiagnosisAgent, diagnosis_transforms

DX_TO_IDX = {
    'mel': 0,    # Melanoma
    'nv': 1,     # Melanocytic nevus
    'bcc': 2,    # Basal cell carcinoma
    'akiec': 3,  # Actinic keratoses
    'bkl': 4,    # Benign keratosis
    'df': 5,     # Dermatofibroma
    'vasc': 6    # Vascular lesion
}

IDX_TO_NAME = {
    0: 'Melanoma',
    1: 'Melanocytic nevus',
    2: 'Basal cell carcinoma',
    3: 'Actinic keratoses',
    4: 'Benign keratosis',
    5: 'Dermatofibroma',
    6: 'Vascular lesion'
}

class SkinLesionDataset(Dataset):
    def __init__(self, metadata_csv: str, images_dir: str, transform=None):
        self.df = pd.read_csv(metadata_csv)
        self.images_dir = images_dir
        self.transform = transform
        
        # Filter to existing images
        self.df['label'] = self.df['dx'].map(DX_TO_IDX)
        self.df['path'] = self.df['image_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
        valid = self.df['path'].apply(os.path.exists)
        self.df = self.df[valid].reset_index(drop=True)
        print(f"[Dataset] Loaded {len(self.df)} images from {metadata_csv}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label

def extract_features(model: DiagnosisAgent, dataloader: DataLoader, device: str = 'cpu'):
    """
    Extract 1280-dim representations using frozen EfficientNet-B0 backbone.
    """
    model.eval()
    model.to(device)
    all_features = []
    all_labels = []

    print("[Training] Extracting features with frozen EfficientNet-B0 backbone...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            # Pass through features and avgpool
            feats = model.backbone.features(images)
            feats = model.backbone.avgpool(feats)
            feats = torch.flatten(feats, 1)
            
            all_features.append(feats.cpu())
            all_labels.append(labels)
            
            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(dataloader):
                elapsed = time.time() - start_time
                processed = min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
                print(f"  Processed {processed}/{len(dataloader.dataset)} images ({elapsed:.1f}s)")
                
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    print(f"[Training] Extracted features shape: {features_tensor.shape} in {time.time() - start_time:.1f}s")
    return features_tensor, labels_tensor

def train_diagnosis_agent(epochs: int = 25, batch_size: int = 64, lr: float = 1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Training] Using device: {device}")
    
    train_metadata = "Health/training/metadata.csv"
    train_images_dir = "Health/training/images"
    
    train_dataset = SkinLesionDataset(train_metadata, train_images_dir, transform=diagnosis_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Calculate inverse class weights
    labels_list = train_dataset.df['label'].values
    counts = np.bincount(labels_list, minlength=7)
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.sum() * 7.0
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[Training] Class counts: {counts.tolist()}")
    print(f"[Training] Class weights: {[round(w, 2) for w in weights.tolist()]}")
    
    # Initialize model
    model = DiagnosisAgent(num_classes=7)
    
    # 1. Feature extraction
    features, labels = extract_features(model, train_loader, device=device)
    
    # 2. Train classifier MLP head
    print("[Training] Optimizing classification head with class-weighted CrossEntropy...")
    classifier = model.backbone.classifier.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset_tensor = torch.utils.data.TensorDataset(features, labels)
    feature_loader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)
    
    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for feats, lbls in feature_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = classifier(feats)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(lbls)
            preds = outputs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += len(lbls)
            
        scheduler.step()
        epoch_loss = total_loss / total
        epoch_acc = correct / total * 100.0
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch [{epoch:2d}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
            
    # Save the complete trained model
    os.makedirs("models", exist_ok=True)
    save_path = "models/diagnosis_agent.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': 7,
        'class_names': IDX_TO_NAME,
        'trained_at': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        'final_train_acc': epoch_acc
    }, save_path)
    print(f"[Success] Saved trained Diagnosis Agent model to {save_path}")

if __name__ == "__main__":
    train_diagnosis_agent(epochs=25, batch_size=64, lr=1e-3)
