import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import zipfile

# Create project structure
PROJECT_DIRS = [
    'data/train',
    'data/val',
    'models',
    'outputs',
    'logs'
]

for directory in PROJECT_DIRS:
    Path(directory).mkdir(parents=True, exist_ok=True)

print("✅ Project directories created")

# HAM10000 metadata - 7 lesion classes
LESION_CLASSES = {
    0: 'Melanoma',
    1: 'Melanocytic nevus',
    2: 'Basal cell carcinoma',
    3: 'Actinic keratoses',
    4: 'Benign keratosis',
    5: 'Dermatofibroma',
    6: 'Vascular lesion'
}

# Save metadata
with open('config.json', 'w') as f:
    json.dump({
        'lesion_classes': LESION_CLASSES,
        'train_samples': 7815,
        'val_samples': 2000,
        'image_size': 224,
        'agents': ['Image Quality Agent', 'Diagnosis Agent']
    }, f, indent=2)

print("✅ Configuration saved")
