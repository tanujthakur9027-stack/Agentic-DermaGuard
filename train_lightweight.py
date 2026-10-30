import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

DX_TO_IDX = {'mel': 0, 'nv': 1, 'bcc': 2, 'akiec': 3, 'bkl': 4, 'df': 5, 'vasc': 6}

def extract_features(img: Image.Image) -> np.ndarray:
    img_rgb = img.convert('RGB').resize((128, 128))
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # 1. Color moments (mean & std)
    mean_r, mean_g, mean_b = float(r.mean()), float(g.mean()), float(b.mean())
    std_r, std_g, std_b = float(r.std()), float(g.std()), float(b.std())
    
    # 2. HSV color representation
    v = np.max(arr, axis=2)
    min_rgb = np.min(arr, axis=2)
    delta = v - min_rgb
    sat = np.where(v > 1e-4, delta / (v + 1e-5), 0.0)
    mean_sat, std_sat = float(sat.mean()), float(sat.std())
    mean_v, std_v = float(v.mean()), float(v.std())
    
    # 3. Pathological color cues
    # Dark melanin (black/dark brown)
    dark_melanin = float(np.sum((r < 0.32) & (g < 0.24) & (b < 0.24)) / arr.size)
    # Erythema / red vascularity
    red_ratio = float(np.sum((r > 0.55) & (r > 1.35 * (g + b) / 2.0)) / arr.size)
    # Blue-gray veil
    blue_gray = float(np.sum((b > g) & (b > r) & (sat > 0.15)) / arr.size)
    
    # 4. Central lesion vs periphery contrast
    h, w = r.shape
    c_y, c_x = h // 4, w // 4
    center_patch = arr[c_y:h-c_y, c_x:w-c_x]
    outer_mask = np.ones((h, w), dtype=bool)
    outer_mask[c_y:h-c_y, c_x:w-c_x] = False
    
    center_mean = center_patch.mean(axis=(0, 1))
    outer_mean = arr[outer_mask].mean(axis=0)
    contrast = float(np.linalg.norm(center_mean - outer_mean))
    
    # 5. Spatial roughness (microtexture)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    roughness = float(np.mean(np.abs(gray[1:, :] - gray[:-1, :])) + np.mean(np.abs(gray[:, 1:] - gray[:, :-1])))

    feats = np.array([
        mean_r, mean_g, mean_b,
        std_r, std_g, std_b,
        mean_sat, std_sat,
        mean_v, std_v,
        dark_melanin, red_ratio, blue_gray,
        contrast, roughness
    ], dtype=np.float32)
    return feats

def train_and_export():
    meta_path = "Health/training/metadata.csv"
    img_dir = "Health/training/images"
    if not os.path.exists(meta_path) or not os.path.exists(img_dir):
        print("Training files missing.")
        return

    df = pd.read_csv(meta_path)
    sample_df = pd.concat([df[df['dx'] == c].sample(min(len(df[df['dx'] == c]), 350), random_state=42) for c in DX_TO_IDX.keys()]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(sample_df)} balanced training images across 7 classes.")

    X = []
    y = []
    for row in sample_df.to_dict('records'):
        p = os.path.join(img_dir, f"{row['image_id']}.jpg")
        if not os.path.exists(p):
            continue
        try:
            im = Image.open(p)
            feats = extract_features(im)
            X.append(feats)
            y.append(DX_TO_IDX[row['dx']])
        except Exception as ex:
            pass

    X = np.array(X)
    y = np.array(y)
    print(f"Extracted features for {len(X)} samples. Shape: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=42)
    clf.fit(X_scaled, y)
    
    preds = clf.predict(X_scaled)
    acc = accuracy_score(y, preds)
    print(f"Training accuracy on feature classifier: {acc*100:.2f}%")

    weights_dict = {
        "coef": clf.coef_.tolist(),        # (7, 15)
        "intercept": clf.intercept_.tolist(),  # (7,)
        "scaler_mean": scaler.mean_.tolist(),  # (15,)
        "scaler_scale": scaler.scale_.tolist(),# (15,)
        "classes": ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
    }

    os.makedirs("models", exist_ok=True)
    out_path = "models/lightweight_weights.json"
    with open(out_path, "w") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"Saved lightweight weights to {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == '__main__':
    train_and_export()
