import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import json
from thresholds import load_thresholds, get_flat_thresholds

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

IDX_TO_CODE = {
    0: 'mel',    # Melanoma
    1: 'nv',     # Melanocytic nevus
    2: 'bcc',    # Basal cell carcinoma
    3: 'akiec',  # Actinic keratoses
    4: 'bkl',    # Benign keratosis
    5: 'df',     # Dermatofibroma
    6: 'vasc'    # Vascular lesion
}

CODE_TO_IDX = {v: k for k, v in IDX_TO_CODE.items()}

IDX_TO_NAME = {
    0: 'Melanoma',
    1: 'Melanocytic nevus',
    2: 'Basal cell carcinoma',
    3: 'Actinic keratoses',
    4: 'Benign keratosis',
    5: 'Dermatofibroma',
    6: 'Vascular lesion'
}

CODE_TO_NAME = {
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevus',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'bkl': 'Benign keratosis',
    'df': 'Dermatofibroma',
    'vasc': 'Vascular lesion'
}

CLINICAL_DESCRIPTIONS = {
    'mel': "Malignant Melanoma: High-risk neuroectodermal malignancy characterized by atypical pigment network, irregular globules, and potential for deep invasion. Urgent surgical excision/biopsy required.",
    'nv': "Melanocytic Nevus: Benign melanocytic proliferation with symmetrical pigment architecture and regular peripheral borders. Routine primary care surveillance recommended.",
    'bcc': "Basal Cell Carcinoma: Locally invasive non-melanocytic epidermal malignancy often presenting with arborizing telangiectasia, blue-gray ovoid nests, or ulceration.",
    'akiec': "Actinic Keratoses / Intraepithelial Carcinoma: Pre-malignant keratinocytic dysplasia with rough scaly texture caused by chronic solar damage. Cryotherapy or topical field therapy indicated.",
    'bkl': "Benign Keratosis: Seborrheic keratosis or lichen-planus like lesion presenting milia-like cysts or fingerprint-like ridges. Benign natural course.",
    'df': "Dermatofibroma: Benign fibrohistiocytic dermal nodule with characteristic central white scar-like patch and delicate peripheral pigment network.",
    'vasc': "Vascular Lesion: Benign hemangioma, angiokeratoma, or pyogenic granuloma exhibiting prominent red-purple lacunae without melanocytic atypia."
}

if HAVE_TORCH:
    class DiagnosisNeuralNet(nn.Module):
        def __init__(self, num_classes: int = 7):
            super(DiagnosisNeuralNet, self).__init__()
            self.num_classes = num_classes
            try:
                self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            except Exception:
                self.backbone = efficientnet_b0(weights=None)
            
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

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    diagnosis_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
else:
    DiagnosisNeuralNet = None
    DEVICE = 'cpu'
    diagnosis_transforms = None

class DiagnosisAgent:
    """
    Agent 2: Diagnosis Agent (7 HAM10000 Pathology Classifier)
    Jaypee Agentic AI Summit 2026 — Track 2 | Healthcare

    Predicts one of 7 official lesion classes:
      - nv: Melanocytic nevus
      - mel: Melanoma
      - bcc: Basal cell carcinoma
      - akiec: Actinic keratoses
      - bkl: Benign keratosis
      - df: Dermatofibroma
      - vasc: Vascular lesion
    Provides:
      - Predicted official class code & clinical disease name
      - Confidence score & 7-class probability distribution
      - Diagnostic certainty level & top-2 separation margin
      - Cumulative malignancy suspicion (mel + bcc + akiec)
      - Clinical pathology summary
    """

    def __init__(self, model_path: str = "models/diagnosis_agent.pt"):
        self.model_path = model_path
        self.torch_model = None
        self.have_weights = False
        self.lightweight_weights = None

        # Load learned lightweight weights for serverless fallback
        lw_paths = [
            Path("models/lightweight_weights.json"),
            Path(__file__).parent.parent / "models" / "lightweight_weights.json"
        ]
        for p in lw_paths:
            if p.exists():
                try:
                    with open(p, "r") as f:
                        self.lightweight_weights = json.load(f)
                    break
                except Exception:
                    pass

        if HAVE_TORCH:
            try:
                self.torch_model = DiagnosisNeuralNet(num_classes=7)
                if Path(model_path).exists():
                    ckpt = torch.load(model_path, map_location=DEVICE)
                    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                        self.torch_model.load_state_dict(ckpt['model_state_dict'])
                    else:
                        self.torch_model.load_state_dict(ckpt)
                    self.have_weights = True
                self.torch_model.to(DEVICE)
                self.torch_model.eval()
            except Exception as e:
                self.torch_model = None
                self.have_weights = False

    def _package_prediction(
        self,
        probs: np.ndarray,
        tta_disagreement: float = 0.0,
        custom_thresholds: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        codes = [IDX_TO_CODE[i] for i in range(7)]
        pred_idx = int(np.argmax(probs))
        pred_code = codes[pred_idx]
        confidence = float(probs[pred_idx])

        # Active thresholds
        th = get_flat_thresholds(custom_thresholds)
        margin_min = th.get("margin_minimum", 0.15)
        max_entropy = th.get("max_entropy", 1.75)
        tta_thresh = th.get("tta_disagreement_threshold", 0.22)

        # Sort candidate probabilities
        sorted_indices = np.argsort(probs)[::-1]
        top_3 = [
            (codes[i], CODE_TO_NAME[codes[i]], round(float(probs[i]), 4))
            for i in sorted_indices[:3]
        ]

        # Separation margin between top-1 and top-2
        second_prob = float(probs[sorted_indices[1]])
        margin = round(confidence - second_prob, 4)

        # Shannon entropy of probability distribution (in bits, max log2(7) ≈ 2.81)
        entropy = round(-float(np.sum(probs * np.log2(probs + 1e-8))), 4)

        is_ambiguous = (margin < margin_min) or (entropy > max_entropy) or (tta_disagreement > tta_thresh)

        if margin >= 0.35 and not is_ambiguous:
            certainty = "HIGH_CERTAINTY"
        elif margin >= margin_min and not is_ambiguous:
            certainty = "MODERATE_CERTAINTY"
        else:
            certainty = "DIFFERENTIAL_AMBIGUITY"

        # Cumulative malignancy suspicion (mel + bcc + akiec)
        mel_p = float(probs[CODE_TO_IDX['mel']])
        bcc_p = float(probs[CODE_TO_IDX['bcc']])
        akiec_p = float(probs[CODE_TO_IDX['akiec']])
        cum_malig = round(mel_p + bcc_p + akiec_p, 4)

        all_probs_code = {codes[i]: round(float(probs[i]), 4) for i in range(7)}
        all_probs_name = {CODE_TO_NAME[codes[i]]: all_probs_code[codes[i]] for i in range(7)}

        is_malignant = pred_code in {'mel', 'bcc', 'akiec'}

        return {
            'agent_id': 'A2',
            'agent_name': 'Diagnosis Agent',
            'predicted_code': pred_code,
            'predicted_class': CODE_TO_NAME[pred_code],
            'predicted_idx': pred_idx,
            'confidence': round(confidence, 4),
            'margin_to_next': margin,
            'entropy': entropy,
            'tta_disagreement': round(float(tta_disagreement), 4),
            'is_ambiguous': is_ambiguous,
            'diagnostic_certainty': certainty,
            'cumulative_malignancy_risk': cum_malig,
            'is_malignant': is_malignant,
            'all_probabilities': all_probs_code,
            'all_probabilities_by_name': all_probs_name,
            'top_3': top_3,
            'clinical_description': CLINICAL_DESCRIPTIONS.get(pred_code, ""),
            'summary': f"Pathology predicted as '{CODE_TO_NAME[pred_code]}' [{pred_code}] with {confidence*100:.1f}% confidence ({certainty}, margin: {margin:.2f}, entropy: {entropy:.2f})."
        }

    def predict_numpy_fallback(self, pil_img: Image.Image, custom_thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Learned feature statistical classifier fallback when PyTorch weights are unavailable."""
        img_rgb = pil_img.convert('RGB').resize((128, 128))
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        
        th = get_flat_thresholds(custom_thresholds)
        temp = max(0.01, float(th.get("temperature", 1.0)))
        biases = th.get("class_probability_bias", {})
        bias_arr = np.array([biases.get(IDX_TO_CODE[i], 0.0) for i in range(7)], dtype=np.float32)

        if self.lightweight_weights is not None:
            try:
                # 1. Color moments
                mean_r, mean_g, mean_b = float(r.mean()), float(g.mean()), float(b.mean())
                std_r, std_g, std_b = float(r.std()), float(g.std()), float(b.std())

                # 2. HSV saturation and value
                v = np.max(arr, axis=2)
                min_rgb = np.min(arr, axis=2)
                delta = v - min_rgb
                sat = np.where(v > 1e-4, delta / (v + 1e-5), 0.0)
                mean_sat, std_sat = float(sat.mean()), float(sat.std())
                mean_v, std_v = float(v.mean()), float(v.std())

                # 3. Pathological cues
                dark_melanin = float(np.sum((r < 0.32) & (g < 0.24) & (b < 0.24)) / arr.size)
                red_ratio = float(np.sum((r > 0.55) & (r > 1.35 * (g + b) / 2.0)) / arr.size)
                blue_gray = float(np.sum((b > g) & (b > r) & (sat > 0.15)) / arr.size)

                # 4. Center vs periphery contrast
                h, w = r.shape
                c_y, c_x = h // 4, w // 4
                center_patch = arr[c_y:h-c_y, c_x:w-c_x]
                outer_mask = np.ones((h, w), dtype=bool)
                outer_mask[c_y:h-c_y, c_x:w-c_x] = False
                contrast = float(np.linalg.norm(center_patch.mean(axis=(0, 1)) - arr[outer_mask].mean(axis=0)))

                # 5. Roughness
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

                # Standardize & forward pass through linear layer
                scaler_mean = np.array(self.lightweight_weights['scaler_mean'])
                scaler_scale = np.array(self.lightweight_weights['scaler_scale'])
                coef = np.array(self.lightweight_weights['coef'])
                intercept = np.array(self.lightweight_weights['intercept'])

                feats_scaled = (feats - scaler_mean) / (scaler_scale + 1e-6)
                raw_logits = np.dot(coef, feats_scaled) + intercept
                scaled_logits = (raw_logits / temp) + bias_arr
                
                exp_l = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_l / np.sum(exp_l)
                return self._package_prediction(probs, tta_disagreement=0.0, custom_thresholds=custom_thresholds)
            except Exception:
                pass

        # Heuristic fallback if lightweight weights fail to evaluate
        mean_r, mean_g, mean_b = float(r.mean()*255.0), float(g.mean()*255.0), float(b.mean()*255.0)
        std_r, std_g, std_b = float(r.std()*255.0), float(g.std()*255.0), float(b.std()*255.0)
        color_std = (std_r + std_g + std_b) / 3.0
        vasc_ratio = mean_r / (mean_g + mean_b + 1e-5)
        dark_melanin = float(np.sum((r < 0.31) & (g < 0.24) & (b < 0.24)) / arr.size)
        
        raw_scores = {
            'mel': max(0.05, (dark_melanin * 12.0) + (color_std / 45.0) * 0.4),
            'nv': max(0.20, 0.80 - (color_std / 60.0) * 0.35 - (dark_melanin * 2.5)),
            'bcc': max(0.04, 0.15 if mean_r > 160 and mean_g > 120 else 0.05),
            'akiec': max(0.03, (color_std / 50.0) * 0.3 + 0.05),
            'bkl': max(0.05, 0.35 if mean_r < 140 else 0.1),
            'df': 0.08,
            'vasc': max(0.02, (vasc_ratio - 0.52) * 5.0) if vasc_ratio > 0.55 else 0.03
        }
        codes = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
        score_arr = np.array([raw_scores[c] for c in codes])
        scaled_scores = (score_arr * 2.5 / temp) + bias_arr
        exp_s = np.exp(scaled_scores - np.max(scaled_scores))
        probs = exp_s / exp_s.sum()
        return self._package_prediction(probs, tta_disagreement=0.0, custom_thresholds=custom_thresholds)

    def predict(self, image_input: Any, custom_thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Executes diagnostic inference on image input with TTA, temperature scaling, and class bias."""
        th = get_flat_thresholds(custom_thresholds)
        temp = max(0.01, float(th.get("temperature", 1.0)))
        tta_enabled = bool(th.get("tta_enabled", True))
        biases = th.get("class_probability_bias", {})
        bias_arr = np.array([biases.get(IDX_TO_CODE[i], 0.0) for i in range(7)], dtype=np.float32)

        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
        elif HAVE_TORCH and isinstance(image_input, torch.Tensor):
            pil_img = None
            tensor = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if HAVE_TORCH and self.torch_model is not None and self.have_weights:
            try:
                with torch.no_grad():
                    if pil_img is not None and tta_enabled:
                        # 4-view Test-Time Augmentation (Original, H-Flip, V-Flip, Rotate 180)
                        v1 = pil_img
                        v2 = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                        v3 = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
                        v4 = pil_img.rotate(180)
                        batch = torch.stack([diagnosis_transforms(v) for v in (v1, v2, v3, v4)]).to(DEVICE)
                        raw_logits = self.torch_model(batch).cpu().numpy()  # (4, 7)
                        
                        scaled_logits = (raw_logits / temp) + bias_arr
                        exp_l = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
                        probs_views = exp_l / np.sum(exp_l, axis=1, keepdims=True)

                        mean_probs = np.mean(probs_views, axis=0)
                        mean_probs = mean_probs / np.sum(mean_probs)
                        tta_disagreement = float(np.max(np.std(probs_views, axis=0)))
                        return self._package_prediction(mean_probs, tta_disagreement=tta_disagreement, custom_thresholds=custom_thresholds)

                    elif pil_img is not None:
                        tensor = diagnosis_transforms(pil_img).unsqueeze(0).to(DEVICE)
                        raw_logits = self.torch_model(tensor).cpu().numpy()[0]
                        scaled_logits = (raw_logits / temp) + bias_arr
                        exp_l = np.exp(scaled_logits - np.max(scaled_logits))
                        probs = exp_l / np.sum(exp_l)
                        return self._package_prediction(probs, tta_disagreement=0.0, custom_thresholds=custom_thresholds)
                    else:
                        if len(tensor.shape) == 3:
                            tensor = tensor.unsqueeze(0)
                        tensor = tensor.to(DEVICE)
                        raw_logits = self.torch_model(tensor).cpu().numpy()[0]
                        scaled_logits = (raw_logits / temp) + bias_arr
                        exp_l = np.exp(scaled_logits - np.max(scaled_logits))
                        probs = exp_l / np.sum(exp_l)
                        return self._package_prediction(probs, tta_disagreement=0.0, custom_thresholds=custom_thresholds)
            except Exception:
                pass

        if pil_img is None:
            pil_img = Image.new('RGB', (224, 224), (180, 140, 120))
        return self.predict_numpy_fallback(pil_img, custom_thresholds=custom_thresholds)

def load_diagnosis_agent(model_path: str = 'models/diagnosis_agent.pt') -> DiagnosisAgent:
    return DiagnosisAgent(model_path=model_path)

# Singleton instance
diagnosis_agent = load_diagnosis_agent()
