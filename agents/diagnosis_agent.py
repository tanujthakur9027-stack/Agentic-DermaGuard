import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import json
import time
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
    'mel': "Melanoma: A serious type of skin cancer originating in pigment cells. Early detection and prompt dermatologist evaluation or biopsy are essential.",
    'nv': "Normal Mole (Nevus): A very common, harmless growth of pigment cells. Typical moles have uniform color and smooth borders, needing only routine self-checks.",
    'bcc': "Basal Cell Carcinoma: The most common type of skin cancer. It usually grows slowly, rarely spreads to other organs, and is highly treatable when checked by a doctor.",
    'akiec': "Actinic Keratosis / Sun Spot: A rough, scaly precancerous patch caused by long-term sun exposure. Early treatment prevents progression to skin cancer.",
    'bkl': "Benign Keratosis (Age/Wisdom Spot): A harmless, non-cancerous skin growth (like a seborrheic keratosis) that commonly appears with age. No treatment needed.",
    'df': "Dermatofibroma: A harmless, firm bump under the skin that often feels like a small button. Completely benign and safe.",
    'vasc': "Vascular Lesion (Blood Vessel Spot): A harmless cluster of small blood vessels (such as a cherry angioma). Non-cancerous and benign."
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
      - Explaining Agent contract with full evidence, margins, and explanation
      - Predicted official class code & clinical disease name
      - Confidence score & 7-class probability distribution
      - Diagnostic certainty level & top-2 separation margin
      - Cumulative malignancy suspicion (mel + bcc + akiec)
    """

    def __init__(self, model_path: str = "models/diagnosis_agent.pt"):
        self.model_path = model_path
        self.torch_model = None
        self.have_weights = False
        self.lightweight_weights = None

        # Load learned lightweight weights for serverless fallback
        base_dir = Path(__file__).resolve().parent
        lw_paths = [
            base_dir.parent / "models" / "lightweight_weights.json",
            base_dir / "models" / "lightweight_weights.json",
            Path.cwd() / "models" / "lightweight_weights.json",
            Path("/var/task/models/lightweight_weights.json"),
            Path("/var/task/api/models/lightweight_weights.json")
        ]
        for p in lw_paths:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
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
        custom_thresholds: Optional[Dict[str, Any]] = None,
        runtime_ms: float = 0.0
    ) -> Dict[str, Any]:
        codes = [IDX_TO_CODE[i] for i in range(7)]
        pred_idx = int(np.argmax(probs))
        pred_code = codes[pred_idx]
        confidence = float(probs[pred_idx])

        # Active thresholds
        th = get_flat_thresholds(custom_thresholds)
        auto_conf_th = float(th.get("auto_triage_confidence", 0.80))
        margin_min = float(th.get("top2_margin_min", th.get("margin_minimum", 0.10)))
        max_entropy = float(th.get("max_entropy", 1.65))
        malig_thresh = float(th.get("malignant_prob_threshold", 0.22))
        mel_thresh = float(th.get("mel_prob_threshold", th.get("mel_threshold", 0.18)))
        tta_thresh = float(th.get("tta_disagreement_threshold", 0.22))

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

        flags = []
        if is_malignant:
            flags.append("MALIGNANCY_PREDICTED")
        if is_ambiguous:
            flags.append("DIFFERENTIAL_AMBIGUITY")
        if confidence >= auto_conf_th and not is_malignant:
            flags.append("HIGH_CONFIDENCE_BENIGN")
        if tta_disagreement > tta_thresh:
            flags.append("TTA_INSTABILITY")

        # Friendly certainty label
        if certainty == "HIGH_CERTAINTY":
            certainty_phrase = "high diagnostic clarity"
        elif certainty == "MODERATE_CERTAINTY":
            certainty_phrase = "moderate diagnostic clarity"
        else:
            certainty_phrase = "borderline ambiguity between multiple possibilities"

        # Formal Explaining Agent contract evidence list
        evidence = [
            {
                "feature": "auto_triage_confidence",
                "value": round(confidence, 4),
                "threshold": auto_conf_th,
                "direction": ">=",
                "passed": confidence >= auto_conf_th,
                "note": f"Primary confidence is {confidence*100:.1f}% (target for safe automated clearance is {auto_conf_th*100:.0f}%)"
            },
            {
                "feature": "top2_margin",
                "value": margin,
                "threshold": margin_min,
                "direction": ">=",
                "passed": margin >= margin_min,
                "note": f"Lead over the runner-up diagnosis is {margin*100:.1f}% (gap between '{pred_code}' and '{codes[sorted_indices[1]]}', minimum target: {margin_min*100:.1f}%)"
            },
            {
                "feature": "shannon_entropy",
                "value": entropy,
                "threshold": max_entropy,
                "direction": "<=",
                "passed": entropy <= max_entropy,
                "note": f"Diagnostic clarity score is {entropy:.2f} bits (lower means more decisive; upper limit: {max_entropy:.2f} bits)"
            },
            {
                "feature": "cumulative_malignancy_risk",
                "value": cum_malig,
                "threshold": malig_thresh,
                "direction": "<=",
                "passed": cum_malig <= malig_thresh,
                "note": f"Combined cancer risk (Melanoma + Basal Cell + Pre-cancer) is {cum_malig*100:.1f}% (safety ceiling: {malig_thresh*100:.1f}%)"
            },
            {
                "feature": "melanoma_risk",
                "value": round(mel_p, 4),
                "threshold": mel_thresh,
                "direction": "<=",
                "passed": mel_p <= mel_thresh,
                "note": f"Melanoma-specific probability is {mel_p*100:.1f}% (safety ceiling: {mel_thresh*100:.1f}%)"
            }
        ]

        thresholds_used = {
            "auto_triage_confidence": auto_conf_th,
            "top2_margin_min": margin_min,
            "max_entropy": max_entropy,
            "malignant_prob_threshold": malig_thresh,
            "mel_prob_threshold": mel_thresh
        }

        margin_to_threshold = {
            "auto_triage_confidence": round(confidence - auto_conf_th, 4),
            "top2_margin_min": round(margin - margin_min, 4),
            "max_entropy": round(max_entropy - entropy, 4),
            "malignant_prob_threshold": round(malig_thresh - cum_malig, 4),
            "mel_prob_threshold": round(mel_thresh - mel_p, 4)
        }

        summary = f"The AI model predicts {CODE_TO_NAME[pred_code]} ({pred_code}) with {confidence*100:.1f}% confidence."

        explanation = (
            f"The Diagnosis Agent identified this spot as {CODE_TO_NAME[pred_code]} ({pred_code}) with {confidence*100:.1f}% primary confidence. "
            f"The top prediction leads the second most likely condition by a margin of {margin*100:.1f}%, showing {certainty_phrase}. "
            f"Total estimated cancer risk across all malignant categories is {cum_malig*100:.1f}% (Melanoma: {mel_p*100:.1f}%, Basal Cell: {bcc_p*100:.1f}%, Actinic Keratosis/Pre-cancer: {akiec_p*100:.1f}%)."
        )

        return {
            # Explaining Agent Contract fields
            "agent": "A2",
            "name": "Diagnosis",
            "status": pred_code,
            "score": round(confidence, 4),
            "summary": summary,
            "evidence": evidence,
            "thresholds_used": thresholds_used,
            "margin_to_threshold": margin_to_threshold,
            "flags": flags,
            "explanation": explanation,
            "confidence_in_this_agent": round(confidence, 4),
            "runtime_ms": runtime_ms,
            # Preserved legacy fields
            "agent_id": "A2",
            "agent_name": "Diagnosis Agent",
            "predicted_code": pred_code,
            "predicted_class": CODE_TO_NAME[pred_code],
            "predicted_idx": pred_idx,
            "confidence": round(confidence, 4),
            "margin_to_next": margin,
            "entropy": entropy,
            "tta_disagreement": round(float(tta_disagreement), 4),
            "is_ambiguous": is_ambiguous,
            "diagnostic_certainty": certainty,
            "cumulative_malignancy_risk": cum_malig,
            "is_malignant": is_malignant,
            "all_probabilities": all_probs_code,
            "all_probabilities_by_name": all_probs_name,
            "top_3": top_3,
            "clinical_description": CLINICAL_DESCRIPTIONS.get(pred_code, "")
        }

    def predict_numpy_fallback(self, pil_img: Image.Image, custom_thresholds: Optional[Dict[str, Any]] = None, t0: float = 0.0) -> Dict[str, Any]:
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
                mean_r, mean_g, mean_b = float(r.mean()), float(g.mean()), float(b.mean())
                std_r, std_g, std_b = float(r.std()), float(g.std()), float(b.std())

                v = np.max(arr, axis=2)
                min_rgb = np.min(arr, axis=2)
                delta = v - min_rgb
                sat = np.where(v > 1e-4, delta / (v + 1e-5), 0.0)
                mean_sat, std_sat = float(sat.mean()), float(sat.std())
                mean_v, std_v = float(v.mean()), float(v.std())

                dark_melanin = float(np.sum((r < 0.32) & (g < 0.24) & (b < 0.24)) / arr.size)
                red_ratio = float(np.sum((r > 0.55) & (r > 1.35 * (g + b) / 2.0)) / arr.size)
                blue_gray = float(np.sum((b > g) & (b > r) & (sat > 0.15)) / arr.size)

                h, w = r.shape
                c_y, c_x = h // 4, w // 4
                center_patch = arr[c_y:h-c_y, c_x:w-c_x]
                outer_mask = np.ones((h, w), dtype=bool)
                outer_mask[c_y:h-c_y, c_x:w-c_x] = False
                contrast = float(np.linalg.norm(center_patch.mean(axis=(0, 1)) - arr[outer_mask].mean(axis=0)))

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

                scaler_mean = np.array(self.lightweight_weights['scaler_mean'])
                scaler_scale = np.array(self.lightweight_weights['scaler_scale'])
                coef = np.array(self.lightweight_weights['coef'])
                intercept = np.array(self.lightweight_weights['intercept'])

                feats_scaled = (feats - scaler_mean) / (scaler_scale + 1e-6)
                raw_logits = np.dot(coef, feats_scaled) + intercept
                scaled_logits = (raw_logits / temp) + bias_arr
                
                exp_l = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_l / np.sum(exp_l)
                runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                return self._package_prediction(probs, tta_disagreement=0.0, custom_thresholds=custom_thresholds, runtime_ms=runtime_ms)
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
        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return self._package_prediction(probs, tta_disagreement=0.0, custom_thresholds=custom_thresholds, runtime_ms=runtime_ms)

    def predict(self, image_input: Any, custom_thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Executes diagnostic inference on image input with TTA, temperature scaling, and class bias."""
        t0 = time.perf_counter()
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
                        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                        return self._package_prediction(mean_probs, tta_disagreement=tta_disagreement, custom_thresholds=custom_thresholds, runtime_ms=runtime_ms)

                    elif pil_img is not None:
                        tensor = diagnosis_transforms(pil_img).unsqueeze(0).to(DEVICE)
                        raw_logits = self.torch_model(tensor).cpu().numpy()[0]
                        scaled_logits = (raw_logits / temp) + bias_arr
                        exp_l = np.exp(scaled_logits - np.max(scaled_logits))
                        probs = exp_l / np.sum(exp_l)
                        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                        return self._package_prediction(probs, tta_disagreement=0.0, custom_thresholds=custom_thresholds, runtime_ms=runtime_ms)
                    else:
                        if len(tensor.shape) == 3:
                            tensor = tensor.unsqueeze(0)
                        tensor = tensor.to(DEVICE)
                        raw_logits = self.torch_model(tensor).cpu().numpy()[0]
                        scaled_logits = (raw_logits / temp) + bias_arr
                        exp_l = np.exp(scaled_logits - np.max(scaled_logits))
                        probs = exp_l / np.sum(exp_l)
                        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                        return self._package_prediction(probs, tta_disagreement=0.0, custom_thresholds=custom_thresholds, runtime_ms=runtime_ms)
            except Exception:
                pass

        if pil_img is None:
            pil_img = Image.new('RGB', (224, 224), (180, 140, 120))
        return self.predict_numpy_fallback(pil_img, custom_thresholds=custom_thresholds, t0=t0)

def load_diagnosis_agent(model_path: str = 'models/diagnosis_agent.pt') -> DiagnosisAgent:
    return DiagnosisAgent(model_path=model_path)

# Singleton instance
diagnosis_agent = load_diagnosis_agent()
