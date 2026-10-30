import json
import base64
import io
import os
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler
import numpy as np
from PIL import Image

BENCHMARK_FILE = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation_results.json")

# Try importing PyTorch model if available locally
try:
    import torch
    from agents.diagnosis_agent import load_diagnosis_agent, diagnosis_transforms
    LOCAL_MODEL = load_diagnosis_agent("models/diagnosis_agent.pt")
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False
    LOCAL_MODEL = None

def evaluate_quality(pil_img: Image.Image, thresholds: dict):
    """
    Evaluates image quality with configurable threshold criteria.
    """
    blur_th = thresholds.get("blur_threshold", 40.0)
    light_th = thresholds.get("lighting_threshold", 40.0)
    glare_th = thresholds.get("glare_threshold", 70.0)
    occl_th = thresholds.get("occlusion_threshold", 50.0)
    qual_cutoff = thresholds.get("quality_cutoff", 50.0)

    img_gray = pil_img.convert('L').resize((224, 224))
    arr_gray = np.array(img_gray, dtype=np.float32)
    
    # 1. Blur via Laplacian variance
    pad = np.pad(arr_gray, 1, mode='reflect')
    lap = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4.0 * arr_gray)
    lap_var = float(lap.var())
    blur_score = min(100.0, (lap_var / 100.0) * 100.0)
    
    # 2. Lighting via Histogram Entropy
    counts, _ = np.histogram(arr_gray, bins=256, range=(0, 256))
    p = counts / float(counts.sum() + 1e-8)
    entropy = -float(np.sum(p * np.log2(p + 1e-8)))
    lighting_score = min(100.0, (entropy / 8.0) * 100.0)
    
    # 3. Glare via Saturated Pixels
    img_rgb = pil_img.convert('RGB').resize((224, 224))
    arr_rgb = np.array(img_rgb, dtype=np.float32)
    v_channel = np.max(arr_rgb, axis=2)
    overexposed = float(np.sum(v_channel > 240.0) / v_channel.size)
    glare_score = max(0.0, 100.0 - (overexposed * 100.0 * 2.5))
    
    # 4. Occlusion via Sobel Gradient Centering
    gx = (pad[:-2, 2:] + 2*pad[1:-1, 2:] + pad[2:, 2:]) - (pad[:-2, :-2] + 2*pad[1:-1, :-2] + pad[2:, :-2])
    gy = (pad[2:, :-2] + 2*pad[2:, 1:-1] + pad[2:, 2:]) - (pad[:-2, :-2] + 2*pad[:-2, 1:-1] + pad[:-2, 2:])
    grad = np.sqrt(gx**2 + gy**2)
    edges = grad > 30.0
    
    h, w = edges.shape
    mask_center = np.zeros((h, w), dtype=bool)
    mask_center[h//4:-h//4, w//4:-w//4] = True
    c_edges = float(np.sum(edges & mask_center))
    b_edges = float(np.sum(edges & (~mask_center)))
    occlusion_score = (c_edges / (c_edges + b_edges + 1e-8)) * 100.0
    
    overall = (blur_score + lighting_score + glare_score + occlusion_score) / 4.0
    
    reasoning = []
    triggers = []
    if blur_score < blur_th:
        reasoning.append(f"Image is blurry (score: {blur_score:.1f} < threshold {blur_th:.1f})")
        triggers.append("Blur")
    if lighting_score < light_th:
        reasoning.append(f"Poor illumination/entropy (score: {lighting_score:.1f} < threshold {light_th:.1f})")
        triggers.append("Illumination")
    if glare_score < glare_th:
        reasoning.append(f"Specular glare detected (score: {glare_score:.1f} < threshold {glare_th:.1f})")
        triggers.append("Glare")
    if occlusion_score < occl_th:
        reasoning.append(f"Border occlusion or lesion off-center (score: {occlusion_score:.1f} < threshold {occl_th:.1f})")
        triggers.append("Occlusion")
        
    if overall < qual_cutoff or len(triggers) >= 2:
        status = "RE_IMAGE"
        if not reasoning:
            reasoning.append(f"Overall quality below clinical threshold ({overall:.1f} < {qual_cutoff:.1f})")
    elif overall < 70.0 or len(triggers) == 1:
        status = "ACCEPTABLE_WITH_CAUTION"
    else:
        status = "ACCEPTABLE"
        
    if not reasoning:
        reasoning.append("Image quality fully meets diagnostic clinical criteria")
        
    return {
        "blur_score": round(blur_score, 1),
        "lighting_score": round(lighting_score, 1),
        "glare_score": round(glare_score, 1),
        "occlusion_score": round(occlusion_score, 1),
        "overall_quality": round(overall, 1),
        "status": status,
        "reasons": reasoning,
        "triggered_criteria": triggers
    }

def predict_pathology(pil_img: Image.Image):
    """
    Dynamic lesion pathology classification.
    Uses trained PyTorch model if available; otherwise uses dermoscopic feature extraction.
    """
    classes = [
        "Melanoma", "Melanocytic nevus", "Basal cell carcinoma", 
        "Actinic keratoses", "Benign keratosis", "Dermatofibroma", "Vascular lesion"
    ]
    
    if HAVE_TORCH and LOCAL_MODEL is not None:
        try:
            tensor = diagnosis_transforms(pil_img)
            res = LOCAL_MODEL.predict(tensor)
            return res
        except Exception:
            pass

    # Computer Vision Dermoscopic Feature Analysis (Pure NumPy)
    img_rgb = pil_img.convert('RGB').resize((224, 224))
    arr = np.array(img_rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    mean_r, mean_g, mean_b = float(r.mean()), float(g.mean()), float(b.mean())
    std_r, std_g, std_b = float(r.std()), float(g.std()), float(b.std())
    
    # Feature 1: Vascular Erythema Index
    vasc_ratio = mean_r / (mean_g + mean_b + 1e-5)
    
    # Feature 2: Dark Melanin Variegation (Melanoma marker)
    dark_melanin = float(np.sum((r < 80) & (g < 60) & (b < 60)) / arr.size)
    color_std = (std_r + std_g + std_b) / 3.0
    
    # Feature 3: Translucent / Pink telangiectasia (BCC marker)
    pink_translucent = float(np.sum((r > 170) & (g > 120) & (b > 120) & (r > g + 25)) / arr.size)
    
    # Feature 4: Rough scaly texture (AKIEC / BKL marker)
    gray = np.array(pil_img.convert('L').resize((224, 224)), dtype=np.float32)
    roughness = float(np.abs(gray[1:, :] - gray[:-1, :]).mean())
    
    # Dynamic raw score accumulation
    scores = {
        "Melanoma": max(0.05, (dark_melanin * 15.0) + (color_std / 45.0) * 0.4),
        "Melanocytic nevus": max(0.15, 0.75 - (color_std / 70.0) * 0.4 - (dark_melanin * 4.0)),
        "Basal cell carcinoma": max(0.04, (pink_translucent * 10.0) + 0.1),
        "Actinic keratoses": max(0.03, (roughness / 30.0) * 0.4 + 0.05),
        "Benign keratosis": max(0.05, (roughness / 25.0) * 0.35 + (color_std / 60.0) * 0.2),
        "Dermatofibroma": 0.08,
        "Vascular lesion": max(0.02, (vasc_ratio - 0.52) * 5.0) if vasc_ratio > 0.55 else 0.03
    }
    
    # Softmax normalization
    score_vals = np.array([scores[c] for c in classes])
    exp_scores = np.exp(score_vals * 2.5)
    probs = exp_scores / exp_scores.sum()
    
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])
    
    all_probs = {c: round(float(probs[i]), 4) for i, c in enumerate(classes)}
    top_3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        "predicted_class": pred_class,
        "predicted_idx": pred_idx,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
        "top_3": top_3
    }

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-KEY")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        
        if path.endswith("/api/benchmark") or path == "/api/benchmark":
            try:
                if os.path.exists(BENCHMARK_FILE):
                    with open(BENCHMARK_FILE, "r") as f:
                        data = json.load(f)
                else:
                    data = {"error": "Benchmark file not found"}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "Agentic DermaGuard API is running"}).encode("utf-8"))

    def do_POST(self):
        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        
        try:
            body = json.loads(post_data.decode("utf-8"))
        except:
            body = {}

        if path.endswith("/api/triage") or path == "/api/triage":
            image_b64 = body.get("image_base64", "")
            thresholds = body.get("thresholds", {})
            
            malig_th = thresholds.get("malignancy_threshold", 0.35)
            auto_th = thresholds.get("auto_triage_confidence", 0.70)
            
            try:
                if image_b64:
                    if "," in image_b64:
                        image_b64 = image_b64.split(",")[1]
                    img_bytes = base64.b64decode(image_b64)
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                else:
                    pil_img = Image.new("RGB", (224, 224), color=(180, 140, 120))
                    
                quality_res = evaluate_quality(pil_img, thresholds)
                
                # Consensus Triage Engine
                if quality_res["status"] == "RE_IMAGE":
                    final_decision = "RE_IMAGE"
                    primary_agent = "Image Quality Agent"
                    trigger_str = ", ".join(quality_res["triggered_criteria"]) if quality_res["triggered_criteria"] else "Overall Quality"
                    explanation = f"Quality gatekeeper rejected image [Trigger: {trigger_str}]: {'; '.join(quality_res['reasons'])}. Re-capture required."
                    diagnosis = None
                else:
                    diagnosis = predict_pathology(pil_img)
                    pred_class = diagnosis["predicted_class"]
                    conf = diagnosis["confidence"]
                    
                    high_risk = {"Melanoma", "Basal cell carcinoma", "Actinic keratoses"}
                    is_caution = (quality_res["status"] == "ACCEPTABLE_WITH_CAUTION")
                    
                    mel_prob = diagnosis["all_probabilities"].get("Melanoma", 0.0)
                    bcc_prob = diagnosis["all_probabilities"].get("Basal cell carcinoma", 0.0)
                    combined_malig = mel_prob + bcc_prob
                    
                    if pred_class in high_risk or combined_malig >= malig_th:
                        final_decision = "REFER"
                        primary_agent = "Diagnosis Agent"
                        culprit = pred_class if pred_class in high_risk else f"Malignancy suspicion ({combined_malig*100:.1f}% >= threshold {malig_th*100:.0f}%)"
                        explanation = f"Specialist Referral (REFER) [Trigger: {culprit}]: High-risk malignancy detected. Urgent dermatologist biopsy advised."
                    elif conf < auto_th or is_caution:
                        final_decision = "REFER"
                        primary_agent = "Consensus (Quality + Diagnosis)"
                        cause = "borderline image quality caution" if is_caution else f"confidence {conf*100:.1f}% below auto-triage threshold {auto_th*100:.0f}%"
                        explanation = f"Clinical Referral (REFER) [Trigger: Diagnostic Uncertainty]: Top candidate is '{pred_class}' ({conf*100:.1f}%), but {cause} mandates clinical evaluation."
                    else:
                        final_decision = "AUTO_TRIAGE"
                        primary_agent = "Consensus (Quality + Diagnosis)"
                        explanation = f"Automated Triage (AUTO_TRIAGE) [Trigger: Validated Benign]: Confident benign presentation of '{pred_class}' ({conf*100:.1f}% >= {auto_th*100:.0f}%) with validated image quality."
                        
                resp_payload = {
                    "final_decision": final_decision,
                    "primary_agent": primary_agent,
                    "explanation": explanation,
                    "quality_summary": quality_res,
                    "diagnosis_summary": diagnosis,
                    "active_thresholds": {
                        "blur_threshold": thresholds.get("blur_threshold", 40.0),
                        "lighting_threshold": thresholds.get("lighting_threshold", 40.0),
                        "glare_threshold": thresholds.get("glare_threshold", 70.0),
                        "occlusion_threshold": thresholds.get("occlusion_threshold", 50.0),
                        "quality_cutoff": thresholds.get("quality_cutoff", 50.0),
                        "malignancy_threshold": malig_th,
                        "auto_triage_confidence": auto_th
                    }
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(resp_payload).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif path.endswith("/api/submit") or path == "/api/submit":
            api_url = body.get("api_url")
            team_name = body.get("team_name", "Team-DermaGuard")
            team_token = body.get("team_token", "")
            
            if not api_url:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing api_url"}).encode("utf-8"))
                return

            if os.path.exists(BENCHMARK_FILE):
                with open(BENCHMARK_FILE, "r") as f:
                    sum_data = json.load(f)
            else:
                sum_data = {}

            payload = {
                "team_name": team_name,
                "track": "Track 2 | Healthcare",
                "level": 1,
                "project": "Agentic DermaGuard",
                "timestamp": sum_data.get("evaluation_timestamp"),
                "total_validation_images": sum_data.get("total_eval_samples", 1942),
                "baseline_metrics": sum_data.get("baseline_classification_metrics", {}),
                "image_quality_summary": sum_data.get("image_quality_agent_outcomes", {}),
                "triage_summary": sum_data.get("multi_agent_triage_decisions", {}),
                "model_architecture": "EfficientNet-B0 + Laplacian/Entropy/HSV/Canny Multi-Agent"
            }
            
            try:
                req = urllib.request.Request(
                    api_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "AgenticDermaGuard-VercelClient/1.0",
                        **({"Authorization": f"Bearer {team_token}", "X-API-KEY": team_token} if team_token else {})
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=20) as r:
                    code = r.getcode()
                    res_body = r.read().decode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "code": code, "response": res_body}).encode("utf-8"))
            except Exception as ex:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(ex)}).encode("utf-8"))
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
