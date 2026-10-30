import json
import base64
import io
import os
import sys
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler
import numpy as np
from PIL import Image

# Add workspace root to sys.path so agents/ and orchestrator can be imported on Vercel
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

BENCHMARK_FILE = os.path.join(WORKSPACE_ROOT, "outputs", "validation_results.json")

# Import the 6-agent orchestrator
try:
    from orchestrator import agentic_dermaguard, AgenticDermaGuard
    HAVE_ORCHESTRATOR = True
except Exception as e:
    print(f"[Warning] Failed to import orchestrator: {e}")
    agentic_dermaguard = None
    HAVE_ORCHESTRATOR = False

# Fallback implementations in case orchestrator module is isolated on Vercel serverless
from agents.image_quality_agent import quality_agent
from agents.diagnosis_agent import diagnosis_agent
from agents.lesion_analysis_agent import lesion_analysis_agent
from agents.context_agent import context_agent
from agents.safety_agent import safety_agent
from agents.coordinator_agent import coordinator_agent

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
        
        if path.endswith("/api/download-submission-sample"):
            sample_csv_path = os.path.join(WORKSPACE_ROOT, "outputs", "final_submission_format_sample.csv")
            if os.path.exists(sample_csv_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", 'attachment; filename="final_submission_format_sample.csv"')
                self._send_cors_headers()
                self.end_headers()
                with open(sample_csv_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Submission sample CSV not found"}).encode("utf-8"))
        elif path.endswith("/api/benchmark") or path == "/api/benchmark":
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
            self.wfile.write(json.dumps({
                "status": "Agentic DermaGuard Six-Agent API is running",
                "version": "Level 2 / Day 2 Architecture",
                "agents": [
                    "A1: Image Quality Agent",
                    "A2: Diagnosis Agent",
                    "A3: Lesion Analysis Agent",
                    "A4: Reasoning / Context Agent",
                    "A5: Safety Agent",
                    "A6: Coordinator Agent"
                ]
            }).encode("utf-8"))

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
            patient_context = body.get("context", {})

            try:
                # Decode image
                if image_b64:
                    if "," in image_b64:
                        image_b64 = image_b64.split(",")[1]
                    img_bytes = base64.b64decode(image_b64)
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                else:
                    # Default placeholder dermoscopic image
                    pil_img = Image.new("RGB", (224, 224), color=(180, 140, 120))

                # Execute Six-Agent Pipeline
                if HAVE_ORCHESTRATOR and agentic_dermaguard is not None:
                    res = agentic_dermaguard.process_image(
                        pil_img,
                        patient_context=patient_context,
                        custom_thresholds=thresholds
                    )
                else:
                    # Direct multi-agent invocation fallback
                    q_res = quality_agent.evaluate(pil_img, custom_thresholds=thresholds)
                    a1_data = {
                        "blur_score": q_res.blur_score,
                        "lighting_score": q_res.lighting_score,
                        "glare_score": q_res.glare_score,
                        "occlusion_score": q_res.occlusion_score,
                        "overall_quality": q_res.overall_quality,
                        "status": q_res.status,
                        "reasoning": q_res.reasoning,
                        "triggered_criteria": q_res.triggered_criteria,
                        "edge_map_b64": q_res.edge_map_b64
                    }
                    diag_output = diagnosis_agent.predict(pil_img)
                    a3_data = lesion_analysis_agent.analyze(pil_img)
                    a4_data = context_agent.evaluate(
                        age=patient_context.get("age"),
                        sex=patient_context.get("sex"),
                        localization=patient_context.get("localization")
                    )
                    a5_data = safety_agent.review(
                        quality_data=a1_data,
                        diagnosis_data=diag_output,
                        lesion_data=a3_data,
                        context_data=a4_data
                    )
                    a6_data = coordinator_agent.coordinate(
                        a1_quality=a1_data,
                        a2_diagnosis=diag_output,
                        a3_lesion=a3_data,
                        a4_context=a4_data,
                        a5_safety=a5_data,
                        custom_auto_conf=thresholds.get("auto_triage_confidence", 0.70)
                    )
                    res = {
                        "final_decision": a6_data["decision"],
                        "prediction": a6_data["prediction"],
                        "prediction_name": a6_data["prediction_name"],
                        "confidence": a6_data["confidence"],
                        "traceable_chain": a6_data["traceable_chain"],
                        "traceable_rationale": a6_data["rationale"],
                        "policy_rule": a6_data["policy_rule"],
                        "agents": {
                            "A1_quality": a1_data,
                            "A2_diagnosis": diag_output,
                            "A3_lesion_analysis": a3_data,
                            "A4_context_reasoning": a4_data,
                            "A5_safety": a5_data,
                            "A6_coordinator": a6_data
                        },
                        "primary_agent": "Coordinator Agent",
                        "predicted_class": a6_data["prediction_name"],
                        "explanation": a6_data["rationale"],
                        "quality_summary": a1_data,
                        "diagnosis_summary": diag_output
                    }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
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
                "level": 2,
                "project": "Agentic DermaGuard (Six-Agent System)",
                "timestamp": sum_data.get("evaluation_timestamp"),
                "total_validation_images": sum_data.get("total_eval_samples", 1942),
                "baseline_metrics": sum_data.get("baseline_classification_metrics", {}),
                "image_quality_summary": sum_data.get("image_quality_agent_outcomes", {}),
                "triage_summary": sum_data.get("multi_agent_triage_decisions", {}),
                "model_architecture": "Six-Agent Consensus: Image Quality + EfficientNet-B0 + ABCD Lesion + Context Reasoning + Safety Sentinel + Coordinator Policy"
            }
            
            try:
                req = urllib.request.Request(
                    api_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "AgenticDermaGuard-VercelClient/2.0",
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
