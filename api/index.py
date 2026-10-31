import json
import base64
import io
import os
import sys
import traceback
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set
from http.server import BaseHTTPRequestHandler
import numpy as np
from PIL import Image, ImageFilter

_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_this_dir, ".."))
_cwd = os.getcwd()

# Ensure all candidate module directories are in sys.path
for _p in [_this_dir, _parent_dir, _cwd, "/var/task", "/var/task/api"]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

WORKSPACE_ROOT = _parent_dir if os.path.exists(os.path.join(_parent_dir, "public")) else _this_dir

# Resolve benchmark and sample csv from either api/outputs/ or root outputs/
BENCHMARK_FILE = os.path.join(_this_dir, "outputs", "validation_results.json")
if not os.path.exists(BENCHMARK_FILE):
    BENCHMARK_FILE = os.path.join(WORKSPACE_ROOT, "outputs", "validation_results.json")

SAMPLE_CSV_FILE = os.path.join(_this_dir, "outputs", "final_submission_format_sample.csv")
if not os.path.exists(SAMPLE_CSV_FILE):
    SAMPLE_CSV_FILE = os.path.join(WORKSPACE_ROOT, "outputs", "final_submission_format_sample.csv")

PUBLIC_DIR = os.path.join(WORKSPACE_ROOT, "public")

# Import centralized thresholds with complete exception protection
try:
    from backend.thresholds import (
        REGISTRY,
        PRESETS,
        get_registry,
        get_presets,
        load_thresholds,
        save_thresholds,
        reset_thresholds,
        get_flat_thresholds
    )
except Exception:
    try:
        from thresholds import (
            REGISTRY,
            PRESETS,
            get_registry,
            get_presets,
            load_thresholds,
            save_thresholds,
            reset_thresholds,
            get_flat_thresholds
        )
    except Exception as _e_th:
        print(f"[Warning] Failed to import thresholds: {_e_th}")
        REGISTRY = []
        PRESETS = {}
        def get_registry(): return []
        def get_presets(): return {}
        def load_thresholds(reload=False): return {}
        def save_thresholds(th): return th
        def reset_thresholds(p=None): return {}
        def get_flat_thresholds(c=None): return {}

# Import orchestrator with non-crashing fallback for serverless import
try:
    from orchestrator import agentic_dermaguard, AgenticDermaGuard
except Exception as _e_orch:
    print(f"[Warning] Failed to import orchestrator at startup: {_e_orch}")
    agentic_dermaguard = None
    AgenticDermaGuard = None

# Import batch service
try:
    from api.batch_service import (
        start_batch_job,
        get_job,
        control_batch_job,
        check_image_matches,
        get_submission_csv_content,
        list_jobs
    )
except Exception:
    try:
        from batch_service import (
            start_batch_job,
            get_job,
            control_batch_job,
            check_image_matches,
            get_submission_csv_content,
            list_jobs
        )
    except Exception as _e_batch:
        print(f"[Warning] Failed to import batch_service: {_e_batch}")
        def start_batch_job(*args, **kwargs): return "mock_job"
        def get_job(jid): return None
        def control_batch_job(jid, act): return False
        def check_image_matches(ids, src): return {}
        def get_submission_csv_content(jid): return None
        def list_jobs(): return []

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
        try:
            self._handle_GET()
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"[ERROR in do_GET] {e}\n{tb_str}", file=sys.stdout)
            sys.stdout.flush()
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                err_payload = {
                    "error": str(e),
                    "type": type(e).__name__,
                    "traceback": tb_str
                }
                self.wfile.write(json.dumps(err_payload).encode("utf-8"))
            except Exception:
                pass

    def _get_request_path(self) -> Tuple[str, str]:
        """
        Extracts the canonical request path and a search corpus across local and Vercel serverless.
        Returns: (primary_path, path_corpus)
        """
        raw_path = getattr(self, "path", "").split("?")[0]
        fwd_uri = (self.headers.get("x-forwarded-uri") or self.headers.get("x-real-path") or "").split("?")[0]
        matched = (self.headers.get("x-matched-path") or self.headers.get("x-vercel-matched-path") or "").split("?")[0]

        candidates = [raw_path, fwd_uri, matched]
        primary_path = raw_path
        for c in candidates:
            if c and c not in ("", "/", "/api", "/api/index.py", "/api/index"):
                primary_path = c
                break

        corpus = f"{raw_path.lower()} {fwd_uri.lower()} {matched.lower()} {primary_path.lower()}"
        return primary_path, corpus

    def _handle_GET(self):
        primary_path, corpus = self._get_request_path()

        # Static assets and HTML pages (for local standalone execution)
        if not primary_path.startswith("/api"):
            rel = "index.html" if primary_path in ("", "/") else primary_path.lstrip("/")
            file_path = os.path.normpath(os.path.join(PUBLIC_DIR, rel))
            if os.path.commonpath([PUBLIC_DIR, file_path]) == PUBLIC_DIR and os.path.isfile(file_path):
                mime = "text/plain"
                if file_path.endswith(".html"):
                    mime = "text/html; charset=utf-8"
                elif file_path.endswith(".css"):
                    mime = "text/css; charset=utf-8"
                elif file_path.endswith(".js"):
                    mime = "application/javascript; charset=utf-8"
                elif file_path.endswith(".json"):
                    mime = "application/json"
                elif file_path.endswith(".png"):
                    mime = "image/png"
                elif file_path.endswith((".jpg", ".jpeg")):
                    mime = "image/jpeg"
                elif file_path.endswith(".svg"):
                    mime = "image/svg+xml"
                elif file_path.endswith(".ico"):
                    mime = "image/x-icon"

                self.send_response(200)
                self.send_header("Content-Type", mime)
                self._send_cors_headers()
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            elif primary_path not in ("", "/"):
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"File not found: {primary_path}"}).encode("utf-8"))
                return

        # --- Batch Endpoints ---
        if "/api/batch/download/" in primary_path:
            job_id = primary_path.split("/api/batch/download/")[1].split("/")[0].split("?")[0]
            csv_text = get_submission_csv_content(job_id)
            if csv_text is not None:
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", 'attachment; filename="submission.csv"')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(csv_text.encode("utf-8"))
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Job {job_id} not found"}).encode("utf-8"))
            return

        if "/api/batch/status/" in primary_path:
            job_id = primary_path.split("/api/batch/status/")[1].split("/")[0].split("?")[0]
            job = get_job(job_id)
            if job:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                resp_payload = {
                    "status": "success",
                    "job_id": job["job_id"],
                    "job_status": job["status"],
                    "total": job["total"],
                    "processed": job["processed"],
                    "elapsed_seconds": job["elapsed_seconds"],
                    "eta_seconds": job["eta_seconds"],
                    "counts": job["counts"],
                    "rows": job["rows"],
                    "is_finished": job["status"] in ["completed", "cancelled"]
                }
                self.wfile.write(json.dumps(resp_payload).encode("utf-8"))
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Job {job_id} not found"}).encode("utf-8"))
            return

        if primary_path == "/api/batch/jobs":
            jobs = list_jobs()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "jobs": jobs}).encode("utf-8"))
            return

        # 1. Download sample submission CSV
        if "download" in corpus or "submission-sample" in corpus:
            if os.path.exists(SAMPLE_CSV_FILE):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", 'attachment; filename="final_submission_format_sample.csv"')
                self._send_cors_headers()
                self.end_headers()
                with open(SAMPLE_CSV_FILE, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Submission sample CSV not found"}).encode("utf-8"))

        # 2. Validation Benchmark Metrics
        elif "benchmark" in corpus:
            try:
                if os.path.exists(BENCHMARK_FILE):
                    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
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

        # 3. Threshold Registry & Presets
        elif "thresholds" in corpus:
            try:
                current_values = load_thresholds(reload=True)
                registry = get_registry()
                presets = get_presets()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "registry": registry,
                    "current": current_values,
                    "presets": presets
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        # 4. Curated Demo Samples
        elif "samples" in corpus:
            try:
                samples = self._get_curated_samples()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "samples": samples}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        # 5. Hidden 200 Test Records & Metadata (Official Level 2 Release)
        elif "hidden-records" in corpus or "hidden_records" in corpus:
            try:
                records = self._get_hidden_records()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "count": len(records), "records": records}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        # 6. Hidden 200 Pre-Computed Six-Agent Audit Trace
        elif "hidden-audit" in corpus or "hidden_audit" in corpus:
            try:
                audit_file = os.path.join(WORKSPACE_ROOT, "outputs", "hidden_200_audit.json")
                if os.path.exists(audit_file):
                    with open(audit_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = {"error": "Audit trace not yet generated. Run run_hidden_batch.py first."}
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

        # 7. Download Official health_submission.csv
        elif "health-submission" in corpus:
            csv_path = os.path.join(WORKSPACE_ROOT, "outputs", "health_submission.csv")
            if os.path.exists(csv_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", 'attachment; filename="health_submission.csv"')
                self._send_cors_headers()
                self.end_headers()
                with open(csv_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "health_submission.csv not found"}).encode("utf-8"))

        # Default info
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "Agentic DermaGuard Six-Agent API is running",
                "version": "Summit 2026 Level 2 Architecture",
                "endpoints": [
                    "POST /api/analyze",
                    "POST /api/triage",
                    "GET /api/thresholds",
                    "POST /api/thresholds",
                    "POST /api/thresholds/reset",
                    "GET /api/benchmark",
                    "GET /api/samples",
                    "POST /api/submit"
                ],
                "agents": [
                    "A1: Image Quality Agent",
                    "A2: Diagnosis Agent",
                    "A3: Lesion Analysis Agent",
                    "A4: Reasoning / Context Agent",
                    "A5: Safety Sentinel Agent",
                    "A6: Coordinator Agent"
                ]
            }).encode("utf-8"))

    def do_POST(self):
        try:
            self._handle_POST()
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"[ERROR in do_POST] {e}\n{tb_str}", file=sys.stdout)
            sys.stdout.flush()
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                err_payload = {
                    "error": str(e),
                    "type": type(e).__name__,
                    "traceback": tb_str
                }
                self.wfile.write(json.dumps(err_payload).encode("utf-8"))
            except Exception:
                pass

    def _handle_POST(self):
        primary_path, corpus = self._get_request_path()
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode("utf-8"))
        except Exception:
            body = {}

        # --- Batch Endpoints ---
        if "/api/batch/start" in primary_path:
            ids = body.get("ids", [])
            image_source = body.get("image_source", {})
            metadata = body.get("metadata", {})
            workers = int(body.get("workers", 3))
            job_id = start_batch_job(ids, image_source, metadata=metadata, workers=workers)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "started",
                "job_id": job_id,
                "total": len(ids)
            }).encode("utf-8"))
            return

        if "/api/batch/control" in primary_path:
            job_id = primary_path.split("/api/batch/control/")[1].split("/")[0].split("?")[0] if "/control/" in primary_path else body.get("job_id")
            action = body.get("action", "pause")
            ok = control_batch_job(job_id, action)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success" if ok else "error", "job_id": job_id, "action": action}).encode("utf-8"))
            return

        if "/api/batch/match-check" in primary_path:
            ids = body.get("ids", [])
            image_source = body.get("image_source", {})
            report = check_image_matches(ids, image_source)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "match_report": report}).encode("utf-8"))
            return

        if "/api/batch/validate" in primary_path:
            job_id = primary_path.split("/api/batch/validate/")[1].split("/")[0].split("?")[0] if "/validate/" in primary_path else body.get("job_id")
            job = get_job(job_id)
            if job:
                from scripts.validate_submission import validate_submission
                passed, errs, stats = validate_submission(job["submission_csv"], ids_path=None)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "job_id": job_id,
                    "passed": passed,
                    "errors": errs,
                    "stats": stats
                }).encode("utf-8"))
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Job {job_id} not found"}).encode("utf-8"))
            return

        if "/api/batch/submit" in primary_path:
            job_id = primary_path.split("/api/batch/submit/")[1].split("/")[0].split("?")[0] if "/submit/" in primary_path else body.get("job_id")
            job = get_job(job_id)
            if job:
                from scripts.submit import execute_submission
                dry_run = bool(body.get("dry_run", False))
                passed, resp_data = execute_submission(job["submission_csv"], dry_run=dry_run, yes=True)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success" if passed else "error",
                    "job_id": job_id,
                    "passed": passed,
                    "response": resp_data
                }).encode("utf-8"))
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Job {job_id} not found"}).encode("utf-8"))
            return

        # 1. Reset Thresholds Endpoint
        if "reset" in corpus:
            try:
                preset = body.get("preset", "Balanced")
                updated = reset_thresholds(preset)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "message": f"Reset thresholds to {preset} preset",
                    "preset": preset,
                    "thresholds": updated
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 2. Save Thresholds Endpoint
        if "thresholds" in corpus and not ("analyze" in corpus or "triage" in corpus):
            try:
                new_th = body.get("thresholds", body)
                updated = save_thresholds(new_th)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "message": "Saved thresholds successfully",
                    "thresholds": updated
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 3. Summit Submission Upload Endpoint
        if "submit" in corpus and not body.get("image_base64"):
            self._handle_summit_submission(body)
            return

        # 4. Primary Image Analysis Pipeline (/api/analyze or /api/triage)
        self._handle_image_analysis(body)

    def _handle_image_analysis(self, body: Dict[str, Any]):
        """Processes a single image via the six-agent orchestrator with full error containment."""
        image_b64 = body.get("image_base64") or body.get("image") or body.get("data") or body.get("image_b64") or ""
        image_id = body.get("image_id", "")
        thresholds = body.get("thresholds", {})
        patient_context = body.get("context", {})

        try:
            pil_img = None
            img_desc = "unknown"

            # 1. Resolve image from base64
            if image_b64:
                if "," in image_b64:
                    image_b64 = image_b64.split(",")[1]
                img_bytes = base64.b64decode(image_b64)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_desc = image_id or "uploaded_image.jpg"

            # 2. Or resolve from local path / image_id
            elif image_id:
                candidates = [
                    os.path.join(WORKSPACE_ROOT, "Health", "hidden", "images", f"{image_id}.jpg"),
                    os.path.join(WORKSPACE_ROOT, "Health", "hidden", "images", image_id),
                    r"C:\Users\Tanuj kumar singh\Desktop\image batch\Health\hidden\images" + f"\\{image_id}.jpg",
                    r"C:\Users\Tanuj kumar singh\Desktop\image batch\Health\hidden\images" + f"\\{image_id}",
                    os.path.join(WORKSPACE_ROOT, "Health", "eval", "images", f"{image_id}.jpg"),
                    os.path.join(WORKSPACE_ROOT, "Health", "eval", "images", image_id),
                    os.path.join(WORKSPACE_ROOT, "Health", "training", "images", f"{image_id}.jpg"),
                    os.path.join(WORKSPACE_ROOT, "Health", "training", "images", image_id),
                    image_id
                ]
                for c in candidates:
                    if os.path.exists(c):
                        pil_img = Image.open(c).convert("RGB")
                        img_desc = os.path.basename(c)
                        break

            # 3. Synthetic or default fallback
            if pil_img is None:
                # Default benign appearance
                pil_img = Image.new("RGB", (224, 224), color=(180, 140, 120))
                img_desc = image_id or "default_placeholder.jpg"

            # Execute full Six-Agent Orchestrator
            global agentic_dermaguard
            if agentic_dermaguard is None:
                try:
                    from orchestrator import agentic_dermaguard as _ad
                    agentic_dermaguard = _ad
                except Exception as _ex:
                    print(f"[Warning] Lazy import orchestrator: {_ex}")

            if agentic_dermaguard is not None:
                res = agentic_dermaguard.process_image(
                    pil_img,
                    patient_context=patient_context,
                    custom_thresholds=thresholds
                )
            else:
                raise RuntimeError("Six-Agent Orchestrator module could not be initialized")
            
            res["image_id"] = image_id or img_desc.replace(".jpg", "")
            raw_dec = res.get("decision", "REFER")
            res["decision"] = raw_dec
            res["decision_hyphenated"] = raw_dec.replace("_", "-")
            res["predicted_label"] = res.get("prediction", "nv")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        except Exception as e:
            # Deterministic fallback row per hackathon rules
            fallback_res = {
                "image_id": image_id or "unknown",
                "final_decision": "REFER",
                "decision": "REFER",
                "prediction": "nv",
                "prediction_name": "Melanocytic nevus",
                "confidence": 0.50,
                "traceable_chain": "A1: ERROR | A2: nv • 0.50 | A3+A4: UNKNOWN | A5: SAFETY VETO | A6: REFER",
                "example_strip": 'A1 ERROR "Processing failed" | A2 nv • 0.50 "Model predicts nevus" | A3+A4 UNKNOWN "Analysis error" | A5 SAFETY VETO "Exception catch" | A6 REFER "Final disposition"',
                "traceable_rationale": f"System error caught during execution ({str(e)}). Conservative clinical referral applied.",
                "policy_rule": "Fallback Rule: Error mandates specialist referral.",
                "error": str(e),
                "agents": {
                    "A1_quality": {"agent": "A1", "name": "Image Quality", "status": "ERROR", "score": 0.0, "summary": "Quality error", "evidence": [], "flags": ["ERROR"], "explanation": str(e), "runtime_ms": 0.0},
                    "A2_diagnosis": {"agent": "A2", "name": "Diagnosis", "status": "nv", "score": 0.50, "summary": "Fallback to nv", "evidence": [], "flags": ["FALLBACK"], "explanation": str(e), "runtime_ms": 0.0},
                    "A3_lesion_analysis": {"agent": "A3", "name": "Lesion Analysis", "status": "INDETERMINATE", "score": 0.50, "summary": "Error in lesion analysis", "evidence": [], "flags": ["ERROR"], "explanation": str(e), "runtime_ms": 0.0},
                    "A4_context_reasoning": {"agent": "A4", "name": "Context & Demographics", "status": "MODERATE", "score": 0.50, "summary": "Error in context evaluation", "evidence": [], "flags": ["ERROR"], "explanation": str(e), "runtime_ms": 0.0},
                    "A5_safety": {"agent": "A5", "name": "Safety Sentinel", "status": "SAFETY_VETO", "score": 1.0, "summary": "Veto on error", "evidence": [], "flags": ["SAFETY_VETO"], "explanation": str(e), "runtime_ms": 0.0},
                    "A6_coordinator": {"agent": "A6", "name": "Coordinator", "status": "REFER", "score": 0.50, "summary": "Conservative fallback", "evidence": [], "flags": ["FALLBACK_REFERRAL"], "explanation": str(e), "runtime_ms": 0.0}
                }
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(fallback_res).encode("utf-8"))

    def _handle_summit_submission(self, body: Dict[str, Any]):
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
            with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
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
                    "User-Agent": "AgenticDermaGuard-Client/2.0",
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

    def _get_curated_samples(self) -> List[Dict[str, Any]]:
        """Returns a list of curated clinical demonstration cases."""
        sample_path = os.path.join(_this_dir, "curated_samples.json")
        if os.path.exists(sample_path):
            try:
                with open(sample_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if isinstance(cached, list) and len(cached) > 0 and cached[0].get("image_b64"):
                        return cached
            except Exception:
                pass

        samples_info = [
            {
                "id": "ISIC_0024341",
                "label": "Case 1: Melanocytic Nevus (nv) — High Confidence Clear Benign",
                "code": "nv",
                "scenario": "Clear Benign Auto-Triage",
                "expected_decision": "AUTO_TRIAGE",
                "context": {"age": 35, "sex": "female", "localization": "lower extremity"},
                "description": "Symmetrical pigment network, regular border, young adult extremity lesion. Passes all safety checks."
            },
            {
                "id": "ISIC_0024332",
                "label": "Case 2: Invasive Melanoma (mel) — High Suspicion ABCD",
                "code": "mel",
                "scenario": "Malignancy Presentation Referral",
                "expected_decision": "REFER",
                "context": {"age": 68, "sex": "male", "localization": "back"},
                "description": "High geometric asymmetry, irregular notched border, color variegation. Mandatory specialist referral."
            },
            {
                "id": "ISIC_0024330",
                "label": "Case 3: Basal Cell Carcinoma (bcc) — Telangiectasia",
                "code": "bcc",
                "scenario": "Malignancy Presentation Referral",
                "expected_decision": "REFER",
                "context": {"age": 72, "sex": "male", "localization": "face"},
                "description": "Translucent nodule on chronic sun-damaged facial zone. Mandatory specialist referral."
            },
            {
                "id": "ISIC_0024317",
                "label": "Case 4: Melanocytic Nevus in Elderly (nv) — High-Risk Context Veto",
                "code": "nv",
                "scenario": "Safety Sentinel Context Veto",
                "expected_decision": "REFER",
                "context": {"age": 78, "sex": "male", "localization": "scalp"},
                "description": "Benign nevus diagnosed in elderly scalp region without 85% confidence hurdle; Safety Rule S4 triggers veto."
            },
            {
                "id": "ISIC_0024340",
                "label": "Case 5: Benign Keratosis (bkl) — Differential Ambiguity",
                "code": "bkl",
                "scenario": "Diagnostic Uncertainty Referral",
                "expected_decision": "REFER",
                "context": {"age": 55, "sex": "female", "localization": "trunk"},
                "description": "Close differential between seborrheic keratosis and atypical nevus. Referred for dermoscopy verification."
            },
            {
                "id": "sample_blurry",
                "label": "Case 6: Out-of-Focus Capture — Quality Gatekeeper RE_IMAGE",
                "code": "nv",
                "scenario": "Image Quality Failure RE_IMAGE",
                "expected_decision": "RE_IMAGE",
                "context": {"age": 45, "sex": "female", "localization": "abdomen"},
                "description": "Severe motion blur (Laplacian variance < 15.0). Policy Rule 1 mandates re-capture while preserving preliminary diagnosis."
            },
            {
                "id": "sample_glare",
                "label": "Case 7: Specular Glare Saturation — Flash Artifact RE_IMAGE",
                "code": "nv",
                "scenario": "Specular Glare Failure RE_IMAGE",
                "expected_decision": "RE_IMAGE",
                "context": {"age": 50, "sex": "male", "localization": "chest"},
                "description": "Flash reflection creates 8% white saturation void across lesion core. Re-imaging requested."
            }
        ]

        # Attach base64 image data for each sample
        for s in samples_info:
            sid = s["id"]
            img = None
            if sid == "sample_blurry":
                # Generate synthetic blurred version of ISIC_0024341
                orig_p = os.path.join(WORKSPACE_ROOT, "Health", "eval", "images", "ISIC_0024341.jpg")
                if os.path.exists(orig_p):
                    img = Image.open(orig_p).convert("RGB").filter(ImageFilter.GaussianBlur(radius=6.0))
            elif sid == "sample_glare":
                # Generate synthetic specular glare
                orig_p = os.path.join(WORKSPACE_ROOT, "Health", "eval", "images", "ISIC_0024341.jpg")
                if os.path.exists(orig_p):
                    img = Image.open(orig_p).convert("RGB")
                    arr = np.array(img)
                    h, w, _ = arr.shape
                    cy, cx = h // 2, w // 2
                    y, x = np.ogrid[:h, :w]
                    glare_mask = ((x - cx)**2 + (y - cy)**2) <= (min(h, w) * 0.18)**2
                    arr[glare_mask] = [255, 255, 255]
                    img = Image.fromarray(arr)
            else:
                p = os.path.join(WORKSPACE_ROOT, "Health", "eval", "images", f"{sid}.jpg")
                if os.path.exists(p):
                    img = Image.open(p).convert("RGB")

            if img is not None:
                img_small = img.resize((224, 224))
                buf = io.BytesIO()
                img_small.save(buf, format="JPEG", quality=85)
                s["image_b64"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            else:
                s["image_b64"] = ""

        return samples_info

    def _get_hidden_records(self) -> List[Dict[str, Any]]:
        candidates = [
            os.path.join(WORKSPACE_ROOT, "Health", "hidden", "records.csv"),
            r"C:\Users\Tanuj kumar singh\Desktop\image batch\Health\hidden\records.csv"
        ]
        records = []
        for csv_path in candidates:
            if os.path.exists(csv_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        records.append({
                            "image_id": str(row["image_id"]).strip(),
                            "age": float(row["age"]) if pd.notnull(row.get("age")) else None,
                            "sex": str(row["sex"]).strip() if pd.notnull(row.get("sex")) else None,
                            "localization": str(row["localization"]).strip() if pd.notnull(row.get("localization")) else None
                        })
                    return records
                except Exception as ex:
                    print(f"Error reading {csv_path}: {ex}")
        return records

if __name__ == "__main__":
    from http.server import HTTPServer
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), handler)
    print(f"[*] Agentic DermaGuard Server running on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server gracefully stopped.")
