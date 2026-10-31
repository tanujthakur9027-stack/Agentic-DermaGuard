"""
api/batch_service.py
Server-side Batch Job Service for Agentic DermaGuard.
Enables running large hidden-test image evaluation jobs in the background,
independent of frontend browser tab or page navigation.
"""

import os
import sys
import time
import json
import uuid
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from orchestrator import agentic_dermaguard, AgenticDermaGuard
from scripts.validate_submission import validate_submission
from scripts.submit import execute_submission
from scripts.run_hidden import build_image_index, resolve_image, process_single_image, ALLOWED_EXTENSIONS, VALID_CODES

# Global in-memory job registry
BATCH_JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
ORCHESTRATOR_LOCK = threading.Lock()

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with JOBS_LOCK:
        return BATCH_JOBS.get(job_id)

def list_jobs() -> List[Dict[str, Any]]:
    with JOBS_LOCK:
        return [
            {
                "job_id": jid,
                "status": job["status"],
                "total": job["total"],
                "processed": job["processed"],
                "created_at": job["created_at"]
            }
            for jid, job in BATCH_JOBS.items()
        ]

def make_thumbnail_b64(img_path: Optional[str]) -> str:
    """Generates a small 64x64 JPEG thumbnail base64 string for the UI table."""
    if not img_path or not os.path.exists(img_path):
        return ""
    try:
        with Image.open(img_path) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((64, 64))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=75)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""

def check_image_matches(ids: List[str], source_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks how many IDs from `ids` match existing files or base URL without running pipeline.
    """
    temp_dirs = []
    source_type = source_info.get("type", "server_folder")
    server_folder = source_info.get("server_folder", "")
    url_template = source_info.get("url_template", "")
    ext_pattern = source_info.get("extension_pattern", ".jpg")

    search_dir = ""
    if source_type == "server_folder" and server_folder:
        candidates = [
            server_folder,
            os.path.join(PROJECT_ROOT, server_folder),
            os.path.join(PROJECT_ROOT, "Health", "hidden", "images"),
            os.path.join(PROJECT_ROOT, "data", "mock_hidden", "images"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                search_dir = c
                break
    elif source_type == "uploaded_files":
        # Files uploaded in request or temp dir
        search_dir = source_info.get("upload_dir", "")

    img_index = {}
    if search_dir and os.path.isdir(search_dir):
        img_index, _ = build_image_index(search_dir, temp_dirs)

    found_count = 0
    missing_ids = []
    sample_matches = []

    for img_id in ids:
        p = resolve_image(img_id, img_index, base_url=url_template)
        if not p and ext_pattern:
            stem = os.path.splitext(img_id)[0]
            cand = f"{stem.lower()}{ext_pattern.lower()}"
            if cand in img_index:
                p = img_index[cand]

        if p:
            found_count += 1
            if len(sample_matches) < 5:
                sample_matches.append({"image_id": img_id, "resolved_path": os.path.basename(p)})
        else:
            missing_ids.append(img_id)

    return {
        "total_ids": len(ids),
        "found_count": found_count,
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids[:50],
        "sample_matches": sample_matches,
        "search_directory": search_dir or "N/A"
    }

def start_batch_job(
    ids: List[str],
    source_info: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    workers: int = 3
) -> str:
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(PROJECT_ROOT, "outputs", "jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    job = {
        "job_id": job_id,
        "status": "running",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(ids),
        "processed": 0,
        "ids": ids,
        "source_info": source_info,
        "metadata": metadata or {},
        "workers": workers,
        "job_dir": job_dir,
        "rows": [],
        "ordered_results": {},
        "counts": {"AUTO_TRIAGE": 0, "REFER": 0, "RE_IMAGE": 0, "errors": 0},
        "start_time": time.time(),
        "elapsed_seconds": 0,
        "eta_seconds": 0,
        "paused": False,
        "cancelled": False,
        "error": None,
        "submission_csv": os.path.join(job_dir, "submission.csv")
    }

    with JOBS_LOCK:
        BATCH_JOBS[job_id] = job

    # Start daemon worker thread
    thread = threading.Thread(target=_batch_job_worker, args=(job_id,), daemon=True)
    thread.start()

    return job_id

def control_batch_job(job_id: str, action: str) -> bool:
    with JOBS_LOCK:
        job = BATCH_JOBS.get(job_id)
        if not job:
            return False

        if action == "pause":
            job["paused"] = True
            job["status"] = "paused"
        elif action == "resume":
            job["paused"] = False
            job["status"] = "running"
        elif action == "cancel":
            job["cancelled"] = True
            job["status"] = "cancelled"
        return True

def _batch_job_worker(job_id: str):
    job = get_job(job_id)
    if not job:
        return

    ids = job["ids"]
    source_info = job["source_info"]
    source_type = source_info.get("type", "server_folder")
    server_folder = source_info.get("server_folder", "")
    url_template = source_info.get("url_template", "")
    ext_pattern = source_info.get("extension_pattern", ".jpg")
    job_dir = job["job_dir"]

    # Handle zip data or uploaded files if provided
    img_dir = ""
    temp_dirs = []
    if source_type == "zip_base64" and source_info.get("zip_data"):
        import zipfile
        zip_bytes = base64.b64decode(source_info["zip_data"])
        img_dir = os.path.join(job_dir, "extracted_images")
        os.makedirs(img_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(img_dir)
    elif source_type == "server_folder" and server_folder:
        candidates = [
            server_folder,
            os.path.join(PROJECT_ROOT, server_folder),
            os.path.join(PROJECT_ROOT, "Health", "hidden", "images"),
            os.path.join(PROJECT_ROOT, "data", "mock_hidden", "images")
        ]
        for c in candidates:
            if os.path.isdir(c):
                img_dir = c
                break
    elif source_info.get("upload_dir"):
        img_dir = source_info["upload_dir"]

    # Pre-index images
    img_index, _ = build_image_index(img_dir, temp_dirs) if img_dir else ({}, "")

    # Pre-resolve map
    resolved_paths: Dict[str, Optional[str]] = {}
    dl_dir = os.path.join(job_dir, "downloads")
    if url_template:
        os.makedirs(dl_dir, exist_ok=True)

    for iid in ids:
        p = resolve_image(iid, img_index, base_url=url_template, dl_dir=dl_dir)
        if not p and ext_pattern:
            stem = os.path.splitext(iid)[0]
            cand = f"{stem.lower()}{ext_pattern.lower()}"
            if cand in img_index:
                p = img_index[cand]
        resolved_paths[iid] = p

    meta_map = job.get("metadata", {})
    workers = min(4, max(1, job.get("workers", 3)))
    orch = agentic_dermaguard or AgenticDermaGuard()

    partial_file = os.path.join(job_dir, "partial.jsonl")

    # Worker inner function
    def do_work(target_id: str):
        path = resolved_paths.get(target_id)
        ctx = meta_map.get(target_id.lower(), {})
        tid, rec, err = process_single_image(target_id, path, ctx, orch, ORCHESTRATOR_LOCK)
        rec["thumbnail"] = make_thumbnail_b64(path)
        return tid, rec, err

    start_t = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {executor.submit(do_work, iid): iid for iid in ids}

        for future in as_completed(future_to_id):
            if job["cancelled"]:
                break
            while job["paused"] and not job["cancelled"]:
                time.sleep(0.5)

            tid, rec, err = future.result()
            with JOBS_LOCK:
                job["ordered_results"][tid] = rec
                job["processed"] += 1
                
                dec = rec.get("decision", "REFER")
                if dec in job["counts"]:
                    job["counts"][dec] += 1
                if err:
                    job["counts"]["errors"] += 1

                elapsed = time.time() - start_t
                job["elapsed_seconds"] = round(elapsed, 1)
                rate = job["processed"] / max(0.001, elapsed)
                remaining = (job["total"] - job["processed"]) / max(0.001, rate)
                job["eta_seconds"] = max(0, int(remaining))

                # Keep live rows array updated in original ids order
                current_rows = []
                for original_id in ids:
                    if original_id in job["ordered_results"]:
                        current_rows.append(job["ordered_results"][original_id])
                job["rows"] = current_rows

            # Append to partial.jsonl
            with open(partial_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    # Generate final submission.csv in exact sample order
    submission_rows = []
    for iid in ids:
        r = job["ordered_results"].get(iid)
        if r:
            submission_rows.append({
                "image_id": r["image_id"],
                "prediction": r["prediction"],
                "decision": r["decision"]
            })
        else:
            submission_rows.append({
                "image_id": iid,
                "prediction": "nv",
                "decision": "REFER"
            })

    import pandas as pd
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(job["submission_csv"], index=False)

    # Also copy to root outputs/submission.csv for convenience
    root_sub = os.path.join(PROJECT_ROOT, "outputs", "submission.csv")
    sub_df.to_csv(root_sub, index=False)

    with JOBS_LOCK:
        if not job["cancelled"]:
            job["status"] = "completed"
            job["eta_seconds"] = 0

def get_submission_csv_content(job_id: str) -> Optional[str]:
    job = get_job(job_id)
    if not job:
        return None

    ids = job["ids"]
    ordered_results = job["ordered_results"]
    lines = ["image_id,prediction,decision"]
    for iid in ids:
        rec = ordered_results.get(iid)
        if rec:
            lines.append(f"{rec['image_id']},{rec['prediction']},{rec['decision']}")
        else:
            # Fallback for partial
            lines.append(f"{iid},nv,REFER")
    return "\n".join(lines) + "\n"
