#!/usr/bin/env python3
"""
scripts/run_hidden.py
CLI for running Agentic DermaGuard on hidden test images.
Jaypee Agentic AI International Summit 2026 - Track 2: Healthcare

Usage:
  python scripts/run_hidden.py --ids sample_submission.csv --images <folder_or_zip> --out outputs/submission.csv \
      [--base-url https://.../{image_id}.jpg] [--metadata optional.csv] [--workers 4] [--resume]
"""

import os
import sys
import argparse
import time
import json
import zipfile
import tempfile
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple, Set
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from orchestrator import AgenticDermaGuard
from scripts.validate_submission import validate_submission, load_expected_ids

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VALID_CODES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]

def parse_args():
    parser = argparse.ArgumentParser(description="Agentic DermaGuard: Run Six-Agent Pipeline on Hidden Test Images")
    parser.add_argument("--ids", type=str, required=True, help="Path to sample CSV containing image IDs")
    parser.add_argument("--images", type=str, required=True, help="Path to folder containing images or a .zip archive")
    parser.add_argument("--out", type=str, default="outputs/submission.csv", help="Path for output submission CSV")
    parser.add_argument("--base-url", type=str, default=None, help="Optional URL template to download missing images (e.g. https://domain.com/imgs/{image_id}.jpg)")
    parser.add_argument("--metadata", type=str, default=None, help="Optional CSV file with patient metadata (image_id, age, sex, localization)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads (default 4)")
    parser.add_argument("--resume", action="store_true", help="Resume from outputs/partial.jsonl, skipping already processed IDs")
    return parser.parse_args()

def read_id_list(ids_path: str) -> List[str]:
    """
    Reads IDs from CSV or single-column file.
    Preserves exact raw IDs, checks for duplicates and errors loudly.
    """
    if not os.path.exists(ids_path):
        print(f"[FATAL ERROR] Sample IDs file not found: {ids_path}", file=sys.stderr)
        sys.exit(1)

    raw_ids: List[str] = []
    try:
        df = pd.read_csv(ids_path, dtype=str)
        matched_col = None
        for col in df.columns:
            if col.strip().lower() == "image_id":
                matched_col = col
                break
        
        if matched_col:
            raw_ids = [str(x).strip() for x in df[matched_col].dropna().tolist() if str(x).strip()]
        elif len(df.columns) == 1:
            first_val = str(df.columns[0]).strip()
            items = [str(x).strip() for x in df.iloc[:, 0].dropna().tolist() if str(x).strip()]
            if first_val.lower() not in {"image_id", "id"}:
                raw_ids = [first_val] + items
            else:
                raw_ids = items
    except Exception as ex:
        print(f"[WARN] Pandas CSV parsing failed ({ex}), falling back to line reader")

    if not raw_ids:
        with open(ids_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines and lines[0].lower().startswith("image_id"):
            lines = lines[1:]
        raw_ids = [l.split(",")[0].strip() for l in lines if l.split(",")[0].strip()]

    if not raw_ids:
        print(f"[FATAL ERROR] No image IDs could be loaded from {ids_path}", file=sys.stderr)
        sys.exit(1)

    # Check for duplicates
    seen = set()
    dups = []
    for img_id in raw_ids:
        if img_id in seen:
            dups.append(img_id)
        seen.add(img_id)

    if dups:
        print(f"\n[FATAL ERROR] Found {len(dups)} DUPLICATE ID(s) in sample IDs CSV:", file=sys.stderr)
        for d in dups[:10]:
            print(f"  - Duplicate: {d}", file=sys.stderr)
        if len(dups) > 10:
            print(f"  ... and {len(dups) - 10} more.", file=sys.stderr)
        sys.exit(1)

    return raw_ids

def build_image_index(images_path: str, temp_dirs: List[str]) -> Tuple[Dict[str, str], str]:
    """
    Builds a lookup map from image stem (and exact name) to full file path.
    Extracts zip archives to a temporary directory if needed.
    """
    search_dir = images_path
    if os.path.isfile(images_path) and images_path.lower().endswith(".zip"):
        td = tempfile.mkdtemp(prefix="dermaguard_zip_")
        temp_dirs.append(td)
        print(f"[INFO] Extracting zip archive {images_path} to {td}...")
        with zipfile.ZipFile(images_path, "r") as z:
            z.extractall(td)
        search_dir = td
    elif not os.path.isdir(search_dir):
        print(f"[WARN] Images path is neither an existing directory nor a .zip: {images_path}")
        return {}, search_dir

    index: Dict[str, str] = {}
    for root, _, files in os.walk(search_dir):
        for f in files:
            full_path = os.path.join(root, f)
            # Map exact name lower
            index[f.lower()] = full_path
            # Map stem lower
            stem, ext = os.path.splitext(f)
            if ext.lower() in ALLOWED_EXTENSIONS or ext == "":
                index[stem.lower()] = full_path

    return index, search_dir

def resolve_image(
    img_id: str,
    index: Dict[str, str],
    base_url: Optional[str] = None,
    dl_dir: Optional[str] = None
) -> Optional[str]:
    """
    Resolves image file path for a given image_id.
    Tries exact match, stem match, extension variants, and optional base-url download.
    """
    id_lower = img_id.lower().strip()
    # 1. Exact match in index
    if id_lower in index:
        return index[id_lower]

    # 2. Stem match without extension
    stem_lower = os.path.splitext(id_lower)[0]
    if stem_lower in index:
        return index[stem_lower]

    # 3. Try with common extensions
    for ext in ALLOWED_EXTENSIONS:
        candidate = f"{stem_lower}{ext}"
        if candidate in index:
            return index[candidate]

    # 4. If base_url provided, attempt download
    if base_url and dl_dir:
        try:
            if "{image_id}" in base_url:
                target_url = base_url.format(image_id=img_id)
            elif "{id}" in base_url:
                target_url = base_url.format(id=img_id)
            else:
                target_url = base_url.rstrip("/") + f"/{img_id}.jpg"

            dest_path = os.path.join(dl_dir, f"{img_id}.jpg")
            urllib.request.urlretrieve(target_url, dest_path)
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                index[id_lower] = dest_path
                index[stem_lower] = dest_path
                return dest_path
        except Exception as e:
            pass

    return None

def load_patient_metadata(metadata_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Loads optional patient metadata indexed by image_id."""
    if not metadata_path or not os.path.exists(metadata_path):
        return {}
    
    meta_map: Dict[str, Dict[str, Any]] = {}
    try:
        df = pd.read_csv(metadata_path, dtype=str)
        # Find image_id col
        id_col = None
        for c in df.columns:
            if c.strip().lower() == "image_id":
                id_col = c
                break
        if not id_col:
            return {}

        for _, row in df.iterrows():
            iid = str(row[id_col]).strip()
            stem = os.path.splitext(iid)[0].lower()
            age_val = None
            if "age" in row and pd.notnull(row["age"]):
                try:
                    age_val = float(row["age"])
                except Exception:
                    pass
            sex_val = str(row["sex"]).strip() if "sex" in row and pd.notnull(row["sex"]) else None
            loc_val = str(row["localization"]).strip() if "localization" in row and pd.notnull(row["localization"]) else None

            ctx = {
                "age": age_val,
                "sex": sex_val,
                "localization": loc_val
            }
            meta_map[iid.lower()] = ctx
            meta_map[stem] = ctx
    except Exception as e:
        print(f"[WARN] Failed to parse metadata file {metadata_path}: {e}")

    return meta_map

def format_eta(seconds: float) -> str:
    if seconds < 0 or seconds > 36000:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def process_single_image(
    img_id: str,
    img_path: Optional[str],
    context: Dict[str, Any],
    orchestrator: AgenticDermaGuard,
    model_lock: threading.Lock
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Executes 6-agent pipeline on a single image.
    Never crashes: on any error returns a deterministic fallback row.
    """
    if not img_path or not os.path.exists(img_path):
        err_msg = f"Image file not found for ID: {img_id}"
        # Deterministic fallback per brief: prediction="nv", decision="REFER"
        fallback_res = {
            "image_id": img_id,
            "prediction": "nv",
            "decision": "REFER",
            "confidence": 0.50,
            "prediction_name": "Melanocytic nevus",
            "traceable_chain": f"A1: ERROR | A2: nv • 0.50 | A3+A4: UNKNOWN | A5: SAFETY VETO | A6: REFER",
            "policy_rule": "Fallback: Missing image file triggers conservative referral.",
            "traceable_rationale": f"Image file could not be located. Conservative referral mandated.",
            "runtime_ms": 0.0,
            "is_fallback": True,
            "error": err_msg,
            "agents": {
                "A1_quality": {"status": "ERROR", "score": 0.0, "flags": ["IMAGE_NOT_FOUND"]},
                "A2_diagnosis": {"status": "nv", "score": 0.50, "flags": ["FALLBACK"], "all_probabilities": {c: 0.1428 for c in VALID_CODES}},
                "A3_lesion_analysis": {"status": "INDETERMINATE", "score": 0.50, "flags": ["MISSING_IMAGE"]},
                "A4_context_reasoning": {"status": "MODERATE", "score": 0.50, "flags": ["NO_CONTEXT"]},
                "A5_safety": {"status": "SAFETY_VETO", "score": 1.0, "flags": ["SAFETY_VETO", "FILE_ERROR"], "triggers": ["Missing file"]},
                "A6_coordinator": {"status": "REFER", "score": 0.50, "flags": ["FALLBACK_REFERRAL"], "policy_rule": "Missing image file referral"}
            }
        }
        return img_id, fallback_res, err_msg

    try:
        with model_lock:
            res = orchestrator.process_image(img_path, patient_context=context)

        # Standardize return dictionary
        diag_a2 = res.get("agents", {}).get("A2_diagnosis", {})
        all_probs = diag_a2.get("all_probabilities", {})

        pred = str(res.get("prediction", "nv")).strip()
        if pred not in VALID_CODES:
            pred = "nv"
        decision = str(res.get("decision", "REFER")).strip()
        if decision not in {"AUTO_TRIAGE", "REFER", "RE_IMAGE"}:
            decision = "REFER"

        clean_record = {
            "image_id": img_id,
            "prediction": pred,
            "decision": decision,
            "confidence": round(float(res.get("confidence", 0.5)), 4),
            "prediction_name": res.get("prediction_name", pred),
            "traceable_chain": res.get("traceable_chain", ""),
            "example_strip": res.get("example_strip", ""),
            "policy_rule": res.get("policy_rule", ""),
            "traceable_rationale": res.get("traceable_rationale", res.get("explanation", "")),
            "runtime_ms": res.get("runtime_ms", 0.0),
            "is_fallback": False,
            "error": None,
            "agents": res.get("agents", {}),
            "all_probabilities": all_probs
        }
        return img_id, clean_record, None

    except Exception as e:
        err_msg = f"Exception processing image {img_id}: {str(e)}"
        fallback_res = {
            "image_id": img_id,
            "prediction": "nv",
            "decision": "REFER",
            "confidence": 0.50,
            "prediction_name": "Melanocytic nevus",
            "traceable_chain": f"A1: ERROR | A2: nv • 0.50 | A3+A4: UNKNOWN | A5: SAFETY VETO | A6: REFER",
            "policy_rule": "Fallback: Processing exception triggers mandatory referral.",
            "traceable_rationale": f"Processing exception ({str(e)}). Conservative referral mandated.",
            "runtime_ms": 0.0,
            "is_fallback": True,
            "error": err_msg,
            "agents": {
                "A1_quality": {"status": "ERROR", "score": 0.0, "flags": ["PIPELINE_EXCEPTION"]},
                "A2_diagnosis": {"status": "nv", "score": 0.50, "flags": ["FALLBACK"], "all_probabilities": {c: 0.1428 for c in VALID_CODES}},
                "A3_lesion_analysis": {"status": "INDETERMINATE", "score": 0.50, "flags": ["EXCEPTION"]},
                "A4_context_reasoning": {"status": "MODERATE", "score": 0.50, "flags": ["EXCEPTION"]},
                "A5_safety": {"status": "SAFETY_VETO", "score": 1.0, "flags": ["SAFETY_VETO", "EXCEPTION"], "triggers": ["Exception"]},
                "A6_coordinator": {"status": "REFER", "score": 0.50, "flags": ["FALLBACK_REFERRAL"], "policy_rule": "Exception referral"}
            }
        }
        return img_id, fallback_res, err_msg

def main():
    args = parse_args()
    t_start = time.time()
    temp_dirs: List[str] = []

    print("=" * 70)
    print("  JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026")
    print("  Track 2: Healthcare | Agentic DermaGuard Hidden Test Pipeline")
    print("=" * 70)

    # 1. Read IDs list
    id_list = read_id_list(args.ids)
    total_ids = len(id_list)
    print(f"[INFO] Successfully loaded {total_ids} test image IDs from: {args.ids}")

    # 2. Index available images
    dl_dir = None
    if args.base_url:
        dl_dir = tempfile.mkdtemp(prefix="dermaguard_dl_")
        temp_dirs.append(dl_dir)

    img_index, base_search_dir = build_image_index(args.images, temp_dirs)

    # Resolve all IDs and build match table
    resolved_paths: Dict[str, Optional[str]] = {}
    found_count = 0
    missing_ids: List[str] = []

    for img_id in id_list:
        p = resolve_image(img_id, img_index, base_url=args.base_url, dl_dir=dl_dir)
        resolved_paths[img_id] = p
        if p:
            found_count += 1
        else:
            missing_ids.append(img_id)

    print("-" * 70)
    print("IMAGE RESOLUTION SUMMARY:")
    print(f"  Total IDs in CSV:    {total_ids}")
    print(f"  Images Found:        {found_count} ({(found_count/total_ids)*100.0:.1f}%)")
    print(f"  Images Not Found:    {len(missing_ids)} ({(len(missing_ids)/total_ids)*100.0:.1f}%)")
    if missing_ids:
        print(f"  Missing IDs sample:  {missing_ids[:5]} (Will use conservative REFER fallback)")
    print("-" * 70)

    # 3. Load metadata
    metadata_map = load_patient_metadata(args.metadata)
    if metadata_map:
        print(f"[INFO] Loaded patient metadata for {len(metadata_map)} entries.")
    else:
        print("[INFO] Running in no-context mode (A4 applies demographic baseline and neutral risk).")

    # Ensure output directories exist
    out_csv = args.out
    out_dir = os.path.dirname(os.path.abspath(out_csv))
    if not out_dir:
        out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    partial_jsonl = os.path.join(out_dir, "partial.jsonl")
    errors_csv = os.path.join(out_dir, "errors.csv")
    detailed_csv = os.path.join(out_dir, "detailed_report.csv")
    detailed_json = os.path.join(out_dir, "detailed_report.json")
    summary_json = os.path.join(out_dir, "summary.json")

    # Check resume
    results_map: Dict[str, Dict[str, Any]] = {}
    error_records: List[Dict[str, str]] = []

    if args.resume and os.path.exists(partial_jsonl):
        print(f"[INFO] --resume enabled. Loading existing partial progress from {partial_jsonl}...")
        try:
            with open(partial_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        if "image_id" in record:
                            results_map[record["image_id"]] = record
            print(f"[INFO] Resumed {len(results_map)} already processed images.")
        except Exception as e:
            print(f"[WARN] Failed to parse {partial_jsonl}: {e}")

    # Prepare list of pending IDs
    pending_ids = [iid for iid in id_list if iid not in results_map]
    print(f"[INFO] Remaining images to process: {len(pending_ids)} / {total_ids}")

    # 4. Load Orchestrator Model ONCE
    print("[INFO] Initializing Six-Agent Orchestrator (loading models into memory once)...")
    orchestrator = AgenticDermaGuard()
    model_lock = threading.Lock()
    print("[INFO] Model loaded successfully. Starting batch execution...")

    partial_file_lock = threading.Lock()
    processed_count = len(results_map)
    start_batch_time = time.time()

    def worker_task(target_id: str):
        path = resolved_paths.get(target_id)
        ctx = metadata_map.get(target_id.lower(), {})
        return process_single_image(target_id, path, ctx, orchestrator, model_lock)

    # Execute with ThreadPoolExecutor
    workers = max(1, min(args.workers, 8))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(worker_task, iid): iid for iid in pending_ids}

        for future in as_completed(future_map):
            target_id, res_dict, err = future.result()
            results_map[target_id] = res_dict
            processed_count += 1

            if err:
                error_records.append({
                    "image_id": target_id,
                    "error_message": err,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

            # Save incrementally to partial.jsonl
            with partial_file_lock:
                with open(partial_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res_dict) + "\n")

            # Update progress bar
            elapsed = time.time() - start_batch_time
            rate = (processed_count - len(id_list) + len(pending_ids)) / max(0.001, elapsed)
            remaining_seconds = (total_ids - processed_count) / max(0.001, rate)
            pct = (processed_count / total_ids) * 100.0

            status_str = f"[{processed_count}/{total_ids} ({pct:5.1f}%)] {target_id} -> {res_dict['prediction']} | {res_dict['decision']} (ETA: {format_eta(remaining_seconds)})"
            print(f"\r{status_str:<90}", end="", flush=True)

    print() # Newline after progress bar

    # 5. Compile outputs in exact original sample-CSV order
    ordered_results: List[Dict[str, Any]] = []
    submission_rows: List[Dict[str, str]] = []
    detailed_rows: List[Dict[str, Any]] = []

    decision_counts = {"AUTO_TRIAGE": 0, "REFER": 0, "RE_IMAGE": 0}
    prediction_counts = {c: 0 for c in VALID_CODES}
    fallback_count = 0

    for iid in id_list:
        rec = results_map.get(iid)
        if not rec:
            # Emergency safeguard fallback
            rec = {
                "image_id": iid,
                "prediction": "nv",
                "decision": "REFER",
                "confidence": 0.50,
                "is_fallback": True,
                "policy_rule": "Emergency missing row fallback",
                "traceable_chain": "A1: ERROR | A2: nv • 0.50 | A3+A4: UNKNOWN | A5: SAFETY VETO | A6: REFER",
                "traceable_rationale": "Missing execution result. Conservative referral mandated.",
                "runtime_ms": 0.0,
                "agents": {},
                "all_probabilities": {c: 0.1428 for c in VALID_CODES}
            }
            error_records.append({
                "image_id": iid,
                "error_message": "Missing execution result recovered by emergency fallback",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

        ordered_results.append(rec)
        submission_rows.append({
            "image_id": rec["image_id"],
            "prediction": rec["prediction"],
            "decision": rec["decision"]
        })

        if rec.get("is_fallback"):
            fallback_count += 1

        d = rec.get("decision", "REFER")
        decision_counts[d] = decision_counts.get(d, 0) + 1
        p = rec.get("prediction", "nv")
        prediction_counts[p] = prediction_counts.get(p, 0) + 1

        # Build detailed CSV row
        ag = rec.get("agents", {})
        a1 = ag.get("A1_quality", {})
        a2 = ag.get("A2_diagnosis", {})
        a3 = ag.get("A3_lesion_analysis", {})
        a4 = ag.get("A4_context_reasoning", {})
        a5 = ag.get("A5_safety", {})
        a6 = ag.get("A6_coordinator", {})

        probs = rec.get("all_probabilities", {}) or {}

        detailed_rows.append({
            "image_id": rec["image_id"],
            "prediction": rec["prediction"],
            "decision": rec["decision"],
            "confidence": rec.get("confidence", 0.5),
            "prob_mel": probs.get("mel", 0.0),
            "prob_nv": probs.get("nv", 0.0),
            "prob_bcc": probs.get("bcc", 0.0),
            "prob_akiec": probs.get("akiec", 0.0),
            "prob_bkl": probs.get("bkl", 0.0),
            "prob_df": probs.get("df", 0.0),
            "prob_vasc": probs.get("vasc", 0.0),
            "a1_status": a1.get("status", ""),
            "a1_score": a1.get("score", 0.0),
            "a1_flags": ";".join(a1.get("flags", [])),
            "a2_status": a2.get("status", ""),
            "a2_score": a2.get("score", 0.0),
            "a2_flags": ";".join(a2.get("flags", [])),
            "a3_status": a3.get("status", ""),
            "a3_score": a3.get("score", 0.0),
            "a3_flags": ";".join(a3.get("flags", [])),
            "a4_status": a4.get("status", ""),
            "a4_score": a4.get("score", 0.0),
            "a4_flags": ";".join(a4.get("flags", [])),
            "a5_status": a5.get("status", ""),
            "a5_score": a5.get("score", 0.0),
            "a5_flags": ";".join(a5.get("flags", [])),
            "a5_fired_rules": ";".join(a5.get("triggers", [])),
            "a6_policy_rule": rec.get("policy_rule", a6.get("policy_rule", "")),
            "trace_string": rec.get("traceable_chain", ""),
            "rationale": rec.get("traceable_rationale", ""),
            "runtime_ms": rec.get("runtime_ms", 0.0),
            "is_fallback": rec.get("is_fallback", False)
        })

    # Write submission CSV
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(out_csv, index=False)
    print(f"\n[OUTPUT] Wrote official submission CSV: {out_csv} ({len(sub_df)} rows)")

    # Write detailed CSV & JSON
    det_df = pd.DataFrame(detailed_rows)
    det_df.to_csv(detailed_csv, index=False)
    print(f"[OUTPUT] Wrote detailed report CSV: {detailed_csv}")

    with open(detailed_json, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=2)
    print(f"[OUTPUT] Wrote detailed audit JSON: {detailed_json}")

    # Write summary JSON
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": total_ids,
        "processed_images": len(ordered_results),
        "fallback_rows": fallback_count,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "decision_counts": decision_counts,
        "predicted_class_distribution": prediction_counts
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[OUTPUT] Wrote summary JSON: {summary_json}")

    # Write errors CSV
    if error_records:
        err_df = pd.DataFrame(error_records)
        err_df.to_csv(errors_csv, index=False)
        print(f"[OUTPUT] Logged {len(error_records)} image errors to: {errors_csv}")
    else:
        # Create empty errors.csv with header
        pd.DataFrame(columns=["image_id", "error_message", "timestamp"]).to_csv(errors_csv, index=False)
        print(f"[OUTPUT] Clean run! Created zero-error log: {errors_csv}")

    # Cleanup temp dirs
    for td in temp_dirs:
        try:
            import shutil
            shutil.rmtree(td, ignore_errors=True)
        except Exception:
            pass

    # 6. Automatic validation check
    print("\n" + "=" * 70)
    print("  RUNNING AUTOMATIC VALIDATOR ON SUBMISSION CSV")
    print("=" * 70)
    passed, errs, stats = validate_submission(out_csv, ids_path=args.ids)
    if passed:
        print("\n>>> VALIDATOR STATUS: PASS <<<")
        print(f"Total time elapsed: {time.time() - t_start:.2f}s")
        sys.exit(0)
    else:
        print("\n>>> VALIDATOR STATUS: FAIL <<<")
        for err in errs:
            print(f"  - {err}")
        sys.exit(1)

if __name__ == "__main__":
    main()
