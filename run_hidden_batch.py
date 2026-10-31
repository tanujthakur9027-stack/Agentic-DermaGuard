#!/usr/bin/env python3
"""
Agentic DermaGuard - Hidden 200-Test Batch Inference & Submission Generator
Jaypee Agentic AI International Summit 2026 - Track 2: Healthcare

Enforces all Summit Level 2 Hidden Test Rules:
- Header: image_id,predicted_label,decision
- 200 rows, one per hidden image_id (no duplicates, no omissions)
- predicted_label in {nv, mel, bkl, bcc, akiec, vasc, df}
- decision in {AUTO-TRIAGE, RE-IMAGE, REFER}
- Preserves full 6-agent evidence and rationale in outputs/hidden_200_audit.json
"""

import os
import sys
import json
import time
import shutil
import pandas as pd
from typing import Dict, Any, List
from orchestrator import agentic_dermaguard
from thresholds import load_thresholds

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))

def find_hidden_paths():
    """Locate the 200 hidden images and records.csv."""
    candidates_csv = [
        os.path.join(WORKSPACE_ROOT, "Health", "hidden", "records.csv"),
        r"C:\Users\Tanuj kumar singh\Desktop\image batch\Health\hidden\records.csv",
    ]
    candidates_img_dir = [
        os.path.join(WORKSPACE_ROOT, "Health", "hidden", "images"),
        r"C:\Users\Tanuj kumar singh\Desktop\image batch\Health\hidden\images",
    ]

    records_csv = None
    for p in candidates_csv:
        if os.path.exists(p):
            records_csv = p
            break

    images_dir = None
    for d in candidates_img_dir:
        if os.path.exists(d):
            images_dir = d
            break

    if not records_csv or not images_dir:
        raise FileNotFoundError(f"Could not locate hidden dataset! records_csv={records_csv}, images_dir={images_dir}")

    return records_csv, images_dir

def run_batch_evaluation():
    t_start = time.time()
    records_csv, images_dir = find_hidden_paths()

    print("=" * 70)
    print("  JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026")
    print("  Track 2: Healthcare | Agentic DermaGuard 200 Hidden Test Inference")
    print("=" * 70)
    print(f"[INFO] Records CSV: {records_csv}")
    print(f"[INFO] Images Dir:  {images_dir}")

    records_df = pd.read_csv(records_csv)
    total_records = len(records_df)
    print(f"[INFO] Total hidden test records: {total_records}")

    submission_rows: List[Dict[str, str]] = []
    full_audit_records: List[Dict[str, Any]] = []

    decision_counts = {"AUTO-TRIAGE": 0, "REFER": 0, "RE-IMAGE": 0}
    prediction_counts = {c: 0 for c in ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']}

    for idx, row in records_df.iterrows():
        img_id = str(row['image_id']).strip()
        img_filename = f"{img_id}.jpg"
        img_path = os.path.join(images_dir, img_filename)

        if not os.path.exists(img_path):
            # Check without extension or case variations
            alt_path = os.path.join(images_dir, img_id)
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                print(f"[WARNING] Image file not found for {img_id}: {img_path}")

        # Extract patient context (Agent 4 input)
        patient_context = {
            "age": float(row['age']) if pd.notnull(row.get('age')) else None,
            "sex": str(row['sex']).strip() if pd.notnull(row.get('sex')) else None,
            "localization": str(row['localization']).strip() if pd.notnull(row.get('localization')) else None
        }

        # Run 6-agent pipeline
        try:
            res = agentic_dermaguard.process_image(
                img_path,
                patient_context=patient_context
            )
        except Exception as e:
            print(f"[ERROR] Exception processing {img_id}: {e}")
            res = {
                "prediction": "nv",
                "final_decision": "REFER",
                "traceable_chain": f"A1: ERROR | A2: nv • 0.50 | A5: SAFETY VETO | A6: REFER",
                "traceable_rationale": f"System error caught ({str(e)}). Conservative fallback.",
                "policy_rule": "Fallback Rule",
                "agents": {
                    "A1_quality": {"status": "ERROR"},
                    "A2_diagnosis": {"predicted_code": "nv"},
                    "A3_lesion_analysis": {"status": "INDETERMINATE"},
                    "A4_context_reasoning": {"status": "MODERATE"},
                    "A5_safety": {"status": "SAFETY_VETO"},
                    "A6_coordinator": {"status": "REFER"}
                }
            }

        # Format decision with hyphen strictly matching Hackathon Portal rules:
        # AUTO-TRIAGE, RE-IMAGE, REFER
        raw_decision = res.get("final_decision") or res.get("decision") or "REFER"
        norm_decision = raw_decision.replace("_", "-").upper()
        if norm_decision not in {"AUTO-TRIAGE", "RE-IMAGE", "REFER"}:
            norm_decision = "REFER"

        # Format prediction
        raw_pred = res.get("prediction", "nv").lower().strip()
        if raw_pred not in {'mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'}:
            raw_pred = "nv"

        decision_counts[norm_decision] = decision_counts.get(norm_decision, 0) + 1
        prediction_counts[raw_pred] = prediction_counts.get(raw_pred, 0) + 1

        submission_rows.append({
            "image_id": img_id,
            "predicted_label": raw_pred,
            "decision": norm_decision
        })

        # Keep per-image full evidence for verification
        full_audit_records.append({
            "index": idx + 1,
            "image_id": img_id,
            "patient_context": patient_context,
            "predicted_label": raw_pred,
            "decision": norm_decision,
            "confidence": res.get("confidence", 0.5),
            "traceable_chain": res.get("traceable_chain", ""),
            "example_strip": res.get("example_strip", ""),
            "traceable_rationale": res.get("traceable_rationale", ""),
            "policy_rule": res.get("policy_rule", ""),
            "agents": res.get("agents", {})
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == total_records:
            print(f"  Processed {idx + 1:>3}/{total_records} [{((idx+1)/total_records)*100:.0f}%] | Current: {img_id} -> {raw_pred:<5} | {norm_decision}")

    total_time = round(time.time() - t_start, 2)
    avg_per_image = round((total_time / total_records) * 1000, 1)

    # 1. Write the official submission CSV
    sub_df = pd.DataFrame(submission_rows)
    
    # Save targets
    out_dir = os.path.join(WORKSPACE_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    primary_csv = os.path.join(out_dir, "health_submission.csv")
    sub_df.to_csv(primary_csv, index=False, columns=["image_id", "predicted_label", "decision"])
    print(f"\n[SAVED] Primary submission CSV -> {primary_csv}")

    # Also copy to user desktop and downloads for instant access
    copy_targets = [
        r"C:\Users\Tanuj kumar singh\Desktop\image batch\health_submission.csv",
        r"C:\Users\Tanuj kumar singh\Downloads\health_submission.csv",
        os.path.join(WORKSPACE_ROOT, "Health", "hidden", "health_submission.csv")
    ]
    for target in copy_targets:
        try:
            target_dir = os.path.dirname(target)
            if os.path.exists(target_dir):
                shutil.copy2(primary_csv, target)
                print(f"[COPIED] -> {target}")
        except Exception as e:
            print(f"[NOTE] Could not copy to {target}: {e}")

    # 2. Write full 6-agent audit trace
    audit_json = os.path.join(out_dir, "hidden_200_audit.json")
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
            "total_records": total_records,
            "total_runtime_seconds": total_time,
            "avg_latency_ms": avg_per_image,
            "decision_distribution": decision_counts,
            "prediction_distribution": prediction_counts,
            "records": full_audit_records
        }, f, indent=2)
    print(f"[SAVED] Full six-agent audit trace -> {audit_json}")

    # 3. Print Summary
    print("\n" + "=" * 70)
    print("  BATCH INFERENCE RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total images evaluated:  {total_records}")
    print(f"Total time elapsed:      {total_time}s ({avg_per_image} ms/image)")
    print("\nDECISION BREAKDOWN (Hyphenated Portal Standard):")
    for dec, count in decision_counts.items():
        pct = (count / total_records) * 100
        print(f"  {dec:<14}: {count:>4} ({pct:>5.1f}%)")

    print("\nPREDICTED LABEL BREAKDOWN:")
    for code, count in prediction_counts.items():
        pct = (count / total_records) * 100
        print(f"  {code:<14}: {count:>4} ({pct:>5.1f}%)")
    print("=" * 70)

    return primary_csv

if __name__ == "__main__":
    run_batch_evaluation()
