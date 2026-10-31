#!/usr/bin/env python3
"""
scripts/make_mock_hidden.py
Generates 200 mock hidden test images and sample submission CSV from eval set.
Jaypee Agentic AI International Summit 2026 - Track 2: Healthcare

Usage:
  python scripts/make_mock_hidden.py [--count 200] [--seed 42] [--evaluate outputs/submission.csv]
"""

import os
import sys
import shutil
import random
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BENIGN_CLASSES = {"nv", "bkl", "df", "vasc"}
MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}
ALL_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]

def generate_mock_dataset(count: int = 200, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    eval_meta_candidates = [
        os.path.join(PROJECT_ROOT, "Health", "eval", "metadata.csv"),
        os.path.join(PROJECT_ROOT, "Health", "eval", "metadata (2).csv"),
    ]
    eval_meta_path = next((p for p in eval_meta_candidates if os.path.exists(p)), None)

    eval_img_dir = os.path.join(PROJECT_ROOT, "Health", "eval", "images")
    if not os.path.isdir(eval_img_dir):
        raise FileNotFoundError(f"Evaluation images folder not found: {eval_img_dir}")

    if not eval_meta_path:
        raise FileNotFoundError("Could not find Health/eval/metadata.csv")

    df = pd.read_csv(eval_meta_path)
    
    # Filter to only images that actually exist on disk
    valid_rows = []
    for _, row in df.iterrows():
        img_id = str(row["image_id"]).strip()
        img_file = os.path.join(eval_img_dir, f"{img_id}.jpg")
        if os.path.exists(img_file):
            valid_rows.append(row)

    valid_df = pd.DataFrame(valid_rows)
    print(f"[INFO] Found {len(valid_df)} verified images in {eval_img_dir}")

    # Stratified or random sample of count items
    if len(valid_df) < count:
        sample_df = valid_df.copy()
    else:
        sample_df = valid_df.sample(n=count, random_state=seed).reset_index(drop=True)

    # Output paths
    mock_dir = os.path.join(PROJECT_ROOT, "data", "mock_hidden")
    mock_img_dir = os.path.join(mock_dir, "images")
    os.makedirs(mock_img_dir, exist_ok=True)

    sample_csv_path = os.path.join(PROJECT_ROOT, "data", "mock_sample_submission.csv")
    gt_csv_path = os.path.join(mock_dir, "ground_truth.csv")

    # Copy images and strip any label metadata
    copied = 0
    id_list = []
    gt_rows = []

    for _, row in sample_df.iterrows():
        img_id = str(row["image_id"]).strip()
        src = os.path.join(eval_img_dir, f"{img_id}.jpg")
        dst = os.path.join(mock_img_dir, f"{img_id}.jpg")
        shutil.copy2(src, dst)
        copied += 1
        id_list.append(img_id)

        gt_rows.append({
            "image_id": img_id,
            "dx": str(row.get("dx", "unknown")).strip(),
            "age": row.get("age"),
            "sex": row.get("sex"),
            "localization": row.get("localization")
        })

    # 1. Produce mock sample submission CSV with ONLY image_id
    sample_sub_df = pd.DataFrame({"image_id": id_list})
    sample_sub_df.to_csv(sample_csv_path, index=False)

    # 2. Save ground truth for performance evaluation
    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(gt_csv_path, index=False)

    print("=" * 65)
    print("  MOCK HIDDEN DATASET GENERATED SUCCESSFULLY")
    print("=" * 65)
    print(f"Total Images Copied:   {copied}")
    print(f"Mock Images Folder:    {mock_img_dir}")
    print(f"Sample Submission CSV: {sample_csv_path} (Header: image_id)")
    print(f"Ground Truth CSV:      {gt_csv_path}")
    print("=" * 65)

    return sample_csv_path, mock_img_dir, gt_csv_path

def evaluate_predictions(submission_csv: str, gt_csv: str):
    """Computes accuracy, macro-F1, correct AUTO_TRIAGE and incorrect AUTO_TRIAGE."""
    if not os.path.exists(submission_csv):
        print(f"[ERROR] Submission CSV does not exist: {submission_csv}")
        return
    if not os.path.exists(gt_csv):
        print(f"[ERROR] Ground truth CSV does not exist: {gt_csv}")
        return

    sub_df = pd.read_csv(submission_csv)
    gt_df = pd.read_csv(gt_csv)

    merged = pd.merge(sub_df, gt_df, on="image_id", how="inner")
    total = len(merged)
    if total == 0:
        print("[ERROR] No overlapping image_ids between submission and ground truth.")
        return

    correct_preds = (merged["prediction"] == merged["dx"]).sum()
    accuracy = (correct_preds / total) * 100.0

    # Calculate Macro-F1 across 7 classes
    f1_list = []
    for cls in ALL_CLASSES:
        tp = ((merged["prediction"] == cls) & (merged["dx"] == cls)).sum()
        fp = ((merged["prediction"] == cls) & (merged["dx"] != cls)).sum()
        fn = ((merged["prediction"] != cls) & (merged["dx"] == cls)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
    macro_f1 = np.mean(f1_list)

    # AUTO_TRIAGE analysis
    # Correct AUTO_TRIAGE: system chose AUTO_TRIAGE on benign lesion and diagnosed it correctly
    correct_auto = ((merged["decision"] == "AUTO_TRIAGE") & (merged["dx"].isin(BENIGN_CLASSES)) & (merged["prediction"] == merged["dx"])).sum()
    
    # Incorrect / Unsafe AUTO_TRIAGE: chose AUTO_TRIAGE on a malignant lesion (critical safety breach!)
    unsafe_auto_malignant = ((merged["decision"] == "AUTO_TRIAGE") & (merged["dx"].isin(MALIGNANT_CLASSES))).sum()
    incorrect_auto_total = ((merged["decision"] == "AUTO_TRIAGE") & ((merged["dx"].isin(MALIGNANT_CLASSES)) | (merged["prediction"] != merged["dx"]))).sum()

    auto_triage_count = (merged["decision"] == "AUTO_TRIAGE").sum()
    refer_count = (merged["decision"] == "REFER").sum()
    reimage_count = (merged["decision"] == "RE_IMAGE").sum()

    print("=" * 65)
    print("  CLINICAL PERFORMANCE EVALUATION REPORT")
    print("=" * 65)
    print(f"Evaluated Cases:             {total}")
    print(f"Overall Classification Acc:  {accuracy:.2f}% ({correct_preds}/{total})")
    print(f"Macro-F1 (7 HAM10000 codes): {macro_f1:.4f}")
    print("-" * 65)
    print("TRIAGE SAFETY DISPOSITION:")
    print(f"  AUTO_TRIAGE Total:         {auto_triage_count} ({(auto_triage_count/total)*100.0:.1f}%)")
    print(f"  - Correct AUTO_TRIAGE:     {correct_auto} (Harmless benign discharged)")
    print(f"  - Unsafe Malignant AUTO:   {unsafe_auto_malignant} {'[CRITICAL SAFETY HAZARD]' if unsafe_auto_malignant > 0 else '[PERFECT 0.0% SENTINEL]'}")
    print(f"  - Incorrect AUTO Total:    {incorrect_auto_total}")
    print(f"  REFER Total:               {refer_count} ({(refer_count/total)*100.0:.1f}%)")
    print(f"  RE_IMAGE Total:            {reimage_count} ({(reimage_count/total)*100.0:.1f}%)")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="Generate and Evaluate Mock Hidden Test Dataset")
    parser.add_argument("--count", type=int, default=200, help="Number of images to sample (default 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--evaluate", type=str, default=None, help="Optional submission CSV to evaluate against mock ground truth")
    args = parser.parse_args()

    gt_path = os.path.join(PROJECT_ROOT, "data", "mock_hidden", "ground_truth.csv")
    if args.evaluate:
        evaluate_predictions(args.evaluate, gt_path)
    else:
        sample_csv, img_dir, gt_csv = generate_mock_dataset(count=args.count, seed=args.seed)

if __name__ == "__main__":
    main()
