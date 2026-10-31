#!/usr/bin/env python3
"""
scripts/validate_submission.py
Validates the final submission CSV against competition requirements:
- Header exactly: image_id,prediction,decision
- Exactly one row per ID (no missing, no duplicate IDs)
- If --ids is provided: IDs and ordering match the sample CSV exactly
- prediction in: {mel, nv, bcc, akiec, bkl, df, vasc}
- decision in: {AUTO_TRIAGE, REFER, RE_IMAGE}
- No blank or null values
- Non-zero exit code on failure, 0 on success
"""

import sys
import os
import argparse
import pandas as pd
from typing import List, Optional, Set, Tuple

VALID_PREDICTIONS: Set[str] = {"mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"}
VALID_DECISIONS: Set[str] = {"AUTO_TRIAGE", "REFER", "RE_IMAGE"}
EXPECTED_HEADER: List[str] = ["image_id", "prediction", "decision"]

def load_expected_ids(ids_path: str) -> List[str]:
    """Reads expected image IDs from a sample CSV or single-column file."""
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"IDs reference file not found: {ids_path}")
    
    # Try reading as CSV first
    try:
        df = pd.read_csv(ids_path, dtype=str)
        # Find column named image_id (case-insensitive)
        matched_col = None
        for col in df.columns:
            if col.strip().lower() == "image_id":
                matched_col = col
                break
        if matched_col:
            raw_ids = df[matched_col].dropna().astype(str).tolist()
            return [i.strip() for i in raw_ids if i.strip()]
        # Single column with header
        if len(df.columns) == 1:
            raw_ids = df.iloc[:, 0].dropna().astype(str).tolist()
            # If the header itself was the first ID
            first_val = str(df.columns[0]).strip()
            if first_val.lower() not in {"image_id", "id"}:
                return [first_val] + [i.strip() for i in raw_ids if i.strip()]
            return [i.strip() for i in raw_ids if i.strip()]
    except Exception:
        pass

    # Fallback to plain line-by-line reading
    with open(ids_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if lines and lines[0].lower() in {"image_id", "image_id,prediction,decision", "image_id,predicted_label,decision"}:
        lines = lines[1:]
    extracted = []
    for line in lines:
        parts = line.split(",")
        extracted.append(parts[0].strip())
    return extracted

def validate_submission(csv_path: str, ids_path: Optional[str] = None, expected_count: Optional[int] = None) -> Tuple[bool, List[str], dict]:
    errors: List[str] = []
    stats: dict = {
        "total_rows": 0,
        "decisions": {},
        "predictions": {},
        "passed": False
    }

    if not os.path.exists(csv_path):
        errors.append(f"Submission file does not exist: {csv_path}")
        return False, errors, stats

    # 1. Check exact raw header from line 1
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    
    actual_cols = [c.strip() for c in first_line.split(",")]
    if actual_cols != EXPECTED_HEADER:
        errors.append(f"Invalid header '{first_line}'. Expected exactly: {','.join(EXPECTED_HEADER)}")

    # 2. Parse CSV
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        errors.append(f"Failed to parse CSV file: {e}")
        return False, errors, stats

    stats["total_rows"] = len(df)

    if len(df) == 0:
        errors.append("Submission CSV is empty (0 rows).")
        return False, errors, stats

    # Verify column presence
    missing_cols = [col for col in EXPECTED_HEADER if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required column(s): {missing_cols}")
        return False, errors, stats

    # 3. Check for Nulls / Blanks
    null_counts = df[EXPECTED_HEADER].isnull().sum()
    for col in EXPECTED_HEADER:
        if null_counts[col] > 0:
            errors.append(f"Column '{col}' contains {null_counts[col]} null or blank values.")

    # 4. Check for duplicate image_ids in submission
    ids_series = df["image_id"].astype(str).str.strip()
    dup_ids = ids_series[ids_series.duplicated()].tolist()
    if dup_ids:
        errors.append(f"Found {len(dup_ids)} duplicate image IDs in submission: {dup_ids[:5]}")

    # 5. Check prediction values
    pred_vals = df["prediction"].astype(str).str.strip().tolist()
    invalid_preds = [p for p in pred_vals if p not in VALID_PREDICTIONS]
    if invalid_preds:
        errors.append(f"Found {len(invalid_preds)} invalid predictions (e.g. {invalid_preds[:5]}). Valid: {sorted(list(VALID_PREDICTIONS))}")

    # 6. Check decision values
    decision_vals = df["decision"].astype(str).str.strip().tolist()
    invalid_decisions = [d for d in decision_vals if d not in VALID_DECISIONS]
    if invalid_decisions:
        errors.append(f"Found {len(invalid_decisions)} invalid decisions (e.g. {invalid_decisions[:5]}). Valid: {sorted(list(VALID_DECISIONS))}")

    # 7. Check against reference IDs if provided
    if ids_path:
        try:
            expected_ids = load_expected_ids(ids_path)
            stats["expected_count"] = len(expected_ids)

            # Check counts
            if len(df) != len(expected_ids):
                errors.append(f"Row count mismatch: Submission has {len(df)} rows, expected {len(expected_ids)} rows.")

            # Check set equality
            sub_ids_list = ids_series.tolist()
            sub_ids_set = set(sub_ids_list)
            exp_ids_set = set(expected_ids)

            missing_from_sub = list(exp_ids_set - sub_ids_set)
            if missing_from_sub:
                errors.append(f"{len(missing_from_sub)} expected IDs are missing from submission (e.g. {missing_from_sub[:5]}).")

            unexpected_in_sub = list(sub_ids_set - exp_ids_set)
            if unexpected_in_sub:
                errors.append(f"{len(unexpected_in_sub)} unexpected IDs found in submission (e.g. {unexpected_in_sub[:5]}).")

            # Check exact order
            if len(sub_ids_list) == len(expected_ids) and sub_ids_list != expected_ids:
                errors.append("IDs are not in the exact same order as the sample submission CSV.")
        except Exception as e:
            errors.append(f"Error validating against expected IDs from {ids_path}: {e}")
    elif expected_count is not None:
        if len(df) != expected_count:
            errors.append(f"Row count mismatch: Found {len(df)} rows, expected exactly {expected_count}.")

    # Record distribution stats
    for d in sorted(list(VALID_DECISIONS)):
        stats["decisions"][d] = decision_vals.count(d)
    for p in sorted(list(VALID_PREDICTIONS)):
        stats["predictions"][p] = pred_vals.count(p)

    stats["passed"] = (len(errors) == 0)
    return stats["passed"], errors, stats

def main():
    parser = argparse.ArgumentParser(description="Validate Agentic DermaGuard Submission CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to submission CSV to validate")
    parser.add_argument("--ids", type=str, default=None, help="Path to sample CSV containing expected image IDs")
    parser.add_argument("--expected-count", type=int, default=None, help="Expected number of test rows")
    args = parser.parse_args()

    passed, errors, stats = validate_submission(args.csv, ids_path=args.ids, expected_count=args.expected_count)

    print("=" * 65)
    print("  AGENTIC DERMAGUARD - SUBMISSION CSV VALIDATION REPORT")
    print("=" * 65)
    print(f"Target File: {args.csv}")
    if args.ids:
        print(f"Reference IDs: {args.ids} ({stats.get('expected_count', '?')} IDs)")
    print(f"Total Rows:  {stats['total_rows']}")
    print("-" * 65)

    if passed:
        print("[PASS] Header exactly: image_id,prediction,decision")
        print("[PASS] Exactly one row per ID, no duplicates, no missing IDs")
        print("[PASS] Predictions all valid: mel, nv, bcc, akiec, bkl, df, vasc")
        print("[PASS] Decisions all valid: AUTO_TRIAGE, REFER, RE_IMAGE")
        if args.ids:
            print("[PASS] ID sequence matches sample CSV exactly")
        print("-" * 65)
        print("DECISION BREAKDOWN:")
        for dec, cnt in stats["decisions"].items():
            pct = (cnt / stats["total_rows"] * 100.0) if stats["total_rows"] else 0.0
            print(f"  {dec:<14}: {cnt:>5} ({pct:>5.1f}%)")
        print("\nPREDICTION BREAKDOWN:")
        for pred, cnt in stats["predictions"].items():
            pct = (cnt / stats["total_rows"] * 100.0) if stats["total_rows"] else 0.0
            print(f"  {pred:<14}: {cnt:>5} ({pct:>5.1f}%)")
        print("=" * 65)
        print("RESULT: >>> PASS <<<")
        sys.exit(0)
    else:
        print("RESULT: >>> FAIL <<<")
        print(f"Encountered {len(errors)} validation failure(s):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        print("=" * 65)
        sys.exit(1)

if __name__ == "__main__":
    main()
