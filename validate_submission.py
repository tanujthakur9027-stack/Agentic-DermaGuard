import os
import sys
import argparse
import pandas as pd

VALID_CODES = {'mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'}
VALID_ACTIONS = {'AUTO_TRIAGE', 'REFER', 'RE_IMAGE'}
EXPECTED_HEADER = ["image_id", "prediction", "decision"]

def validate_submission_file(csv_path: str, expected_count: int = None) -> bool:
    print("=" * 65)
    print("  JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026")
    print("  Track 2: Healthcare | Submission CSV Validation Check")
    print("=" * 65)
    print(f"Inspecting file: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"[FAIL] Submission file '{csv_path}' does not exist!")
        return False

    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    header_cols = [c.strip() for c in first_line.split(",")]
    if header_cols != EXPECTED_HEADER:
        print(f"[FAIL] Invalid CSV Header: {header_cols}")
        print(f"       Expected Header: {EXPECTED_HEADER}")
        return False
    print(f"[PASS] Header matches exact schema: {EXPECTED_HEADER}")

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"[INFO] Total submission rows: {total_rows}")

    if total_rows == 0:
        print("[FAIL] Submission CSV contains 0 data rows!")
        return False

    # Check missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"[FAIL] Null/NaN values found in submission:\n{null_counts}")
        return False
    print("[PASS] Zero null / NaN entries across all columns.")

    # Check duplicates in image_id
    duplicate_ids = df[df['image_id'].duplicated()]['image_id'].tolist()
    if duplicate_ids:
        print(f"[FAIL] Found {len(duplicate_ids)} duplicate image_id entries: {duplicate_ids[:5]} ...")
        return False
    print("[PASS] All image_ids are unique (no duplicate stems).")

    # Check prediction codes
    invalid_preds = df[~df['prediction'].isin(VALID_CODES)]
    if len(invalid_preds) > 0:
        print(f"[FAIL] Found {len(invalid_preds)} rows with invalid prediction codes:\n{invalid_preds.head()}")
        return False
    print(f"[PASS] All {total_rows} predictions belong to permitted 7 codes ({', '.join(sorted(VALID_CODES))}).")

    # Check decision actions
    invalid_decisions = df[~df['decision'].isin(VALID_ACTIONS)]
    if len(invalid_decisions) > 0:
        print(f"[FAIL] Found {len(invalid_decisions)} rows with invalid decision actions:\n{invalid_decisions.head()}")
        return False
    print(f"[PASS] All {total_rows} decisions belong to permitted 3 actions ({', '.join(sorted(VALID_ACTIONS))}).")

    # Check expected count if given
    if expected_count is not None and total_rows != expected_count:
        print(f"[FAIL] Row count mismatch: Found {total_rows}, expected exactly {expected_count}!")
        return False

    # Print distribution breakdown
    print("-" * 65)
    print("SUBMISSION DISPOSITION BREAKDOWN:")
    decision_counts = df['decision'].value_counts()
    for action in ['AUTO_TRIAGE', 'REFER', 'RE_IMAGE']:
        count = decision_counts.get(action, 0)
        pct = (count / total_rows) * 100
        print(f"  {action:<14}: {count:>5} ({pct:>5.1f}%)")

    print("\nPREDICTION CLASS BREAKDOWN:")
    pred_counts = df['prediction'].value_counts()
    for code in sorted(VALID_CODES):
        count = pred_counts.get(code, 0)
        pct = (count / total_rows) * 100
        print(f"  {code:<8}: {count:>5} ({pct:>5.1f}%)")

    print("=" * 65)
    print("[SUCCESS] Submission file verified and strictly compliant with summit rules!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Agentic DermaGuard Submission CSV")
    parser.add_argument("--csv", type=str, default="outputs/final_submission_format_sample.csv", help="Path to submission CSV")
    parser.add_argument("--expected-count", type=int, default=None, help="Expected number of test rows (e.g. 200)")
    args = parser.parse_args()

    success = validate_submission_file(args.csv, expected_count=args.expected_count)
    sys.exit(0 if success else 1)
