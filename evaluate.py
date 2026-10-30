import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, classification_report
)

from orchestrator import agentic_dermaguard, AgenticDermaGuard
from agents.diagnosis_agent import load_diagnosis_agent, IDX_TO_CODE, CODE_TO_IDX, CODE_TO_NAME

DX_TO_IDX = CODE_TO_IDX
IDX_TO_NAME = {
    0: 'Melanoma',
    1: 'Melanocytic nevus',
    2: 'Basal cell carcinoma',
    3: 'Actinic keratoses',
    4: 'Benign keratosis',
    5: 'Dermatofibroma',
    6: 'Vascular lesion'
}

def run_evaluation(
    eval_metadata: str = "Health/eval/metadata.csv",
    eval_images_dir: str = "Health/eval/images",
    max_samples: int = None
):
    print("=" * 70)
    print("  JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026")
    print("  Track 2: Healthcare | Agentic DermaGuard Level 2 Evaluation")
    print("  Evaluating Full Six-Agent System on Validation Dataset")
    print("=" * 70)
    start_time = time.time()

    # Load trained model weights if present
    if os.path.exists("models/diagnosis_agent.pt"):
        diag_model = load_diagnosis_agent("models/diagnosis_agent.pt")
        agentic_dermaguard.diagnosis_agent = diag_model
        print("[Eval] Loaded fine-tuned Diagnosis Agent from models/diagnosis_agent.pt")

    if not os.path.exists(eval_metadata):
        raise FileNotFoundError(f"Metadata file '{eval_metadata}' not found!")

    df = pd.read_csv(eval_metadata)
    if max_samples:
        df = df.iloc[:max_samples]

    results = []
    submission_rows = []

    print(f"[Eval] Processing {len(df)} validation images through Six-Agent pipeline...")

    for idx, row in df.iterrows():
        img_id = str(row['image_id']).strip()
        img_path = os.path.join(eval_images_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            continue

        patient_ctx = {
            "age": row.get("age"),
            "sex": row.get("sex"),
            "localization": row.get("localization")
        }

        # Process through Six-Agent Pipeline
        res = agentic_dermaguard.process_image(
            img_path,
            patient_context=patient_ctx
        )

        pred_code = res['prediction']
        decision = res['final_decision']
        confidence = res['confidence']
        true_dx = row['dx']

        pred_idx = CODE_TO_IDX.get(pred_code, 1)
        true_idx = CODE_TO_IDX.get(true_dx, 1)

        is_correct = (pred_code == true_dx)
        is_auto_triage = (decision == "AUTO_TRIAGE")
        auto_triage_correct = is_auto_triage and is_correct
        auto_triage_incorrect = is_auto_triage and (not is_correct)

        # Official CSV submission row
        submission_rows.append({
            'image_id': img_id,
            'prediction': pred_code,
            'decision': decision
        })

        # Detailed record for analysis & panel presentation
        agents = res['agents']
        a1 = agents['A1_quality']
        a3 = agents['A3_lesion_analysis']
        a4 = agents['A4_context_reasoning']
        a5 = agents['A5_safety']
        a6 = agents['A6_coordinator']

        results.append({
            'image_id': img_id,
            'ground_truth_dx': true_dx,
            'ground_truth_class': IDX_TO_NAME.get(true_idx, true_dx),
            'prediction': pred_code,
            'predicted_class': res['prediction_name'],
            'confidence': confidence,
            'is_correct': is_correct,
            'final_decision': decision,
            'auto_triage_correct': auto_triage_correct,
            'auto_triage_incorrect': auto_triage_incorrect,
            'traceable_chain': res['traceable_chain'],
            'policy_rule': a6.get('policy_rule', ''),
            'explanation': res['traceable_rationale'],
            # Agent 1 Quality
            'a1_quality_status': a1.get('status'),
            'a1_overall_quality': a1.get('overall_quality'),
            'a1_blur_score': a1.get('blur_score'),
            'a1_lighting_score': a1.get('lighting_score'),
            'a1_glare_score': a1.get('glare_score'),
            'a1_occlusion_score': a1.get('occlusion_score'),
            # Agent 3 Lesion Analysis
            'a3_suspicion_score': a3.get('suspicion_score'),
            'a3_suspicion_level': a3.get('suspicion_level'),
            # Agent 4 Context Reasoning
            'a4_risk_indicator': a4.get('risk_indicator'),
            'a4_risk_score': a4.get('risk_score'),
            'a4_missing_flags': " | ".join(a4.get('missing_field_flags', [])),
            # Agent 5 Safety
            'a5_safety_flag': a5.get('safety_flag'),
            'a5_veto': a5.get('veto_auto_triage'),
            'a5_trigger': a5.get('primary_trigger'),
            'a5_justification': a5.get('justification')
        })

        if (idx + 1) % 250 == 0 or (idx + 1) == len(df):
            elapsed = time.time() - start_time
            print(f"  Processed {idx + 1}/{len(df)} images ({elapsed:.1f}s)")

    res_df = pd.DataFrame(results)
    sub_df = pd.DataFrame(submission_rows)

    os.makedirs("outputs", exist_ok=True)

    # Save official submission CSV format
    sub_df.to_csv("outputs/final_submission_format_sample.csv", index=False)
    res_df.to_csv("outputs/validation_detailed.csv", index=False)

    # --- Metrics Computation ---
    y_true = [CODE_TO_IDX.get(d, 1) for d in res_df['ground_truth_dx']]
    y_pred = [CODE_TO_IDX.get(p, 1) for p in res_df['prediction']]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(7))).tolist()

    # Triage decision counts
    triage_counts = res_df['final_decision'].value_counts().to_dict()
    total_samples = len(res_df)

    correct_auto_triage = int(res_df['auto_triage_correct'].sum())
    incorrect_auto_triage = int(res_df['auto_triage_incorrect'].sum())
    num_refers = int((res_df['final_decision'] == 'REFER').sum())
    num_reimage = int((res_df['final_decision'] == 'RE_IMAGE').sum())

    # Per-class classification report
    target_names = [IDX_TO_NAME[i] for i in range(7)]
    cls_report = classification_report(y_true, y_pred, labels=list(range(7)), target_names=target_names, output_dict=True, zero_division=0)

    summary = {
        "event": "JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026",
        "track": "Track 2 | Healthcare",
        "project": "Agentic DermaGuard (Six-Agent System)",
        "stage": "Level 2 / Day 2 Multi-Agent Evaluation",
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_eval_samples": total_samples,
        "classification_metrics": {
            "overall_accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1_score": round(float(f1_macro), 4),
            "weighted_f1_score": round(float(f1_weighted), 4),
            "class_metrics": {
                name: {
                    "code": IDX_TO_CODE[i],
                    "precision": round(cls_report[name]["precision"], 4),
                    "recall": round(cls_report[name]["recall"], 4),
                    "f1_score": round(cls_report[name]["f1-score"], 4),
                    "support": int(cls_report[name]["support"])
                }
                for i, name in enumerate(target_names)
            },
            "confusion_matrix": cm
        },
        "live_demo_reporting_counts": {
            "correct_auto_triage": correct_auto_triage,
            "incorrect_auto_triage": incorrect_auto_triage,
            "number_of_refers": num_refers,
            "number_of_re_image": num_reimage,
            "triage_distribution": triage_counts
        },
        "six_agent_summary": {
            "A1_image_quality": {
                "mean_quality": round(float(res_df['a1_overall_quality'].mean()), 2),
                "mean_blur": round(float(res_df['a1_blur_score'].mean()), 2),
                "mean_lighting": round(float(res_df['a1_lighting_score'].mean()), 2),
                "mean_glare": round(float(res_df['a1_glare_score'].mean()), 2),
                "mean_occlusion": round(float(res_df['a1_occlusion_score'].mean()), 2),
                "status_breakdown": res_df['a1_quality_status'].value_counts().to_dict()
            },
            "A3_lesion_analysis": {
                "mean_suspicion_score": round(float(res_df['a3_suspicion_score'].mean()), 3),
                "suspicion_levels": res_df['a3_suspicion_level'].value_counts().to_dict()
            },
            "A4_context_reasoning": {
                "risk_levels": res_df['a4_risk_indicator'].value_counts().to_dict()
            },
            "A5_safety": {
                "safety_flags": res_df['a5_safety_flag'].value_counts().to_dict(),
                "vetoes_triggered": int(res_df['a5_veto'].sum())
            }
        }
    }

    with open("outputs/validation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate complete Markdown Summary
    md_content = f"""# Level 2 / Day 2 Evaluation Summary: Six-Agent DermaGuard

**Jaypee Agentic AI International Summit 2026 — Track 2 | Healthcare**
**Evaluation Date:** {time.strftime('%d October %Y %H:%M:%S')}

---

## 1. Final Live Demonstration Metrics (Required Reporting)

- **Total Images Evaluated:** {total_samples}
- **Overall Classification Accuracy:** {acc*100:.2f}%
- **Balanced Accuracy:** {bal_acc*100:.2f}%
- **Macro F1-Score:** {f1_macro:.4f}
- **Weighted F1-Score:** {f1_weighted:.4f}

### Triage Disposition Counts
| Triage Disposition | Count | Percentage | Description |
|---|---|---|---|
| **Correct AUTO_TRIAGE** | {correct_auto_triage} | {correct_auto_triage/total_samples*100:.1f}% | Automated screening matches ground truth label |
| **Incorrect AUTO_TRIAGE** | {incorrect_auto_triage} | {incorrect_auto_triage/total_samples*100:.1f}% | Automated screening misclassified |
| **Number of REFERS** | {num_refers} | {num_refers/total_samples*100:.1f}% | Suspicion, diagnostic uncertainty, or safety veto |
| **Number of RE_IMAGE** | {num_reimage} | {num_reimage/total_samples*100:.1f}% | Inadequate capture quality (blur/glare/centering) |

---

## 2. Per-Class Classification Performance (7 Permitted Codes)

| Code | Class Name | Precision | Recall (Sensitivity) | F1-Score | Support |
|---|---|---|---|---|---|
"""
    for i, name in enumerate(target_names):
        code = IDX_TO_CODE[i]
        stats = cls_report[name]
        md_content += f"| **{code}** | {name} | {stats['precision']*100:.2f}% | {stats['recall']*100:.2f}% | {stats['f1-score']:.4f} | {stats['support']} |\n"

    md_content += f"""
---

## 3. Six-Agent Contribution & Cross-Agent Safety Audit

- **Agent 1 (Image Quality):** Mean Quality: {res_df['a1_overall_quality'].mean():.2f}/100. Status: {res_df['a1_quality_status'].value_counts().to_dict()}
- **Agent 2 (Diagnosis):** Evaluated all {total_samples} images across all 7 permitted classes.
- **Agent 3 (Lesion Analysis ABCD):** Mean Suspicion Score: {res_df['a3_suspicion_score'].mean():.3f}.
- **Agent 4 (Reasoning / Context):** Stratified by age, sex, and anatomical localization.
- **Agent 5 (Safety Sentinel):** Vetoes Triggered: {int(res_df['a5_veto'].sum())} unsafe autonomous discharges blocked.
- **Agent 6 (Coordinator):** Applied deterministic precedence policy ensuring explainability and audit traceability.

---
*CSV submission generated at `outputs/final_submission_format_sample.csv` (Columns: `image_id,prediction,decision`).*
"""

    with open("outputs/validation_summary.md", "w") as f:
        f.write(md_content)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Level 2 Six-Agent Evaluation Complete!")
    print(f"  Accuracy: {acc*100:.2f}% | Balanced Accuracy: {bal_acc*100:.2f}% | Macro F1: {f1_macro:.4f}")
    print(f"  Correct AUTO_TRIAGE: {correct_auto_triage} | Incorrect: {incorrect_auto_triage}")
    print(f"  Number of REFERS: {num_refers} | Number of RE_IMAGE: {num_reimage}")
    print(f"  Saved submission sample: outputs/final_submission_format_sample.csv")
    print(f"  Saved validation summary: outputs/validation_summary.md")
    print("=" * 70)
    return summary

def predict_hidden_test(
    hidden_dir: str,
    output_csv: str = "outputs/submission.csv"
):
    """
    Inference tool for the 200 hidden-test images released at 10:00 AM by organizers.
    Generates exact 3-column CSV: image_id, prediction, decision
    """
    print(f"[Hidden Test] Running Six-Agent inference on images in: {hidden_dir}")
    if not os.path.exists(hidden_dir):
        print(f"[Error] Directory '{hidden_dir}' does not exist.")
        return

    # Find all jpg/png files
    files = [f for f in os.listdir(hidden_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[Hidden Test] Found {len(files)} test images.")

    rows = []
    for f in sorted(files):
        img_id = os.path.splitext(f)[0]
        img_path = os.path.join(hidden_dir, f)
        res = agentic_dermaguard.process_image(img_path)
        rows.append({
            'image_id': img_id,
            'prediction': res['prediction'],
            'decision': res['final_decision']
        })

    sub_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    sub_df.to_csv(output_csv, index=False)
    print(f"[Success] Saved hidden test submission to '{output_csv}' ({len(sub_df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic DermaGuard Level 2 Evaluation & Hidden Test Inference")
    parser.add_argument("--hidden-dir", type=str, default=None, help="Path to 200 hidden test images directory")
    parser.add_argument("--output-csv", type=str, default="outputs/submission.csv", help="Output path for final CSV submission")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick test run")
    args = parser.parse_args()

    if args.hidden_dir:
        predict_hidden_test(args.hidden_dir, args.output_csv)
    else:
        run_evaluation(max_samples=args.max_samples)
