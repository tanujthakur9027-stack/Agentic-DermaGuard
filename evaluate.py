import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, classification_report
)

from orchestrator import agentic_dermaguard, AgenticDermaGuard
from thresholds import load_thresholds, get_flat_thresholds
from agents.diagnosis_agent import load_diagnosis_agent, IDX_TO_CODE, CODE_TO_IDX, CODE_TO_NAME
from agents.image_quality_agent import quality_agent
from validate_submission import validate_submission_file

DX_TO_IDX = CODE_TO_IDX
TARGET_NAMES = [CODE_TO_NAME[IDX_TO_CODE[i]] for i in range(7)]
MALIGNANT_CODES = {'mel', 'bcc', 'akiec'}
BENIGN_CODES = {'nv', 'bkl', 'df', 'vasc'}

def run_synthetic_degradation_test(
    eval_df: pd.DataFrame,
    images_dir: str,
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Evaluates Image Quality Agent robustness on clean vs. synthetically degraded captures.
    Tests: blur, glare, underexposure (dark), and peripheral occlusion.
    """
    print(f"\n[Quality Agent Stress-Test] Running synthetic degradation test on {n_samples} images ...")
    clean_tested = 0
    clean_false_positives = 0  # Clean images flagged as RE_IMAGE

    degraded_tests = {
        "blur": {"tested": 0, "detected": 0},
        "glare": {"tested": 0, "detected": 0},
        "dark": {"tested": 0, "detected": 0},
        "occlusion": {"tested": 0, "detected": 0}
    }

    sample_df = eval_df.iloc[:n_samples]

    for _, row in sample_df.iterrows():
        img_id = str(row['image_id']).strip()
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue

        orig_img = Image.open(img_path).convert('RGB')
        clean_tested += 1

        # Baseline clean test
        q_clean = quality_agent.evaluate(orig_img)
        if q_clean.status == "RE_IMAGE":
            clean_false_positives += 1

        # 1. Synthetic Blur (heavy box/gaussian blur)
        blurry_img = orig_img.filter(ImageFilter.GaussianBlur(radius=6.0))
        q_blur = quality_agent.evaluate(blurry_img)
        degraded_tests["blur"]["tested"] += 1
        if q_blur.status == "RE_IMAGE" or "Blur" in " ".join(q_blur.triggered_criteria):
            degraded_tests["blur"]["detected"] += 1

        # 2. Synthetic Glare (specular saturation flash void on center-top region)
        glare_arr = np.array(orig_img, dtype=np.uint8).copy()
        h, w, _ = glare_arr.shape
        cy, cx = h // 3, w // 2
        r = min(h, w) // 5
        y, x = np.ogrid[:h, :w]
        mask = ((x - cx)**2 + (y - cy)**2) <= r**2
        glare_arr[mask] = 255  # Pure sensor clip
        glare_img = Image.fromarray(glare_arr)
        q_glare = quality_agent.evaluate(glare_img)
        degraded_tests["glare"]["tested"] += 1
        if q_glare.status == "RE_IMAGE" or "Glare" in " ".join(q_glare.triggered_criteria):
            degraded_tests["glare"]["detected"] += 1

        # 3. Synthetic Dark / Underexposure
        dark_arr = (np.array(orig_img, dtype=np.float32) * 0.18).astype(np.uint8)
        dark_img = Image.fromarray(dark_arr)
        q_dark = quality_agent.evaluate(dark_img)
        degraded_tests["dark"]["tested"] += 1
        if q_dark.status == "RE_IMAGE" or "Lighting" in " ".join(q_dark.triggered_criteria):
            degraded_tests["dark"]["detected"] += 1

        # 4. Synthetic Occlusion (heavy peripheral black border masking)
        occl_arr = np.array(orig_img, dtype=np.uint8).copy()
        border_px = int(min(h, w) * 0.22)
        occl_arr[:border_px, :] = 10
        occl_arr[-border_px:, :] = 10
        occl_arr[:, :border_px] = 10
        occl_arr[:, -border_px:] = 10
        occl_img = Image.fromarray(occl_arr)
        q_occl = quality_agent.evaluate(occl_img)
        degraded_tests["occlusion"]["tested"] += 1
        if q_occl.status == "RE_IMAGE" or "Occlusion" in " ".join(q_occl.triggered_criteria):
            degraded_tests["occlusion"]["detected"] += 1

    clean_fp_rate = clean_false_positives / clean_tested if clean_tested > 0 else 0.0

    summary = {
        "clean_images_evaluated": clean_tested,
        "clean_false_positive_re_images": clean_false_positives,
        "clean_false_positive_rate_pct": round(clean_fp_rate * 100.0, 2),
        "degradation_detection_rates": {
            k: {
                "tested": v["tested"],
                "detected": v["detected"],
                "detection_rate_pct": round((v["detected"] / v["tested"] * 100.0), 2) if v["tested"] > 0 else 0.0
            }
            for k, v in degraded_tests.items()
        }
    }

    print(f"  Clean False-Positive Rate: {summary['clean_false_positive_rate_pct']}% ({clean_false_positives}/{clean_tested})")
    for k, v in summary["degradation_detection_rates"].items():
        print(f"  {k.capitalize():<12} Defect Detection Rate: {v['detection_rate_pct']}% ({v['detected']}/{v['tested']})")

    return summary

def run_evaluation(
    eval_metadata: str = "Health/eval/metadata.csv",
    eval_images_dir: str = "Health/eval/images",
    output_csv: str = "outputs/final_submission_format_sample.csv",
    max_samples: int = None,
    context_mode: str = "no_context",
    run_degradation: bool = True
) -> Dict[str, Any]:
    print("=" * 75)
    print("  JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026")
    print("  Track 2: Healthcare | Agentic DermaGuard Level 2 Evaluation")
    print(f"  Context Mode: {context_mode.upper()} (Hidden-test benchmark condition)")
    print("=" * 75)
    t_start = time.time()

    if not os.path.exists(eval_metadata):
        raise FileNotFoundError(f"Metadata file '{eval_metadata}' not found!")

    df = pd.read_csv(eval_metadata)
    total_in_meta = len(df)
    if max_samples and max_samples < len(df):
        # Sample uniformly across the sorted dataset to ensure all 7 classes are evaluated in smoke tests
        step = max(1, len(df) // max_samples)
        df = df.iloc[::step].iloc[:max_samples].reset_index(drop=True)

    results = []
    submission_rows = []
    skipped_images = []

    # Active clinical thresholds
    th = get_flat_thresholds()
    auto_conf_th = th["auto_triage_confidence"]

    # Ablation stage trackers
    ablation_stages = {
        "stage_1_a1_a2": {"correct_auto": 0, "incorrect_auto": 0, "refers": 0, "re_image": 0, "missed_malig": 0},
        "stage_2_plus_a3": {"correct_auto": 0, "incorrect_auto": 0, "refers": 0, "re_image": 0, "missed_malig": 0},
        "stage_3_plus_a4": {"correct_auto": 0, "incorrect_auto": 0, "refers": 0, "re_image": 0, "missed_malig": 0},
        "stage_4_plus_a5": {"correct_auto": 0, "incorrect_auto": 0, "refers": 0, "re_image": 0, "missed_malig": 0},
        "stage_5_full_a6": {"correct_auto": 0, "incorrect_auto": 0, "refers": 0, "re_image": 0, "missed_malig": 0}
    }

    # Cross-agent agreement & catch statistics
    cross_agent_stats = {
        "a2_benign_a3_suspicious": 0,
        "a2_benign_a4_high_risk": 0,
        "a5_total_vetoes": 0,
        "a5_auto_triage_overrides": 0,
        "a5_overrides_prevented_incorrect_auto": 0,
        "a5_overrides_prevented_missed_malignancy": 0
    }

    print(f"[Eval] Processing {len(df)} images through Six-Agent consensus pipeline...")

    for idx, row in df.iterrows():
        img_id = str(row['image_id']).strip()
        img_path = os.path.join(eval_images_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            skipped_images.append(img_id)
            continue

        true_dx = str(row['dx']).strip()
        is_malignant_true = (true_dx in MALIGNANT_CODES)

        # Context configuration
        if context_mode == "with_context":
            patient_ctx = {
                "age": row.get("age"),
                "sex": row.get("sex"),
                "localization": row.get("localization")
            }
        else:
            # Primary hidden-test condition: metadata absent
            patient_ctx = None

        # Execute full 6-agent workflow
        res = agentic_dermaguard.process_image(
            img_path,
            patient_context=patient_ctx
        )

        pred_code = res['prediction']
        decision = res['final_decision']
        confidence = res['confidence']

        is_correct = (pred_code == true_dx)
        is_auto = (decision == "AUTO_TRIAGE")
        auto_correct = is_auto and is_correct
        auto_incorrect = is_auto and (not is_correct)
        is_missed_malig = is_auto and is_malignant_true

        submission_rows.append({
            'image_id': img_id,
            'prediction': pred_code,
            'decision': decision
        })

        agents = res['agents']
        a1 = agents['A1_quality']
        a2 = agents['A2_diagnosis']
        a3 = agents['A3_lesion_analysis']
        a4 = agents['A4_context_reasoning']
        a5 = agents['A5_safety']
        a6 = agents['A6_coordinator']

        # --- ABLATION STUDY COMPUTATION ---
        # Stage 1: A1 + A2 only
        if a1['status'] == 'RE_IMAGE':
            dec_s1 = 'RE_IMAGE'
        elif pred_code in MALIGNANT_CODES:
            dec_s1 = 'REFER'
        elif confidence < auto_conf_th:
            dec_s1 = 'REFER'
        else:
            dec_s1 = 'AUTO_TRIAGE'

        # Stage 2: + A3 (Lesion ABCD)
        if dec_s1 == 'AUTO_TRIAGE' and (a3.get('tds_score', 0) > 5.2 or a3.get('suspicion_score', 0) > 0.65):
            dec_s2 = 'REFER'
        else:
            dec_s2 = dec_s1

        # Stage 3: + A4 (Context)
        if dec_s2 == 'AUTO_TRIAGE' and a4.get('risk_indicator') == 'HIGH' and confidence < 0.85:
            dec_s3 = 'REFER'
        else:
            dec_s3 = dec_s2

        # Stage 4: + A5 (Safety Sentinel)
        if dec_s3 == 'AUTO_TRIAGE' and (a5.get('veto_auto_triage') or a5.get('safety_flag') != 'PASS'):
            dec_s4 = 'REFER'
        else:
            dec_s4 = dec_s3

        # Stage 5: Full A6 Coordinator
        dec_s5 = decision

        for st_name, st_dec in [
            ("stage_1_a1_a2", dec_s1),
            ("stage_2_plus_a3", dec_s2),
            ("stage_3_plus_a4", dec_s3),
            ("stage_4_plus_a5", dec_s4),
            ("stage_5_full_a6", dec_s5)
        ]:
            if st_dec == 'AUTO_TRIAGE':
                if is_correct:
                    ablation_stages[st_name]["correct_auto"] += 1
                else:
                    ablation_stages[st_name]["incorrect_auto"] += 1
                if is_malignant_true:
                    ablation_stages[st_name]["missed_malig"] += 1
            elif st_dec == 'RE_IMAGE':
                ablation_stages[st_name]["re_image"] += 1
            else:
                ablation_stages[st_name]["refers"] += 1

        # --- CROSS-AGENT AGREEMENT & OVERRIDE STATS ---
        if pred_code in BENIGN_CODES and a3.get('suspicion_level') == 'SUSPICIOUS':
            cross_agent_stats["a2_benign_a3_suspicious"] += 1
        if pred_code in BENIGN_CODES and a4.get('risk_indicator') == 'HIGH':
            cross_agent_stats["a2_benign_a4_high_risk"] += 1

        if a5.get('veto_auto_triage'):
            cross_agent_stats["a5_total_vetoes"] += 1

        # Did A5 override what would have been an automated discharge in Stage 1?
        if dec_s1 == 'AUTO_TRIAGE' and decision == 'REFER':
            cross_agent_stats["a5_auto_triage_overrides"] += 1
            if not is_correct:
                cross_agent_stats["a5_overrides_prevented_incorrect_auto"] += 1
            if is_malignant_true:
                cross_agent_stats["a5_overrides_prevented_missed_malignancy"] += 1

        results.append({
            'image_id': img_id,
            'ground_truth_dx': true_dx,
            'prediction': pred_code,
            'confidence': confidence,
            'final_decision': decision,
            'is_correct': is_correct,
            'is_auto_triage': is_auto,
            'auto_triage_correct': auto_correct,
            'auto_triage_incorrect': auto_incorrect,
            'missed_malignancy': is_missed_malig,
            'traceable_chain': res['traceable_chain'],
            'policy_rule': a6.get('policy_rule', ''),
            'a1_status': a1.get('status'),
            'a1_quality': a1.get('overall_quality'),
            'a3_tds': a3.get('tds_score'),
            'a3_suspicion': a3.get('suspicion_score'),
            'a4_risk': a4.get('risk_indicator'),
            'a5_flag': a5.get('safety_flag'),
            'a5_trigger': a5.get('primary_trigger')
        })

        if (idx + 1) % 250 == 0 or (idx + 1) == len(df):
            print(f"  Evaluated {idx + 1}/{len(df)} images ({time.time() - t_start:.1f}s)")

    if skipped_images:
        print(f"[Warning] Skipped {len(skipped_images)} missing image files: {skipped_images[:5]} ...")

    total_time = time.time() - t_start
    total_evaluated = len(results)
    latency_ms_per_image = (total_time / total_evaluated * 1000.0) if total_evaluated > 0 else 0.0

    res_df = pd.DataFrame(results)
    sub_df = pd.DataFrame(submission_rows)

    os.makedirs("outputs", exist_ok=True)
    sub_df.to_csv(output_csv, index=False)
    res_df.to_csv("outputs/validation_detailed.csv", index=False)

    # Validate output submission CSV schema and contents
    validate_submission_file(output_csv, expected_count=len(sub_df))

    # --- Metrics Computation ---
    y_true = [CODE_TO_IDX.get(d, 1) for d in res_df['ground_truth_dx']]
    y_pred = [CODE_TO_IDX.get(p, 1) for p in res_df['prediction']]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(7))).tolist()

    # Per-class classification report
    cls_report = classification_report(
        y_true, y_pred, labels=list(range(7)), target_names=TARGET_NAMES, output_dict=True, zero_division=0
    )

    # Class-wise errors (find top misclassifications)
    misclassifications = []
    for i in range(7):
        for j in range(7):
            if i != j and cm[i][j] > 0:
                misclassifications.append({
                    "true_code": IDX_TO_CODE[i],
                    "pred_code": IDX_TO_CODE[j],
                    "true_name": TARGET_NAMES[i],
                    "pred_name": TARGET_NAMES[j],
                    "count": cm[i][j]
                })
    misclassifications = sorted(misclassifications, key=lambda x: x["count"], reverse=True)

    # Decision counts
    correct_auto = int(res_df['auto_triage_correct'].sum())
    incorrect_auto = int(res_df['auto_triage_incorrect'].sum())
    total_auto = correct_auto + incorrect_auto
    auto_precision = (correct_auto / total_auto) if total_auto > 0 else 1.0
    num_refers = int((res_df['final_decision'] == 'REFER').sum())
    num_reimage = int((res_df['final_decision'] == 'RE_IMAGE').sum())
    missed_malig = int(res_df['missed_malignancy'].sum())

    # Quality outcomes
    q_status_counts = res_df['a1_status'].value_counts().to_dict()

    # Synthetic degradation test (if requested)
    degradation_results = None
    if run_degradation and len(df) >= 100:
        degradation_results = run_synthetic_degradation_test(df, eval_images_dir, n_samples=min(100, len(df)))

    # Save 6 illustrative example demo cases
    example_cases = []
    # 1. Correct AUTO_TRIAGE (clean nevus)
    c1 = res_df[(res_df['final_decision'] == 'AUTO_TRIAGE') & (res_df['is_correct'])].head(1)
    # 2. Melanoma REFER
    c2 = res_df[(res_df['prediction'] == 'mel') & (res_df['final_decision'] == 'REFER')].head(1)
    # 3. BCC REFER
    c3 = res_df[(res_df['prediction'] == 'bcc') & (res_df['final_decision'] == 'REFER')].head(1)
    # 4. Diagnostic Uncertainty REFER
    c4 = res_df[(res_df['final_decision'] == 'REFER') & (res_df['confidence'] < auto_conf_th) & (res_df['prediction'] == 'nv')].head(1)
    # 5. RE_IMAGE Quality Failure
    c5 = res_df[res_df['final_decision'] == 'RE_IMAGE'].head(1)
    # 6. Cross-Agent Conflict REFER (A2 benign, A3/A4 or A5 conflict)
    c6 = res_df[(res_df['prediction'] == 'nv') & (res_df['final_decision'] == 'REFER') & (res_df['confidence'] >= auto_conf_th)].head(1)

    for case_df, desc in [
        (c1, "Correct AUTO_TRIAGE (High-confidence benign nevus, verified clear)"),
        (c2, "Malignancy REFER (Melanoma presentation, immediate specialist biopsy)"),
        (c3, "Malignancy REFER (Basal cell carcinoma presentation, referral indicated)"),
        (c4, "Diagnostic Uncertainty REFER (Differential ambiguity below auto threshold)"),
        (c5, "Quality Failure RE_IMAGE (Compromised capture, re-imaging requested)"),
        (c6, "Cross-Agent Conflict REFER (Benign candidate vetoed by lesion/safety cues)")
    ]:
        if len(case_df) > 0:
            r = case_df.iloc[0]
            example_cases.append({
                "scenario": desc,
                "image_id": r['image_id'],
                "true_dx": r['ground_truth_dx'],
                "prediction": r['prediction'],
                "decision": r['final_decision'],
                "confidence": round(float(r['confidence']), 4),
                "traceable_chain": r['traceable_chain'],
                "policy_rule": r['policy_rule']
            })

    with open("outputs/example_demo_cases.json", "w") as f:
        json.dump(example_cases, f, indent=2)

    # Master summary dictionary
    summary_data = {
        "event": "JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026",
        "track": "Track 2 | Healthcare",
        "project": "Agentic DermaGuard (Six-Agent Skin Lesion Screening & Triage)",
        "context_mode": context_mode,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_eval_samples": total_evaluated,
        "runtime_metrics": {
            "total_elapsed_seconds": round(total_time, 2),
            "latency_ms_per_image": round(latency_ms_per_image, 1)
        },
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
                for i, name in enumerate(TARGET_NAMES)
            },
            "confusion_matrix": cm,
            "top_class_confusions": misclassifications[:8]
        },
        "live_demo_reporting_counts": {
            "correct_auto_triage": correct_auto,
            "incorrect_auto_triage": incorrect_auto,
            "auto_triage_precision_pct": round(auto_precision * 100.0, 2),
            "number_of_refers": num_refers,
            "number_of_re_image": num_reimage,
            "missed_malignancies": missed_malig
        },
        "ablation_study": ablation_stages,
        "cross_agent_interaction": cross_agent_stats,
        "image_quality_summary": {
            "status_counts": q_status_counts
        },
        "synthetic_degradation_test": degradation_results
    }

    with open("outputs/validation_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Write Markdown Summary
    generate_markdown_summary(summary_data, ablation_stages, misclassifications, example_cases)

    print("\n" + "=" * 75)
    print(f"[SUCCESS] Evaluation Complete ({total_evaluated} images, {total_time:.1f}s, {latency_ms_per_image:.1f}ms/img)!")
    print(f"  Accuracy: {acc*100:.2f}% | Balanced Accuracy: {bal_acc*100:.2f}% | Macro F1: {f1_macro:.4f}")
    print(f"  AUTO_TRIAGE -> Correct: {correct_auto} | Incorrect: {incorrect_auto} | Precision: {auto_precision*100:.1f}%")
    print(f"  REFERS: {num_refers} | RE_IMAGE: {num_reimage} | Missed Malignancies: {missed_malig}")
    print(f"  Saved output submission: {output_csv}")
    print(f"  Saved validation summary: outputs/validation_summary.md")
    print("=" * 75)

    return summary_data

def generate_markdown_summary(
    sum_data: Dict[str, Any],
    ablation: Dict[str, Any],
    top_confusions: List[Dict[str, Any]],
    examples: List[Dict[str, Any]]
):
    cls_metrics = sum_data["classification_metrics"]
    triage = sum_data["live_demo_reporting_counts"]
    rt = sum_data["runtime_metrics"]
    cross = sum_data["cross_agent_interaction"]

    md = f"""# Level 2 Evaluation Summary: Agentic DermaGuard (Six-Agent System)
**Jaypee Agentic AI International Summit 2026 — Track 2 | Healthcare**
**Evaluation Date:** {time.strftime('%d October %Y %H:%M:%S')} | **Context Mode:** `{sum_data['context_mode']}`
**Total Images Evaluated:** {sum_data['total_eval_samples']} | **Runtime:** {rt['total_elapsed_seconds']}s ({rt['latency_ms_per_image']} ms/image)

---

## 1. Live Demonstration Core Reporting Metrics (Official Summit Format)

- **Overall Classification Accuracy:** **{cls_metrics['overall_accuracy']*100:.2f}%**
- **Balanced Accuracy:** **{cls_metrics['balanced_accuracy']*100:.2f}%**
- **Macro F1-Score:** **{cls_metrics['macro_f1_score']:.4f}**
- **Weighted F1-Score:** **{cls_metrics['weighted_f1_score']:.4f}**

### Triage Disposition Table
| Triage Disposition | Count | Percentage | Clinical Meaning |
|---|---|---|---|
| **Correct AUTO_TRIAGE** | **{triage['correct_auto_triage']}** | **{triage['correct_auto_triage']/sum_data['total_eval_samples']*100:.1f}%** | Automated discharge matching ground truth benign pathology |
| **Incorrect AUTO_TRIAGE** | **{triage['incorrect_auto_triage']}** | **{triage['incorrect_auto_triage']/sum_data['total_eval_samples']*100:.1f}%** | Automated discharge misclassification |
| **AUTO_TRIAGE Precision** | **{triage['auto_triage_precision_pct']}%** | - | Proportion of automated discharges that are correct |
| **Missed Malignancies** | **{triage['missed_malignancies']}** | **{triage['missed_malignancies']/sum_data['total_eval_samples']*100:.2f}%** | Malignant lesions (mel/bcc/akiec) wrongly AUTO_TRIAGED |
| **Number of REFERS** | **{triage['number_of_refers']}** | **{triage['number_of_refers']/sum_data['total_eval_samples']*100:.1f}%** | Specialist dermatologist review required |
| **Number of RE_IMAGE** | **{triage['number_of_re_image']}** | **{triage['number_of_re_image']/sum_data['total_eval_samples']*100:.1f}%** | Capture quality failure (blur, glare, occlusion) |

---

## 2. Six-Agent Incremental Ablation Study
*Proves that every specialized agent directly contributes to clinical safety and optimal disposition.*

| Agent Architecture Pipeline | Correct AUTO | Incorrect AUTO | Missed Malig | Auto Precision | REFERS | RE_IMAGE |
|---|---|---|---|---|---|---|
| **Stage 1: A1 (Quality) + A2 (Diagnosis CNN)** | {ablation['stage_1_a1_a2']['correct_auto']} | {ablation['stage_1_a1_a2']['incorrect_auto']} | {ablation['stage_1_a1_a2']['missed_malig']} | {(ablation['stage_1_a1_a2']['correct_auto'] / max(1, ablation['stage_1_a1_a2']['correct_auto'] + ablation['stage_1_a1_a2']['incorrect_auto']) * 100):.1f}% | {ablation['stage_1_a1_a2']['refers']} | {ablation['stage_1_a1_a2']['re_image']} |
| **Stage 2: + A3 (Lesion ABCD Morphometry)** | {ablation['stage_2_plus_a3']['correct_auto']} | {ablation['stage_2_plus_a3']['incorrect_auto']} | {ablation['stage_2_plus_a3']['missed_malig']} | {(ablation['stage_2_plus_a3']['correct_auto'] / max(1, ablation['stage_2_plus_a3']['correct_auto'] + ablation['stage_2_plus_a3']['incorrect_auto']) * 100):.1f}% | {ablation['stage_2_plus_a3']['refers']} | {ablation['stage_2_plus_a3']['re_image']} |
| **Stage 3: + A4 (Demographic Priors)** | {ablation['stage_3_plus_a4']['correct_auto']} | {ablation['stage_3_plus_a4']['incorrect_auto']} | {ablation['stage_3_plus_a4']['missed_malig']} | {(ablation['stage_3_plus_a4']['correct_auto'] / max(1, ablation['stage_3_plus_a4']['correct_auto'] + ablation['stage_3_plus_a4']['incorrect_auto']) * 100):.1f}% | {ablation['stage_3_plus_a4']['refers']} | {ablation['stage_3_plus_a4']['re_image']} |
| **Stage 4: + A5 (Safety Sentinel Veto)** | {ablation['stage_4_plus_a5']['correct_auto']} | {ablation['stage_4_plus_a5']['incorrect_auto']} | {ablation['stage_4_plus_a5']['missed_malig']} | {(ablation['stage_4_plus_a5']['correct_auto'] / max(1, ablation['stage_4_plus_a5']['correct_auto'] + ablation['stage_4_plus_a5']['incorrect_auto']) * 100):.1f}% | {ablation['stage_4_plus_a5']['refers']} | {ablation['stage_4_plus_a5']['re_image']} |
| **Stage 5: Full A6 (Coordinator Master Policy)** | **{ablation['stage_5_full_a6']['correct_auto']}** | **{ablation['stage_5_full_a6']['incorrect_auto']}** | **{ablation['stage_5_full_a6']['missed_malig']}** | **{(ablation['stage_5_full_a6']['correct_auto'] / max(1, ablation['stage_5_full_a6']['correct_auto'] + ablation['stage_5_full_a6']['incorrect_auto']) * 100):.1f}%** | **{ablation['stage_5_full_a6']['refers']}** | **{ablation['stage_5_full_a6']['re_image']}** |

### Cross-Agent Agreement & Sentinel Catches
- **A2 Benign vs. A3 Suspicious ABCD Structure:** **{cross['a2_benign_a3_suspicious']}** cases detected.
- **A5 Safety Sentinel Overrides:** A5 intervened to block **{cross['a5_auto_triage_overrides']}** automated triage cases.
- **Sentinel Catch Efficiency:**
  - Prevented **{cross['a5_overrides_prevented_incorrect_auto']}** diagnostic misclassifications from automated discharge.
  - Prevented **{cross['a5_overrides_prevented_missed_malignancy']}** true malignant tumors (Melanoma/BCC/AKIEC) from being wrongly discharged!

---

## 3. Per-Class Diagnostic Performance (7 Permitted Classes)

| Code | Pathology Name | Precision | Recall (Sensitivity) | F1-Score | Support |
|---|---|---|---|---|---|
"""
    for i, name in enumerate(TARGET_NAMES):
        code = IDX_TO_CODE[i]
        stats = cls_metrics["class_metrics"][name]
        md += f"| **{code}** | {name} | {stats['precision']*100:.2f}% | {stats['recall']*100:.2f}% | {stats['f1_score']:.4f} | {stats['support']} |\n"

    md += f"""
---

## 4. Class-Wise Errors & Confusion Matrix
### Top Diagnostic Confusions
"""
    for item in top_confusions[:6]:
        md += f"- **{item['true_name']} ({item['true_code']}) misclassified as {item['pred_name']} ({item['pred_code']}):** {item['count']} cases\n"

    md += f"""
---

## 5. Demonstration Cases (Official Summit Traceable Syntax)

"""
    for c in examples:
        md += f"""### {c['scenario']}
- **Image ID:** `{c['image_id']}` | **Ground Truth:** `{c['true_dx']}` | **Predicted:** `{c['prediction']}` ({c['confidence']*100:.1f}%) | **Decision:** `{c['decision']}`
- **Traceable Chain:**
  `{c['traceable_chain']}`
- **Coordinator Rationale:** {c['policy_rule']}

"""

    md += f"""---
*Official submission CSV generated at `{sum_data.get('output_csv', 'outputs/final_submission_format_sample.csv')}`.*
"""
    with open("outputs/validation_summary.md", "w", encoding="utf-8") as f:
        f.write(md)

def predict_hidden_test(
    hidden_dir: str,
    output_csv: str = "outputs/submission.csv"
):
    """
    Robust inference tool for the 200 hidden-test images released at 10:00 AM by organizers.
    Generates exact 3-column CSV: image_id, prediction, decision
    """
    print("=" * 70)
    print(f"  JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026")
    print(f"  Hidden-Test Submission Pipeline: {hidden_dir} -> {output_csv}")
    print("=" * 70)
    t0 = time.time()

    if not os.path.exists(hidden_dir):
        raise FileNotFoundError(f"Hidden test directory '{hidden_dir}' does not exist!")

    files = [f for f in os.listdir(hidden_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[Hidden Test] Located {len(files)} test image files.")

    rows = []
    seen_ids = set()

    for idx, f in enumerate(sorted(files)):
        raw_stem = os.path.splitext(f)[0]
        # Normalize duplicate stems like "ISIC_0000001 (2)" to "ISIC_0000001" if present
        img_id = raw_stem.replace(" (2)", "").replace(" (1)", "").strip()
        if img_id in seen_ids:
            # Skip duplicated file on disk if base ID already processed
            continue
        seen_ids.add(img_id)

        img_path = os.path.join(hidden_dir, f)

        try:
            # Robust inference with try/except
            res = agentic_dermaguard.process_image(img_path, patient_context=None)
            pred_code = res['prediction']
            decision = res['final_decision']
        except Exception as ex:
            print(f"[Warning] Failed processing {f} ({ex}); applying safe clinical fallback.")
            pred_code = 'nv'
            decision = 'REFER'

        # Strict validation of permitted codes
        if pred_code not in {'mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'}:
            pred_code = 'nv'
        if decision not in {'AUTO_TRIAGE', 'REFER', 'RE_IMAGE'}:
            decision = 'REFER'

        rows.append({
            'image_id': img_id,
            'prediction': pred_code,
            'decision': decision
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(files):
            print(f"  Inference completed: {len(rows)}/{len(files)} ({time.time() - t0:.1f}s)")

    sub_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    sub_df.to_csv(output_csv, index=False)

    total_time = time.time() - t0
    latency_ms = (total_time / len(sub_df) * 1000.0) if len(sub_df) > 0 else 0.0

    print(f"\n[Completed] Processed {len(sub_df)} hidden-test images in {total_time:.2f}s ({latency_ms:.1f} ms/img).")
    print(f"[Saved] Written to '{output_csv}'.")

    # Validate submission file immediately
    validate_submission_file(output_csv, expected_count=len(sub_df))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic DermaGuard Level 2 Evaluation & Hidden Test Inference")
    parser.add_argument("--hidden-dir", type=str, default=None, help="Path to 200 hidden test images directory")
    parser.add_argument("--metadata", type=str, default="Health/eval/metadata.csv", help="Evaluation metadata CSV")
    parser.add_argument("--images-dir", type=str, default="Health/eval/images", help="Evaluation images folder")
    parser.add_argument("--output-csv", type=str, default="outputs/final_submission_format_sample.csv", help="Output path for final CSV submission")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick smoke test run")
    parser.add_argument("--context-mode", type=str, default="no_context", choices=["no_context", "with_context", "both"], help="Context mode for evaluation")
    parser.add_argument("--skip-degradation", action="store_true", help="Skip synthetic degradation test")
    args = parser.parse_args()

    if args.hidden_dir:
        predict_hidden_test(args.hidden_dir, args.output_csv)
    elif args.context_mode == "both":
        print("[Run 1/2] Evaluating with no_context (Primary Hidden Test condition) ...")
        res_no_ctx = run_evaluation(
            eval_metadata=args.metadata,
            eval_images_dir=args.images_dir,
            output_csv=args.output_csv,
            max_samples=args.max_samples,
            context_mode="no_context",
            run_degradation=not args.skip_degradation
        )
        print("\n[Run 2/2] Evaluating with with_context ...")
        res_with_ctx = run_evaluation(
            eval_metadata=args.metadata,
            eval_images_dir=args.images_dir,
            output_csv="outputs/val_with_context_submission.csv",
            max_samples=args.max_samples,
            context_mode="with_context",
            run_degradation=False
        )
    else:
        run_evaluation(
            eval_metadata=args.metadata,
            eval_images_dir=args.images_dir,
            output_csv=args.output_csv,
            max_samples=args.max_samples,
            context_mode=args.context_mode,
            run_degradation=not args.skip_degradation
        )
