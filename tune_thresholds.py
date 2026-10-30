import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from thresholds import load_thresholds, save_thresholds, get_flat_thresholds
from agents.diagnosis_agent import (
    diagnosis_agent, IDX_TO_CODE, CODE_TO_IDX, CODE_TO_NAME, DEVICE, diagnosis_transforms, HAVE_TORCH
)
from agents.image_quality_agent import quality_agent
from agents.lesion_analysis_agent import lesion_analysis_agent
from agents.context_agent import context_agent
from agents.safety_agent import safety_agent
from agents.coordinator_agent import coordinator_agent

CACHE_FILE = "outputs/val_cached_features.pkl"
EVAL_METADATA = "Health/eval/metadata.csv"
EVAL_IMAGES_DIR = "Health/eval/images"

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)

def extract_and_cache_validation_data(force_refresh: bool = False) -> List[Dict[str, Any]]:
    os.makedirs("outputs", exist_ok=True)
    if not force_refresh and os.path.exists(CACHE_FILE):
        print(f"[Cache] Loading cached features from {CACHE_FILE} ...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print(f"[Cache] Processing all validation samples and caching features into {CACHE_FILE} ...")
    df = pd.read_csv(EVAL_METADATA)
    cached_records = []
    t0 = time.time()

    import torch
    import torch.nn.functional as F

    for idx, row in df.iterrows():
        img_id = str(row['image_id']).strip()
        img_path = os.path.join(EVAL_IMAGES_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue

        true_dx = row['dx']
        true_label = CODE_TO_IDX.get(true_dx, 1)
        lesion_id = str(row.get('lesion_id', img_id)).strip()

        pil_img = Image.open(img_path).convert('RGB')

        # 1. Image Quality
        q_metrics = quality_agent.evaluate(pil_img)
        a1_data = {
            "blur_score": q_metrics.blur_score,
            "lighting_score": q_metrics.lighting_score,
            "glare_score": q_metrics.glare_score,
            "occlusion_score": q_metrics.occlusion_score,
            "overall_quality": q_metrics.overall_quality,
            "status": q_metrics.status,
            "triggered_criteria": q_metrics.triggered_criteria
        }

        # 2. Lesion ABCD Analysis
        a3_data = lesion_analysis_agent.analyze(pil_img)

        # 3. Diagnosis Agent raw logits & TTA
        raw_logits_views = None
        tta_disag = 0.0

        if HAVE_TORCH and diagnosis_agent.torch_model is not None and diagnosis_agent.have_weights:
            try:
                v1 = pil_img
                v2 = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                v3 = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
                v4 = pil_img.rotate(180)
                batch = torch.stack([diagnosis_transforms(v) for v in (v1, v2, v3, v4)]).to(DEVICE)
                with torch.no_grad():
                    logits_tensor = diagnosis_agent.torch_model(batch)
                    raw_logits_views = logits_tensor.cpu().numpy()  # (4, 7)
                    probs_views = F.softmax(logits_tensor, dim=1).cpu().numpy()
                    tta_disag = float(np.max(np.std(probs_views, axis=0)))
            except Exception:
                pass

        if raw_logits_views is None:
            # Fallback representation
            pred_dict = diagnosis_agent.predict_numpy_fallback(pil_img)
            probs = np.array([pred_dict['all_probabilities'][IDX_TO_CODE[i]] for i in range(7)])
            logits = np.log(probs + 1e-8)
            raw_logits_views = np.tile(logits, (4, 1))
            tta_disag = 0.0

        record = {
            "image_id": img_id,
            "lesion_id": lesion_id,
            "true_dx": true_dx,
            "true_label": true_label,
            "raw_logits_views": raw_logits_views,  # shape (4, 7)
            "tta_disagreement": tta_disag,
            "a1_data": a1_data,
            "a3_data": a3_data,
            "context": {
                "age": row.get("age"),
                "sex": row.get("sex"),
                "localization": row.get("localization")
            }
        }
        cached_records.append(record)

        if (idx + 1) % 200 == 0 or (idx + 1) == len(df):
            print(f"  Cached {idx + 1}/{len(df)} samples ({time.time() - t0:.1f}s)")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cached_records, f)
    print(f"[Cache] Successfully saved {len(cached_records)} records to {CACHE_FILE} ({time.time() - t0:.1f}s)")
    return cached_records

def get_mean_logits(record: Dict[str, Any]) -> np.ndarray:
    return np.mean(record["raw_logits_views"], axis=0)

def compute_probs_from_logits(logits: np.ndarray, temperature: float = 1.0, bias: np.ndarray = None) -> np.ndarray:
    scaled = (logits / max(0.01, temperature))
    if bias is not None:
        scaled = scaled + bias
    exp_l = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    return probs

def tune_temperature(records_a: List[Dict[str, Any]], records_b: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Optimizes temperature to minimize NLL on Set A; reports ECE on Set A and Set B."""
    logits_a = np.array([get_mean_logits(r) for r in records_a])
    labels_a = np.array([r["true_label"] for r in records_a])

    logits_b = np.array([get_mean_logits(r) for r in records_b])
    labels_b = np.array([r["true_label"] for r in records_b])

    # Baseline ECE at T = 1.0
    probs_a_base = compute_probs_from_logits(logits_a, temperature=1.0)
    ece_a_base = compute_ece(probs_a_base, labels_a)

    probs_b_base = compute_probs_from_logits(logits_b, temperature=1.0)
    ece_b_base = compute_ece(probs_b_base, labels_b)

    # Grid search for best temperature
    best_t = 1.0
    best_nll = float('inf')

    t_candidates = np.linspace(0.6, 2.2, 81)
    for t in t_candidates:
        probs = compute_probs_from_logits(logits_a, temperature=t)
        # NLL
        correct_probs = probs[np.arange(len(labels_a)), labels_a]
        nll = -np.mean(np.log(np.clip(correct_probs, 1e-12, 1.0)))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    probs_a_cal = compute_probs_from_logits(logits_a, temperature=best_t)
    ece_a_cal = compute_ece(probs_a_cal, labels_a)

    probs_b_cal = compute_probs_from_logits(logits_b, temperature=best_t)
    ece_b_cal = compute_ece(probs_b_cal, labels_b)

    print(f"\n[Temperature Calibration]")
    print(f"  Optimal Temperature: {best_t:.2f}")
    print(f"  Set A (Tune) ECE:         {ece_a_base*100:.2f}% -> {ece_a_cal*100:.2f}%")
    print(f"  Set B (Confirmation) ECE: {ece_b_base*100:.2f}% -> {ece_b_cal*100:.2f}%")

    return best_t, ece_a_cal, ece_b_cal

def tune_class_biases(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    temperature: float = 1.0,
    rounds: int = 4
) -> Tuple[np.ndarray, float, float]:
    """Coordinate ascent optimization of per-class probability biases to maximize Macro-F1."""
    logits_a = np.array([get_mean_logits(r) for r in records_a])
    labels_a = np.array([r["true_label"] for r in records_a])

    logits_b = np.array([get_mean_logits(r) for r in records_b])
    labels_b = np.array([r["true_label"] for r in records_b])

    # Baseline Macro-F1
    probs_a_base = compute_probs_from_logits(logits_a, temperature=temperature)
    f1_a_base = f1_score(labels_a, np.argmax(probs_a_base, axis=1), average='macro', zero_division=0)

    probs_b_base = compute_probs_from_logits(logits_b, temperature=temperature)
    f1_b_base = f1_score(labels_b, np.argmax(probs_b_base, axis=1), average='macro', zero_division=0)

    biases = np.zeros(7, dtype=np.float32)
    step_sizes = np.linspace(-1.2, 1.2, 49)

    best_macro_f1 = f1_a_base

    for rnd in range(rounds):
        improved = False
        # Permute or iterate over classes
        for c in range(7):
            best_delta = 0.0
            best_c_f1 = best_macro_f1
            orig_val = biases[c]

            for delta in step_sizes:
                biases[c] = orig_val + delta
                probs = compute_probs_from_logits(logits_a, temperature=temperature, bias=biases)
                preds = np.argmax(probs, axis=1)
                score = f1_score(labels_a, preds, average='macro', zero_division=0)
                if score > best_c_f1:
                    best_c_f1 = score
                    best_delta = delta

            if best_c_f1 > best_macro_f1:
                biases[c] = orig_val + best_delta
                best_macro_f1 = best_c_f1
                improved = True
            else:
                biases[c] = orig_val

        if not improved:
            break

    # Evaluate on Confirmation Set B
    probs_b_tuned = compute_probs_from_logits(logits_b, temperature=temperature, bias=biases)
    f1_b_tuned = f1_score(labels_b, np.argmax(probs_b_tuned, axis=1), average='macro', zero_division=0)

    print(f"\n[Coordinate Ascent Class Bias Tuning]")
    print(f"  Biases: { {IDX_TO_CODE[i]: round(float(biases[i]), 3) for i in range(7)} }")
    print(f"  Set A (Tune) Macro-F1:         {f1_a_base:.4f} -> {best_macro_f1:.4f}")
    print(f"  Set B (Confirmation) Macro-F1: {f1_b_base:.4f} -> {f1_b_tuned:.4f}")

    return biases, best_macro_f1, f1_b_tuned

def simulate_pipeline_single(
    record: Dict[str, Any],
    temperature: float,
    bias: np.ndarray,
    th: Dict[str, Any]
) -> Dict[str, Any]:
    """Runs fast vectorized simulation of A1-A6 on cached features."""
    raw_logits = get_mean_logits(record)
    probs = compute_probs_from_logits(raw_logits, temperature=temperature, bias=bias)
    pred_idx = int(np.argmax(probs))
    pred_code = IDX_TO_CODE[pred_idx]
    confidence = float(probs[pred_idx])

    sorted_p = np.sort(probs)[::-1]
    margin = float(confidence - sorted_p[1])
    entropy = float(-np.sum(probs * np.log2(probs + 1e-8)))
    tta_disag = record.get("tta_disagreement", 0.0)

    # Malignancy mass
    mel_p = float(probs[CODE_TO_IDX['mel']])
    bcc_p = float(probs[CODE_TO_IDX['bcc']])
    akiec_p = float(probs[CODE_TO_IDX['akiec']])
    tot_malig = float(mel_p + bcc_p + akiec_p)

    # A1 Quality
    a1 = record["a1_data"]
    q_status = a1["status"]

    # A3 Lesion
    a3 = record["a3_data"]
    susp_score = a3.get("suspicion_score", 0.0)
    tds = a3.get("tds_score", 3.5)
    border_pts = a3.get("abcd_breakdown", {}).get("B_border", {}).get("border_score", 1)
    asym_pts = a3.get("abcd_breakdown", {}).get("A_asymmetry", {}).get("axis_score", 0)

    # A4 Context (neutral if missing or evaluated)
    # If tuning for primary no_context mode: context_risk = "LOW"
    context_risk = "LOW"
    age = record["context"].get("age")
    loc = record["context"].get("localization")
    sex = record["context"].get("sex")

    # Safety Veto Check
    veto = False
    is_critical = False

    # Rule S0: Direct malignancy
    if pred_code in {'mel', 'bcc', 'akiec'}:
        veto = True
        is_critical = True

    # Rule S1: Residual malignancy
    elif tot_malig >= th["malignant_prob_threshold"] or mel_p >= th["mel_threshold"]:
        veto = True

    # Rule S2: Lesion ABCD discordance
    if pred_code in {'nv', 'bkl', 'df', 'vasc'}:
        if (tds > 5.20) or (border_pts >= 6 and asym_pts >= 1) or (susp_score >= th.get("suspicion_discordance", 0.65)):
            veto = True
            is_critical = True

    # Rule S3: Diagnostic ambiguity
    if pred_code in {'nv', 'bkl', 'df', 'vasc'}:
        if (margin < th["margin_minimum"]) or (entropy > th["max_entropy"]):
            veto = True

    # Rule S5: Quality precaution
    if q_status == 'RE_IMAGE':
        veto = True
        is_critical = True
    elif q_status == 'ACCEPTABLE_WITH_CAUTION' and confidence < th["caution_confidence_required"]:
        veto = True

    # Rule S6: TTA Disagreement
    if tta_disag > th.get("tta_disagreement_threshold", 0.22):
        veto = True

    # Coordinator Decision Precedence
    auto_conf_thresh = th["auto_triage_confidence"]

    if q_status == 'RE_IMAGE':
        decision = 'RE_IMAGE'
    elif pred_code in {'mel', 'bcc', 'akiec'}:
        decision = 'REFER'
    elif veto:
        decision = 'REFER'
    elif confidence < auto_conf_thresh:
        decision = 'REFER'
    elif pred_code in {'nv', 'bkl', 'df', 'vasc'} and confidence >= auto_conf_thresh and not veto:
        decision = 'AUTO_TRIAGE'
    else:
        decision = 'REFER'

    true_dx = record["true_dx"]
    is_correct = (pred_code == true_dx)
    is_malignant_true = (true_dx in {'mel', 'bcc', 'akiec'})

    return {
        "pred_code": pred_code,
        "decision": decision,
        "confidence": confidence,
        "is_correct": is_correct,
        "is_malignant_true": is_malignant_true,
        "missed_malignancy": (decision == "AUTO_TRIAGE" and is_malignant_true)
    }

def run_grid_search(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    temperature: float,
    bias: np.ndarray
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Sweeps clinical thresholds targeting auto_triage_precision >= 0.95 and missed_malignancies = 0."""
    print("\n[Grid Search Optimization] Sweeping clinical operating points ...")

    sweep_records = []

    conf_values = [0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85]
    margin_values = [0.10, 0.15, 0.20]
    malig_values = [0.22, 0.25, 0.28, 0.30]
    mel_values = [0.18, 0.20, 0.22]
    entropy_values = [1.65, 1.75, 1.85]

    base_th = get_flat_thresholds()

    for auto_conf in conf_values:
        for malig_t in malig_values:
            for mel_t in mel_values:
                for margin_min in margin_values:
                    for ent_max in entropy_values:
                        current_th = base_th.copy()
                        current_th["auto_triage_confidence"] = auto_conf
                        current_th["malignant_prob_threshold"] = malig_t
                        current_th["mel_threshold"] = mel_t
                        current_th["margin_minimum"] = margin_min
                        current_th["max_entropy"] = ent_max

                        # Evaluate on Set A
                        sims_a = [simulate_pipeline_single(r, temperature, bias, current_th) for r in records_a]
                        total_a = len(sims_a)
                        auto_triages_a = [s for s in sims_a if s["decision"] == "AUTO_TRIAGE"]
                        correct_auto_a = sum(1 for s in auto_triages_a if s["is_correct"])
                        incorrect_auto_a = sum(1 for s in auto_triages_a if not s["is_correct"])
                        missed_malig_a = sum(1 for s in sims_a if s["missed_malignancy"])
                        refers_a = sum(1 for s in sims_a if s["decision"] == "REFER")
                        re_image_a = sum(1 for s in sims_a if s["decision"] == "RE_IMAGE")
                        prec_a = (correct_auto_a / len(auto_triages_a)) if auto_triages_a else 1.0
                        re_image_rate_a = re_image_a / total_a

                        # Evaluate on Set B (confirmation)
                        sims_b = [simulate_pipeline_single(r, temperature, bias, current_th) for r in records_b]
                        total_b = len(sims_b)
                        auto_triages_b = [s for s in sims_b if s["decision"] == "AUTO_TRIAGE"]
                        correct_auto_b = sum(1 for s in auto_triages_b if s["is_correct"])
                        incorrect_auto_b = sum(1 for s in auto_triages_b if not s["is_correct"])
                        missed_malig_b = sum(1 for s in sims_b if s["missed_malignancy"])
                        refers_b = sum(1 for s in sims_b if s["decision"] == "REFER")
                        re_image_b = sum(1 for s in sims_b if s["decision"] == "RE_IMAGE")
                        prec_b = (correct_auto_b / len(auto_triages_b)) if auto_triages_b else 1.0

                        sweep_records.append({
                            "auto_triage_confidence": auto_conf,
                            "malignant_prob_threshold": malig_t,
                            "mel_threshold": mel_t,
                            "margin_minimum": margin_min,
                            "max_entropy": ent_max,
                            # Set A (Tuning)
                            "tune_correct_auto": correct_auto_a,
                            "tune_incorrect_auto": incorrect_auto_a,
                            "tune_auto_precision": round(prec_a, 4),
                            "tune_refers": refers_a,
                            "tune_re_image": re_image_a,
                            "tune_missed_malignancy": missed_malig_a,
                            "tune_re_image_rate": round(re_image_rate_a, 4),
                            # Set B (Confirmation)
                            "conf_correct_auto": correct_auto_b,
                            "conf_incorrect_auto": incorrect_auto_b,
                            "conf_auto_precision": round(prec_b, 4),
                            "conf_refers": refers_b,
                            "conf_re_image": re_image_b,
                            "conf_missed_malignancy": missed_malig_b,
                        })

    sweep_df = pd.DataFrame(sweep_records)
    sweep_df.to_csv("outputs/threshold_sweep.csv", index=False)
    print(f"[Grid Search] Evaluated {len(sweep_df)} points. Saved to outputs/threshold_sweep.csv")

    # Filter candidates meeting constraints on Set A:
    # 1. missed_malignancy == 0 (HARD safety priority!)
    # 2. auto_precision >= 0.95 (or highest available)
    # 3. re_image_rate <= 0.05
    feasible = sweep_df[
        (sweep_df["tune_missed_malignancy"] == 0) &
        (sweep_df["tune_re_image_rate"] <= 0.05)
    ].copy()

    if len(feasible) == 0:
        print("[Warning] No points with missed_malignancy == 0. Finding minimum missed.")
        min_missed = sweep_df["tune_missed_malignancy"].min()
        feasible = sweep_df[sweep_df["tune_missed_malignancy"] == min_missed].copy()

    # Prefer highest precision >= 0.95, then maximize correct_auto_triage
    high_prec = feasible[feasible["tune_auto_precision"] >= 0.95]
    if len(high_prec) > 0:
        best_row = high_prec.sort_values(by=["tune_correct_auto", "tune_auto_precision"], ascending=[False, False]).iloc[0]
    else:
        best_row = feasible.sort_values(by=["tune_auto_precision", "tune_correct_auto"], ascending=[False, False]).iloc[0]

    chosen = {
        "auto_triage_confidence": float(best_row["auto_triage_confidence"]),
        "malignant_prob_threshold": float(best_row["malignant_prob_threshold"]),
        "mel_threshold": float(best_row["mel_threshold"]),
        "margin_minimum": float(best_row["margin_minimum"]),
        "max_entropy": float(best_row["max_entropy"])
    }

    print("\n" + "=" * 65)
    print("  OPTIMAL CLINICAL OPERATING POINT SELECTED")
    print("=" * 65)
    print(f"  auto_triage_confidence:    {chosen['auto_triage_confidence']:.2f}")
    print(f"  malignant_prob_threshold:  {chosen['malignant_prob_threshold']:.2f}")
    print(f"  mel_threshold:             {chosen['mel_threshold']:.2f}")
    print(f"  margin_minimum:            {chosen['margin_minimum']:.2f}")
    print(f"  max_entropy:               {chosen['max_entropy']:.2f}")
    print("-" * 65)
    print(f"  [Set A Tune]         Correct: {int(best_row['tune_correct_auto'])}, Incorrect: {int(best_row['tune_incorrect_auto'])}, Prec: {best_row['tune_auto_precision']*100:.1f}%, Missed Malig: {int(best_row['tune_missed_malignancy'])}")
    print(f"  [Set B Confirmation] Correct: {int(best_row['conf_correct_auto'])}, Incorrect: {int(best_row['conf_incorrect_auto'])}, Prec: {best_row['conf_auto_precision']*100:.1f}%, Missed Malig: {int(best_row['conf_missed_malignancy'])}")
    print("=" * 65)

    # Plot Tradeoff
    plot_tradeoff(sweep_df, chosen)

    return chosen, sweep_df

def plot_tradeoff(df: pd.DataFrame, chosen: Dict[str, Any]):
    plt.figure(figsize=(10, 6), dpi=150)
    plt.style.use('dark_background')

    # Scatter plot: Auto Triage Precision vs Correct Auto Triage count
    sc = plt.scatter(
        df["tune_correct_auto"],
        df["tune_auto_precision"] * 100,
        c=df["tune_missed_malignancy"],
        cmap="coolwarm_r",
        alpha=0.65,
        s=40,
        edgecolors='none'
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Missed Malignancies (Mel/BCC/AKIEC)", color="white")

    # Target line: 95% Precision
    plt.axhline(95, color="#10b981", linestyle="--", linewidth=1.5, label="Target Precision (95%)")

    # Highlight chosen point
    chosen_pts = df[
        (df["auto_triage_confidence"] == chosen["auto_triage_confidence"]) &
        (df["malignant_prob_threshold"] == chosen["malignant_prob_threshold"]) &
        (df["mel_threshold"] == chosen["mel_threshold"]) &
        (df["margin_minimum"] == chosen["margin_minimum"]) &
        (df["max_entropy"] == chosen["max_entropy"])
    ]
    if len(chosen_pts) > 0:
        pt = chosen_pts.iloc[0]
        plt.scatter(
            [pt["tune_correct_auto"]],
            [pt["tune_auto_precision"] * 100],
            color="#06b6d4",
            s=160,
            marker='*',
            edgecolors='white',
            linewidths=1.5,
            label=f"Selected Point (Conf={chosen['auto_triage_confidence']:.2f}, Prec={pt['tune_auto_precision']*100:.1f}%)"
        )

    plt.title("Clinical Screening Triage Trade-Off (Jaypee Summit 2026)", fontsize=13, fontweight='bold', color='white', pad=12)
    plt.xlabel("Correct AUTO_TRIAGE Cases (Set A)", fontsize=11, color='white')
    plt.ylabel("AUTO_TRIAGE Precision (%)", fontsize=11, color='white')
    plt.grid(True, linestyle=":", alpha=0.3)
    plt.legend(loc="lower left", framealpha=0.8)
    plt.tight_layout()

    out_path = "outputs/threshold_tradeoff.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Saved threshold tradeoff plot to {out_path}")

def update_centralized_thresholds(
    temperature: float,
    biases: np.ndarray,
    chosen: Dict[str, Any]
):
    th = load_thresholds()
    th["diagnosis"]["temperature"] = round(float(temperature), 3)
    th["diagnosis"]["auto_triage_confidence"] = round(float(chosen["auto_triage_confidence"]), 3)
    th["diagnosis"]["margin_minimum"] = round(float(chosen["margin_minimum"]), 3)
    th["diagnosis"]["max_entropy"] = round(float(chosen["max_entropy"]), 3)
    th["diagnosis"]["class_probability_bias"] = {
        IDX_TO_CODE[i]: round(float(biases[i]), 4) for i in range(7)
    }
    th["malignancy"]["malignant_prob_threshold"] = round(float(chosen["malignant_prob_threshold"]), 3)
    th["malignancy"]["mel_threshold"] = round(float(chosen["mel_threshold"]), 3)

    save_thresholds(th)

def main():
    print("=" * 65)
    print("  JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026")
    print("  Track 2: Healthcare | Automatic Clinical Threshold Tuning")
    print("=" * 65)

    # 1. Cache or load validation features
    records = extract_and_cache_validation_data(force_refresh=False)

    # 2. 50/50 Grouped Stratified Split (by lesion_id, stratified by dx)
    lesion_ids = [r["lesion_id"] for r in records]
    labels = [r["true_label"] for r in records]

    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(records, labels, groups=lesion_ids))

    records_a = [records[i] for i in train_idx]
    records_b = [records[i] for i in test_idx]

    # Verify 0 group overlap
    lesions_a = set(r["lesion_id"] for r in records_a)
    lesions_b = set(r["lesion_id"] for r in records_b)
    overlap = lesions_a.intersection(lesions_b)
    print(f"\n[Split Verification]")
    print(f"  Set A (Tuning Set):       {len(records_a)} samples across {len(lesions_a)} unique lesions")
    print(f"  Set B (Confirmation Set): {len(records_b)} samples across {len(lesions_b)} unique lesions")
    print(f"  Lesion Overlap:           {len(overlap)} (Zero data leakage confirmed)")

    # 3. Temperature Scaling
    opt_temp, ece_a, ece_b = tune_temperature(records_a, records_b)

    # 4. Coordinate Ascent Class Bias Tuning
    biases, f1_a, f1_b = tune_class_biases(records_a, records_b, temperature=opt_temp)

    # 5. Grid Search over Triage Parameters
    chosen, sweep_df = run_grid_search(records_a, records_b, temperature=opt_temp, bias=biases)

    # 6. Save tuned values to thresholds.json
    update_centralized_thresholds(opt_temp, biases, chosen)

    print("\n[SUCCESS] Threshold tuning completed and persisted to thresholds.json!")

if __name__ == "__main__":
    main()
