# Agentic-DermaGuard: Six-Agent Skin Lesion Screening & Triage

**Jaypee Agentic AI International Summit 2026 — Track 2 | Healthcare**

Agentic DermaGuard is an explainable, safety-first multi-agent clinical decision support system. It combines deep learning diagnosis, computer vision edge/border mapping, demographic context reasoning, and clinical safety vetoes into a traceable final triage disposition: **`AUTO_TRIAGE`**, **`REFER`**, or **`RE_IMAGE`**.

---

## 🏗️ Six-Agent Architecture (Level 2 Specification)

```
                            [ Dermoscopic Image + Patient Demographics ]
                                                 │
         ┌───────────────────┬───────────────────┼───────────────────┐
         │                   │                   │                   │
         ▼                   ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   1. Quality    │ │  2. Diagnosis   │ │3. Lesion Analys.│ │4. Context/Reason│
│      Agent      │ │      Agent      │ │      Agent      │ │      Agent      │
│(Blur/Glare/Edge)│ │ (7 Class Codes) │ │(ABCD Cues/Edges)│ │ (Age/Sex/Site)  │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │                   │
         └───────────────────┼───────────────────┴───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    5. Safety    │
                    │      Agent      │
                    │(Sentinel & Veto)│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ 6. Coordinator  │
                    │      Agent      │
                    │(Policy Engine)  │
                    └────────┬────────┘
                             │
                             ▼
    [ Final Decision: AUTO_TRIAGE / REFER / RE_IMAGE ]
    [ Prediction Code: nv, mel, bcc, akiec, bkl, df, vasc ]
    [ Traceable Audit: A1: PASS | A2: nv • 0.82 | A3+A4: BENIGN_CUES | A5: PASS | A6: AUTO_TRIAGE ]
```

### The Six Specialized Roles

1. **Agent 1: Image Quality Agent (`agents/image_quality_agent.py`)**
   - **Metrics:** Combined Laplacian variance & Tenengrad gradient for blur/sharpness, histogram entropy for lighting dynamic range, HSV V-channel specular glare saturation, and edge-centering obstruction.
   - **Outputs:** Quality status (`PASS`, `ACCEPTABLE_WITH_CAUTION`, `RE_IMAGE`), scores (0–100), defect triggers, and visual sharpness gradient map.

2. **Agent 2: Diagnosis Agent (`agents/diagnosis_agent.py`)**
   - **Classification:** 7 official HAM10000 disease class codes:
     - `mel`: Melanoma (malignant)
     - `nv`: Melanocytic nevus (benign mole)
     - `bcc`: Basal cell carcinoma (malignant)
     - `akiec`: Actinic keratoses (pre-malignant)
     - `bkl`: Benign keratosis-like lesions
     - `df`: Dermatofibroma (benign nodule)
     - `vasc`: Vascular lesions (angioma)
   - **Outputs:** Predicted class code, full probability distribution, confidence score, and top-3 differential diagnosis.

3. **Agent 3: Lesion Analysis Agent (`agents/lesion_analysis_agent.py`)**
   - **ABCD Visual Cues:**
     - **Border Irregularity (B):** Compactness / Isoperimetric quotient \(P^2 / 4\pi A\) (circle = 1.0; jagged/notched borders > 1.6), and radial distance variance.
     - **Asymmetry (A):** Non-overlapping area fraction across horizontal and vertical centroid folds.
     - **Color Variation (C):** RGB channel standard deviation and dark melanin / erythema variegation.
   - **Edge Mapping:** Computes and exports high-contrast boundary contour overlay (PNG base64).
   - **Outputs:** Suspicion score (0.0 to 1.0), suspicion level (`BENIGN_CUES`, `INDETERMINATE`, `SUSPICIOUS`), and reliability flags.

4. **Agent 4: Reasoning / Context Agent (`agents/context_agent.py`)**
   - **Clinical Context:** Stratifies cumulative solar exposure and malignancy incidence across `age`, `sex`/gender, and anatomical `localization`.
   - **Missing Fields:** Detects and flags missing demographic data (`MISSING_AGE`, `MISSING_SEX`, `MISSING_LOCALIZATION`).
   - **Outputs:** Risk indicator (`LOW`, `MODERATE`, `HIGH`, `VERY_HIGH`), demographic risk score, and clinical evidence narrative.

5. **Agent 5: Safety Agent (`agents/safety_agent.py`)**
   - **Sentinel & Veto:** Cross-checks diagnosis against visual cues, patient demographics, and image quality.
   - **Veto Rules:**
     - Overrides `AUTO_TRIAGE` if benign prediction conflicts with high border irregularity or asymmetry.
     - Overrides `AUTO_TRIAGE` if benign prediction occurs in high-risk demographic context (e.g. elderly patient with scalp/facial lesion).
     - Overrides `AUTO_TRIAGE` on degraded images (`RE_IMAGE` or quality caution).
   - **Outputs:** Safety flag (`PASS`, `SAFETY_FLAG`, `SAFETY_VETO`), auto-triage veto boolean, and clinical justification.

6. **Agent 6: Coordinator Agent (`agents/coordinator_agent.py`)**
   - **Deterministic Policy:** Applies documented clinical triage policy:
     1. Quality failure &rarr; `RE_IMAGE` (with diagnostic inference preserved per track rule).
     2. Predicted malignancy (`mel`, `bcc`, `akiec`) or safety veto &rarr; `REFER`.
     3. High-confidence benign (`nv`, `bkl`, `df`, `vasc`) with clear safety checks &rarr; `AUTO_TRIAGE`.
     4. Diagnostic uncertainty &rarr; `REFER`.
   - **Outputs:** Final predicted label, triage decision, and standardized traceable chain:
     `A1: PASS | A2: nv • 0.73 | A3+A4: SUSPICIOUS | A5: SAFETY FLAG | A6: REFER`.

---

## 🎛️ Centralized Thresholds & Automatic Tuning

All tunable operational parameters and clinical cutoffs are centralized in a single source of truth: `thresholds.json` (loaded dynamically with fallback defaults via `thresholds.py`).

### 1. Parameter Specifications

| Category | Parameter | Tuned Default | Description / Clinical Rationale |
|---|---|---|---|
| **Quality** | `blur_threshold`<br>`lighting_threshold`<br>`glare_threshold`<br>`occlusion_threshold`<br>`quality_cutoff`<br>`caution_overall_threshold` | 40.0<br>40.0<br>70.0<br>50.0<br>50.0<br>72.0 | Combined Laplacian + Tenengrad sharpness cutoff.<br>Histogram entropy lighting dynamic range.<br>Specular saturation sensor-clip penalty (min(R,G,B) >= 246).<br>Center vs. perimeter edge density.<br>Overall composite quality required for clinical screening.<br>Upper bound of cautionary evaluation band. |
| **Diagnosis** | `auto_triage_confidence`<br>`margin_minimum`<br>`max_entropy`<br>`temperature`<br>`tta_enabled`<br>`tta_disagreement_threshold`<br>`class_probability_bias` | 0.80<br>0.10<br>1.65<br>1.14<br>true<br>0.22<br>mel: +0.30<br>nv: +0.75<br>bcc: -0.40<br>akiec: -0.85<br>df: +0.80 | Minimum confidence required for autonomous benign triage.<br>Top-1 vs top-2 candidate probability margin separation.<br>Shannon entropy threshold guarding differential ambiguity.<br>Probability calibration scaling factor.<br>4-view Test-Time Augmentation (Original, H-Flip, V-Flip, Rot180).<br>Max standard deviation across TTA views signaling morphology instability.<br>Coordinate ascent probability biases tuned for Macro-F1. |
| **Malignancy** | `malignant_prob_threshold`<br>`mel_threshold` | 0.22<br>0.18 | Cumulative malignancy probability mass (p_mel + p_bcc + p_akiec).<br>Strict separate threshold for invasive melanoma suspicion. |
| **Lesion ABCD** | `suspicion_cutoffs`<br>`tds_cutoffs`<br>`asymmetry_weight`<br>`border_weight`<br>`color_weight` | Benign: <0.35, Suspicious: >0.55<br>TDS <4.75 Benign, >5.45 Suspicious<br>0.35, 0.35, 0.30 | Continuous morphometry suspicion cutoffs.<br>Stolz Total Dermoscopy Score: 1.3*A + 0.1*B + 0.5*C + 0.5*D.<br>Clinical ABCD weighting factors. |
| **Context** | `risk_cutoffs`<br>`age_cutoffs`<br>`neutral_missing_prior` | Low: <0.45, High: >=0.70<br>25, 40, 60, 75<br>0.20 | Epidemiological risk accumulation thresholds.<br>Pediatric, Adult, Senior, Geriatric incidence cohorts.<br>Neutral prior when demographic fields are omitted (hidden test mode). |
| **Safety S1-S6** | `s1_enabled`..`s6_enabled`<br>`caution_confidence_required`<br>`high_context_risk_conf_req` | true (all enabled)<br>0.70<br>0.85 | Named sentinel rules enforcing patient safety and vetoing AUTO_TRIAGE.<br>Required confidence when quality is borderline.<br>Required confidence in high-risk elderly/sun-damaged demographics. |

### 2. Automated Tuning Methodology (`tune_thresholds.py`)

1. **50/50 Grouped Stratified Split on `lesion_id`:**
   - Evaluated across validation set (1,942 images, 1,456 unique lesions).
   - Set A (Tuning): 971 images across 728 lesions.
   - Set B (Confirmation): 971 images across 728 lesions.
   - **Lesion Overlap: EXACTLY 0 (Zero data leakage)**.
2. **Logit Calibration & Temperature Scaling:**
   - Optimized temperature $T = 1.14$ via NLL minimization.
   - Set A Expected Calibration Error (ECE): **3.83% &rarr; 1.64%**.
3. **Coordinate Ascent Per-Class Bias Tuning:**
   - Maximized multi-class Macro-F1 across 7 disease codes.
   - Set A Macro-F1: **0.5414 &rarr; 0.6140** (confirmed on Set B at 0.5937).
4. **Clinical Operating Point Grid Search:**
   - Evaluated 756 combinations of confidence, margins, and malignancy thresholds.
   - **Tuning Set A Performance:** Precision = **96.0%**, Correct AUTO_TRIAGE = 72.
   - **Confirmation Set B Performance:** Precision = **98.4%**, Correct AUTO_TRIAGE = 60, **Missed Malignancies = 0**.
   - Output artifacts: `outputs/threshold_sweep.csv` and `outputs/threshold_tradeoff.png`.

---

## 🌐 Vercel Web Deployment

The application is deployed on Vercel as a full-stack serverless web application:
- **Frontend (`public/index.html`):** Tailwind CSS + Chart.js responsive single-page web app. Features real-time edge mapping visualization, patient context selectors, threshold tuning sliders, benchmark analytics dashboard, and API submission portal.
- **Backend API (`api/index.py`):** Serverless HTTP endpoints:
  - `POST /api/triage`: Accepts image and clinical context; executes all 6 agents; returns decision and edge map overlay.
  - `GET /api/benchmark`: Serves validation benchmark metrics and confusion matrix.
  - `POST /api/submit`: Uploads benchmark results or CSV to summit endpoints.
- **Configuration:** `vercel.json` rewrites all `/api/*` requests to `/api/index.py`.

---

## 🚀 CLI Commands & Usage

### 1. Run Six-Agent Validation Benchmark
```bash
python evaluate.py
```
Outputs:
- `outputs/final_submission_format_sample.csv`: Sample submission file in exact competition format (`image_id,prediction,decision`).
- `outputs/validation_detailed.csv`: Complete table of all 1,942 images with outputs from all 6 agents.
- `outputs/validation_results.json`: Benchmark metrics and confusion matrix.
- `outputs/validation_summary.md`: Formatted report for the evaluation panel.

### 2. Run Inference on 200 Hidden Test Images (Released at 10:00 AM)
```bash
python evaluate.py --hidden-dir <path_to_hidden_images_folder> --output-csv outputs/submission.csv
```
Generates the exact required 3-column CSV file:
```csv
image_id,prediction,decision
ISIC_0035001,nv,AUTO_TRIAGE
ISIC_0035002,mel,REFER
ISIC_0035003,bkl,RE_IMAGE
```

### 3. Upload Submission CSV to Summit API
```bash
python submit.py --csv outputs/submission.csv --api-url <ORGANIZER_API_URL> --team-name Team-DermaGuard
```

### 4. Run Local Web Server
```bash
python -m http.server 8000 --directory public
```
Or with Vercel CLI:
```bash
vercel dev
```
