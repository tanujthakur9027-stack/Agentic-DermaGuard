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
