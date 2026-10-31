# Agentic-DermaGuard: Six-Agent Skin Lesion Screening & Triage

**Jaypee Agentic AI International Summit 2026 — Track 2 | Healthcare**

Agentic DermaGuard is an explainable, deterministic, safety-first multi-agent clinical decision support system for dermoscopic skin lesion screening. It coordinates six specialized agents to deliver traceable, accountable decisions: **`AUTO_TRIAGE`**, **`REFER`**, or **`RE_IMAGE`**, while preserving diagnostic predictions across all inputs.

---

## 🏛️ Six-Agent Architecture & Explaining Agent Contract

Each agent operates with a distinct clinical responsibility and strictly outputs a standard **Explaining Agent Contract**:
```json
{
  "agent": "A3",
  "name": "Lesion Analysis",
  "status": "SUSPICIOUS",
  "score": 0.71,
  "summary": "Elevated border compactness and pigment asymmetry observed.",
  "evidence": [
    {
      "feature": "border_irregularity",
      "value": 1.82,
      "threshold": 1.60,
      "direction": ">",
      "passed": false,
      "note": "Border is jagged (compactness 1.82 > 1.60)"
    }
  ],
  "thresholds_used": { "border_compactness_high": 1.60 },
  "margin_to_threshold": { "border_compactness_high": 0.22 },
  "flags": ["LOW_RELIABILITY"],
  "explanation": "Lesion Analysis Agent identified abnormal geometric border notches...",
  "confidence_in_this_agent": 0.80,
  "runtime_ms": 12.4
}
```

### The Six Specialized Agents

1. **A1: Image Quality Agent (`agents/image_quality_agent.py`)**
   - **Role:** Gatekeeper for capture integrity.
   - **Evaluated Dimensions:** Laplacian variance & Tenengrad gradient (sharpness/blur), grayscale histogram entropy (illumination dynamic range), HSV V-channel specular saturation clip (flash glare voids), and perimeter edge density (obstruction/vignetting).
   - **Status:** `PASS`, `ACCEPTABLE_WITH_CAUTION`, `RE_IMAGE`.

2. **A2: Diagnosis Agent (`agents/diagnosis_agent.py`)**
   - **Role:** Pathology classifier for 7 official HAM10000 disease classes:
     - `mel`: Melanoma (malignant)
     - `nv`: Melanocytic nevus (benign)
     - `bcc`: Basal cell carcinoma (malignant)
     - `akiec`: Actinic keratoses (pre-malignant)
     - `bkl`: Benign keratosis-like lesions
     - `df`: Dermatofibroma (benign)
     - `vasc`: Vascular lesions (benign)
   - **Outputs:** Predicted class code, confidence score, differential probability distribution, Shannon entropy, top-2 separation margin, and cumulative malignancy risk mass (\(p_{mel} + p_{bcc} + p_{akiec}\)).

3. **A3: Lesion Analysis Agent (`agents/lesion_analysis_agent.py`)**
   - **Role:** ABCD visual morphometry and contour boundary detection.
   - **Features:** Compactness quotient (\(P^2 / 4\pi A\)), radial distance variance, asymmetry fold ratio across horizontal/vertical centroid axes, channel color standard deviation, and Stolz Total Dermoscopy Score (TDS).
   - **Status:** `BENIGN_CUES`, `INDETERMINATE`, `SUSPICIOUS`.

4. **A4: Reasoning / Context Agent (`agents/context_agent.py`)**
   - **Role:** Epidemiological context and patient demographics reasoning.
   - **Features:** Stratification by age cohorts, biological sex weights, and high-risk anatomical sites (face, scalp, ear, neck, acral extremities). Handles missing fields cleanly with neutral priors and missing-field flags.
   - **Status:** `LOW`, `MODERATE`, `HIGH`, `VERY_HIGH`.

5. **A5: Safety Sentinel (`agents/safety_agent.py`)**
   - **Role:** Cross-agent conflict detector and safety veto engine.
   - **Formal Rules (S1–S6):**
     - **S1 (Malignancy Probability Mass):** Benign prediction vetoed if combined malignancy risk exceeds cutoff.
     - **S2 (Lesion ABCD Discordance):** Benign prediction vetoed if visual ABCD cues are SUSPICIOUS.
     - **S3 (Diagnostic Ambiguity):** Vetoed if top-2 margin is narrow or entropy is high.
     - **S4 (High-Risk Context Discordance):** Benign prediction in elderly high-risk site without 85% confidence hurdle vetoed.
     - **S5 (Image Quality Caution):** High caution on borderline captures.
     - **S6 (Extreme Conflict Triad):** Simultaneous high age + high-risk site + border notch triggers mandatory specialist escalation.

6. **A6: Coordinator Agent (`agents/coordinator_agent.py`)**
   - **Role:** Policy arbiter applying clinical decision rules to produce final predicted label, triage disposition, and traceable audit trail.
   - **Golden Rule:** Every image receives BOTH a diagnosis prediction AND a decision. Diagnostic predictions are NEVER altered by triage logic (even under `RE_IMAGE`, preliminary diagnosis is strictly preserved).
   - **Example Strip:**
     `A1 PASS "Image is usable" | A2 nv • 0.73 "Model predicts nevus" | A3+A4 SUSPICIOUS "Other evidence raises concern" | A5 SAFETY FLAG "Conflict detected" | A6 REFER "Final disposition"`

---

## 🎛️ Centralized Threshold Registry (`backend/thresholds.py`)

All 51 operational parameters across all six agents are centralized in a unified registry (`backend/thresholds.py` + `thresholds.json`). Each parameter specifies:
- `key`, `agent`, `label`, `default`, `min`, `max`, `step`, `unit`
- `description`: Plain-English explanation of what the parameter controls
- `effect_up`: Plain-English clinical consequence of raising the value
- `effect_down`: Plain-English clinical consequence of lowering the value

### Presets
- **Conservative:** Higher safety margins, lower auto-triage volume, near-zero missed malignancy risk.
- **Balanced (Default):** Calibrated operating point on 1,942 HAM10000 validation cases.
- **Aggressive:** Lower hurdles for automated discharge in high-throughput screening clinics.

Per-request threshold overrides can be passed via `{"thresholds": { ... }}` to any API call without mutating global state.

---

## 💻 Multi-Page Frontend Architecture

A pure HTML/CSS/vanilla JS frontend (no npm build steps required) served from `public/`:

- **Interactive Screening (`/` or `/index.html`):** Single-image diagnostic console with dual image viewports, edge contour mapping, patient context selector, 7 curated demonstration cases, master disposition banner, required UI example strip, and 6 expandable agent cards with live evidence tables.
- **Batch Evaluation & 200 Hidden Test (`/batch.html`):** Multi-file browser runner designed specifically for the hidden test. Dispatches calls to `/api/analyze` with a **strict browser concurrency of 3 workers** and progress bar, preventing Vercel serverless timeouts. Supports optional ground truth CSV upload to compute live Panel KPIs:
  - Correct AUTO_TRIAGE count
  - Incorrect AUTO_TRIAGE count
  - Number of REFERs
  - Number of RE_IMAGEs
  - Accuracy & Macro-F1
  - One-click export of official 3-column submission CSV (`image_id,prediction,decision`).
- **Threshold Registry Explorer (`/thresholds.html`):** Full parameter tuning console showing all 51 thresholds grouped by agent, with interactive sliders, unit badges, effect tooltips, preset switches, and persistent save/reset controls.
- **Validation Benchmark Dashboard (`/benchmark.html`):** Level 1 benchmark reporting across all 1,942 held-out images. Features KPI scorecards, interactive 7x7 confusion matrix, class-wise metrics, and 5-stage ablation study.

Shared assets:
- `public/css/app.css`: Medical UI dark/light design system, status badges, evidence tables.
- `public/js/app.js`: Universal API client (`apiRequest`), navigation state, toast notifications, UI example strip renderer.
- `public/js/charts.js`: Chart.js helpers for probability bars, quality radar, and confusion matrices.

---

## 🔌 REST API Endpoints (`api/index.py`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Single image analysis (accepts base64 or image ID, context, thresholds override). Returns explaining agent contract. |
| `POST` | `/api/triage` | Backward-compatible alias for `/api/analyze`. |
| `GET` | `/api/thresholds` | Returns parameter registry, current values, and presets. |
| `POST` | `/api/thresholds` | Updates and persists threshold values. |
| `POST` | `/api/thresholds/reset` | Resets thresholds to defaults or a chosen preset (`Balanced`, `Conservative`, `Aggressive`). |
| `GET` | `/api/samples` | Serves 7 curated clinical demonstration cases with embedded base64 imagery. |
| `GET` | `/api/benchmark` | Serves validation benchmark metrics, confusion matrix, and ablation breakdown. |
| `GET` | `/api/download` | Downloads sample submission CSV. |
| `POST` | `/api/submit` | Relays benchmark or submission payload to summit API endpoint. |

---

## 🏃 Running the Application

### Option 1: Standalone Python Server (Recommended for Local Dev)
```bash
python api/index.py
```
Starts the full server at `http://localhost:8000`, serving both the multi-page frontend and all API endpoints.

### Option 2: Legacy Streamlit UI
```bash
streamlit run app.py
```

### Option 3: Vercel CLI
```bash
vercel dev
```

---

## 🧪 Hidden Test Execution (10:00 AM 200 Images Release)

### Method A: Browser Batch Page (`/batch.html`)
1. Open `http://localhost:8000/batch.html` in your browser.
2. Drag and drop all 200 test images (or select folder).
3. Click **"🚀 Start Batch Processing"**.
4. The worker pool will process images with concurrency 3, displaying live progress and disposition counts.
5. Click **"📥 Download Submission CSV"** to save `submission.csv`.

### Method B: CLI Inference (`evaluate.py`)
```bash
python evaluate.py --hidden-dir <path_to_200_images> --output-csv outputs/submission.csv
```
The script processes all images, automatically handles potential file errors with safe clinical fallbacks, and validates the output with `validate_submission.py`.

### Submission CSV Format Verification
Header must be exactly:
```csv
image_id,prediction,decision
ISIC_0035001,nv,AUTO_TRIAGE
ISIC_0035002,mel,REFER
ISIC_0035003,bkl,RE_IMAGE
```
Strict validation criteria:
- Exactly 3 columns: `image_id`, `prediction`, `decision`
- Exactly one row per image, no missing values, no duplicates
- Predictions restricted to: `mel`, `nv`, `bcc`, `akiec`, `bkl`, `df`, `vasc`
- Decisions restricted to: `AUTO_TRIAGE`, `REFER`, `RE_IMAGE`

Run validation manually anytime:
```bash
python validate_submission.py --csv outputs/submission.csv --expected-count 200
```

---

## 📝 Design Decisions & Ambiguities Resolved

1. **Prediction Independence:** Per competition rules, diagnostic prediction is strictly separated from triage decisions. When an image fails quality checks (`RE_IMAGE`), the underlying diagnosis model still runs and records its best preliminary prediction rather than being replaced with null or an error code.
2. **Deterministic Fallbacks:** The orchestrator never silently fails on corrupted or unreadable images. Any exception triggers a deterministic fallback row: `prediction = 'nv'`, `decision = 'REFER'`, with error flags recorded in the audit trace.
3. **Offline Explanations:** Explanations and clinical summaries are generated deterministically via templates and computed numeric features; no external LLM APIs are invoked during inference.
4. **Vercel Serverless Limits:** Monolithic batch POSTs are completely avoided. The batch evaluation workflow uses browser-orchestrated queuing with concurrency 3, keeping each request payload lightweight and within execution timeouts.
