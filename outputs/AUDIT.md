# Agentic DermaGuard: System Audit & Codebase Inspection Report
**Jaypee Agentic AI International Summit 2026 — Track 2 | Healthcare**
**Date:** 31 October 2026 | **Author:** Senior Lead Developer, Team DermaGuard

---

## 1. Executive Summary
This audit reviews all six agents, orchestrator, training, evaluation, submission scripts, and Vercel serverless integration prior to implementing centralized thresholds, automated tuning, and ablation reporting. The architecture adheres to clinical dermoscopy guidelines (ABCD rule, Stolz Total Dermoscopy Score, demographic prior interaction, and sentinel veto).

---

## 2. Core Mandatory Audit Checks

### (a) Train / Validation Lesion Leakage Audit
- **Dataset Analysis:**
  - Training dataset (`Health/training/metadata.csv`): **7,762** samples across **5,815** unique `lesion_id` records.
  - Validation dataset (`Health/eval/metadata.csv`): **1,942** samples across **1,456** unique `lesion_id` records.
  - Intersection between train and validation `lesion_id` sets: **EXACTLY 0 (ZERO LEAKAGE)**.
- **Finding:** The train/validation split is completely patient/lesion disjoint. Multiple captures of the same anatomical lesion do not cross the partition boundary. Diagnostic performance metrics on validation reflect true generalization.

### (b) Validation Dataset Size Discrepancy (1,942 vs. 2,000)
- **Metadata Analysis:** `Health/eval/metadata.csv` defines exactly **1,942 rows**.
- **Disk Analysis:** `Health/eval/images/` contains **3,884 image files**. Every image was saved twice on disk: once as `{image_id}.jpg` and once with Windows duplicate naming `{image_id} (2).jpg` (1,942 base images + 1,942 duplicate files = 3,884 files).
- **HAM10000 Breakdown:**
  - The complete canonical HAM10000 dataset contains 10,015 images.
  - Training set provided: 7,762 images.
  - Validation set provided: 1,942 images.
  - Total provided images: $7,762 + 1,942 = 9,704$ images.
  - The remaining **311 images** were held out by the summit organizers (from which the 200 hidden test images released at 10:00 AM are sampled).

### (c) Model Serving Parity: `api/index.py` vs. `evaluate.py`
- **Local Environment:**
  - `evaluate.py` initializes `DiagnosisAgent(model_path="models/diagnosis_agent.pt")` running PyTorch EfficientNet-B0 (17.65 MB).
  - `api/index.py` imports `agentic_dermaguard` (or `diagnosis_agent`), which loads the identical `models/diagnosis_agent.pt` whenever PyTorch and CUDA/CPU are available.
- **Vercel Serverless Production Environment:**
  - Vercel functions have a strict bundle size constraint (~250 MB decompressed), precluding `torch` and `torchvision` (~1.5 GB).
  - On Vercel, `HAVE_TORCH` evaluates to `False`. `DiagnosisAgent` gracefully dispatches to `predict_numpy_fallback()` using `models/lightweight_weights.json` (4.1 KB, 15 extracted clinical features standardized with StandardScaler + Logistic Regression coefficients).
- **Action Required:** Ensure probability threshold tuning is calibrated for the PyTorch model on local/evaluation environments while keeping the lightweight fallback fully operational and calibrated with the same centralized threshold contract.

---

## 3. Current Agent Rules, Formulas & Thresholds

| Agent | Current Parameters & Defaults | Mathematical Formulation / Logic | Status / Weaknesses |
|---|---|---|---|
| **A1: Image Quality** | `blur_threshold`: 40.0<br>`lighting_threshold`: 40.0<br>`glare_threshold`: 70.0<br>`occlusion_threshold`: 50.0<br>`quality_cutoff`: 50.0 | $\text{blur} = 0.55 \frac{\text{Var}(\text{Lap})}{55} \times 100 + 0.45 \frac{\text{Tenengrad}}{800} \times 100$<br>$\text{glare} = 100 - (\text{specular\_ratio} \times 1800)$ where specular is $\min(R,G,B) \ge 246$<br>$\text{occl} = 0.90 \times \text{center} + 20 \times (1 - 2.5 \times \text{outer})$ | Well calibrated. Glare and blur no longer produce false RE_IMAGE on normal fair skin. Caution band (`overall_quality < 72.0`) active. |
| **A2: Diagnosis** | `auto_triage_confidence`: 0.70<br>`margin_minimum`: 0.15<br>`malignancy_threshold`: 0.28 | Softmax probabilities over 7 classes (`mel`, `nv`, `bcc`, `akiec`, `bkl`, `df`, `vasc`). Top-2 margin $= p_{(1)} - p_{(2)}$. | Lacks per-class probability calibration/bias for imbalanced classes (e.g. `df`, `vasc`). Needs Test-Time Augmentation (TTA) for uncertainty estimation. |
| **A3: Lesion Analysis** | `asymmetry_weight`: 0.35<br>`border_weight`: 0.35<br>`color_weight`: 0.30<br>TDS thresholds: $<4.75$ Benign, $4.75-5.45$ Indeterminate, $>5.45$ Suspicious | $\text{TDS} = 1.3 A + 0.1 B + 0.5 C + 0.5 D$<br>Asymmetry $A \in \{0, 1, 2\}$ via horizontal & vertical contour folding.<br>Border $B \in \{0..8\}$ via 8 octant radial gradients.<br>Color $C \in \{1..6\}$ dermoscopic color detection. | Accurate dermoscopic ABCD segmentation; generates visual edge overlay and crosshairs. |
| **A4: Context Reasoning** | `age_cutoffs`: 25, 40, 60, 75<br>`risk_indicator`: $<0.45$ LOW, $0.45-0.70$ MODERATE, $\ge 0.70$ HIGH | Risk score accumulation: Age (0.05-0.45) + Sex (0.10-0.20) + Site (0.10-0.35). High-risk triad bonus (+0.15) for elderly male with trunk/head lesion. Missing fields flagged with neutral 0.20 prior. | Robust, but needs explicit handling when ALL demographic context is omitted (`no_context` test scenario). |
| **A5: Safety Sentinel** | Rule S1: Pred Malignancy<br>Rule S2: Residual Malignancy $\ge 0.28$<br>Rule S3: ABCD Discordance<br>Rule S4: Demographic Discordance<br>Rule S5: Quality Failure/Caution<br>Rule S6: Uncertainty $< 0.65$ | Checks for discordance between A2 and A3/A4. Vetoes `AUTO_TRIAGE` if any conflict or risk rule triggers. | Triggers were partially hardcoded; need centralized flag toggles and systematic naming S1-S6 with traceable rule IDs. |
| **A6: Coordinator** | Decision Precedence:<br>1. Quality Failure $\to$ `RE_IMAGE`<br>2. Malignancy Presentation $\to$ `REFER`<br>3. Sentinel Veto $\to$ `REFER`<br>4. Diagnostic Uncertainty $\to$ `REFER`<br>5. Verified Benign $\to$ `AUTO_TRIAGE` | Evaluates inputs sequentially. Formats traceable chain: `A1: PASS \| A2: nv • 0.73 \| A3+A4: SUSPICIOUS \| A5: SAFETY FLAG \| A6: REFER`. | Documented policy logic works cleanly; needs centralized configuration for auto triage confidence and safety margins. |

---

## 4. Key Bugs Identified & Remediation Plan

1. **Threshold Fragmentation:**
   - Thresholds are currently duplicated between `AgenticDermaGuard.__init__`, `ImageQualityAgent.__init__`, `CoordinatorAgent.__init__`, `SafetyAgent`, and `api/index.py`.
   - *Fix:* Create centralized `thresholds.json` and `thresholds.py` as single source of truth.
2. **Missing Demographic Context Robustness:**
   - On the hidden test set, organizers may provide only image files without demographic tables.
   - *Fix:* Verify Agent 4 outputs neutral risk and `MISSING_*` flags without raising exceptions when all metadata is None; benchmark `evaluate.py` in both `with_context` and `no_context` modes.
3. **Imbalanced Class Predictions & Calibration:**
   - Rare classes (`df`, `vasc`, `akiec`) suffer from class imbalance under raw softmax.
   - *Fix:* In Step 2, run coordinate ascent bias tuning on validation set A to optimize Macro-F1 on `prediction`.
4. **Test-Time Augmentation & Uncertainty:**
   - Currently, A2 does a single pass. Adding 4-view TTA (original, hflip, vflip, rot180) provides both higher classification accuracy and an explicit disagreement signal to Agent 5.
5. **Submission Generation Edge Cases:**
   - Hidden test inference must handle `.jpg`, `.jpeg`, `.png`, prevent skipping failed images, handle duplicate stems gracefully, and validate row counts and columns.

---
**Audit Approved for Progression to Step 1.**
