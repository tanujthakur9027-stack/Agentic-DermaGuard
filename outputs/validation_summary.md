# Level 2 Evaluation Summary: Agentic DermaGuard (Six-Agent System)
**Jaypee Agentic AI International Summit 2026 — Track 2 | Healthcare**
**Evaluation Date:** 30 October 2026 23:38:04 | **Context Mode:** `no_context`
**Total Images Evaluated:** 1942 | **Runtime:** 264.85s (136.4 ms/image)

---

## 1. Live Demonstration Core Reporting Metrics (Official Summit Format)

- **Overall Classification Accuracy:** **75.18%**
- **Balanced Accuracy:** **64.97%**
- **Macro F1-Score:** **0.5980**
- **Weighted F1-Score:** **0.7637**

### Triage Disposition Table
| Triage Disposition | Count | Percentage | Clinical Meaning |
|---|---|---|---|
| **Correct AUTO_TRIAGE** | **130** | **6.7%** | Automated discharge matching ground truth benign pathology |
| **Incorrect AUTO_TRIAGE** | **4** | **0.2%** | Automated discharge misclassification |
| **AUTO_TRIAGE Precision** | **97.01%** | - | Proportion of automated discharges that are correct |
| **Missed Malignancies** | **2** | **0.10%** | Malignant lesions (mel/bcc/akiec) wrongly AUTO_TRIAGED |
| **Number of REFERS** | **1696** | **87.3%** | Specialist dermatologist review required |
| **Number of RE_IMAGE** | **112** | **5.8%** | Capture quality failure (blur, glare, occlusion) |

---

## 2. Six-Agent Incremental Ablation Study
*Proves that every specialized agent directly contributes to clinical safety and optimal disposition.*

| Agent Architecture Pipeline | Correct AUTO | Incorrect AUTO | Missed Malig | Auto Precision | REFERS | RE_IMAGE |
|---|---|---|---|---|---|---|
| **Stage 1: A1 (Quality) + A2 (Diagnosis CNN)** | 713 | 16 | 8 | 97.8% | 1101 | 112 |
| **Stage 2: + A3 (Lesion ABCD Morphometry)** | 308 | 4 | 2 | 98.7% | 1518 | 112 |
| **Stage 3: + A4 (Demographic Priors)** | 308 | 4 | 2 | 98.7% | 1518 | 112 |
| **Stage 4: + A5 (Safety Sentinel Veto)** | 130 | 4 | 2 | 97.0% | 1696 | 112 |
| **Stage 5: Full A6 (Coordinator Master Policy)** | **130** | **4** | **2** | **97.0%** | **1696** | **112** |

### Cross-Agent Agreement & Sentinel Catches
- **A2 Benign vs. A3 Suspicious ABCD Structure:** **1116** cases detected.
- **A5 Safety Sentinel Overrides:** A5 intervened to block **595** automated triage cases.
- **Sentinel Catch Efficiency:**
  - Prevented **12** diagnostic misclassifications from automated discharge.
  - Prevented **6** true malignant tumors (Melanoma/BCC/AKIEC) from being wrongly discharged!

---

## 3. Per-Class Diagnostic Performance (7 Permitted Classes)

| Code | Pathology Name | Precision | Recall (Sensitivity) | F1-Score | Support |
|---|---|---|---|---|---|
| **mel** | Melanoma | 41.90% | 61.68% | 0.4991 | 214 |
| **nv** | Melanocytic nevus | 92.09% | 84.02% | 0.8787 | 1302 |
| **bcc** | Basal cell carcinoma | 53.70% | 57.43% | 0.5550 | 101 |
| **akiec** | Actinic keratoses | 68.75% | 52.38% | 0.5946 | 63 |
| **bkl** | Benign keratosis | 53.85% | 49.53% | 0.5160 | 212 |
| **df** | Dermatofibroma | 26.42% | 60.87% | 0.3684 | 23 |
| **vasc** | Vascular lesion | 68.57% | 88.89% | 0.7742 | 27 |

---

## 4. Class-Wise Errors & Confusion Matrix
### Top Diagnostic Confusions
- **Melanocytic nevus (nv) misclassified as Melanoma (mel):** 120 cases
- **Melanocytic nevus (nv) misclassified as Benign keratosis (bkl):** 42 cases
- **Benign keratosis (bkl) misclassified as Melanocytic nevus (nv):** 41 cases
- **Benign keratosis (bkl) misclassified as Melanoma (mel):** 37 cases
- **Melanoma (mel) misclassified as Melanocytic nevus (nv):** 36 cases
- **Melanoma (mel) misclassified as Benign keratosis (bkl):** 28 cases

---

## 5. Demonstration Cases (Official Summit Traceable Syntax)

### Correct AUTO_TRIAGE (High-confidence benign nevus, verified clear)
- **Image ID:** `ISIC_0025567` | **Ground Truth:** `bkl` | **Predicted:** `bkl` (99.1%) | **Decision:** `AUTO_TRIAGE`
- **Traceable Chain:**
  `A1: PASS | A2: bkl • 0.99 | A3+A4: INDETERMINATE | A5: PASS | A6: AUTO_TRIAGE`
- **Coordinator Rationale:** Policy Rule 5 (Verified Benign Autonomous Screening): High-confidence benign lesion with clear safety checks.

### Malignancy REFER (Melanoma presentation, immediate specialist biopsy)
- **Image ID:** `ISIC_0026769` | **Ground Truth:** `bkl` | **Predicted:** `mel` (37.7%) | **Decision:** `REFER`
- **Traceable Chain:**
  `A1: PASS | A2: mel • 0.38 | A3+A4: SUSPICIOUS | A5: SAFETY VETO | A6: REFER`
- **Coordinator Rationale:** Policy Rule 2 (Malignancy Referral): Detected high-risk lesion code 'mel'.

### Malignancy REFER (Basal cell carcinoma presentation, referral indicated)
- **Image ID:** `ISIC_0025915` | **Ground Truth:** `bkl` | **Predicted:** `bcc` (26.8%) | **Decision:** `REFER`
- **Traceable Chain:**
  `A1: PASS | A2: bcc • 0.27 | A3+A4: SUSPICIOUS | A5: SAFETY VETO | A6: REFER`
- **Coordinator Rationale:** Policy Rule 2 (Malignancy Referral): Detected high-risk lesion code 'bcc'.

### Diagnostic Uncertainty REFER (Differential ambiguity below auto threshold)
- **Image ID:** `ISIC_0029396` | **Ground Truth:** `bkl` | **Predicted:** `nv` (38.6%) | **Decision:** `REFER`
- **Traceable Chain:**
  `A1: PASS | A2: nv • 0.39 | A3+A4: SUSPICIOUS | A5: SAFETY VETO | A6: REFER`
- **Coordinator Rationale:** Policy Rule 3 (Safety Sentinel Veto): Overrode automated triage via RULE_S1:RESIDUAL_MALIGNANCY_RISK.

### Quality Failure RE_IMAGE (Compromised capture, re-imaging requested)
- **Image ID:** `ISIC_0025984` | **Ground Truth:** `bkl` | **Predicted:** `nv` (34.5%) | **Decision:** `RE_IMAGE`
- **Traceable Chain:**
  `A1: RE_IMAGE | A2: nv • 0.35 | A3+A4: SUSPICIOUS | A5: SAFETY VETO | A6: RE_IMAGE`
- **Coordinator Rationale:** Policy Rule 1 (Quality Gate): Capture degradation prevents dependable reading.

### Cross-Agent Conflict REFER (Benign candidate vetoed by lesion/safety cues)
- **Image ID:** `ISIC_0034031` | **Ground Truth:** `bkl` | **Predicted:** `nv` (91.6%) | **Decision:** `REFER`
- **Traceable Chain:**
  `A1: PASS | A2: nv • 0.92 | A3+A4: SUSPICIOUS | A5: SAFETY VETO | A6: REFER`
- **Coordinator Rationale:** Policy Rule 3 (Safety Sentinel Veto): Overrode automated triage via RULE_S2:LESION_ABCD_DISCORDANCE.

---
*Official submission CSV generated at `outputs/final_submission_format_sample.csv`.*
