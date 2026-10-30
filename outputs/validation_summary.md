# Level 2 / Day 2 Evaluation Summary: Six-Agent DermaGuard

**Jaypee Agentic AI International Summit 2026 — Track 2 | Healthcare**
**Evaluation Date:** 30 October 2026 22:36:29

---

## 1. Final Live Demonstration Metrics (Required Reporting)

- **Total Images Evaluated:** 1942
- **Overall Classification Accuracy:** 69.82%
- **Balanced Accuracy:** 64.82%
- **Macro F1-Score:** 0.5516
- **Weighted F1-Score:** 0.7254

### Triage Disposition Counts
| Triage Disposition | Count | Percentage | Description |
|---|---|---|---|
| **Correct AUTO_TRIAGE** | 134 | 6.9% | Automated screening matches ground truth label |
| **Incorrect AUTO_TRIAGE** | 9 | 0.5% | Automated screening misclassified |
| **Number of REFERS** | 1647 | 84.8% | Suspicion, diagnostic uncertainty, or safety veto |
| **Number of RE_IMAGE** | 152 | 7.8% | Inadequate capture quality (blur/glare/centering) |

---

## 2. Per-Class Classification Performance (7 Permitted Codes)

| Code | Class Name | Precision | Recall (Sensitivity) | F1-Score | Support |
|---|---|---|---|---|---|
| **mel** | Melanoma | 37.08% | 66.36% | 0.4757 | 214.0 |
| **nv** | Melanocytic nevus | 96.62% | 74.58% | 0.8418 | 1302.0 |
| **bcc** | Basal cell carcinoma | 39.47% | 59.41% | 0.4743 | 101.0 |
| **akiec** | Actinic keratoses | 40.78% | 66.67% | 0.5060 | 63.0 |
| **bkl** | Benign keratosis | 47.75% | 50.00% | 0.4885 | 212.0 |
| **df** | Dermatofibroma | 31.43% | 47.83% | 0.3793 | 23.0 |
| **vasc** | Vascular lesion | 57.14% | 88.89% | 0.6957 | 27.0 |

---

## 3. Six-Agent Contribution & Cross-Agent Safety Audit

- **Agent 1 (Image Quality):** Mean Quality: 76.49/100. Status: {'ACCEPTABLE': 1177, 'ACCEPTABLE_WITH_CAUTION': 613, 'RE_IMAGE': 152}
- **Agent 2 (Diagnosis):** Evaluated all 1942 images across all 7 permitted classes.
- **Agent 3 (Lesion Analysis ABCD):** Mean Suspicion Score: 0.640.
- **Agent 4 (Reasoning / Context):** Stratified by age, sex, and anatomical localization.
- **Agent 5 (Safety Sentinel):** Vetoes Triggered: 1796 unsafe autonomous discharges blocked.
- **Agent 6 (Coordinator):** Applied deterministic precedence policy ensuring explainability and audit traceability.

---
*CSV submission generated at `outputs/final_submission_format_sample.csv` (Columns: `image_id,prediction,decision`).*
