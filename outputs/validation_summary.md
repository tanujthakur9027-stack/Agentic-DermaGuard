# Day 1 / Level 1 Evaluation Summary: Agentic DermaGuard

**Jaypee Agentic AI International Summit 2026 — Track 2 | Healthcare**
**Submission Date:** 30 October 2026 17:18:47

---

## 1. Baseline Classification Performance (Diagnosis Agent)

- **Total Validation Images:** 1942
- **Overall Accuracy:** 69.82%
- **Balanced Accuracy:** 64.82%
- **Macro F1-Score:** 0.5516
- **Weighted F1-Score:** 0.7254

### Per-Class Performance
| Lesion Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Melanoma** | 0.3708 | 0.6636 | 0.4757 | 214.0 |
| **Melanocytic nevus** | 0.9662 | 0.7458 | 0.8418 | 1302.0 |
| **Basal cell carcinoma** | 0.3947 | 0.5941 | 0.4743 | 101.0 |
| **Actinic keratoses** | 0.4078 | 0.6667 | 0.5060 | 63.0 |
| **Benign keratosis** | 0.4775 | 0.5000 | 0.4885 | 212.0 |
| **Dermatofibroma** | 0.3143 | 0.4783 | 0.3793 | 23.0 |
| **Vascular lesion** | 0.5714 | 0.8889 | 0.6957 | 27.0 |

---

## 2. Image Quality Agent Outcomes

- **Mean Overall Quality Score:** 79.77 / 100
- **Mean Blur Score (Sharpness):** 89.61 / 100
- **Mean Lighting Score (Entropy):** 76.62 / 100
- **Mean Glare Score (V-channel):** 94.39 / 100
- **Mean Occlusion Score (Centering):** 58.47 / 100

### Quality Status Breakdown
- **ACCEPTABLE:** 1742 images (89.7%)
- **ACCEPTABLE_WITH_CAUTION:** 187 images (9.6%)
- **RE_IMAGE:** 13 images (0.7%)

---

## 3. Multi-Agent Triage Decisions (`AUTO_TRIAGE`, `REFER`, `RE_IMAGE`)

The multi-agent system synthesizes the findings of the Image Quality Agent and Diagnosis Agent into an explainable decision:

| Triage Decision | Count | Percentage |
|---|---|---|
| **REFER** | 1144 | 58.9% |
| **AUTO_TRIAGE** | 785 | 40.4% |
| **RE_IMAGE** | 13 | 0.7% |

---
*Results saved to `outputs/validation_results.json` and `outputs/validation_detailed.csv`.*
