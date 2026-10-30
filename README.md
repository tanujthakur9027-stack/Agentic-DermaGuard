# Agentic-DermaGuard: Multi-Agent Skin Lesion Screening & Triage

**Jaypee Agentic AI International Summit 2026 — Track 2 | Healthcare**

Agentic DermaGuard is a collaborative multi-agent clinical decision support prototype that analyzes dermoscopic skin-lesion images, classifies 7 lesion pathologies from the HAM10000 dataset, and reaches an explainable final tri-state clinical triage decision: **`AUTO_TRIAGE`**, **`REFER`**, or **`RE_IMAGE`**.

---

## 🏗️ Multi-Agent Architecture

```
                      [ Dermoscopic Image Input ]
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │    1. Image Quality Agent     │
                   │ (Blur, Lighting, Glare, Occl) │
                   └───────────────┬───────────────┘
                                   │
                    Quality Failed? (RE_IMAGE)
                                   ├──────────────────────────► [ RE_IMAGE Decision ]
                                   │                              (Re-capture required)
                        Quality Passed (ACCEPTABLE)
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │      2. Diagnosis Agent       │
                   │ (7-Class Pathologies + Conf)  │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │    Multi-Agent Orchestrator   │
                   │   Consensus Decision Engine   │
                   └───────┬───────────────┬───────┘
                           │               │
        Malignant / Low Conf               Confident Benign
                           │               │
                           ▼               ▼
                   [ REFER Decision ]   [ AUTO_TRIAGE Decision ]
                 (Specialist Referral)  (Primary Care Follow-up)
```

### 1. Image Quality Agent (`agents/image_quality_agent.py`)
* **Blur:** Evaluates high-frequency sharpness using Laplacian variance ($\sigma^2$).
* **Illumination:** Measures dynamic range via grayscale histogram entropy ($H$).
* **Glare:** Detects specular reflection saturation in HSV Value channel ($V > 240$).
* **Occlusion:** Analyzes Canny edge concentration comparing lesion center against peripheral borders.
* **Explainability:** Rejects degraded captures (`RE_IMAGE`) before diagnostic errors occur.

### 2. Diagnosis Agent (`agents/diagnosis_agent.py`)
* Fine-tuned **EfficientNet-B0** classifier trained with class-weighted CrossEntropy loss across the 7 HAM10000 classes:
  * Melanoma (`mel`), Melanocytic nevus (`nv`), Basal cell carcinoma (`bcc`), Actinic keratoses (`akiec`), Benign keratosis (`bkl`), Dermatofibroma (`df`), Vascular lesion (`vasc`).
* Yields full probability distribution, top-3 differential diagnosis, and clinical confidence score.

### 3. Orchestrator (`orchestrator.py`)
* Synthesizes findings into a clinically actionable recommendation:
  * **`RE_IMAGE`:** Quality below acceptable diagnostic threshold.
  * **`REFER`:** High risk of malignancy (Melanoma, BCC, AKIEC) or high diagnostic uncertainty.
  * **`AUTO_TRIAGE`:** High-confidence benign lesion on verified high-quality dermoscopy.

---

## 📊 Level 1 Validation Benchmark (1,942 Eval Images)

* **Overall Accuracy:** **69.82%**
* **Balanced Accuracy:** **64.82%**
* **Macro F1-Score:** **0.5516**
* **Weighted F1-Score:** **0.7254**

| Lesion Class | Category | Precision | Recall | F1-Score | Support |
|---|---|:---:|:---:|:---:|:---:|
| **Melanoma (`mel`)** | Malignant / High Risk | 0.3708 | **66.36%** | 0.4757 | 214 |
| **Melanocytic nevus (`nv`)** | Benign | 0.9662 | 74.58% | **0.8418** | 1,302 |
| **Basal cell carcinoma (`bcc`)** | Malignant / High Risk | 0.3947 | 59.41% | 0.4743 | 101 |
| **Actinic keratoses (`akiec`)** | Pre-malignant | 0.4078 | 66.67% | 0.5060 | 63 |
| **Benign keratosis (`bkl`)** | Benign | 0.4775 | 50.00% | 0.4885 | 212 |
| **Dermatofibroma (`df`)** | Benign | 0.3143 | 47.83% | 0.3793 | 23 |
| **Vascular lesion (`vasc`)** | Benign | 0.5714 | **88.89%** | **0.6957** | 27 |

### Multi-Agent Triage Outcomes
* **`REFER`:** 1,144 images (58.91%)
* **`AUTO_TRIAGE`:** 785 images (40.42%)
* **`RE_IMAGE`:** 13 images (0.67%)

---

## 🚀 Running Locally

### Streamlit Interactive App:
```bash
python -m streamlit run app.py
```

### Running Validation Benchmark:
```bash
python evaluate.py
```

### Submitting to Competition API:
```bash
python submit.py --api-url "<API_URL>" --team-name "<TEAM_NAME>" --token "<TOKEN>"
```

---

## 🌐 Vercel Deployment
This repository is configured for serverless deployment on Vercel:
* `vercel.json`: Edge routing configuration
* `api/index.py`: Serverless Python backend with lightweight NumPy/PIL quality agent
* `public/index.html`: Responsive clinical screening dashboard with Tailwind CSS & Chart.js

Deploy instantly:
```bash
npx vercel --prod
```
