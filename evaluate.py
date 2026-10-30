import os
import json
import time
import pandas as pd
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, classification_report
)

from agents.image_quality_agent import quality_agent
from agents.diagnosis_agent import load_diagnosis_agent, diagnosis_transforms
from orchestrator import AgenticDermaGuard

DX_TO_IDX = {
    'mel': 0,
    'nv': 1,
    'bcc': 2,
    'akiec': 3,
    'bkl': 4,
    'df': 5,
    'vasc': 6
}

IDX_TO_NAME = {
    0: 'Melanoma',
    1: 'Melanocytic nevus',
    2: 'Basal cell carcinoma',
    3: 'Actinic keratoses',
    4: 'Benign keratosis',
    5: 'Dermatofibroma',
    6: 'Vascular lesion'
}

def run_evaluation(eval_metadata="Health/eval/metadata.csv", eval_images_dir="Health/eval/images"):
    print("[Eval] Starting Level 1 validation run...")
    start_time = time.time()
    
    # Reload trained model
    diagnosis_agent = load_diagnosis_agent("models/diagnosis_agent.pt")
    orchestrator = AgenticDermaGuard()
    orchestrator.diagnosis_agent = diagnosis_agent

    df = pd.read_csv(eval_metadata)
    df['target_idx'] = df['dx'].map(DX_TO_IDX)
    df['target_name'] = df['target_idx'].map(IDX_TO_NAME)
    
    results = []
    
    print(f"[Eval] Processing {len(df)} validation images through multi-agent pipeline...")
    for idx, row in df.iterrows():
        img_id = row['image_id']
        img_path = os.path.join(eval_images_dir, f"{img_id}.jpg")
        
        if not os.path.exists(img_path):
            continue
            
        # Run orchestrator
        eval_res = orchestrator.process_image(img_path)
        
        # In case quality was RE_IMAGE, still obtain diagnosis for pure baseline classification stats
        if eval_res['diagnosis_summary'] is None:
            img = Image.open(img_path).convert('RGB')
            tensor = diagnosis_transforms(img)
            diag = diagnosis_agent.predict(tensor)
        else:
            diag = eval_res['diagnosis_summary']
            
        qual = eval_res['quality_summary']
        
        results.append({
            'image_id': img_id,
            'ground_truth_dx': row['dx'],
            'ground_truth_class': row['target_name'],
            'ground_truth_idx': row['target_idx'],
            'predicted_class': diag['predicted_class'],
            'predicted_idx': diag['predicted_idx'],
            'confidence': diag['confidence'],
            'is_correct': (diag['predicted_idx'] == row['target_idx']),
            'final_decision': eval_res['final_decision'],
            'primary_agent': eval_res['primary_agent'],
            'explanation': eval_res['explanation'],
            'quality_status': qual['status'],
            'overall_quality': qual['overall_quality'],
            'blur_score': qual['blur_score'],
            'lighting_score': qual['lighting_score'],
            'glare_score': qual['glare_score'],
            'occlusion_score': qual['occlusion_score'],
            'quality_reasons': " | ".join(qual['reasons'])
        })
        
        if (idx + 1) % 250 == 0 or (idx + 1) == len(df):
            elapsed = time.time() - start_time
            print(f"  Evaluated {idx + 1}/{len(df)} images ({elapsed:.1f}s)")
            
    res_df = pd.DataFrame(results)
    
    # --- Metrics Computation ---
    y_true = res_df['ground_truth_idx'].values
    y_pred = res_df['predicted_idx'].values
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7))).tolist()
    
    # Quality breakdown
    qual_status_counts = res_df['quality_status'].value_counts().to_dict()
    triage_counts = res_df['final_decision'].value_counts().to_dict()
    
    # Class-wise report
    target_names = [IDX_TO_NAME[i] for i in range(7)]
    cls_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    
    summary = {
        "event": "JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026",
        "track": "Track 2 | Healthcare",
        "project": "Agentic DermaGuard: Multi-Agent Skin Lesion Screening & Triage",
        "stage": "Day 1 / Level 1 Benchmark Submission",
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_eval_samples": len(res_df),
        "baseline_classification_metrics": {
            "overall_accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1_score": round(float(f1_macro), 4),
            "weighted_f1_score": round(float(f1_weighted), 4),
            "class_metrics": {
                name: {
                    "precision": round(cls_report[name]["precision"], 4),
                    "recall": round(cls_report[name]["recall"], 4),
                    "f1_score": round(cls_report[name]["f1-score"], 4),
                    "support": int(cls_report[name]["support"])
                }
                for name in target_names
            },
            "confusion_matrix": cm
        },
        "image_quality_agent_outcomes": {
            "mean_overall_quality": round(float(res_df['overall_quality'].mean()), 2),
            "mean_blur_score": round(float(res_df['blur_score'].mean()), 2),
            "mean_lighting_score": round(float(res_df['lighting_score'].mean()), 2),
            "mean_glare_score": round(float(res_df['glare_score'].mean()), 2),
            "mean_occlusion_score": round(float(res_df['occlusion_score'].mean()), 2),
            "quality_status_distribution": qual_status_counts
        },
        "multi_agent_triage_decisions": {
            "distribution": triage_counts,
            "percentages": {k: round(v / len(res_df) * 100, 2) for k, v in triage_counts.items()}
        }
    }
    
    # Save outputs
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/validation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    res_df.to_csv("outputs/validation_detailed.csv", index=False)
    
    # Generate clean Markdown summary
    md_content = f"""# Day 1 / Level 1 Evaluation Summary: Agentic DermaGuard

**Jaypee Agentic AI International Summit 2026 — Track 2 | Healthcare**
**Submission Date:** {time.strftime('%d October %Y %H:%M:%S')}

---

## 1. Baseline Classification Performance (Diagnosis Agent)

- **Total Validation Images:** {len(res_df)}
- **Overall Accuracy:** {acc*100:.2f}%
- **Balanced Accuracy:** {bal_acc*100:.2f}%
- **Macro F1-Score:** {f1_macro:.4f}
- **Weighted F1-Score:** {f1_weighted:.4f}

### Per-Class Performance
| Lesion Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
"""
    for name in target_names:
        stats = cls_report[name]
        md_content += f"| **{name}** | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1-score']:.4f} | {stats['support']} |\n"
        
    md_content += f"""
---

## 2. Image Quality Agent Outcomes

- **Mean Overall Quality Score:** {res_df['overall_quality'].mean():.2f} / 100
- **Mean Blur Score (Sharpness):** {res_df['blur_score'].mean():.2f} / 100
- **Mean Lighting Score (Entropy):** {res_df['lighting_score'].mean():.2f} / 100
- **Mean Glare Score (V-channel):** {res_df['glare_score'].mean():.2f} / 100
- **Mean Occlusion Score (Centering):** {res_df['occlusion_score'].mean():.2f} / 100

### Quality Status Breakdown
"""
    for k, v in qual_status_counts.items():
        md_content += f"- **{k}:** {v} images ({v/len(res_df)*100:.1f}%)\n"
        
    md_content += f"""
---

## 3. Multi-Agent Triage Decisions (`AUTO_TRIAGE`, `REFER`, `RE_IMAGE`)

The multi-agent system synthesizes the findings of the Image Quality Agent and Diagnosis Agent into an explainable decision:

| Triage Decision | Count | Percentage |
|---|---|---|
"""
    for k, v in triage_counts.items():
        md_content += f"| **{k}** | {v} | {v/len(res_df)*100:.1f}% |\n"

    md_content += f"""
---
*Results saved to `outputs/validation_results.json` and `outputs/validation_detailed.csv`.*
"""

    with open("outputs/validation_summary.md", "w") as f:
        f.write(md_content)
        
    print(f"\n[Success] Validation complete! Accuracy: {acc*100:.2f}% | Balanced Acc: {bal_acc*100:.2f}%")
    print(f"Results saved to outputs/validation_results.json and outputs/validation_summary.md")
    return summary

if __name__ == "__main__":
    run_evaluation()
