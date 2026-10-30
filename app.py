import os
import json
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

from agents.image_quality_agent import quality_agent
from agents.diagnosis_agent import load_diagnosis_agent, diagnosis_transforms
from orchestrator import AgenticDermaGuard

st.set_page_config(
    page_title="Agentic DermaGuard | Multi-Agent Skin Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .decision-badge {
        padding: 16px 24px;
        border-radius: 12px;
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1.2rem;
        letter-spacing: 0.02em;
    }
    .badge-refer {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 2px solid #EF4444;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
    }
    .badge-auto {
        background-color: #F0FDF4;
        color: #166534;
        border: 2px solid #22C55E;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.15);
    }
    .badge-reimage {
        background-color: #FFFBEB;
        color: #92400E;
        border: 2px solid #F59E0B;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);
    }
    .agent-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .criteria-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_orchestrator():
    model = load_diagnosis_agent("models/diagnosis_agent.pt")
    orch = AgenticDermaGuard()
    orch.diagnosis_agent = model
    return orch

orchestrator = get_orchestrator()

# --- SIDEBAR: NAVIGATION & THRESHOLD CRITERIA ---
st.sidebar.image("https://img.icons8.com/color/96/medical-heart.png", width=64)
st.sidebar.title("DermaGuard Control")
st.sidebar.caption("Track 2 | Healthcare — Summit 2026")

nav = st.sidebar.radio(
    "Navigation",
    ["🔬 Interactive Screening", "⚙️ Threshold Criteria Settings", "📊 Level 1 Benchmark Dashboard", "🚀 API Submission"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Live Threshold Tuning")

# Threshold Controls in Sidebar
blur_th = st.sidebar.slider("Blur Cutoff (Variance)", 10.0, 100.0, 40.0, 5.0, help="Lower = stricter blur detection")
light_th = st.sidebar.slider("Lighting Cutoff (Entropy)", 10.0, 100.0, 40.0, 5.0, help="Histogram entropy minimum")
glare_th = st.sidebar.slider("Glare Cutoff (V-Channel)", 50.0, 100.0, 70.0, 5.0, help="Lower score indicates specular reflection")
occl_th = st.sidebar.slider("Centering Cutoff (%)", 20.0, 100.0, 50.0, 5.0, help="Min edge concentration in central quadrant")
qual_cutoff = st.sidebar.slider("Re-Image Cutoff Score", 20.0, 80.0, 50.0, 5.0, help="Images scoring below this are flagged RE_IMAGE")

st.sidebar.markdown("**Clinical Decision Thresholds:**")
malig_th = st.sidebar.slider("Malignancy Referral Trigger", 0.10, 0.80, 0.35, 0.05, help="Referral triggered if Melanoma/BCC probability exceeds this")
auto_th = st.sidebar.slider("Auto-Triage Confidence Min", 0.50, 0.95, 0.70, 0.05, help="Minimum confidence needed to discharge benign lesion without referral")

active_thresholds = {
    "blur_threshold": blur_th,
    "lighting_threshold": light_th,
    "glare_threshold": glare_th,
    "occlusion_threshold": occl_th,
    "quality_cutoff": qual_cutoff,
    "malignancy_threshold": malig_th,
    "auto_triage_confidence": auto_th
}

# -------------------------------------------------------------
# PAGE 1: INTERACTIVE SCREENING
# -------------------------------------------------------------
if nav == "🔬 Interactive Screening":
    st.markdown('<div class="main-title">Multi-Agent Skin Lesion Screening & Triage</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Two collaborative agents gatekeep capture quality and classify 7 dermoscopic pathologies to reach an explainable clinical decision.</div>', unsafe_allow_html=True)
    
    col_input, col_view = st.columns([1, 1])
    
    sample_options = {
        "Sample 1: Melanocytic Nevus (Benign Mole)": "Health/eval/images/ISIC_0025661.jpg",
        "Sample 2: Melanoma (Malignant Lesion)": "Health/eval/images/ISIC_0031023.jpg",
        "Sample 3: Basal Cell Carcinoma (Malignant)": "Health/eval/images/ISIC_0028155.jpg",
        "Sample 4: Vascular Lesion (Benign Angioma)": "Health/eval/images/ISIC_0031197.jpg",
        "Sample 5: Actinic Keratoses (Pre-Malignant)": "Health/eval/images/ISIC_0029360.jpg",
        "Sample 6: Benign Keratosis (Seborrheic Keratosis)": "Health/eval/images/ISIC_0026769.jpg",
        "Sample 7: Dermatofibroma (Benign Nodule)": "Health/eval/images/ISIC_0032138.jpg",
        "Sample 8: Degraded Motion Blur (Re-Image)": "Health/eval/images/ISIC_0030291.jpg",
        "Sample 9: Severe Specular Glare (Re-Image)": "Health/eval/images/ISIC_0026859.jpg",
        "Sample 10: Peripheral Occlusion (Re-Image)": "Health/eval/images/ISIC_0030775.jpg"
    }

    with col_input:
        input_source = st.radio("Select input mode:", ["Preset Clinical Cases", "Upload Custom Image"], horizontal=True)
        temp_img_path = None
        
        if input_source == "Preset Clinical Cases":
            selected_case = st.selectbox("Select Dermoscopy Case:", list(sample_options.keys()))
            temp_img_path = sample_options[selected_case]
            pil_image = Image.open(temp_img_path).convert('RGB')
        else:
            uploaded_file = st.file_uploader("Upload Dermoscopy Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                pil_image = Image.open(uploaded_file).convert('RGB')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    pil_image.save(tmp.name)
                    temp_img_path = tmp.name
            else:
                pil_image = None

    with col_view:
        if pil_image is not None:
            st.image(pil_image, caption="Dermoscopic Input (224x224)", use_container_width=True)
            run_btn = st.button("🚀 Execute Multi-Agent Screening", type="primary", use_container_width=True)
        else:
            st.info("Upload or select an image on the left to begin.")
            run_btn = False

    if pil_image is not None and (run_btn or input_source == "Preset Clinical Cases"):
        st.markdown("---")
        with st.spinner("Agents analyzing image quality and pathology..."):
            res = orchestrator.process_image(temp_img_path, custom_thresholds=active_thresholds)
            
        final_decision = res['final_decision']
        
        # Display Decision Banner
        if final_decision == "REFER":
            st.markdown(f'<div class="decision-badge badge-refer">🚨 FINAL DECISION: {final_decision} (Specialist Referral Required)</div>', unsafe_allow_html=True)
        elif final_decision == "AUTO_TRIAGE":
            st.markdown(f'<div class="decision-badge badge-auto">✅ FINAL DECISION: {final_decision} (Routine Follow-up / Discharge)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="decision-badge badge-reimage">⚠️ FINAL DECISION: {final_decision} (Re-imaging Required)</div>', unsafe_allow_html=True)
            
        col_meta1, col_meta2 = st.columns([1, 3])
        with col_meta1:
            st.write(f"**Lead Agent:** `{res['primary_agent']}`")
        with col_meta2:
            st.write(f"**Clinical Explanation:** {res['explanation']}")
            
        st.markdown("### Agent Collaboration Breakdown")
        col_ag1, col_ag2 = st.columns(2)
        
        # Agent 1 Card
        with col_ag1:
            st.markdown('<div class="agent-card">', unsafe_allow_html=True)
            st.subheader("1️⃣ Image Quality Agent (Gatekeeper)")
            q = res['quality_summary']
            
            st.metric("Overall Quality Score", f"{q['overall_quality']:.1f} / 100", q['status'])
            
            st.write("**Evaluated Quality Dimensions vs Thresholds:**")
            st.progress(float(min(100.0, q['blur_score'])) / 100.0, text=f"Sharpness: {q['blur_score']:.1f} (Cutoff: {blur_th:.1f})")
            st.progress(float(min(100.0, q['lighting_score'])) / 100.0, text=f"Illumination: {q['lighting_score']:.1f} (Cutoff: {light_th:.1f})")
            st.progress(float(min(100.0, q['glare_score'])) / 100.0, text=f"Glare Control: {q['glare_score']:.1f} (Cutoff: {glare_th:.1f})")
            st.progress(float(min(100.0, q['occlusion_score'])) / 100.0, text=f"Lesion Centering: {q['occlusion_score']:.1f} (Cutoff: {occl_th:.1f})")
            
            st.write("**Active Quality Findings:**")
            for r in q['reasons']:
                st.markdown(f"- {r}")
                
            if q.get('triggered_criteria'):
                st.write("**Triggered Defect Flags:**", ", ".join(q['triggered_criteria']))
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Agent 2 Card
        with col_ag2:
            st.markdown('<div class="agent-card">', unsafe_allow_html=True)
            st.subheader("2️⃣ Diagnosis Agent (Pathology Classifier)")
            
            if res['diagnosis_summary'] is not None:
                d = res['diagnosis_summary']
                is_high_risk = d['predicted_class'] in orchestrator.high_risk_classes
                risk_tag = "🔴 Malignant / High Risk" if is_high_risk else "🟢 Benign / Low Risk"
                
                st.metric("Predicted Pathology", d['predicted_class'], f"{d['confidence']*100:.1f}% confidence")
                st.markdown(f"**Risk Category:** {risk_tag}")
                
                probs_df = pd.DataFrame(list(d['all_probabilities'].items()), columns=["Lesion Class", "Probability"]).sort_values(by="Probability", ascending=True)
                st.bar_chart(probs_df.set_index("Lesion Class"))
                
                st.write("**Differential Diagnosis:**")
                for cls_name, prob in d['top_3']:
                    st.write(f"- **{cls_name}**: {prob*100:.2f}%")
            else:
                st.warning("⚠️ Diagnostic inference bypassed by Agent 1 to avoid inaccurate clinical predictions on degraded imagery.")
            st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 2: THRESHOLD CRITERIA SETTINGS
# -------------------------------------------------------------
elif nav == "⚙️ Threshold Criteria Settings":
    st.markdown('<div class="main-title">Threshold Criteria & Decision Rules</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed specification of mathematical cutoffs, justification rules, and multi-agent consensus logic.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. Image Quality Agent Rules
    The Image Quality Agent evaluates captures along 4 critical medical imaging axes:
    
    | Axis | Metric & Algorithm | Standard Threshold | Clinical Rationale |
    |---|---|:---:|---|
    | **Sharpness** | Laplacian Variance ($\sigma^2$) on grayscale | $\ge 40.0$ | Blurred edges obscure delicate pigment networks and globules. |
    | **Illumination** | Grayscale Histogram Entropy ($H$) | $\ge 40.0$ | Ensures dynamic range without severe underexposure or shadows. |
    | **Specular Glare** | Saturated pixels in HSV Value ($V > 240$) | $\ge 70.0$ | Dermoscopic fluid/flash glare creates white reflection voids. |
    | **Centering** | Center vs Peripheral Sobel edge density | $\ge 50.0$ | Off-center lesions risk truncated borders or vignette artifacts. |
    | **Overall Quality** | Composite arithmetic mean | $\ge 50.0$ | Composite quality required to authorize diagnostic analysis. |
    
    ---
    ### 2. Diagnosis Agent & Triage Consensus Rules
    The Multi-Agent Orchestrator applies sequential consensus criteria:
    
    ```
    IF Quality < Cutoff OR Multiple Defect Triggers THEN:
        DECISION = "RE_IMAGE"
        RATIONALE = "Poor capture quality prevents trustworthy clinical diagnosis."
        
    ELSE IF Predicted Pathology in {Melanoma, BCC, AKIEC} OR Combined Malignancy Prob >= Malignancy Threshold THEN:
        DECISION = "REFER"
        RATIONALE = "Suspected malignancy requires urgent biopsy/dermatology evaluation."
        
    ELSE IF Confidence < Auto-Triage Threshold OR Borderline Quality Caution THEN:
        DECISION = "REFER"
        RATIONALE = "Diagnostic uncertainty mandates specialist inspection."
        
    ELSE:
        DECISION = "AUTO_TRIAGE"
        RATIONALE = "High-confidence benign presentation validated on high-quality image."
    ```
    """)

# -------------------------------------------------------------
# PAGE 3: LEVEL 1 BENCHMARK DASHBOARD
# -------------------------------------------------------------
elif nav == "📊 Level 1 Benchmark Dashboard":
    st.markdown('<div class="main-title">Level 1 Benchmark Performance Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluated across all 1,942 held-out validation images from the official HAM10000 evaluation partition.</div>', unsafe_allow_html=True)
    
    if os.path.exists("outputs/validation_results.json"):
        with open("outputs/validation_results.json", "r") as f:
            summary = json.load(f)
            
        m1, m2, m3, m4 = st.columns(4)
        base = summary["baseline_classification_metrics"]
        m1.metric("Overall Accuracy", f"{base['overall_accuracy']*100:.2f}%")
        m2.metric("Balanced Accuracy", f"{base['balanced_accuracy']*100:.2f}%")
        m3.metric("Macro F1-Score", f"{base['macro_f1_score']:.4f}")
        m4.metric("Weighted F1-Score", f"{base['weighted_f1_score']:.4f}")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Multi-Agent Triage Decisions")
            triage_dist = summary["multi_agent_triage_decisions"]["distribution"]
            tdf = pd.DataFrame(list(triage_dist.items()), columns=["Triage Decision", "Count"])
            st.dataframe(tdf, use_container_width=True, hide_index=True)
            
        with c2:
            st.subheader("Image Quality Agent Outcomes")
            q_stats = summary["image_quality_agent_outcomes"]
            st.write(f"- **Mean Quality Score:** {q_stats['mean_overall_quality']} / 100")
            st.write(f"- **Mean Blur Score:** {q_stats['mean_blur_score']} / 100")
            st.write(f"- **Mean Glare Score:** {q_stats['mean_glare_score']} / 100")
            st.write(f"- **Mean Lighting Score:** {q_stats['mean_lighting_score']} / 100")
            st.write(f"- **Mean Centering Score:** {q_stats['mean_occlusion_score']} / 100")
            st.write("**Quality Breakdown:**", q_stats["quality_status_distribution"])
            
        st.markdown("---")
        st.subheader("Per-Class Classification Metrics (7 Lesion Classes)")
        cls_metrics = base["class_metrics"]
        rows = []
        for cname, stats in cls_metrics.items():
            rows.append({
                "Lesion Class": cname,
                "Precision": f"{stats['precision']*100:.2f}%",
                "Recall (Sensitivity)": f"{stats['recall']*100:.2f}%",
                "F1-Score": f"{stats['f1_score']:.4f}",
                "Validation Samples": stats["support"]
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        
        if os.path.exists("outputs/validation_detailed.csv"):
            st.markdown("---")
            st.subheader("Browse Validation Image Predictions (1,942 Samples)")
            det_df = pd.read_csv("outputs/validation_detailed.csv")
            filter_decision = st.multiselect("Filter by Triage Decision", options=det_df['final_decision'].unique(), default=det_df['final_decision'].unique())
            filtered = det_df[det_df['final_decision'].isin(filter_decision)]
            st.dataframe(filtered[['image_id', 'ground_truth_class', 'predicted_class', 'confidence', 'final_decision', 'overall_quality', 'explanation']], use_container_width=True)
    else:
        st.error("Validation summary not found. Run evaluate.py first.")

# -------------------------------------------------------------
# PAGE 4: API SUBMISSION
# -------------------------------------------------------------
elif nav == "🚀 API Submission":
    st.markdown('<div class="main-title">Level 1 API Submission Client</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Submit your benchmark summary payload before the summit deadline.</div>', unsafe_allow_html=True)
    
    if os.path.exists("outputs/validation_results.json"):
        with open("outputs/validation_results.json", "r") as f:
            summary_data = json.load(f)
            
        col_form, col_prev = st.columns([1, 1])
        with col_form:
            api_url = st.text_input("Summit API URL (from WhatsApp group)", placeholder="https://api.summit2026.example.com/submit/level1")
            team_name = st.text_input("Team Name", value="Team-DermaGuard")
            team_token = st.text_input("Team API Key / Token (if provided)", type="password")
            
            payload = {
                "team_name": team_name,
                "track": "Track 2 | Healthcare",
                "level": 1,
                "project": "Agentic DermaGuard",
                "timestamp": summary_data.get("evaluation_timestamp"),
                "total_validation_images": summary_data.get("total_eval_samples"),
                "baseline_metrics": {
                    "overall_accuracy": summary_data["baseline_classification_metrics"]["overall_accuracy"],
                    "balanced_accuracy": summary_data["baseline_classification_metrics"]["balanced_accuracy"],
                    "macro_f1": summary_data["baseline_classification_metrics"]["macro_f1_score"],
                    "weighted_f1": summary_data["baseline_classification_metrics"]["weighted_f1_score"]
                },
                "image_quality_summary": summary_data.get("image_quality_agent_outcomes"),
                "triage_summary": summary_data.get("multi_agent_triage_decisions"),
                "model_architecture": "EfficientNet-B0 + Laplacian/Entropy/HSV/Canny Multi-Agent"
            }
            
            sub_btn = st.button("🚀 Submit Level 1 Results Now", type="primary")
            if sub_btn:
                if not api_url:
                    st.error("Please enter the Submission API URL provided by the organizers.")
                else:
                    import urllib.request
                    import urllib.error
                    
                    payload_bytes = json.dumps(payload).encode('utf-8')
                    headers = {
                        "Content-Type": "application/json",
                        "User-Agent": "AgenticDermaGuard-SubmitClient/1.0"
                    }
                    if team_token:
                        headers["Authorization"] = f"Bearer {team_token}"
                        headers["X-API-KEY"] = team_token
                        
                    req = urllib.request.Request(api_url, data=payload_bytes, headers=headers, method="POST")
                    try:
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            code = resp.getcode()
                            body = resp.read().decode('utf-8')
                            st.success(f"Successfully submitted! Server returned HTTP {code}")
                            st.json(body)
                    except Exception as ex:
                        st.error(f"Submission failed: {str(ex)}")
                        
        with col_prev:
            st.subheader("Payload Preview")
            st.json(payload)
    else:
        st.error("Validation summary not found. Run evaluate.py first.")
