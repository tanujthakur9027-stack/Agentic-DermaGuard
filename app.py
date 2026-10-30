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

# Streamlit Page Config
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
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .decision-badge {
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .badge-refer {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 2px solid #F87171;
    }
    .badge-auto {
        background-color: #DCFCE7;
        color: #166534;
        border: 2px solid #4ADE80;
    }
    .badge-reimage {
        background-color: #FEF3C7;
        color: #92400E;
        border: 2px solid #FBBF24;
    }
    .agent-box {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 1rem;
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

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/medical-heart.png", width=64)
st.sidebar.title("Agentic DermaGuard")
st.sidebar.caption("Track 2 | Healthcare — Summit 2026")
st.sidebar.markdown("---")

nav = st.sidebar.radio(
    "Navigation",
    ["🔬 Single Image Screening", "📊 Level 1 Benchmark Dashboard", "🚀 API Submission Client"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Agent Roles:**")
st.sidebar.info(
    "**1. Image Quality Agent:** Gatekeeper checking blur, illumination, glare, and occlusion.\n\n"
    "**2. Diagnosis Agent:** 7-class EfficientNet classifier predicting malignancy probabilities and clinical confidence."
)

# -------------------------------------------------------------
# PAGE 1: SINGLE IMAGE SCREENING
# -------------------------------------------------------------
if nav == "🔬 Single Image Screening":
    st.markdown('<div class="main-title">Multi-Agent Skin Lesion Screening & Triage</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Two collaborative agents gatekeep image quality and classify dermoscopic pathologies to reach an explainable triage decision.</div>', unsafe_allow_html=True)
    
    col_input, col_view = st.columns([1, 1])
    
    with col_input:
        input_source = st.radio("Choose input method:", ["Preset Validation Samples", "Upload Image"])
        temp_img_path = None
        
        sample_dict = {
            "Sample 1: Melanocytic Nevus (Benign Mole)": "Health/eval/images/ISIC_0025661.jpg",
            "Sample 2: Melanoma (Malignant Lesion)": "Health/eval/images/ISIC_0024349.jpg",
            "Sample 3: Basal Cell Carcinoma (Malignant)": "Health/eval/images/ISIC_0024341.jpg",
            "Sample 4: Vascular Lesion (Benign Angioma)": "Health/eval/images/ISIC_0024340.jpg",
            "Sample 5: Borderline / Occluded Presentation": "Health/eval/images/ISIC_0024354.jpg"
        }
        
        if input_source == "Preset Validation Samples":
            selected_sample = st.selectbox("Select sample image:", list(sample_dict.keys()))
            temp_img_path = sample_dict[selected_sample]
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
            st.image(pil_image, caption="Dermoscopic Input", use_container_width=True)
            run_btn = st.button("🚀 Run Multi-Agent Triage Pipeline", type="primary", use_container_width=True)
        else:
            st.warning("Please upload or select an image to inspect.")
            run_btn = False

    if pil_image is not None and (run_btn or input_source == "Preset Validation Samples"):
        st.markdown("---")
        with st.spinner("Executing Image Quality Agent & Diagnosis Agent..."):
            res = orchestrator.process_image(temp_img_path)
            
        final_decision = res['final_decision']
        
        # Display Decision Banner
        if final_decision == "REFER":
            st.markdown(f'<div class="decision-badge badge-refer">🚨 FINAL DECISION: {final_decision} (Specialist Referral Required)</div>', unsafe_allow_html=True)
        elif final_decision == "AUTO_TRIAGE":
            st.markdown(f'<div class="decision-badge badge-auto">✅ FINAL DECISION: {final_decision} (Routine Follow-up / Discharge)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="decision-badge badge-reimage">⚠️ FINAL DECISION: {final_decision} (Re-imaging Required)</div>', unsafe_allow_html=True)
            
        st.markdown(f"**Lead Agent:** `{res['primary_agent']}`")
        st.markdown(f"**Synthesized Rationale:** {res['explanation']}")
        
        # Agents Breakdown Columns
        st.markdown("### Agent Collaboration Breakdown")
        col_ag1, col_ag2 = st.columns(2)
        
        # Agent 1
        with col_ag1:
            st.markdown('<div class="agent-box">', unsafe_allow_html=True)
            st.subheader("1️⃣ Image Quality Agent")
            q = res['quality_summary']
            
            st.metric("Overall Quality Score", f"{q['overall_quality']:.1f} / 100", q['status'])
            
            st.write("**Quality Dimensions:**")
            st.progress(float(q['blur_score']) / 100.0, text=f"Sharpness / Blur: {q['blur_score']:.1f}/100")
            st.progress(float(q['lighting_score']) / 100.0, text=f"Illumination / Entropy: {q['lighting_score']:.1f}/100")
            st.progress(float(q['glare_score']) / 100.0, text=f"Glare / Exposure: {q['glare_score']:.1f}/100")
            st.progress(float(q['occlusion_score']) / 100.0, text=f"Centering / Edge Focus: {q['occlusion_score']:.1f}/100")
            
            st.write("**Quality Findings & Reasons:**")
            for r in q['reasons']:
                st.markdown(f"- {r}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Agent 2
        with col_ag2:
            st.markdown('<div class="agent-box">', unsafe_allow_html=True)
            st.subheader("2️⃣ Diagnosis Agent")
            
            if res['diagnosis_summary'] is not None:
                d = res['diagnosis_summary']
                is_high_risk = d['predicted_class'] in orchestrator.high_risk_classes
                risk_tag = "🔴 Malignant / High Risk" if is_high_risk else "🟢 Benign / Low Risk"
                
                st.metric("Predicted Lesion", d['predicted_class'], f"{d['confidence']*100:.1f}% confidence")
                st.markdown(f"**Clinical Category:** {risk_tag}")
                
                # Bar chart for probabilities
                probs_df = pd.DataFrame(list(d['all_probabilities'].items()), columns=["Lesion Class", "Probability"]).sort_values(by="Probability", ascending=True)
                st.bar_chart(probs_df.set_index("Lesion Class"))
                
                st.write("**Top 3 Differential Candidates:**")
                for cls_name, prob in d['top_3']:
                    st.write(f"- **{cls_name}**: {prob*100:.2f}%")
            else:
                st.warning("Diagnosis bypassed because Image Quality Agent marked image as RE_IMAGE.")
            st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 2: LEVEL 1 BENCHMARK DASHBOARD
# -------------------------------------------------------------
elif nav == "📊 Level 1 Benchmark Dashboard":
    st.markdown('<div class="main-title">Level 1 Benchmark Performance Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluation across all 1,942 held-out validation images from the official HAM10000 evaluation partition.</div>', unsafe_allow_html=True)
    
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
# PAGE 3: API SUBMISSION CLIENT
# -------------------------------------------------------------
elif nav == "🚀 API Submission Client":
    st.markdown('<div class="main-title">Level 1 API Submission Client</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Submit your benchmark summary payload before the 8:00 PM summit deadline.</div>', unsafe_allow_html=True)
    
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
