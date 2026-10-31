import os
import time
from PIL import Image
from typing import Dict, Any, Optional

from thresholds import load_thresholds, get_flat_thresholds
from agents.image_quality_agent import quality_agent, QualityMetrics
from agents.diagnosis_agent import diagnosis_agent, DiagnosisAgent
from agents.lesion_analysis_agent import lesion_analysis_agent, LesionAnalysisAgent
from agents.context_agent import context_agent, ContextAgent
from agents.safety_agent import safety_agent, SafetyAgent
from agents.coordinator_agent import coordinator_agent, CoordinatorAgent

class AgenticDermaGuard:
    """
    Agentic DermaGuard: Six-Agent Skin Lesion Screening & Triage Orchestrator.
    Jaypee Agentic AI International Summit 2026 — Track 2 | Healthcare

    Coordinates all SIX specialized roles:
      1. Image Quality Agent (A1): Blur, lighting, glare, obstruction, edge mapping.
      2. Diagnosis Agent (A2): 7 HAM10000 classes (nv, mel, bcc, akiec, bkl, df, vasc).
      3. Lesion Analysis Agent (A3): Visual cues (ABCD: Asymmetry, Border Irregularity, Color Variation).
      4. Reasoning / Context Agent (A4): Demographic priors (Age, Sex, Site) & missing field handling.
      5. Safety Agent (A5): Sentinel & veto (Cross-checks for discordance & unsafe automation).
      6. Coordinator Agent (A6): Master decision policy synthesizing all signals.
    """

    def __init__(
        self,
        quality_cutoff: Optional[float] = None,
        blur_threshold: Optional[float] = None,
        lighting_threshold: Optional[float] = None,
        glare_threshold: Optional[float] = None,
        occlusion_threshold: Optional[float] = None,
        malignancy_threshold: Optional[float] = None,
        auto_triage_confidence: Optional[float] = None
    ):
        self.quality_agent = quality_agent
        self.diagnosis_agent = diagnosis_agent
        self.lesion_agent = lesion_analysis_agent
        self.context_agent = context_agent
        self.safety_agent = safety_agent
        self.coordinator_agent = coordinator_agent

        defaults = get_flat_thresholds()
        self.quality_cutoff = quality_cutoff if quality_cutoff is not None else defaults["quality_score_reimage_below"]
        self.blur_threshold = blur_threshold if blur_threshold is not None else defaults["blur_min"]
        self.lighting_threshold = lighting_threshold if lighting_threshold is not None else defaults["entropy_min"]
        self.glare_threshold = glare_threshold if glare_threshold is not None else defaults["glare_max_pct"]
        self.occlusion_threshold = occlusion_threshold if occlusion_threshold is not None else defaults["obstruction_max_pct"]
        self.malignancy_threshold = malignancy_threshold if malignancy_threshold is not None else defaults["malignant_prob_threshold"]
        self.auto_triage_confidence = auto_triage_confidence if auto_triage_confidence is not None else defaults["auto_triage_confidence"]

    def process_image(
        self,
        image_input: Any,
        patient_context: Optional[Dict[str, Any]] = None,
        custom_thresholds: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes full six-agent workflow on a single dermoscopic capture.
        Always returns both diagnosis prediction AND triage decision.
        Deterministic error handling: On any failure, returns a valid fallback row.
        """
        t_start = time.perf_counter()
        img_path = "memory://in-memory-image.jpg"
        fallback_pred = "nv"
        
        try:
            if isinstance(image_input, str):
                img_path = image_input
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image not found: {image_input}")
                pil_img = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")

            active_th = get_flat_thresholds(custom_thresholds)

            # --- AGENT 1: Image Quality Agent ---
            q_metrics = self.quality_agent.evaluate(pil_img, custom_thresholds=active_th)
            a1_data = q_metrics.to_dict()

            # --- AGENT 2: Diagnosis Agent ---
            diag_output = self.diagnosis_agent.predict(pil_img, custom_thresholds=active_th)
            fallback_pred = diag_output.get("status", diag_output.get("predicted_code", "nv"))

            # --- AGENT 3: Lesion Analysis Agent ---
            a3_data = self.lesion_agent.analyze(pil_img, custom_thresholds=active_th)

            # --- AGENT 4: Reasoning / Context Agent ---
            ctx = patient_context or {}
            a4_data = self.context_agent.evaluate(
                age=ctx.get("age"),
                sex=ctx.get("sex"),
                localization=ctx.get("localization"),
                custom_thresholds=active_th
            )

            # --- AGENT 5: Safety Agent ---
            a5_data = self.safety_agent.review(
                quality_data=a1_data,
                diagnosis_data=diag_output,
                lesion_data=a3_data,
                context_data=a4_data,
                custom_thresholds=active_th
            )

            # --- AGENT 6: Coordinator Agent ---
            a6_data = self.coordinator_agent.coordinate(
                a1_quality=a1_data,
                a2_diagnosis=diag_output,
                a3_lesion=a3_data,
                a4_context=a4_data,
                a5_safety=a5_data,
                custom_auto_conf=active_th.get("auto_triage_confidence"),
                custom_thresholds=active_th
            )

            total_runtime_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

            return {
                "image_path": img_path,
                # Master Level 2 outputs
                "final_decision": a6_data["decision"],
                "decision": a6_data["decision"],
                "prediction": a6_data["prediction"],
                "prediction_name": a6_data["prediction_name"],
                "confidence": a6_data["confidence"],
                "traceable_chain": a6_data["traceable_chain"],
                "example_strip": a6_data["example_strip"],
                "traceable_rationale": a6_data["rationale"],
                "policy_rule": a6_data["policy_rule"],
                "runtime_ms": total_runtime_ms,
                # Detailed individual agent contracts
                "agents": {
                    "A1_quality": a1_data,
                    "A2_diagnosis": diag_output,
                    "A3_lesion_analysis": a3_data,
                    "A4_context_reasoning": a4_data,
                    "A5_safety": a5_data,
                    "A6_coordinator": a6_data
                },
                "explaining_agents": [
                    a1_data,
                    diag_output,
                    a3_data,
                    a4_data,
                    a5_data,
                    a6_data
                ],
                # Backward-compatible mappings
                "primary_agent": a6_data["agent_name"],
                "predicted_class": a6_data["prediction_name"],
                "explanation": a6_data["rationale"],
                "quality_summary": a1_data,
                "diagnosis_summary": diag_output,
                "error": None
            }

        except Exception as e:
            # Deterministic fallback row per hackathon rules:
            # "Never silently skip an image: on any error return a fallback row (prediction = best available else "nv", decision = REFER, error noted)."
            total_runtime_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
            err_msg = str(e)
            print(f"[Orchestrator Fallback] Error processing image {img_path}: {err_msg}")

            fallback_chain = f"A1: ERROR | A2: {fallback_pred} • 0.50 | A3+A4: UNKNOWN | A5: SAFETY VETO | A6: REFER"
            fallback_strip = f'A1 ERROR "Processing failed" | A2 {fallback_pred} • 0.50 "Fallback prediction" | A3+A4 UNKNOWN "Analysis error" | A5 SAFETY VETO "Exception catch" | A6 REFER "Safety referral"'

            return {
                "image_path": img_path,
                "final_decision": "REFER",
                "decision": "REFER",
                "prediction": fallback_pred,
                "prediction_name": "Melanocytic nevus" if fallback_pred == "nv" else fallback_pred,
                "confidence": 0.50,
                "traceable_chain": fallback_chain,
                "example_strip": fallback_strip,
                "traceable_rationale": f"System encountered processing error ({err_msg}). Conservative referral mandated.",
                "policy_rule": "Fallback Rule: System error triggers mandatory specialist referral.",
                "runtime_ms": total_runtime_ms,
                "agents": {
                    "A1_quality": {"agent": "A1", "name": "Image Quality", "status": "ERROR", "score": 0.0, "summary": "Error in image quality assessment", "evidence": [], "flags": ["SYSTEM_ERROR"], "explanation": err_msg, "runtime_ms": 0.0},
                    "A2_diagnosis": {"agent": "A2", "name": "Diagnosis", "status": fallback_pred, "score": 0.50, "summary": f"Fallback to {fallback_pred}", "evidence": [], "flags": ["FALLBACK"], "explanation": err_msg, "runtime_ms": 0.0},
                    "A3_lesion_analysis": {"agent": "A3", "name": "Lesion Analysis", "status": "INDETERMINATE", "score": 0.50, "summary": "Analysis skipped due to error", "evidence": [], "flags": ["SYSTEM_ERROR"], "explanation": err_msg, "runtime_ms": 0.0},
                    "A4_context_reasoning": {"agent": "A4", "name": "Context & Demographics", "status": "MODERATE", "score": 0.50, "summary": "Demographic baseline", "evidence": [], "flags": ["SYSTEM_ERROR"], "explanation": err_msg, "runtime_ms": 0.0},
                    "A5_safety": {"agent": "A5", "name": "Safety Sentinel", "status": "SAFETY_VETO", "score": 1.0, "summary": "Safety veto triggered by error", "evidence": [], "flags": ["SAFETY_VETO", "SYSTEM_ERROR"], "explanation": err_msg, "runtime_ms": 0.0},
                    "A6_coordinator": {"agent": "A6", "name": "Coordinator", "status": "REFER", "score": 0.50, "summary": "Conservative fallback referral", "evidence": [], "flags": ["FALLBACK_REFERRAL"], "explanation": err_msg, "runtime_ms": 0.0}
                },
                "explaining_agents": [
                    {"agent": "A1", "name": "Image Quality", "status": "ERROR", "score": 0.0, "summary": "Error in image quality assessment", "evidence": [], "flags": ["SYSTEM_ERROR"], "explanation": err_msg, "runtime_ms": 0.0},
                    {"agent": "A2", "name": "Diagnosis", "status": fallback_pred, "score": 0.50, "summary": f"Fallback to {fallback_pred}", "evidence": [], "flags": ["FALLBACK"], "explanation": err_msg, "runtime_ms": 0.0},
                    {"agent": "A3", "name": "Lesion Analysis", "status": "INDETERMINATE", "score": 0.50, "summary": "Analysis skipped due to error", "evidence": [], "flags": ["SYSTEM_ERROR"], "explanation": err_msg, "runtime_ms": 0.0},
                    {"agent": "A4", "name": "Context & Demographics", "status": "MODERATE", "score": 0.50, "summary": "Demographic baseline", "evidence": [], "flags": ["SYSTEM_ERROR"], "explanation": err_msg, "runtime_ms": 0.0},
                    {"agent": "A5", "name": "Safety Sentinel", "status": "SAFETY_VETO", "score": 1.0, "summary": "Safety veto triggered by error", "evidence": [], "flags": ["SAFETY_VETO", "SYSTEM_ERROR"], "explanation": err_msg, "runtime_ms": 0.0},
                    {"agent": "A6", "name": "Coordinator", "status": "REFER", "score": 0.50, "summary": "Conservative fallback referral", "evidence": [], "flags": ["FALLBACK_REFERRAL"], "explanation": err_msg, "runtime_ms": 0.0}
                ],
                "primary_agent": "Coordinator Agent",
                "predicted_class": "Melanocytic nevus" if fallback_pred == "nv" else fallback_pred,
                "explanation": f"Conservative fallback referral applied due to processing error: {err_msg}",
                "quality_summary": {},
                "diagnosis_summary": {},
                "error": err_msg
            }

# Global singleton orchestrator
agentic_dermaguard = AgenticDermaGuard()
