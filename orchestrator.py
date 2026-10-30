import os
from PIL import Image
from typing import Dict, Any, Optional

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
        quality_cutoff: float = 50.0,
        blur_threshold: float = 40.0,
        lighting_threshold: float = 40.0,
        glare_threshold: float = 70.0,
        occlusion_threshold: float = 50.0,
        malignancy_threshold: float = 0.35,
        auto_triage_confidence: float = 0.70
    ):
        self.quality_agent = quality_agent
        self.diagnosis_agent = diagnosis_agent
        self.lesion_agent = lesion_analysis_agent
        self.context_agent = context_agent
        self.safety_agent = safety_agent
        self.coordinator_agent = coordinator_agent

        self.quality_cutoff = quality_cutoff
        self.blur_threshold = blur_threshold
        self.lighting_threshold = lighting_threshold
        self.glare_threshold = glare_threshold
        self.occlusion_threshold = occlusion_threshold
        self.malignancy_threshold = malignancy_threshold
        self.auto_triage_confidence = auto_triage_confidence

    def process_image(
        self,
        image_input: Any,
        patient_context: Optional[Dict[str, Any]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Executes full six-agent workflow on a single dermoscopic capture.
        Always returns both diagnosis prediction AND triage decision.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            pil_img = Image.open(image_input).convert('RGB')
            img_path = image_input
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
            img_path = "memory://in-memory-image.jpg"
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        th = custom_thresholds or {}
        active_th = {
            "blur_threshold": th.get("blur_threshold", self.blur_threshold),
            "lighting_threshold": th.get("lighting_threshold", self.lighting_threshold),
            "glare_threshold": th.get("glare_threshold", self.glare_threshold),
            "occlusion_threshold": th.get("occlusion_threshold", self.occlusion_threshold),
            "quality_cutoff": th.get("quality_cutoff", self.quality_cutoff),
            "malignancy_threshold": th.get("malignancy_threshold", self.malignancy_threshold),
            "auto_triage_confidence": th.get("auto_triage_confidence", self.auto_triage_confidence)
        }

        # --- AGENT 1: Image Quality Agent ---
        q_metrics = self.quality_agent.evaluate(pil_img, custom_thresholds=active_th)
        a1_data = {
            "blur_score": q_metrics.blur_score,
            "lighting_score": q_metrics.lighting_score,
            "glare_score": q_metrics.glare_score,
            "occlusion_score": q_metrics.occlusion_score,
            "overall_quality": q_metrics.overall_quality,
            "status": q_metrics.status,
            "reasoning": q_metrics.reasoning,
            "triggered_criteria": q_metrics.triggered_criteria,
            "edge_map_b64": q_metrics.edge_map_b64
        }

        # --- AGENT 2: Diagnosis Agent ---
        # NOTE: Track golden rule: Every image still requires a diagnosis prediction!
        diag_output = self.diagnosis_agent.predict(pil_img)

        # --- AGENT 3: Lesion Analysis Agent ---
        a3_data = self.lesion_agent.analyze(pil_img)

        # --- AGENT 4: Reasoning / Context Agent ---
        ctx = patient_context or {}
        a4_data = self.context_agent.evaluate(
            age=ctx.get("age"),
            sex=ctx.get("sex"),
            localization=ctx.get("localization")
        )

        # --- AGENT 5: Safety Agent ---
        a5_data = self.safety_agent.review(
            quality_data=a1_data,
            diagnosis_data=diag_output,
            lesion_data=a3_data,
            context_data=a4_data
        )

        # --- AGENT 6: Coordinator Agent ---
        a6_data = self.coordinator_agent.coordinate(
            a1_quality=a1_data,
            a2_diagnosis=diag_output,
            a3_lesion=a3_data,
            a4_context=a4_data,
            a5_safety=a5_data,
            custom_auto_conf=active_th["auto_triage_confidence"]
        )

        return {
            "image_path": img_path,
            # Master Level 2 outputs
            "final_decision": a6_data["decision"],
            "prediction": a6_data["prediction"],
            "prediction_name": a6_data["prediction_name"],
            "confidence": a6_data["confidence"],
            "traceable_chain": a6_data["traceable_chain"],
            "traceable_rationale": a6_data["rationale"],
            "policy_rule": a6_data["policy_rule"],
            # Detailed individual agent cards
            "agents": {
                "A1_quality": a1_data,
                "A2_diagnosis": diag_output,
                "A3_lesion_analysis": a3_data,
                "A4_context_reasoning": a4_data,
                "A5_safety": a5_data,
                "A6_coordinator": a6_data
            },
            # Backward-compatible mappings
            "primary_agent": a6_data["agent_name"],
            "predicted_class": a6_data["prediction_name"],
            "explanation": a6_data["rationale"],
            "quality_summary": a1_data,
            "diagnosis_summary": diag_output
        }

# Global singleton orchestrator
agentic_dermaguard = AgenticDermaGuard()
