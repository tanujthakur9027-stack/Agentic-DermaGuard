import os
from PIL import Image
from typing import Dict, Any, Optional
from agents.image_quality_agent import quality_agent, QualityMetrics
from agents.diagnosis_agent import diagnosis_agent, diagnosis_transforms

class AgenticDermaGuard:
    """
    Multi-Agent Skin Lesion Screening & Triage Orchestrator.
    Coordinates:
      1. Image Quality Agent: Gates low-quality images (blur, illumination, glare, occlusion).
      2. Diagnosis Agent: Evaluates lesion class probabilities (7 HAM10000 classes).
    Synthesizes an explainable final decision: AUTO_TRIAGE, REFER, or RE_IMAGE.
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
        
        # Criteria Thresholds
        self.quality_cutoff = quality_cutoff
        self.blur_threshold = blur_threshold
        self.lighting_threshold = lighting_threshold
        self.glare_threshold = glare_threshold
        self.occlusion_threshold = occlusion_threshold
        self.malignancy_threshold = malignancy_threshold
        self.auto_triage_confidence = auto_triage_confidence
        
        self.high_risk_classes = {'Melanoma', 'Basal cell carcinoma', 'Actinic keratoses'}
        self.benign_classes = {'Melanocytic nevus', 'Benign keratosis', 'Dermatofibroma', 'Vascular lesion'}

    def process_image(self, image_path: str, custom_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Executes multi-agent workflow on a single dermoscopic image with configurable threshold criteria.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        th = custom_thresholds or {}
        q_cutoff = th.get("quality_cutoff", self.quality_cutoff)
        b_th = th.get("blur_threshold", self.blur_threshold)
        l_th = th.get("lighting_threshold", self.lighting_threshold)
        g_th = th.get("glare_threshold", self.glare_threshold)
        o_th = th.get("occlusion_threshold", self.occlusion_threshold)
        m_th = th.get("malignancy_threshold", self.malignancy_threshold)
        a_th = th.get("auto_triage_confidence", self.auto_triage_confidence)

        # Update quality agent thresholds dynamically
        self.quality_agent.blur_threshold = 100.0
        
        # --- Step 1: Agent 1 (Image Quality Assessment) ---
        quality_metrics = self.quality_agent.evaluate(image_path)
        
        # Check specific threshold criteria
        triggers = []
        if quality_metrics.blur_score < b_th:
            triggers.append(f"Blur ({quality_metrics.blur_score:.1f} < {b_th:.1f})")
        if quality_metrics.lighting_score < l_th:
            triggers.append(f"Lighting ({quality_metrics.lighting_score:.1f} < {l_th:.1f})")
        if quality_metrics.glare_score < g_th:
            triggers.append(f"Glare ({quality_metrics.glare_score:.1f} < {g_th:.1f})")
        if quality_metrics.occlusion_score < o_th:
            triggers.append(f"Occlusion ({quality_metrics.occlusion_score:.1f} < {o_th:.1f})")
            
        is_reimage = (quality_metrics.overall_quality < q_cutoff) or (len(triggers) >= 2)
        has_quality_caution = (quality_metrics.status == 'ACCEPTABLE_WITH_CAUTION') or (len(triggers) == 1)

        # Gatekeeper Rule: If quality fails, short-circuit to RE_IMAGE
        if is_reimage:
            reasons_str = "; ".join(triggers if triggers else quality_metrics.reasoning)
            return {
                'image_path': image_path,
                'final_decision': 'RE_IMAGE',
                'primary_agent': 'Image Quality Agent',
                'confidence': round(1.0 - (quality_metrics.overall_quality / 100.0), 3),
                'explanation': f"Quality check rejected [Trigger: {', '.join(triggers) if triggers else 'Overall Score'}]: {reasons_str}. Re-imaging is required before safe clinical analysis.",
                'quality_summary': {
                    'blur_score': quality_metrics.blur_score,
                    'lighting_score': quality_metrics.lighting_score,
                    'glare_score': quality_metrics.glare_score,
                    'occlusion_score': quality_metrics.occlusion_score,
                    'overall_quality': quality_metrics.overall_quality,
                    'status': 'RE_IMAGE',
                    'reasons': quality_metrics.reasoning,
                    'triggered_criteria': triggers
                },
                'diagnosis_summary': None
            }

        # --- Step 2: Agent 2 (Diagnosis & Probability Prediction) ---
        img = Image.open(image_path).convert('RGB')
        tensor = diagnosis_transforms(img)
        diag_output = self.diagnosis_agent.predict(tensor)
        
        pred_class = diag_output['predicted_class']
        diag_conf = diag_output['confidence']
        
        mel_prob = diag_output['all_probabilities'].get('Melanoma', 0.0)
        bcc_prob = diag_output['all_probabilities'].get('Basal cell carcinoma', 0.0)
        combined_malig = mel_prob + bcc_prob

        # --- Step 3: Multi-Agent Clinical Triage Synthesis ---
        if pred_class in self.high_risk_classes or combined_malig >= m_th:
            final_decision = 'REFER'
            lead_agent = 'Diagnosis Agent'
            culprit = pred_class if pred_class in self.high_risk_classes else f"Combined malignancy suspicion ({combined_malig*100:.1f}% >= {m_th*100:.0f}%)"
            explanation = (
                f"Urgent Specialist Referral (REFER) [Trigger: {culprit}]: High risk lesion presentation "
                f"with {diag_conf * 100:.1f}% primary confidence. Quality: {quality_metrics.overall_quality:.1f}/100."
            )
        elif diag_conf < a_th or has_quality_caution:
            final_decision = 'REFER'
            lead_agent = 'Consensus (Quality + Diagnosis)'
            caution_text = f" (along with quality criteria: {', '.join(triggers)})" if triggers else ""
            explanation = (
                f"Clinical Referral (REFER) [Trigger: Diagnostic Uncertainty]: "
                f"Top prediction '{pred_class}' confidence ({diag_conf * 100:.1f}%) is below auto-triage threshold ({a_th * 100:.0f}%){caution_text}."
            )
        else:
            final_decision = 'AUTO_TRIAGE'
            lead_agent = 'Consensus (Quality + Diagnosis)'
            explanation = (
                f"Automated Triage (AUTO_TRIAGE) [Trigger: Validated Benign]: High confidence benign '{pred_class}' "
                f"({diag_conf * 100:.1f}% >= {a_th * 100:.0f}%) validated on high-quality dermoscopy ({quality_metrics.overall_quality:.1f}/100)."
            )

        return {
            'image_path': image_path,
            'final_decision': final_decision,
            'primary_agent': lead_agent,
            'predicted_class': pred_class,
            'confidence': diag_conf,
            'explanation': explanation,
            'quality_summary': {
                'blur_score': quality_metrics.blur_score,
                'lighting_score': quality_metrics.lighting_score,
                'glare_score': quality_metrics.glare_score,
                'occlusion_score': quality_metrics.occlusion_score,
                'overall_quality': quality_metrics.overall_quality,
                'status': 'ACCEPTABLE_WITH_CAUTION' if has_quality_caution else 'ACCEPTABLE',
                'reasons': quality_metrics.reasoning,
                'triggered_criteria': triggers
            },
            'diagnosis_summary': diag_output
        }

# Global singleton
agentic_dermaguard = AgenticDermaGuard()
