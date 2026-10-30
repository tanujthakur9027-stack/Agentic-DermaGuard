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
    def __init__(self, quality_threshold: float = 50.0):
        self.quality_agent = quality_agent
        self.diagnosis_agent = diagnosis_agent
        self.quality_threshold = quality_threshold
        
        self.high_risk_classes = {'Melanoma', 'Basal cell carcinoma', 'Actinic keratoses'}
        self.benign_classes = {'Melanocytic nevus', 'Benign keratosis', 'Dermatofibroma', 'Vascular lesion'}

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Executes multi-agent workflow on a single dermoscopic image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # --- Step 1: Agent 1 (Image Quality Assessment) ---
        quality_metrics = self.quality_agent.evaluate(image_path)
        
        # Gatekeeper Rule: If quality fails, short-circuit to RE_IMAGE
        if quality_metrics.status == 'RE_IMAGE' or quality_metrics.overall_quality < self.quality_threshold:
            reasons_str = "; ".join(quality_metrics.reasoning)
            return {
                'image_path': image_path,
                'final_decision': 'RE_IMAGE',
                'primary_agent': 'Image Quality Agent',
                'confidence': 1.0 - (quality_metrics.overall_quality / 100.0),
                'explanation': f"Quality check failed ({quality_metrics.status}, score {quality_metrics.overall_quality:.1f}/100): {reasons_str}. Re-imaging is required before safe clinical analysis.",
                'quality_summary': {
                    'blur_score': quality_metrics.blur_score,
                    'lighting_score': quality_metrics.lighting_score,
                    'glare_score': quality_metrics.glare_score,
                    'occlusion_score': quality_metrics.occlusion_score,
                    'overall_quality': quality_metrics.overall_quality,
                    'status': quality_metrics.status,
                    'reasons': quality_metrics.reasoning
                },
                'diagnosis_summary': None
            }

        # --- Step 2: Agent 2 (Diagnosis & Probability Prediction) ---
        img = Image.open(image_path).convert('RGB')
        tensor = diagnosis_transforms(img)
        diag_output = self.diagnosis_agent.predict(tensor)
        
        pred_class = diag_output['predicted_class']
        diag_conf = diag_output['confidence']
        
        # --- Step 3: Multi-Agent Clinical Triage Synthesis ---
        has_quality_caution = (quality_metrics.status == 'ACCEPTABLE_WITH_CAUTION')
        
        if pred_class in self.high_risk_classes:
            final_decision = 'REFER'
            lead_agent = 'Diagnosis Agent'
            explanation = (
                f"Urgent Specialist Referral (REFER): High risk lesion identified as '{pred_class}' "
                f"with {diag_conf * 100:.1f}% confidence. Quality: {quality_metrics.status} ({quality_metrics.overall_quality:.1f}/100)."
            )
        elif diag_conf < 0.65 or has_quality_caution:
            final_decision = 'REFER'
            lead_agent = 'Consensus (Quality + Diagnosis)'
            caution_text = " with borderline image quality caution" if has_quality_caution else ""
            explanation = (
                f"Clinical Referral (REFER): Diagnostic uncertainty{caution_text}. "
                f"Top prediction '{pred_class}' has moderate confidence ({diag_conf * 100:.1f}%). Dermatologist inspection recommended."
            )
        else:
            final_decision = 'AUTO_TRIAGE'
            lead_agent = 'Consensus (Quality + Diagnosis)'
            explanation = (
                f"Automated Triage (AUTO_TRIAGE): High confidence benign '{pred_class}' "
                f"({diag_conf * 100:.1f}% confidence) validated on high-quality dermoscopy ({quality_metrics.overall_quality:.1f}/100)."
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
                'status': quality_metrics.status,
                'reasons': quality_metrics.reasoning
            },
            'diagnosis_summary': diag_output
        }

# Global singleton
agentic_dermaguard = AgenticDermaGuard()
