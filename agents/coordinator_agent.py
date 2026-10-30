from typing import Dict, Any

class CoordinatorAgent:
    """
    Agent 6: Coordinator Agent (Master Clinical Decision Orchestrator)
    Receives all agent outputs:
      A1: Image Quality Agent
      A2: Diagnosis Agent
      A3: Lesion Analysis Agent
      A4: Reasoning / Context Agent
      A5: Safety Agent
    Applies a documented, deterministic decision policy to produce:
      - final predicted label (official code: nv, mel, bcc, akiec, bkl, df, vasc)
      - final triage decision (AUTO_TRIAGE, REFER, or RE_IMAGE)
      - traceable rationale and traceable evidence chain
    """

    def __init__(self, auto_triage_confidence: float = 0.70):
        self.auto_triage_confidence = auto_triage_confidence
        self.benign_codes = {'nv', 'bkl', 'df', 'vasc'}
        self.malignant_codes = {'mel', 'bcc', 'akiec'}

    def coordinate(
        self,
        a1_quality: Dict[str, Any],
        a2_diagnosis: Dict[str, Any],
        a3_lesion: Dict[str, Any],
        a4_context: Dict[str, Any],
        a5_safety: Dict[str, Any],
        custom_auto_conf: float = None
    ) -> Dict[str, Any]:
        auto_conf_threshold = custom_auto_conf if custom_auto_conf is not None else self.auto_triage_confidence

        # 1. Extract inputs from all agents
        # A1
        q_status = a1_quality.get("status", "ACCEPTABLE")
        q_score = a1_quality.get("overall_quality", 80.0)
        a1_token = "RE_IMAGE" if q_status == "RE_IMAGE" else ("CAUTION" if q_status == "ACCEPTABLE_WITH_CAUTION" else "PASS")

        # A2
        pred_code = a2_diagnosis.get("predicted_code", "nv")
        pred_name = a2_diagnosis.get("predicted_class", "Melanocytic nevus")
        conf = a2_diagnosis.get("confidence", 0.70)
        a2_token = f"{pred_code} • {conf:.2f}"

        # A3 + A4
        lesion_score = a3_lesion.get("suspicion_score", 0.0)
        lesion_level = a3_lesion.get("suspicion_level", "BENIGN_CUES")
        context_risk = a4_context.get("risk_indicator", "LOW")
        
        if lesion_score >= 0.50 or context_risk == "HIGH":
            a3_a4_token = "SUSPICIOUS"
        elif lesion_score >= 0.35 or context_risk == "MODERATE":
            a3_a4_token = "INDETERMINATE"
        else:
            a3_a4_token = "BENIGN_CUES"

        # A5
        safety_flag = a5_safety.get("safety_flag", "PASS")
        safety_veto = a5_safety.get("veto_auto_triage", False)
        a5_token = "SAFETY_VETO" if safety_flag == "SAFETY_VETO" else ("SAFETY_FLAG" if safety_flag == "SAFETY_FLAG" else "PASS")

        # --- Documented Decision Policy Execution ---
        decision = "REFER"
        policy_rule = ""
        detailed_rationale = []

        # Policy Branch 1: Quality Failure -> RE_IMAGE
        if q_status == "RE_IMAGE":
            decision = "RE_IMAGE"
            policy_rule = "Rule 1 (Quality Gate): Capture quality inadequate for dependable clinical reading."
            triggers = a1_quality.get("triggered_criteria", ["Low composite quality"])
            detailed_rationale.append(f"Image quality failed acceptance ({', '.join(triggers)}). Re-imaging required.")
            detailed_rationale.append(f"Preliminary diagnostic inference is '{pred_code}' ({conf*100:.1f}%), recorded per competition guidelines.")

        # Policy Branch 2: Malignant Prediction -> Immediate REFER
        elif pred_code in self.malignant_codes:
            decision = "REFER"
            policy_rule = f"Rule 2 (Malignancy Referral): Detected high-risk lesion code '{pred_code}'."
            detailed_rationale.append(f"Diagnosis agent identified '{pred_name}' [{pred_code}] with {conf*100:.1f}% confidence.")
            detailed_rationale.append(f"Immediate specialist biopsy/dermatology evaluation indicated.")

        # Policy Branch 3: Safety Agent Veto -> Mandatory REFER
        elif safety_veto or safety_flag in {"SAFETY_VETO", "SAFETY_FLAG"}:
            decision = "REFER"
            trigger_reason = a5_safety.get("primary_trigger", "Clinical conflict detected")
            policy_rule = f"Rule 3 (Safety Sentinel Veto): Overrode automated triage due to: {trigger_reason}."
            detailed_rationale.append(f"Safety agent vetoed discharge: {a5_safety.get('justification', '')}")

        # Policy Branch 4: Low Diagnostic Confidence -> REFER
        elif conf < auto_conf_threshold:
            decision = "REFER"
            policy_rule = "Rule 4 (Diagnostic Uncertainty): Confidence below discharge threshold."
            detailed_rationale.append(f"Diagnostic confidence ({conf*100:.1f}%) is below auto-triage requirement ({auto_conf_threshold*100:.0f}%).")
            detailed_rationale.append("Human dermatologist review required to resolve diagnostic ambiguity.")

        # Policy Branch 5: Verified High-Confidence Benign Presentation -> AUTO_TRIAGE
        elif pred_code in self.benign_codes and conf >= auto_conf_threshold and not safety_veto:
            decision = "AUTO_TRIAGE"
            policy_rule = "Rule 5 (Verified Benign Autonomous Screening): High-confidence benign presentation with clear safety checks."
            detailed_rationale.append(f"Confident benign '{pred_name}' [{pred_code}] ({conf*100:.1f}% >= {auto_conf_threshold*100:.0f}%).")
            detailed_rationale.append(f"Supported by symmetric lesion features (suspicion: {lesion_score:.2f}) and validated image quality ({q_score:.1f}/100).")

        else:
            decision = "REFER"
            policy_rule = "Rule 6 (Conservative Fallback): Evidence insufficient for automated discharge."
            detailed_rationale.append("Conservative clinical triage applied due to incomplete consensus.")

        # Construct traceable chain matching brief example:
        # A1: PASS | A2: nv • 0.73 | A3 + A4: SUSPICIOUS | A5: SAFETY FLAG | A6: REFER
        traceable_chain = (
            f"A1: {a1_token} | "
            f"A2: {a2_token} | "
            f"A3+A4: {a3_a4_token} | "
            f"A5: {a5_token} | "
            f"A6: {decision}"
        )

        return {
            "agent_id": "A6",
            "agent_name": "Coordinator Agent",
            "prediction": pred_code,
            "prediction_name": pred_name,
            "confidence": conf,
            "decision": decision,
            "traceable_chain": traceable_chain,
            "policy_rule": policy_rule,
            "rationale": " ".join(detailed_rationale),
            "tokens": {
                "a1": a1_token,
                "a2": a2_token,
                "a3_a4": a3_a4_token,
                "a5": a5_token,
                "a6": decision
            }
        }

# Singleton instance
coordinator_agent = CoordinatorAgent()
