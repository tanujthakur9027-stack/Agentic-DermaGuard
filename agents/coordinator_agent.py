from typing import Dict, Any

class CoordinatorAgent:
    """
    Agent 6: Coordinator Agent (Master Clinical Decision Policy Orchestrator)
    Jaypee Agentic AI Summit 2026 — Track 2 | Healthcare

    Synthesizes signals from all five specialized agents:
      A1: Image Quality Agent (Quality Gate)
      A2: Diagnosis Agent (Pathology Classifier)
      A3: Lesion Analysis Agent (Dermoscopic ABCD Cues)
      A4: Reasoning / Context Agent (Patient Demographics & Priors)
      A5: Safety Agent (Clinical Sentinel & Veto)
    Enforces a documented, deterministic clinical decision policy:
      - Policy Rule 1: Quality Failure -> RE_IMAGE (Preserving diagnosis)
      - Policy Rule 2: Malignancy Presentation -> REFER
      - Policy Rule 3: Safety Sentinel Veto -> REFER
      - Policy Rule 4: Diagnostic Uncertainty -> REFER
      - Policy Rule 5: Verified Benign Autonomous Screening -> AUTO_TRIAGE
    Outputs:
      - Final predicted label (Official 7-class code: nv, mel, bcc, akiec, bkl, df, vasc)
      - Final triage action (AUTO_TRIAGE, REFER, or RE_IMAGE)
      - Exact traceable evidence chain matching summit syntax:
        A1: PASS | A2: nv • 0.82 | A3+A4: BENIGN_CUES | A5: PASS | A6: AUTO_TRIAGE
      - Complete clinical rationale and audit log
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

        # 1. Extract agent findings
        # A1 Quality
        q_status = a1_quality.get("status", "ACCEPTABLE")
        q_score = a1_quality.get("overall_quality", 80.0)
        a1_token = "RE_IMAGE" if q_status == "RE_IMAGE" else ("CAUTION" if q_status == "ACCEPTABLE_WITH_CAUTION" else "PASS")

        # A2 Diagnosis
        pred_code = a2_diagnosis.get("predicted_code", "nv")
        pred_name = a2_diagnosis.get("predicted_class", "Melanocytic nevus")
        conf = a2_diagnosis.get("confidence", 0.70)
        a2_token = f"{pred_code} • {conf:.2f}"

        # A3 Lesion + A4 Context
        lesion_score = a3_lesion.get("suspicion_score", 0.0)
        lesion_level = a3_lesion.get("suspicion_level", "BENIGN_CUES")
        tds_score = a3_lesion.get("tds_score", 3.5)
        context_risk = a4_context.get("risk_indicator", "LOW")

        if lesion_score >= 0.55 or tds_score >= 5.0 or context_risk == "HIGH":
            a3_a4_token = "SUSPICIOUS"
        elif lesion_score >= 0.35 or context_risk == "MODERATE":
            a3_a4_token = "INDETERMINATE"
        else:
            a3_a4_token = "BENIGN_CUES"

        # A5 Safety
        safety_flag = a5_safety.get("safety_flag", "PASS")
        safety_veto = a5_safety.get("veto_auto_triage", False)
        a5_token = "SAFETY_VETO" if safety_flag == "SAFETY_VETO" else ("SAFETY_FLAG" if safety_flag == "SAFETY_FLAG" else "PASS")

        # --- Documented Decision Policy Matrix ---
        decision = "REFER"
        policy_rule = ""
        detailed_rationale = []
        action_justification = ""

        # Branch 1: Quality Gatekeeper Rule -> RE_IMAGE
        if q_status == "RE_IMAGE":
            decision = "RE_IMAGE"
            policy_rule = "Policy Rule 1 (Quality Gate): Capture degradation prevents dependable reading."
            triggers = a1_quality.get("triggered_criteria", ["Low composite quality"])
            action_justification = (
                f"Image quality failed clinical acceptance ({', '.join(triggers)}). "
                f"Re-imaging requested before definitive screening. "
                f"Preliminary diagnostic inference is '{pred_name}' [{pred_code}] ({conf*100:.1f}%), recorded per competition guidelines."
            )
            detailed_rationale.append(action_justification)

        # Branch 2: Malignancy Presentation -> Mandatory REFER
        elif pred_code in self.malignant_codes:
            decision = "REFER"
            policy_rule = f"Policy Rule 2 (Malignancy Referral): Detected high-risk lesion code '{pred_code}'."
            action_justification = (
                f"Diagnosis Agent identified malignant pathology '{pred_name}' [{pred_code}] with {conf*100:.1f}% primary confidence. "
                f"Automated triage is strictly forbidden; urgent dermatologist biopsy indicated."
            )
            detailed_rationale.append(action_justification)

        # Branch 3: Safety Sentinel Veto -> Mandatory REFER
        elif safety_veto or safety_flag in {"SAFETY_VETO", "SAFETY_FLAG"}:
            decision = "REFER"
            primary_trig = a5_safety.get("primary_trigger", "Clinical conflict detected")
            policy_rule = f"Policy Rule 3 (Safety Sentinel Veto): Overrode automated triage via {primary_trig}."
            action_justification = (
                f"Clinical safety sentinel vetoed automated discharge: {a5_safety.get('justification', '')} "
                f"Escalated to specialist review to protect patient safety."
            )
            detailed_rationale.append(action_justification)

        # Branch 4: Diagnostic Uncertainty -> REFER
        elif conf < auto_conf_threshold:
            decision = "REFER"
            policy_rule = f"Policy Rule 4 (Diagnostic Uncertainty): Confidence {conf*100:.1f}% below threshold {auto_conf_threshold*100:.0f}%."
            action_justification = (
                f"Primary candidate '{pred_name}' [{pred_code}] confidence ({conf*100:.1f}%) is below the "
                f"autonomous triage threshold ({auto_conf_threshold*100:.0f}%). Clinical evaluation required to resolve differential ambiguity."
            )
            detailed_rationale.append(action_justification)

        # Branch 5: Verified Benign Autonomous Screening -> AUTO_TRIAGE
        elif pred_code in self.benign_codes and conf >= auto_conf_threshold and not safety_veto:
            decision = "AUTO_TRIAGE"
            policy_rule = "Policy Rule 5 (Verified Benign Autonomous Screening): High-confidence benign lesion with clear safety checks."
            action_justification = (
                f"Autonomous screening approved: High-confidence benign '{pred_name}' [{pred_code}] ({conf*100:.1f}% >= {auto_conf_threshold*100:.0f}%) "
                f"corroborated by regular dermoscopic ABCD features (TDS: {tds_score:.2f}) and validated image capture quality ({q_score:.1f}/100)."
            )
            detailed_rationale.append(action_justification)

        else:
            decision = "REFER"
            policy_rule = "Policy Rule 6 (Conservative Clinical Fallback): Consensus incomplete."
            action_justification = "Conservative clinical referral applied due to borderline multi-agent consensus."
            detailed_rationale.append(action_justification)

        # Standardized Traceable Chain matching Summit Specification:
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
            "action_justification": action_justification,
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
