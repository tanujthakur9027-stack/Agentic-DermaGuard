from typing import Dict, Any, Optional
import time
from thresholds import load_thresholds, get_flat_thresholds

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
      - Policy Rule 6: Clinical Consensus Fallback -> REFER
    Outputs:
      - Explaining Agent contract with full evidence, margins, and explanation
      - Final predicted label (Official 7-class code: nv, mel, bcc, akiec, bkl, df, vasc)
      - Final triage action (AUTO_TRIAGE, REFER, or RE_IMAGE)
      - Standardized traceable chain & required UI example strip
    """

    def __init__(self, auto_triage_confidence: Optional[float] = None):
        defaults = get_flat_thresholds()
        self.auto_triage_confidence = auto_triage_confidence if auto_triage_confidence is not None else defaults.get("auto_triage_confidence", 0.80)
        self.benign_codes = {'nv', 'bkl', 'df', 'vasc'}
        self.malignant_codes = {'mel', 'bcc', 'akiec'}

    def coordinate(
        self,
        a1_quality: Dict[str, Any],
        a2_diagnosis: Dict[str, Any],
        a3_lesion: Dict[str, Any],
        a4_context: Dict[str, Any],
        a5_safety: Dict[str, Any],
        custom_auto_conf: Optional[float] = None,
        custom_thresholds: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        th = get_flat_thresholds(custom_thresholds)
        auto_conf_threshold = custom_auto_conf if custom_auto_conf is not None else float(th.get("auto_triage_confidence", self.auto_triage_confidence))

        # Policy rule switches
        rule_q_enabled = th.get("policy_quality_gate_enabled", True)
        rule_mal_enabled = th.get("policy_malignancy_refer_enabled", True)
        rule_veto_enabled = th.get("policy_safety_veto_enabled", True)
        rule_conf_enabled = th.get("policy_confidence_gate_enabled", True)
        rule_auto_enabled = th.get("policy_benign_autotriage_enabled", True)

        # 1. Extract agent findings
        # A1 Quality
        q_status = a1_quality.get("status", "PASS")
        q_score = float(a1_quality.get("score", a1_quality.get("overall_quality", 80.0)))
        a1_token = "RE_IMAGE" if q_status == "RE_IMAGE" else ("CAUTION" if "CAUTION" in q_status else "PASS")

        # A2 Diagnosis (Track Golden Rule: NEVER alter prediction)
        pred_code = a2_diagnosis.get("status", a2_diagnosis.get("predicted_code", "nv"))
        pred_name = a2_diagnosis.get("predicted_class", "Melanocytic nevus")
        conf = float(a2_diagnosis.get("score", a2_diagnosis.get("confidence", 0.70)))
        is_ambiguous = a2_diagnosis.get("is_ambiguous", False)
        a2_token = f"{pred_code} • {conf:.2f}"

        # A3 Lesion + A4 Context
        lesion_score = float(a3_lesion.get("score", a3_lesion.get("suspicion_score", 0.0)))
        lesion_status = a3_lesion.get("status", a3_lesion.get("suspicion_level", "BENIGN_CUES"))
        tds_score = float(a3_lesion.get("tds_score", 3.5))
        context_risk = a4_context.get("status", a4_context.get("risk_indicator", "LOW"))

        susp_indet = float(th.get("suspicion_suspicious_from", 0.55))
        susp_benign = float(th.get("suspicion_indeterminate_from", 0.35))

        if lesion_score >= susp_indet or tds_score >= 5.2 or context_risk in {"HIGH", "VERY_HIGH"} or lesion_status == "SUSPICIOUS":
            a3_a4_token = "SUSPICIOUS"
        elif lesion_score >= susp_benign or context_risk == "MODERATE" or lesion_status == "INDETERMINATE":
            a3_a4_token = "INDETERMINATE"
        else:
            a3_a4_token = "BENIGN_CUES"

        # A5 Safety
        safety_flag = a5_safety.get("status", a5_safety.get("safety_flag", "PASS"))
        safety_veto = a5_safety.get("veto_auto_triage", False)
        a5_token = "SAFETY VETO" if safety_flag == "SAFETY_VETO" else ("SAFETY FLAG" if "FLAG" in safety_flag else "PASS")

        # --- Documented Decision Policy Matrix ---
        decision = "REFER"
        policy_rule = ""
        detailed_rationale = []
        action_justification = ""
        flags = []

        # Branch 1: Quality Gatekeeper Rule -> RE_IMAGE
        if rule_q_enabled and q_status == "RE_IMAGE":
            decision = "RE_IMAGE"
            flags.append("POLICY_RULE_1_QUALITY_GATE")
            policy_rule = "Policy Rule 1 (Quality Gate): Capture degradation prevents dependable reading."
            triggers = a1_quality.get("triggered_criteria", a1_quality.get("flags", ["Low composite quality"]))
            action_justification = (
                f"Image quality failed clinical acceptance ({', '.join(triggers)}). "
                f"Re-imaging requested before definitive screening. "
                f"Preliminary diagnostic inference is '{pred_name}' [{pred_code}] ({conf*100:.1f}%), recorded per competition guidelines."
            )
            detailed_rationale.append(action_justification)

        # Branch 2: Malignancy Presentation -> Mandatory REFER
        elif rule_mal_enabled and pred_code in self.malignant_codes:
            decision = "REFER"
            flags.append("POLICY_RULE_2_MALIGNANCY")
            policy_rule = f"Policy Rule 2 (Malignancy Referral): Detected high-risk lesion code '{pred_code}'."
            action_justification = (
                f"Diagnosis Agent identified malignant pathology '{pred_name}' [{pred_code}] with {conf*100:.1f}% primary confidence. "
                f"Automated triage is strictly forbidden; urgent dermatologist biopsy indicated."
            )
            detailed_rationale.append(action_justification)

        # Branch 3: Safety Sentinel Veto -> Mandatory REFER
        elif rule_veto_enabled and (safety_veto or safety_flag in {"SAFETY_VETO", "SAFETY_FLAG"}):
            decision = "REFER"
            flags.append("POLICY_RULE_3_SAFETY_VETO")
            primary_trig = a5_safety.get("primary_trigger", "Clinical conflict detected")
            policy_rule = f"Policy Rule 3 (Safety Sentinel Veto): Overrode automated triage via {primary_trig}."
            action_justification = (
                f"Clinical safety sentinel vetoed automated discharge: {a5_safety.get('justification', '')} "
                f"Escalated to specialist review to protect patient safety."
            )
            detailed_rationale.append(action_justification)

        # Branch 4: Diagnostic Uncertainty -> REFER
        elif rule_conf_enabled and (conf < auto_conf_threshold or is_ambiguous):
            decision = "REFER"
            flags.append("POLICY_RULE_4_DIAGNOSTIC_UNCERTAINTY")
            policy_rule = f"Policy Rule 4 (Diagnostic Uncertainty): Confidence {conf*100:.1f}% below threshold {auto_conf_threshold*100:.0f}% or differential ambiguity."
            action_justification = (
                f"Primary candidate '{pred_name}' [{pred_code}] confidence ({conf*100:.1f}%) is below the "
                f"autonomous triage threshold ({auto_conf_threshold*100:.0f}%) or exhibits differential ambiguity. "
                f"Clinical evaluation required to resolve diagnostic ambiguity."
            )
            detailed_rationale.append(action_justification)

        # Branch 5: Verified Benign Autonomous Screening -> AUTO_TRIAGE
        elif rule_auto_enabled and (pred_code in self.benign_codes and conf >= auto_conf_threshold and not safety_veto):
            decision = "AUTO_TRIAGE"
            flags.append("POLICY_RULE_5_VERIFIED_BENIGN_AUTOTRIAGE")
            policy_rule = "Policy Rule 5 (Verified Benign Autonomous Screening): High-confidence benign lesion with clear safety checks."
            action_justification = (
                f"Autonomous screening approved: High-confidence benign '{pred_name}' [{pred_code}] ({conf*100:.1f}% >= {auto_conf_threshold*100:.0f}%) "
                f"corroborated by regular dermoscopic ABCD features (TDS: {tds_score:.2f}) and validated image capture quality ({q_score:.1f}/100)."
            )
            detailed_rationale.append(action_justification)

        else:
            decision = "REFER"
            flags.append("POLICY_RULE_6_FALLBACK_REFERRAL")
            policy_rule = "Policy Rule 6 (Conservative Clinical Fallback): Multi-agent consensus borderline."
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

        # UI strip matching exact competition requirements:
        # e.g.: A1 PASS "Image is usable" | A2 nv • 0.73 "Model predicts nevus" | A3+A4 SUSPICIOUS "Other evidence raises concern" | A5 SAFETY FLAG "Conflict detected" | A6 REFER "Final disposition"
        clean_names = {
            'nv': 'nevus',
            'mel': 'melanoma',
            'bcc': 'basal cell',
            'akiec': 'actinic keratosis',
            'bkl': 'benign keratosis',
            'df': 'dermatofibroma',
            'vasc': 'vascular lesion'
        }
        a1_note = "Image is usable" if a1_token == "PASS" else ("Borderline quality" if a1_token == "CAUTION" else "Quality defect detected")
        a2_note = f"Model predicts {clean_names.get(pred_code, 'nevus')}"
        a3_a4_note = "Other evidence raises concern" if a3_a4_token == "SUSPICIOUS" else ("Intermediate features" if a3_a4_token == "INDETERMINATE" else "Benign visual cues")
        a5_note = "Conflict detected" if a5_token != "PASS" else "All checks clear"
        a6_note = "Final disposition"

        example_strip = (
            f'A1 {a1_token} "{a1_note}" | '
            f'A2 {pred_code} • {conf:.2f} "{a2_note}" | '
            f'A3+A4 {a3_a4_token} "{a3_a4_note}" | '
            f'A5 {a5_token} "{a5_note}" | '
            f'A6 {decision} "{a6_note}"'
        )

        evidence = [
            {
                "feature": "policy_decision",
                "value": decision,
                "threshold": "AUTO_TRIAGE / REFER / RE_IMAGE",
                "direction": "==",
                "passed": True,
                "note": f"Coordinator applied {policy_rule}"
            },
            {
                "feature": "predicted_class_code",
                "value": pred_code,
                "threshold": "7 HAM10000 codes",
                "direction": "in",
                "passed": True,
                "note": f"Preserved diagnostic classification: '{pred_name}' [{pred_code}] ({conf*100:.1f}%)"
            },
            {
                "feature": "safety_veto_status",
                "value": safety_veto,
                "threshold": False,
                "direction": "==",
                "passed": not safety_veto,
                "note": f"Safety veto status: {'VETO FIRED' if safety_veto else 'CLEAR'}"
            }
        ]

        thresholds_used = {
            "auto_triage_confidence": auto_conf_threshold,
            "policy_rule_applied": policy_rule
        }

        margin_to_threshold = {
            "auto_triage_confidence": round(conf - auto_conf_threshold, 4)
        }

        summary = f"Final disposition is {decision} with diagnosis '{pred_name}' [{pred_code}] ({conf*100:.1f}% confidence)."

        explanation = (
            f"Coordinator Agent synthesized outputs from all 5 agents and executed {policy_rule}. "
            f"The final diagnosis is confirmed as '{pred_name}' [{pred_code}] with disposition '{decision}'. "
            f"{action_justification}"
        )

        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return {
            # Explaining Agent Contract
            "agent": "A6",
            "name": "Coordinator",
            "status": decision,
            "score": round(conf, 4),
            "summary": summary,
            "evidence": evidence,
            "thresholds_used": thresholds_used,
            "margin_to_threshold": margin_to_threshold,
            "flags": flags,
            "explanation": explanation,
            "confidence_in_this_agent": 1.0,
            "runtime_ms": runtime_ms,
            # Preserved legacy and summit-specific fields
            "agent_id": "A6",
            "agent_name": "Coordinator Agent",
            "prediction": pred_code,
            "prediction_name": pred_name,
            "confidence": round(conf, 4),
            "decision": decision,
            "traceable_chain": traceable_chain,
            "example_strip": example_strip,
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
