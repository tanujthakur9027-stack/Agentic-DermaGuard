from typing import Dict, Any, List, Optional
import time
from thresholds import load_thresholds, get_flat_thresholds

class SafetyAgent:
    """
    Agent 5: Safety Agent (Clinical Safety Sentinel & Veto)
    Jaypee Agentic AI Summit 2026 — Track 2 | Healthcare

    Performs rigorous cross-agent safety auditing:
      - Reviews diagnosis against ABCD structural cues, patient demographics, and capture quality.
      - Blocks unsafe autonomous discharge (AUTO_TRIAGE) whenever clinical conflict or risk is detected.
      - Enforces the 'Primum Non Nocere' (First, do no harm) principle in medical AI.

    Named Clinical Safety Rules:
      S1: Residual / Predicted Malignancy (Malignant prediction or residual prob mass >= cutoff)
      S2: ABCD Lesion Structural Discordance (Benign prediction + SUSPICIOUS lesion cues / TDS)
      S3: Diagnostic Ambiguity (Small top1-top2 margin or high Shannon entropy)
      S4: Demographic Context Discordance (Benign prediction in high-risk demographic presentation)
      S5: Capture Degradation Discharge Block (Quality RE_IMAGE or CAUTION + non-definitive confidence)
      S6: Test-Time Augmentation Disagreement (High variance across rotation/flip views)
    """

    def __init__(self):
        self.malignant_classes = {'mel', 'bcc', 'akiec', 'Melanoma', 'Basal cell carcinoma', 'Actinic keratoses'}

    def review(
        self,
        quality_data: Dict[str, Any],
        diagnosis_data: Dict[str, Any],
        lesion_data: Dict[str, Any],
        context_data: Dict[str, Any],
        custom_thresholds: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        th = get_flat_thresholds(custom_thresholds)
        
        # Rule toggles
        s1_enabled = th.get("safety_s1_enabled", th.get("s1_enabled", True))
        s2_enabled = th.get("safety_s2_enabled", th.get("s2_enabled", True))
        s3_enabled = th.get("safety_s3_enabled", th.get("s3_enabled", True))
        s4_enabled = th.get("safety_s4_enabled", th.get("s4_enabled", True))
        s5_enabled = th.get("safety_s5_enabled", th.get("s5_enabled", True))
        s6_enabled = th.get("safety_s6_enabled", th.get("s6_enabled", True))

        # Cutoffs
        malig_thresh = float(th.get("malignant_prob_threshold", 0.22))
        mel_thresh = float(th.get("mel_prob_threshold", th.get("mel_threshold", 0.18)))
        margin_min = float(th.get("top2_margin_min", th.get("margin_minimum", 0.10)))
        max_entropy = float(th.get("max_entropy", 1.65))
        tta_thresh = float(th.get("tta_disagreement_threshold", 0.22))
        caution_conf_req = float(th.get("caution_confidence_required", 0.70))
        context_conf_req = float(th.get("high_context_risk_confidence_required", 0.85))

        veto = False
        triggers = []
        conflicts = []
        justifications = []
        rules_evaluated = []

        # Extract findings from other agents
        pred_code = diagnosis_data.get("predicted_code", diagnosis_data.get("status", "nv"))
        pred_class = diagnosis_data.get("predicted_class", "Melanocytic nevus")
        confidence = float(diagnosis_data.get("confidence", diagnosis_data.get("score", 0.0)))
        probs = diagnosis_data.get("all_probabilities", {})
        margin = float(diagnosis_data.get("margin_to_next", 0.5))
        entropy = float(diagnosis_data.get("entropy", 0.0))
        tta_disag = float(diagnosis_data.get("tta_disagreement", 0.0))

        mel_p = float(probs.get("mel", probs.get("Melanoma", 0.0)))
        bcc_p = float(probs.get("bcc", probs.get("Basal cell carcinoma", 0.0)))
        akiec_p = float(probs.get("akiec", probs.get("Actinic keratoses", 0.0)))
        total_malignancy_risk = round(mel_p + bcc_p + akiec_p, 4)

        quality_status = quality_data.get("status", "PASS")
        triggered_defects = quality_data.get("triggered_criteria", [])

        lesion_suspicion = float(lesion_data.get("suspicion_score", lesion_data.get("score", 0.0)))
        lesion_status = lesion_data.get("status", lesion_data.get("suspicion_level", "BENIGN_CUES"))
        tds_score = float(lesion_data.get("tds_score", 3.5))
        abcd = lesion_data.get("abcd_breakdown", {})
        border_pts = abcd.get("B_border", {}).get("border_score", 1)
        asym_pts = abcd.get("A_asymmetry", {}).get("axis_score", 0)

        context_risk = context_data.get("risk_indicator", context_data.get("status", "LOW"))
        parsed_fields = context_data.get("parsed_fields", {})
        age = parsed_fields.get("age")
        loc = parsed_fields.get("localization")

        # --- Rule S1: Residual / Predicted Malignancy Risk ---
        s1_fired = False
        s1_reason = "No abnormal malignancy probability detected."
        if s1_enabled:
            if pred_code in {'mel', 'bcc', 'akiec'} or pred_class in self.malignant_classes:
                s1_fired = True
                s1_reason = f"Primary predicted class is malignant '{pred_class}' [{pred_code}] ({confidence*100:.1f}% confidence)."
                veto = True
                triggers.append("S1:PREDICTED_MALIGNANCY")
                conflicts.append({
                    "type": "MALIGNANCY_PRESENTATION",
                    "severity": "CRITICAL",
                    "detail": s1_reason
                })
                justifications.append(f"Model predicts malignant lesion '{pred_class}'; automated triage strictly forbidden.")
            elif total_malignancy_risk >= malig_thresh or mel_p >= mel_thresh:
                s1_fired = True
                s1_reason = f"Cumulative malignant prob mass {total_malignancy_risk*100:.1f}% >= {malig_thresh*100:.0f}% or Mel prob {mel_p*100:.1f}% >= {mel_thresh*100:.0f}%."
                veto = True
                triggers.append("S1:RESIDUAL_MALIGNANCY_RISK")
                conflicts.append({
                    "type": "SUBTHRESHOLD_MALIGNANCY_RISK",
                    "severity": "HIGH",
                    "detail": s1_reason
                })
                justifications.append(f"Residual malignancy probability ({total_malignancy_risk*100:.1f}%) exceeds safety threshold.")
        rules_evaluated.append({
            "id": "S1",
            "name": "Residual / Predicted Malignancy",
            "fired": s1_fired,
            "reason": s1_reason
        })

        # --- Rule S2: Benign Diagnosis vs Suspicious ABCD Lesion Cues ---
        s2_fired = False
        s2_reason = "Lesion morphometry is concordant with benign diagnosis."
        if s2_enabled and pred_code in {'nv', 'bkl', 'df', 'vasc'}:
            has_abcd_discordance = (tds_score > 5.20) or (border_pts >= 6 and asym_pts >= 1) or (lesion_suspicion >= 0.65) or (lesion_status == "SUSPICIOUS")
            if has_abcd_discordance:
                s2_fired = True
                s2_reason = f"Diagnostic classifier predicts benign '{pred_code}', but ABCD analysis reveals suspicious cues (TDS={tds_score:.2f}, Border={border_pts}/8, Asymmetry={asym_pts}/2)."
                veto = True
                triggers.append("S2:LESION_ABCD_DISCORDANCE")
                conflicts.append({
                    "type": "STRUCTURAL_DISCORDANCE",
                    "severity": "HIGH",
                    "detail": s2_reason
                })
                justifications.append(f"Structural conflict detected: Diagnosis predicts benign '{pred_class}', but visual ABCD analysis raises suspicion (TDS: {tds_score:.2f}).")
        rules_evaluated.append({
            "id": "S2",
            "name": "ABCD Lesion Structural Discordance",
            "fired": s2_fired,
            "reason": s2_reason
        })

        # --- Rule S3: Diagnostic Ambiguity (Small Top-2 Margin or High Entropy) ---
        s3_fired = False
        s3_reason = "Diagnostic differential shows decisive margin and controlled entropy."
        if s3_enabled and pred_code in {'nv', 'bkl', 'df', 'vasc'}:
            is_ambiguous = (margin < margin_min) or (entropy > max_entropy)
            if is_ambiguous:
                s3_fired = True
                s3_reason = f"Top-2 margin ({margin:.2f} < {margin_min:.2f}) or Shannon entropy ({entropy:.2f} > {max_entropy:.2f}) indicates differential ambiguity."
                veto = True
                triggers.append("S3:DIAGNOSTIC_AMBIGUITY")
                conflicts.append({
                    "type": "DIAGNOSTIC_AMBIGUITY",
                    "severity": "MODERATE",
                    "detail": s3_reason
                })
                justifications.append(f"Diagnostic ambiguity detected: Top-2 margin ({margin:.2f}) or entropy ({entropy:.2f} bits) exceeds safety tolerance.")
        rules_evaluated.append({
            "id": "S3",
            "name": "Diagnostic Ambiguity",
            "fired": s3_fired,
            "reason": s3_reason
        })

        # --- Rule S4: Demographic Context Discordance ---
        s4_fired = False
        s4_reason = "Demographic context is concordant with benign diagnosis."
        if s4_enabled and pred_code in {'nv', 'bkl', 'df'} and context_risk in {"HIGH", "VERY_HIGH"} and confidence < context_conf_req:
            s4_fired = True
            s4_reason = f"Benign '{pred_code}' diagnosed in high-risk demographic (Age: {age if age else 'Unk'}, Site: '{loc if loc else 'Unk'}') with confidence {confidence*100:.1f}% < {context_conf_req*100:.0f}%."
            veto = True
            triggers.append("S4:CONTEXT_DEMOGRAPHIC_DISCORDANCE")
            conflicts.append({
                "type": "DEMOGRAPHIC_DISCORDANCE",
                "severity": "MODERATE",
                "detail": s4_reason
            })
            justifications.append(f"Context conflict detected: Automated benign diagnosis in high-risk demographic presentation requires {context_conf_req*100:.0f}% confidence.")
        rules_evaluated.append({
            "id": "S4",
            "name": "Demographic Context Discordance",
            "fired": s4_fired,
            "reason": s4_reason
        })

        # --- Rule S5: Image Quality Caution / Degradation Discharge Block ---
        s5_fired = False
        s5_reason = "Image quality meets screening requirements."
        if s5_enabled:
            if quality_status == 'RE_IMAGE':
                s5_fired = True
                s5_reason = f"Image quality gatekeeper failed capture ({', '.join(triggered_defects) if triggered_defects else 'RE_IMAGE'})."
                veto = True
                triggers.append("S5:QUALITY_FAILURE_BLOCK")
                conflicts.append({
                    "type": "QUALITY_FAILURE",
                    "severity": "CRITICAL",
                    "detail": s5_reason
                })
                justifications.append("Capture quality failed clinical acceptance; autonomous triage prohibited on compromised imagery.")
            elif quality_status == 'ACCEPTABLE_WITH_CAUTION' and confidence < caution_conf_req:
                s5_fired = True
                s5_reason = f"Borderline capture quality combined with non-definitive confidence ({confidence*100:.1f}% < {caution_conf_req*100:.0f}%)."
                veto = True
                triggers.append("S5:QUALITY_CAUTION_DISCHARGE_BLOCK")
                conflicts.append({
                    "type": "QUALITY_CAUTION",
                    "severity": "LOW",
                    "detail": s5_reason
                })
                justifications.append(f"Quality caution requires heightened confidence ({caution_conf_req*100:.0f}%); specialist referral advised.")
        rules_evaluated.append({
            "id": "S5",
            "name": "Capture Degradation Discharge Block",
            "fired": s5_fired,
            "reason": s5_reason
        })

        # --- Rule S6: Test-Time Augmentation Disagreement ---
        s6_fired = False
        s6_reason = "Test-time view predictions are geometrically stable."
        if s6_enabled and tta_disag > tta_thresh:
            s6_fired = True
            s6_reason = f"TTA view standard deviation ({tta_disag:.3f}) exceeds threshold ({tta_thresh:.3f})."
            veto = True
            triggers.append("S6:TTA_DISAGREEMENT")
            conflicts.append({
                "type": "TTA_INSTABILITY",
                "severity": "MODERATE",
                "detail": s6_reason
            })
            justifications.append(f"Model instability across rotations/flips ({tta_disag:.3f} > {tta_thresh:.3f}) indicates borderline morphology.")
        rules_evaluated.append({
            "id": "S6",
            "name": "Test-Time Augmentation Disagreement",
            "fired": s6_fired,
            "reason": s6_reason
        })

        # Determine flag severity
        if veto:
            is_critical = any(c.get("severity") == "CRITICAL" for c in conflicts) or any("S1" in t or "S2" in t for t in triggers)
            safety_flag = "SAFETY_VETO" if is_critical else "SAFETY_FLAG"
            score = 1.0 if is_critical else 0.70
        else:
            safety_flag = "PASS"
            score = 0.0
            justifications.append("Cross-agent audit passed: No diagnostic, structural, quality, or demographic conflicts detected.")

        # Build Explaining Agent Contract evidence list
        evidence = [
            {
                "feature": r["id"],
                "value": r["fired"],
                "threshold": False,
                "direction": "==",
                "passed": not r["fired"],
                "note": f"{r['name']}: {r['reason']}"
            }
            for r in rules_evaluated
        ]

        thresholds_used = {
            "malignant_prob_threshold": malig_thresh,
            "mel_prob_threshold": mel_thresh,
            "top2_margin_min": margin_min,
            "max_entropy": max_entropy,
            "caution_confidence_required": caution_conf_req,
            "high_context_risk_confidence_required": context_conf_req,
            "tta_disagreement_threshold": tta_thresh
        }

        margin_to_threshold = {
            "malignant_prob_threshold": round(malig_thresh - total_malignancy_risk, 4),
            "top2_margin_min": round(margin - margin_min, 4),
            "max_entropy": round(max_entropy - entropy, 4)
        }

        fired_rule_ids = [r["id"] for r in rules_evaluated if r["fired"]]

        if safety_flag == "PASS":
            summary = "All clinical safety checks passed; no cross-agent conflicts detected."
        else:
            summary = f"Safety Sentinel issued {safety_flag}: {triggers[0]}."

        explanation = (
            f"Safety Agent audited cross-agent consistency across 6 sentinel rules resulting in '{safety_flag}'. "
            f"{'Autonomous triage clearance approved with zero clinical conflicts.' if not veto else f'{len(conflicts)} conflict(s) detected: ' + '; '.join(justifications[:2])} "
            f"Primary trigger: {triggers[0] if triggers else 'NONE'}."
        )

        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return {
            # Explaining Agent Contract
            "agent": "A5",
            "name": "Safety Sentinel",
            "status": safety_flag,
            "score": score,
            "summary": summary,
            "evidence": evidence,
            "thresholds_used": thresholds_used,
            "margin_to_threshold": margin_to_threshold,
            "flags": fired_rule_ids,
            "explanation": explanation,
            "confidence_in_this_agent": 1.0,
            "runtime_ms": runtime_ms,
            # Preserved legacy fields
            "agent_id": "A5",
            "agent_name": "Safety Agent",
            "safety_flag": safety_flag,
            "veto_auto_triage": veto,
            "triggers": triggers,
            "primary_trigger": triggers[0] if triggers else "NONE",
            "active_conflicts": conflicts,
            "total_conflicts": len(conflicts),
            "justification": " | ".join(justifications),
            "rules": rules_evaluated
        }

# Singleton instance
safety_agent = SafetyAgent()
