from typing import Dict, Any, List, Optional
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
      RULE_S0:PREDICTED_MALIGNANCY (Direct predicted malignancy code mel, bcc, akiec)
      RULE_S1:RESIDUAL_MALIGNANCY_RISK (Benign prediction but malignant prob mass >= threshold)
      RULE_S2:LESION_ABCD_DISCORDANCE (Benign prediction + SUSPICIOUS lesion cues / TDS)
      RULE_S3:DIAGNOSTIC_AMBIGUITY (Top-1 vs top-2 margin too small or entropy too high)
      RULE_S4:CONTEXT_DEMOGRAPHIC_DISCORDANCE (Benign prediction in high-risk demographic presentation)
      RULE_S5:QUALITY_CAUTION_DISCHARGE_BLOCK (Quality caution + non-definitive confidence)
      RULE_S6:TTA_DISAGREEMENT (Test-time augmentation views disagree significantly)
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
        th = get_flat_thresholds(custom_thresholds)
        
        # Rule toggles
        s1_enabled = th.get("s1_enabled", True)
        s2_enabled = th.get("s2_enabled", True)
        s3_enabled = th.get("s3_enabled", True)
        s4_enabled = th.get("s4_enabled", True)
        s5_enabled = th.get("s5_enabled", True)
        s6_enabled = th.get("s6_enabled", True)

        # Cutoffs
        malig_thresh = th.get("malignant_prob_threshold", 0.28)
        mel_thresh = th.get("mel_threshold", 0.20)
        margin_min = th.get("margin_minimum", 0.15)
        max_entropy = th.get("max_entropy", 1.75)
        tta_thresh = th.get("tta_disagreement_threshold", 0.22)
        caution_conf_req = th.get("caution_confidence_required", 0.70)
        context_conf_req = th.get("high_context_risk_confidence_required", 0.85)

        veto = False
        triggers = []
        conflicts = []
        justifications = []

        # Extract findings
        pred_code = diagnosis_data.get("predicted_code", "")
        pred_class = diagnosis_data.get("predicted_class", "")
        confidence = diagnosis_data.get("confidence", 0.0)
        probs = diagnosis_data.get("all_probabilities", {})
        margin = diagnosis_data.get("margin_to_next", 0.5)
        entropy = diagnosis_data.get("entropy", 0.0)
        tta_disag = diagnosis_data.get("tta_disagreement", 0.0)

        mel_p = probs.get("mel", probs.get("Melanoma", 0.0))
        bcc_p = probs.get("bcc", probs.get("Basal cell carcinoma", 0.0))
        akiec_p = probs.get("akiec", probs.get("Actinic keratoses", 0.0))
        total_malignancy_risk = round(mel_p + bcc_p + akiec_p, 4)

        quality_status = quality_data.get("status", "ACCEPTABLE")
        triggered_defects = quality_data.get("triggered_criteria", [])

        lesion_suspicion = lesion_data.get("suspicion_score", 0.0)
        tds_score = lesion_data.get("tds_score", 3.5)
        abcd = lesion_data.get("abcd_breakdown", {})
        border_pts = abcd.get("B_border", {}).get("border_score", 1)
        asym_pts = abcd.get("A_asymmetry", {}).get("axis_score", 0)

        context_risk = context_data.get("risk_indicator", "LOW")
        parsed_fields = context_data.get("parsed_fields", {})
        age = parsed_fields.get("age")
        loc = parsed_fields.get("localization")

        # --- Rule S0: Direct Malignancy Presentation ---
        if pred_code in {'mel', 'bcc', 'akiec'} or pred_class in self.malignant_classes:
            veto = True
            triggers.append("RULE_S0:PREDICTED_MALIGNANCY")
            conflicts.append({
                "type": "MALIGNANCY_PRESENTATION",
                "severity": "CRITICAL",
                "detail": f"Pathology predicted as '{pred_class}' [{pred_code}] ({confidence*100:.1f}% confidence)."
            })
            justifications.append(
                f"Model predicts malignant/pre-malignant pathology '{pred_class}' ({confidence*100:.1f}% confidence). "
                f"Automated triage strictly contraindicated; immediate specialist biopsy/dermatology referral mandated."
            )

        # --- Rule S1: Residual Malignancy Probability Mass ---
        elif s1_enabled and (total_malignancy_risk >= malig_thresh or mel_p >= mel_thresh):
            veto = True
            triggers.append("RULE_S1:RESIDUAL_MALIGNANCY_RISK")
            conflicts.append({
                "type": "SUBTHRESHOLD_MALIGNANCY_RISK",
                "severity": "HIGH",
                "detail": f"Cumulative malignant prob {total_malignancy_risk*100:.1f}% >= {malig_thresh*100:.0f}% or Mel prob {mel_p*100:.1f}% >= {mel_thresh*100:.0f}%."
            })
            justifications.append(
                f"While primary candidate is '{pred_class}', cumulative malignancy probability "
                f"is elevated ({total_malignancy_risk*100:.1f}%: Melanoma {mel_p*100:.1f}%, BCC {bcc_p*100:.1f}%, AKIEC {akiec_p*100:.1f}%). "
                f"Safety veto blocks automated dismissal."
            )

        # --- Rule S2: Cross-Agent Conflict (Benign Diagnosis vs Suspicious ABCD Structural Score) ---
        if s2_enabled and pred_code in {'nv', 'bkl', 'df', 'vasc'}:
            has_abcd_discordance = (tds_score > 5.20) or (border_pts >= 6 and asym_pts >= 1) or (lesion_suspicion >= 0.65)
            if has_abcd_discordance:
                veto = True
                triggers.append("RULE_S2:LESION_ABCD_DISCORDANCE")
                conflicts.append({
                    "type": "STRUCTURAL_DISCORDANCE",
                    "severity": "HIGH",
                    "detail": f"Diagnostic classifier predicts benign '{pred_code}', but ABCD analysis reveals elevated Total Dermoscopy Score (TDS={tds_score:.2f}) with {asym_pts} asymmetry axes and border score {border_pts}/8."
                })
                justifications.append(
                    f"Structural conflict detected: Diagnosis predicts benign '{pred_class}', but lesion visual analysis reveals "
                    f"concerning dermoscopic architecture (TDS: {tds_score:.2f}, Border: {border_pts}/8, Asymmetry: {asym_pts}/2). "
                    f"Veto prevents missing atypical dysplastic nevi or early-stage melanomas."
                )

        # --- Rule S3: Diagnostic Ambiguity (Margin Too Small or Entropy Too High) ---
        if s3_enabled and pred_code in {'nv', 'bkl', 'df', 'vasc'}:
            is_ambiguous = (margin < margin_min) or (entropy > max_entropy)
            if is_ambiguous:
                veto = True
                triggers.append("RULE_S3:DIAGNOSTIC_AMBIGUITY")
                conflicts.append({
                    "type": "DIAGNOSTIC_AMBIGUITY",
                    "severity": "MODERATE",
                    "detail": f"Top-2 margin ({margin:.2f} < {margin_min:.2f}) or Shannon entropy ({entropy:.2f} > {max_entropy:.2f}) indicates differential ambiguity."
                })
                justifications.append(
                    f"Diagnostic ambiguity detected: Top-2 margin ({margin:.2f}) is below minimum ({margin_min:.2f}) "
                    f"or Shannon entropy ({entropy:.2f} bits) exceeds safety limit ({max_entropy:.2f} bits). Specialist review required."
                )

        # --- Rule S4: Demographic Context Discordance ---
        if s4_enabled and pred_code in {'nv', 'bkl'} and context_risk == "HIGH" and confidence < context_conf_req:
            veto = True
            triggers.append("RULE_S4:CONTEXT_DEMOGRAPHIC_DISCORDANCE")
            conflicts.append({
                "type": "DEMOGRAPHIC_DISCORDANCE",
                "severity": "MODERATE",
                "detail": f"Benign nevus diagnosed in high-risk demographic (Age: {age if age else 'Unk'}, Site: '{loc if loc else 'Unk'}') with confidence {confidence*100:.1f}% < {context_conf_req*100:.0f}%."
            })
            justifications.append(
                f"Context conflict detected: Automated benign diagnosis '{pred_class}' in high-risk demographic presentation "
                f"(Age: {age if age else 'Unk'} at chronic UV site: '{loc if loc else 'Unk'}'). New or evolving lesions in this cohort "
                f"exhibit high prior for lentigo maligna or non-melanoma skin cancer."
            )

        # --- Rule S5: Image Quality Caution Precaution ---
        if s5_enabled:
            if quality_status == 'RE_IMAGE':
                veto = True
                triggers.append("RULE_S5:QUALITY_FAILURE_BLOCK")
                conflicts.append({
                    "type": "QUALITY_FAILURE",
                    "severity": "CRITICAL",
                    "detail": f"Image quality gatekeeper failed capture ({', '.join(triggered_defects) if triggered_defects else 'Below cutoff'})."
                })
                justifications.append(
                    f"Image quality failed clinical acceptance (Status: RE_IMAGE; defects: {', '.join(triggered_defects)}). "
                    f"Automated triage invalid on compromised capture."
                )
            elif quality_status == 'ACCEPTABLE_WITH_CAUTION' and confidence < caution_conf_req:
                veto = True
                triggers.append("RULE_S5:QUALITY_CAUTION_DISCHARGE_BLOCK")
                conflicts.append({
                    "type": "QUALITY_CAUTION",
                    "severity": "LOW",
                    "detail": f"Borderline capture quality combined with non-definitive confidence ({confidence*100:.1f}% < {caution_conf_req*100:.0f}%)."
                })
                justifications.append(
                    f"Image quality caution combined with non-definitive diagnostic confidence ({confidence*100:.1f}% < {caution_conf_req*100:.0f}%). "
                    f"Specialist referral advised to prevent false reassurance."
                )

        # --- Rule S6: Test-Time Augmentation Disagreement ---
        if s6_enabled and tta_disag > tta_thresh:
            veto = True
            triggers.append("RULE_S6:TTA_DISAGREEMENT")
            conflicts.append({
                "type": "TTA_INSTABILITY",
                "severity": "MODERATE",
                "detail": f"Test-Time Augmentation view standard deviation ({tta_disag:.3f}) exceeds threshold ({tta_thresh:.3f})."
            })
            justifications.append(
                f"TTA instability detected: Model probability distribution shifted significantly across rotation/flip views ({tta_disag:.3f} > {tta_thresh:.3f}), "
                f"indicating borderline lesion morphology or orientational artifact."
            )

        # Determine flag severity
        if veto:
            is_critical = any(c.get("severity") == "CRITICAL" for c in conflicts) or any("S0" in t or "S2" in t for t in triggers)
            safety_flag = "SAFETY_VETO" if is_critical else "SAFETY_FLAG"
        else:
            safety_flag = "PASS"
            justifications.append("Cross-agent audit passed: No diagnostic, structural, quality, or demographic conflicts detected.")

        return {
            "agent_id": "A5",
            "agent_name": "Safety Agent",
            "safety_flag": safety_flag,
            "veto_auto_triage": veto,
            "triggers": triggers,
            "primary_trigger": triggers[0] if triggers else "NONE",
            "active_conflicts": conflicts,
            "total_conflicts": len(conflicts),
            "justification": " | ".join(justifications),
            "summary": (
                f"[{safety_flag}] {triggers[0] if triggers else 'All checks clear'}: "
                f"{justifications[0]}"
            )
        }

# Singleton instance
safety_agent = SafetyAgent()
