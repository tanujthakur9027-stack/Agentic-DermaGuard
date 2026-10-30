from typing import Dict, Any, List

class SafetyAgent:
    """
    Agent 5: Safety Agent (Clinical Safety Sentinel & Veto)
    Jaypee Agentic AI Summit 2026 — Track 2 | Healthcare

    Performs rigorous cross-agent safety auditing:
      - Reviews diagnosis against ABCD structural cues, patient demographics, and capture quality.
      - Blocks unsafe autonomous discharge (AUTO_TRIAGE) whenever clinical conflict or risk is detected.
      - Enforces the 'Primum Non Nocere' (First, do no harm) principle in medical AI.
    Provides:
      - safety_flag: PASS | SAFETY_FLAG | SAFETY_VETO
      - veto_auto_triage: bool (strictly forbids AUTO_TRIAGE if True)
      - active_conflicts: List of specific discordance items
      - primary_trigger: Root rule that fired
      - justification: Medical rationale formatted for audit panel inspection
    """

    def __init__(self):
        self.malignant_classes = {'mel', 'bcc', 'akiec', 'Melanoma', 'Basal cell carcinoma', 'Actinic keratoses'}

    def review(
        self,
        quality_data: Dict[str, Any],
        diagnosis_data: Dict[str, Any],
        lesion_data: Dict[str, Any],
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        veto = False
        triggers = []
        conflicts = []
        justifications = []

        # Extract findings
        pred_code = diagnosis_data.get("predicted_code", "")
        pred_class = diagnosis_data.get("predicted_class", "")
        confidence = diagnosis_data.get("confidence", 0.0)
        probs = diagnosis_data.get("all_probabilities", {})

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

        # --- Rule S1: Direct Malignancy Detection ---
        if pred_code in {'mel', 'bcc', 'akiec'} or pred_class in self.malignant_classes:
            veto = True
            triggers.append("RULE_S1:PREDICTED_MALIGNANCY")
            conflicts.append({
                "type": "MALIGNANCY_PRESENTATION",
                "severity": "CRITICAL",
                "detail": f"Pathology predicted as '{pred_class}' [{pred_code}] ({confidence*100:.1f}% confidence)."
            })
            justifications.append(
                f"Model predicts malignant/pre-malignant pathology '{pred_class}' ({confidence*100:.1f}% confidence). "
                f"Automated triage strictly contraindicated; immediate specialist biopsy/dermatology referral mandated."
            )

        # --- Rule S2: Cumulative Malignancy Residual Probability ---
        elif total_malignancy_risk >= 0.28:
            veto = True
            triggers.append("RULE_S2:ELEVATED_MALIGNANCY_RESIDUAL_PROBABILITY")
            conflicts.append({
                "type": "SUBTHRESHOLD_MALIGNANCY_RISK",
                "severity": "HIGH",
                "detail": f"Cumulative malignant probability {total_malignancy_risk*100:.1f}% exceeds 28% clinical threshold."
            })
            justifications.append(
                f"While primary candidate is '{pred_class}', cumulative malignancy probability "
                f"is dangerously elevated ({total_malignancy_risk*100:.1f}%: Melanoma {mel_p*100:.1f}%, BCC {bcc_p*100:.1f}%, AKIEC {akiec_p*100:.1f}%). "
                f"Safety veto blocks automated dismissal."
            )

        # --- Rule S3: Cross-Agent Conflict (Benign Diagnosis vs High ABCD Structural Score) ---
        if pred_code in {'nv', 'bkl', 'df', 'vasc'}:
            has_abcd_discordance = (tds_score > 5.20) or (border_pts >= 6 and asym_pts >= 1) or (lesion_suspicion >= 0.65)
            if has_abcd_discordance:
                veto = True
                triggers.append("RULE_S3:DIAGNOSIS_VS_ABCD_STRUCTURE_DISCORDANCE")
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

        # --- Rule S4: Cross-Agent Conflict (Benign Diagnosis vs High-Risk Patient Demographics) ---
        if pred_code in {'nv', 'bkl'} and context_risk == "HIGH" and confidence < 0.85:
            veto = True
            triggers.append("RULE_S4:DIAGNOSIS_VS_DEMOGRAPHIC_CONTEXT_DISCORDANCE")
            conflicts.append({
                "type": "DEMOGRAPHIC_DISCORDANCE",
                "severity": "MODERATE",
                "detail": f"Benign nevus diagnosed in high-risk patient demographic (Age: {age if age else 'Unk'}, Site: '{loc if loc else 'Unk'}')."
            })
            justifications.append(
                f"Context conflict detected: Automated benign diagnosis '{pred_class}' in high-risk demographic presentation "
                f"(Age: {age if age else 'Unk'} at chronic UV site: '{loc if loc else 'Unk'}'). New or evolving lesions in this group "
                f"exhibit high prior for lentigo maligna or squamous cell carcinoma."
            )

        # --- Rule S5: Image Quality Precaution ---
        if quality_status == 'RE_IMAGE':
            veto = True
            triggers.append("RULE_S5:UNSAFE_ON_DEFECTIVE_IMAGE")
            conflicts.append({
                "type": "QUALITY_FAILURE",
                "severity": "CRITICAL",
                "detail": f"Image quality gatekeeper failed capture ({', '.join(triggered_defects) if triggered_defects else 'Below threshold'})."
            })
            justifications.append(
                f"Image quality failed clinical acceptance (Status: RE_IMAGE; defects: {', '.join(triggered_defects)}). "
                f"Automated triage invalid on compromised capture."
            )
        elif quality_status == 'ACCEPTABLE_WITH_CAUTION' and confidence < 0.70:
            veto = True
            triggers.append("RULE_S5:QUALITY_CAUTION_DISCHARGE_BLOCK")
            conflicts.append({
                "type": "QUALITY_CAUTION",
                "severity": "LOW",
                "detail": "Borderline capture quality combined with non-definitive confidence."
            })
            justifications.append(
                f"Image quality caution combined with borderline diagnostic confidence ({confidence*100:.1f}%). "
                f"Specialist referral advised to prevent false reassurance."
            )

        # --- Rule S6: Diagnostic Uncertainty ---
        if confidence < 0.65 and not veto:
            veto = True
            triggers.append("RULE_S6:DIAGNOSTIC_UNCERTAINTY")
            conflicts.append({
                "type": "DIAGNOSTIC_UNCERTAINTY",
                "severity": "MODERATE",
                "detail": f"Top candidate confidence ({confidence*100:.1f}%) is below 65% autonomous safety threshold."
            })
            justifications.append(
                f"Diagnostic confidence ({confidence*100:.1f}%) is below safe autonomous threshold (65.0%). "
                f"Requires clinical dermatologist evaluation to resolve differential ambiguity."
            )

        # Determine flag severity
        if veto:
            is_critical = any(c.get("severity") == "CRITICAL" for c in conflicts) or any("S1" in t or "S3" in t for t in triggers)
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
