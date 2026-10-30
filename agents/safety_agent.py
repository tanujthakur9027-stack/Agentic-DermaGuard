from typing import Dict, Any, List

class SafetyAgent:
    """
    Agent 5: Safety Agent (Clinical Safety Sentinel & Veto)
    Reviews outputs across all prior agents:
      - Agent 1: Image Quality
      - Agent 2: Diagnosis & Probabilities
      - Agent 3: Lesion Visual Cues (ABCD)
      - Agent 4: Reasoning / Context
    Detects:
      - Conflicts between diagnostic classification and visual cues (e.g., nv with jagged borders)
      - Demographic contradictions (e.g., elderly patient sun-damaged site with benign diagnosis)
      - Unsafe automatic triage attempts on compromised images
      - Probability margin ambiguity
    Provides:
      - safety_flag: PASS | SAFETY_FLAG | SAFETY_VETO
      - veto_auto_triage: bool (blocks AUTO_TRIAGE if True)
      - trigger: specific rule/conflict that fired
      - justification: clinical rationale for panel inspection
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
        justifications = []

        # Extract findings
        pred_code = diagnosis_data.get("predicted_code", "")
        pred_class = diagnosis_data.get("predicted_class", "")
        confidence = diagnosis_data.get("confidence", 0.0)
        probs = diagnosis_data.get("all_probabilities", {})

        mel_p = probs.get("mel", probs.get("Melanoma", 0.0))
        bcc_p = probs.get("bcc", probs.get("Basal cell carcinoma", 0.0))
        akiec_p = probs.get("akiec", probs.get("Actinic keratoses", 0.0))
        total_malignancy_risk = mel_p + bcc_p + akiec_p

        quality_status = quality_data.get("status", "ACCEPTABLE")
        triggered_defects = quality_data.get("triggered_criteria", [])

        lesion_suspicion = lesion_data.get("suspicion_score", 0.0)
        lesion_level = lesion_data.get("suspicion_level", "BENIGN_CUES")
        evidence = lesion_data.get("feature_evidence", {})
        border_status = evidence.get("border_irregularity", {}).get("status", "REGULAR")
        asym_status = evidence.get("asymmetry", {}).get("status", "SYMMETRIC")

        context_risk = context_data.get("risk_indicator", "LOW")
        parsed_fields = context_data.get("parsed_fields", {})
        age = parsed_fields.get("age")
        loc = parsed_fields.get("localization")

        # --- Rule 1: Direct Malignancy Detection ---
        if pred_code in {'mel', 'bcc', 'akiec'} or pred_class in self.malignant_classes:
            veto = True
            triggers.append("PREDICTED_MALIGNANCY")
            justifications.append(
                f"Model predicts malignant/pre-malignant pathology '{pred_class}' ({confidence*100:.1f}% confidence). "
                f"Automated triage is strictly contraindicated; immediate specialist referral mandated."
            )

        # --- Rule 2: Sub-threshold Malignancy Suspicion ---
        elif total_malignancy_risk >= 0.28:
            veto = True
            triggers.append("ELEVATED_MALIGNANCY_RESIDUAL_PROBABILITY")
            justifications.append(
                f"Although primary prediction is '{pred_class}', cumulative malignancy probability "
                f"is dangerously elevated ({total_malignancy_risk*100:.1f}%: Melanoma {mel_p*100:.1f}%, BCC {bcc_p*100:.1f}%, AKIEC {akiec_p*100:.1f}%). "
                f"Safety veto blocks automated dismissal."
            )

        # --- Rule 3: Conflict: Benign Diagnosis vs Suspicious ABCD Lesion Cues ---
        if pred_code in {'nv', 'bkl', 'df', 'vasc'} and (lesion_suspicion >= 0.65 or (border_status == "IRREGULAR" and asym_status == "MARKED_ASYMMETRY")):
            veto = True
            triggers.append("DIAGNOSIS_VS_LESION_STRUCTURE_CONFLICT")
            justifications.append(
                f"Conflict detected: Diagnosis predicts benign '{pred_class}', but visual structural analysis reveals "
                f"marked suspicion (score: {lesion_suspicion:.2f}, Border: {border_status}, Asymmetry: {asym_status}). "
                f"High risk of missed amelanotic or atypical melanoma."
            )

        # --- Rule 4: Conflict: Benign Diagnosis vs High-Risk Patient Demographics ---
        if pred_code in {'nv', 'bkl'} and context_risk == "HIGH" and confidence < 0.85:
            veto = True
            triggers.append("DIAGNOSIS_VS_DEMOGRAPHIC_CONTEXT_CONFLICT")
            justifications.append(
                f"Conflict detected: Automated benign diagnosis '{pred_class}' in high-risk demographic presentation "
                f"(Age: {age if age else 'Unk'} at site: '{loc if loc else 'Unk'}'). New/changing lesions in this demographic "
                f"carry high prior for lentigo maligna or squamous neoplasm."
            )

        # --- Rule 5: Borderline Image Quality Precaution ---
        if quality_status == 'RE_IMAGE':
            veto = True
            triggers.append("UNSAFE_ON_DEFECTIVE_IMAGE")
            justifications.append(
                f"Image quality failed clinical acceptance (Status: RE_IMAGE; defects: {', '.join(triggered_defects)}). "
                f"Automated triage completely invalid."
            )
        elif quality_status == 'ACCEPTABLE_WITH_CAUTION' and confidence < 0.70:
            veto = True
            triggers.append("QUALITY_CAUTION_DISCHARGE_BLOCK")
            justifications.append(
                f"Image has quality caution ({', '.join(triggered_defects) if triggered_defects else 'Borderline sharpness/lighting'}) "
                f"combined with non-definitive confidence ({confidence*100:.1f}%). Automated triage vetoed."
            )

        # --- Rule 6: Diagnostic Uncertainty ---
        if confidence < 0.65 and not veto:
            veto = True
            triggers.append("DIAGNOSTIC_UNCERTAINTY")
            justifications.append(
                f"Primary diagnostic confidence ({confidence*100:.1f}%) is below safe autonomous threshold (65.0%). "
                f"Requires human expert confirmation."
            )

        # Determine flag level
        if veto:
            safety_flag = "SAFETY_VETO" if any("VETO" in t or "MALIGNANCY" in t or "CONFLICT" in t for t in triggers) else "SAFETY_FLAG"
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
            "justification": " | ".join(justifications),
            "summary": (
                f"[{safety_flag}] {triggers[0] if triggers else 'All checks clear'}: "
                f"{justifications[0]}"
            )
        }

# Singleton instance
safety_agent = SafetyAgent()
