from typing import Dict, Any, List, Optional
import math

class ContextAgent:
    """
    Agent 4: Reasoning / Context Agent
    Evaluates clinical and demographic patient context:
      - Age: Evaluates age-stratified malignancy priors.
      - Sex / Gender: Accounts for sex-specific anatomical risk profiles.
      - Lesion Localization: Analyzes anatomical site sun-exposure and risk.
      - Missing Fields: Identifies and flags missing, NaN, or unspecified fields.
    Provides:
      - Context evidence (structured demographic rationale)
      - Risk indicator (LOW, MODERATE, HIGH, VERY_HIGH)
      - Missing-field flags
    """

    def __init__(self):
        # Known high-risk anatomical sites for UV damage & aggressive melanoma/BCC
        self.high_risk_sites = {'face', 'scalp', 'ear', 'neck', 'back'}
        self.moderate_risk_sites = {'chest', 'abdomen', 'trunk', 'upper extremity'}
        self.lower_risk_sites = {'lower extremity', 'hand', 'foot', 'genital', 'acral'}

    def evaluate(
        self, 
        age: Optional[float] = None, 
        sex: Optional[str] = None, 
        localization: Optional[str] = None
    ) -> Dict[str, Any]:
        missing_flags = []
        evidence_points = []
        risk_score = 0.0  # Range: 0.0 to 1.0

        # --- 1. Evaluate Age ---
        parsed_age = None
        if age is None or (isinstance(age, float) and (math.isnan(age) or age <= 0)):
            missing_flags.append("MISSING_AGE: Patient age is unspecified or invalid")
            evidence_points.append("Age unavailable; assuming default adult population prior (neutral risk).")
            risk_score += 0.25
        else:
            try:
                parsed_age = float(age)
                if parsed_age >= 70:
                    risk_score += 0.45
                    evidence_points.append(f"Advanced age ({parsed_age:.0f} yrs) significantly elevates incidence of BCC, melanoma, and actinic keratoses.")
                elif parsed_age >= 50:
                    risk_score += 0.30
                    evidence_points.append(f"Mature age ({parsed_age:.0f} yrs) represents elevated cumulative UV exposure and moderate lesion risk.")
                elif parsed_age >= 30:
                    risk_score += 0.15
                    evidence_points.append(f"Adult age ({parsed_age:.0f} yrs) correlates with typical melanocytic nevus presentation.")
                else:
                    risk_score += 0.05
                    evidence_points.append(f"Young age ({parsed_age:.0f} yrs) strongly correlates with benign congenital/acquired nevi.")
            except (ValueError, TypeError):
                missing_flags.append("INVALID_AGE: Could not parse age value")
                risk_score += 0.25

        # --- 2. Evaluate Sex / Gender ---
        parsed_sex = None
        if not sex or str(sex).strip().lower() in {'unknown', 'nan', 'none', ''}:
            missing_flags.append("MISSING_SEX: Patient gender not documented")
            evidence_points.append("Patient sex not provided; using gender-neutral baseline.")
            risk_score += 0.15
        else:
            parsed_sex = str(sex).strip().lower()
            if parsed_sex in {'male', 'm'}:
                risk_score += 0.20
                evidence_points.append("Male sex statistically exhibits higher rates of invasive melanoma and head/neck/trunk lesions.")
            elif parsed_sex in {'female', 'f'}:
                risk_score += 0.10
                evidence_points.append("Female sex noted; lower statistical mortality across population-wide dermoscopy cohorts.")
            else:
                missing_flags.append(f"UNRECOGNIZED_SEX: '{sex}'")
                risk_score += 0.15

        # --- 3. Evaluate Anatomical Localization ---
        parsed_loc = None
        if not localization or str(localization).strip().lower() in {'unknown', 'nan', 'none', ''}:
            missing_flags.append("MISSING_LOCALIZATION: Lesion anatomical site not recorded")
            evidence_points.append("Lesion anatomical site unknown; missing critical UV exposure contextual factor.")
            risk_score += 0.25
        else:
            parsed_loc = str(localization).strip().lower()
            if parsed_loc in self.high_risk_sites:
                risk_score += 0.35
                evidence_points.append(f"High-risk anatomical site '{parsed_loc}': High chronic/intermittent solar UV exposure site.")
            elif parsed_loc in self.moderate_risk_sites:
                risk_score += 0.20
                evidence_points.append(f"Moderate-risk anatomical site '{parsed_loc}': Covered/torso region with intermittent exposure.")
            else:
                risk_score += 0.10
                evidence_points.append(f"Standard anatomical site '{parsed_loc}': Typically lower relative incidence of sun-induced malignancies.")

        # Combined interaction: Elderly male with head/neck/back lesion is high-priority red flag
        if parsed_age and parsed_age >= 60 and parsed_sex in {'male', 'm'} and parsed_loc in self.high_risk_sites:
            risk_score += 0.20
            evidence_points.append("HIGH-RISK CONTEXT TRIAD: Older male with head/neck/back lesion mandates strict clinical caution.")

        # Normalize risk score to [0.0, 1.0]
        risk_score = round(min(1.0, risk_score), 3)

        if risk_score >= 0.70:
            risk_indicator = "HIGH"
        elif risk_score >= 0.45:
            risk_indicator = "MODERATE"
        else:
            risk_indicator = "LOW"

        return {
            "agent_id": "A4",
            "agent_name": "Reasoning / Context Agent",
            "risk_score": risk_score,
            "risk_indicator": risk_indicator,
            "missing_field_flags": missing_flags,
            "has_missing_data": len(missing_flags) > 0,
            "context_evidence": evidence_points,
            "parsed_fields": {
                "age": parsed_age,
                "sex": parsed_sex,
                "localization": parsed_loc
            },
            "summary": (
                f"Demographic & anatomical context suggests {risk_indicator} risk (score: {risk_score:.2f}). "
                f"Missing fields: {len(missing_flags)} flagged."
            )
        }

# Singleton instance
context_agent = ContextAgent()
