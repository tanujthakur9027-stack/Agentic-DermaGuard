from typing import Dict, Any, List, Optional
import math
from thresholds import load_thresholds, get_flat_thresholds

class ContextAgent:
    """
    Agent 4: Reasoning / Context Agent (Epidemiological & Demographic Priors)
    Jaypee Agentic AI Summit 2026 — Track 2 | Healthcare

    Analyzes clinical context based on epidemiological dermatology guidelines:
      - Age: Cumulative solar damage & malignancy incidence curves.
      - Sex: Anatomical site predisposition and mortality differentials.
      - Localization: Chronic vs. intermittent solar UV radiation patterns.
      - Missing Field Audit: Explicitly detects and flags incomplete clinical charts.
    Provides:
      - Context Evidence: Medical narrative linking demographics to pathology
      - Risk Indicator: LOW, MODERATE, HIGH, VERY_HIGH
      - Missing-Field Flags: Traceable missing data flags
      - Epidemiological Prior Bias for Safety Sentinel
    """

    def __init__(self):
        th = load_thresholds().get("context", {})
        self.chronic_uv_sites = set(th.get("chronic_uv_sites", ['face', 'scalp', 'ear', 'neck']))
        self.intermittent_uv_sites = set(th.get("intermittent_uv_sites", ['back', 'chest', 'abdomen', 'trunk']))
        self.extremity_sites = set(th.get("extremity_sites", ['upper extremity', 'lower extremity', 'hand', 'foot', 'genital', 'acral']))
        self.age_cutoffs = th.get("age_cutoffs", {"pediatric": 25, "adult": 40, "senior": 60, "geriatric": 75})
        self.high_risk_triad_bonus = th.get("high_risk_triad_bonus", 0.15)
        self.neutral_missing_prior = th.get("neutral_missing_prior", 0.20)

    def evaluate(
        self,
        age: Optional[float] = None,
        sex: Optional[str] = None,
        localization: Optional[str] = None,
        custom_thresholds: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        th = get_flat_thresholds(custom_thresholds)
        low_upper = th.get("risk_low_upper", 0.45)
        mod_upper = th.get("risk_moderate_upper", 0.70)
        triad_bonus = th.get("high_risk_triad_bonus", self.high_risk_triad_bonus)
        neutral_prior = th.get("neutral_missing_prior", self.neutral_missing_prior)

        missing_flags = []
        evidence_points = []
        risk_score = 0.0

        # --- 1. Evaluate Age ---
        parsed_age = None
        if age is None or (isinstance(age, float) and (math.isnan(age) or age <= 0)) or str(age).strip().lower() in {'unknown', 'nan', 'none', ''}:
            missing_flags.append("MISSING_AGE: Patient age is unspecified")
            evidence_points.append("Age unspecified; using population median baseline (neutral prior).")
            risk_score += 0.20
        else:
            try:
                parsed_age = float(age)
                if parsed_age >= self.age_cutoffs.get("geriatric", 75):
                    risk_score += 0.45
                    evidence_points.append(f"Geriatric cohort ({parsed_age:.0f} yrs): Marked cumulative solar elastosis; high incidence of lentigo maligna, BCC, and SCC.")
                elif parsed_age >= self.age_cutoffs.get("senior", 60):
                    risk_score += 0.35
                    evidence_points.append(f"Senior cohort ({parsed_age:.0f} yrs): Elevated baseline prior for non-melanocytic and melanocytic skin neoplasms.")
                elif parsed_age >= self.age_cutoffs.get("adult", 40):
                    risk_score += 0.20
                    evidence_points.append(f"Mature adult ({parsed_age:.0f} yrs): Moderate cumulative UV history; active surveillance recommended for atypical lesions.")
                elif parsed_age >= self.age_cutoffs.get("pediatric", 25):
                    risk_score += 0.10
                    evidence_points.append(f"Young adult ({parsed_age:.0f} yrs): Benign melanocytic nevi are prevalent; new or evolving lesions warrant monitoring.")
                else:
                    risk_score += 0.05
                    evidence_points.append(f"Adolescent / pediatric cohort ({parsed_age:.0f} yrs): Very low spontaneous malignancy prior; congenital or acquired nevi predominate.")
            except (ValueError, TypeError):
                missing_flags.append("INVALID_AGE: Could not parse age numeric")
                risk_score += 0.20

        # --- 2. Evaluate Sex / Gender ---
        parsed_sex = None
        if not sex or str(sex).strip().lower() in {'unknown', 'nan', 'none', ''}:
            missing_flags.append("MISSING_SEX: Patient gender not recorded")
            evidence_points.append("Patient gender unrecorded; using gender-neutral clinical baseline.")
            risk_score += 0.10
        else:
            parsed_sex = str(sex).strip().lower()
            if parsed_sex in {'male', 'm'}:
                risk_score += 0.20
                evidence_points.append("Male sex: Statistically higher melanoma mortality and propensity for aggressive trunk/scalp presentations.")
            elif parsed_sex in {'female', 'f'}:
                risk_score += 0.10
                evidence_points.append("Female sex: Lower population-adjusted melanoma mortality; higher prevalence of extremity lesions.")
            else:
                missing_flags.append(f"UNRECOGNIZED_SEX: '{sex}'")
                risk_score += 0.10

        # --- 3. Evaluate Anatomical Localization ---
        parsed_loc = None
        if not localization or str(localization).strip().lower() in {'unknown', 'nan', 'none', ''}:
            missing_flags.append("MISSING_LOCALIZATION: Lesion anatomical site not recorded")
            evidence_points.append("Anatomical site unknown; solar exposure pattern cannot be determined.")
            risk_score += 0.20
        else:
            parsed_loc = str(localization).strip().lower()
            if parsed_loc in self.chronic_uv_sites:
                risk_score += 0.35
                evidence_points.append(f"High-risk anatomical site '{parsed_loc}': Chronic solar ultraviolet exposure zone prone to actinic keratoses and lentigo maligna.")
            elif parsed_loc in self.intermittent_uv_sites:
                risk_score += 0.25
                evidence_points.append(f"Intermittent UV exposure site '{parsed_loc}': Classic anatomical distribution for superficial spreading melanoma in adults.")
            else:
                risk_score += 0.10
                evidence_points.append(f"Standard anatomical site '{parsed_loc}': Lower comparative solar UV density relative to head and trunk.")

        # High-Risk Triad Interaction: Elderly male with head/neck/back lesion
        if parsed_age and parsed_age >= self.age_cutoffs.get("senior", 60) and parsed_sex in {'male', 'm'} and (parsed_loc in self.chronic_uv_sites or parsed_loc == 'back'):
            risk_score += triad_bonus
            evidence_points.append("CRITICAL RISK TRIAD: Older male presenting with head/neck/back lesion triggers heightened diagnostic vigilance.")

        # Complete missing context scenario (all 3 demographic fields unprovided)
        if len(missing_flags) >= 3:
            risk_score = neutral_prior
            risk_indicator = "LOW"
            evidence_points = ["Demographic metadata absent; evaluated with neutral prior baseline (0.20)."]
        else:
            # Normalize risk score to [0.0, 1.0]
            risk_score = round(min(1.0, max(0.05, risk_score)), 3)
            if risk_score >= mod_upper:
                risk_indicator = "HIGH"
            elif risk_score >= low_upper:
                risk_indicator = "MODERATE"
            else:
                risk_indicator = "LOW"

        # Context completeness score
        completeness_pct = round((3 - len(missing_flags)) / 3.0 * 100.0, 1)

        summary_narrative = (
            f"Context profile indicates {risk_indicator} risk (score: {risk_score:.2f}). "
            f"Demographic completeness: {completeness_pct}% ({len(missing_flags)} missing fields flagged). "
            f"{evidence_points[0]}"
        )

        return {
            "agent_id": "A4",
            "agent_name": "Reasoning / Context Agent",
            "risk_score": risk_score,
            "risk_indicator": risk_indicator,
            "missing_field_flags": missing_flags,
            "has_missing_data": len(missing_flags) > 0,
            "completeness_percentage": completeness_pct,
            "context_evidence": evidence_points,
            "parsed_fields": {
                "age": parsed_age,
                "sex": parsed_sex,
                "localization": parsed_loc
            },
            "summary": summary_narrative
        }

# Singleton instance
context_agent = ContextAgent()
