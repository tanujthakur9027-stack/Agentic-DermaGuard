from typing import Dict, Any, List, Optional
import math
import time
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
      - Explaining Agent contract with full evidence, margins, and explanation
      - Context Evidence: Medical narrative linking demographics to pathology
      - Risk Indicator: LOW, MODERATE, HIGH, VERY_HIGH
      - Missing-Field Flags: Traceable missing data flags
      - Epidemiological Prior Bias for Safety Sentinel
    """

    def __init__(self):
        defaults = get_flat_thresholds()
        self.chronic_uv_sites = set(defaults.get("high_risk_sites", ['face', 'scalp', 'ear', 'neck', 'acral']))
        self.intermittent_uv_sites = {'back', 'chest', 'abdomen', 'trunk'}
        self.extremity_sites = {'upper extremity', 'lower extremity', 'hand', 'foot', 'genital'}
        self.age_cutoffs = {
            "pediatric": 25,
            "adult": 40,
            "senior": int(defaults.get("age_high_risk", 60)),
            "geriatric": int(defaults.get("age_very_high_risk", 75))
        }
        self.high_risk_triad_bonus = 0.15
        self.neutral_missing_prior = float(defaults.get("missing_field_penalty", 0.15))

    def evaluate(
        self,
        age: Optional[float] = None,
        sex: Optional[str] = None,
        localization: Optional[str] = None,
        custom_thresholds: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        th = get_flat_thresholds(custom_thresholds)
        low_upper = float(th.get("risk_moderate_from", th.get("risk_low_upper", 0.45)))
        mod_upper = float(th.get("risk_high_from", th.get("risk_moderate_upper", 0.70)))
        very_high_thresh = float(th.get("risk_very_high_from", 0.85))
        triad_bonus = float(th.get("high_risk_triad_bonus", self.high_risk_triad_bonus))
        neutral_prior = float(th.get("neutral_missing_prior", th.get("missing_field_penalty", self.neutral_missing_prior)))
        age_high = int(th.get("age_high_risk", 60))
        age_very_high = int(th.get("age_very_high_risk", 75))
        high_risk_sites = set(th.get("high_risk_sites", ['face', 'scalp', 'ear', 'neck', 'acral']))
        sex_w = float(th.get("sex_weight", 0.10))

        missing_flags = []
        evidence_points = []
        flags = []
        risk_score = 0.0

        # --- 1. Evaluate Age ---
        parsed_age = None
        if age is None or (isinstance(age, float) and (math.isnan(age) or age <= 0)) or str(age).strip().lower() in {'unknown', 'nan', 'none', ''}:
            missing_flags.append("MISSING_AGE: Patient age is unspecified")
            flags.append("MISSING_AGE")
            evidence_points.append("Age unspecified; using population median baseline (neutral prior).")
            risk_score += neutral_prior
        else:
            try:
                parsed_age = float(age)
                if parsed_age >= age_very_high:
                    risk_score += 0.45
                    flags.append("GERIATRIC_COHORT")
                    evidence_points.append(f"Geriatric cohort ({parsed_age:.0f} yrs): Marked cumulative solar elastosis; high incidence of lentigo maligna, BCC, and SCC.")
                elif parsed_age >= age_high:
                    risk_score += 0.35
                    flags.append("SENIOR_COHORT")
                    evidence_points.append(f"Senior cohort ({parsed_age:.0f} yrs): Elevated baseline prior for non-melanocytic and melanocytic skin neoplasms.")
                elif parsed_age >= 40:
                    risk_score += 0.20
                    evidence_points.append(f"Mature adult ({parsed_age:.0f} yrs): Moderate cumulative UV history; active surveillance recommended for atypical lesions.")
                elif parsed_age >= 25:
                    risk_score += 0.10
                    evidence_points.append(f"Young adult ({parsed_age:.0f} yrs): Benign melanocytic nevi are prevalent; new or evolving lesions warrant monitoring.")
                else:
                    risk_score += 0.05
                    evidence_points.append(f"Adolescent / pediatric cohort ({parsed_age:.0f} yrs): Very low spontaneous malignancy prior; congenital or acquired nevi predominate.")
            except (ValueError, TypeError):
                missing_flags.append("INVALID_AGE: Could not parse age numeric")
                flags.append("MISSING_AGE")
                risk_score += neutral_prior

        # --- 2. Evaluate Sex / Gender ---
        parsed_sex = None
        if not sex or str(sex).strip().lower() in {'unknown', 'nan', 'none', ''}:
            missing_flags.append("MISSING_SEX: Patient gender not recorded")
            flags.append("MISSING_SEX")
            evidence_points.append("Patient gender unrecorded; using gender-neutral clinical baseline.")
            risk_score += (neutral_prior * 0.7)
        else:
            parsed_sex = str(sex).strip().lower()
            if parsed_sex in {'male', 'm'}:
                risk_score += (sex_w * 2.0)
                evidence_points.append("Male sex: Statistically higher melanoma mortality and propensity for aggressive trunk/scalp presentations.")
            elif parsed_sex in {'female', 'f'}:
                risk_score += sex_w
                evidence_points.append("Female sex: Lower population-adjusted melanoma mortality; higher prevalence of extremity lesions.")
            else:
                missing_flags.append(f"UNRECOGNIZED_SEX: '{sex}'")
                flags.append("MISSING_SEX")
                risk_score += (neutral_prior * 0.7)

        # --- 3. Evaluate Anatomical Localization ---
        parsed_loc = None
        if not localization or str(localization).strip().lower() in {'unknown', 'nan', 'none', ''}:
            missing_flags.append("MISSING_LOCALIZATION: Lesion anatomical site not recorded")
            flags.append("MISSING_LOCALIZATION")
            evidence_points.append("Anatomical site unknown; solar exposure pattern cannot be determined.")
            risk_score += neutral_prior
        else:
            parsed_loc = str(localization).strip().lower()
            if parsed_loc in high_risk_sites or parsed_loc in self.chronic_uv_sites:
                risk_score += 0.35
                flags.append("CHRONIC_UV_SITE")
                evidence_points.append(f"High-risk anatomical site '{parsed_loc}': Chronic solar ultraviolet exposure zone prone to actinic keratoses and lentigo maligna.")
            elif parsed_loc in self.intermittent_uv_sites:
                risk_score += 0.25
                evidence_points.append(f"Intermittent UV exposure site '{parsed_loc}': Classic anatomical distribution for superficial spreading melanoma in adults.")
            else:
                risk_score += 0.10
                evidence_points.append(f"Standard anatomical site '{parsed_loc}': Lower comparative solar UV density relative to head and trunk.")

        # High-Risk Triad Interaction: Elderly male with head/neck/back lesion
        if parsed_age and parsed_age >= age_high and parsed_sex in {'male', 'm'} and (parsed_loc in high_risk_sites or parsed_loc == 'back'):
            risk_score += triad_bonus
            flags.append("HIGH_RISK_TRIAD")
            evidence_points.append("CRITICAL RISK TRIAD: Older male presenting with head/neck/back lesion triggers heightened diagnostic vigilance.")

        # Complete missing context scenario (all 3 demographic fields unprovided)
        if len(missing_flags) >= 3:
            risk_score = neutral_prior
            risk_indicator = "LOW"
            evidence_points = ["Demographic metadata absent; evaluated with neutral prior baseline."]
        else:
            risk_score = round(min(1.0, max(0.05, risk_score)), 3)
            if risk_score >= very_high_thresh:
                risk_indicator = "VERY_HIGH"
            elif risk_score >= mod_upper:
                risk_indicator = "HIGH"
            elif risk_score >= low_upper:
                risk_indicator = "MODERATE"
            else:
                risk_indicator = "LOW"

        completeness_pct = round((3 - len(missing_flags)) / 3.0 * 100.0, 1)

        evidence = [
            {
                "feature": "patient_age_risk",
                "value": parsed_age if parsed_age is not None else "Unspecified",
                "threshold": age_high,
                "direction": "<",
                "passed": parsed_age is not None and parsed_age < age_high,
                "note": f"Age is {f'{parsed_age:.0f} yrs' if parsed_age else 'Unspecified'} (senior cutoff: {age_high} yrs, geriatric cutoff: {age_very_high} yrs)"
            },
            {
                "feature": "anatomical_site_risk",
                "value": parsed_loc if parsed_loc is not None else "Unrecorded",
                "threshold": list(high_risk_sites),
                "direction": "not in",
                "passed": parsed_loc is not None and parsed_loc not in high_risk_sites,
                "note": f"Location is '{parsed_loc or 'Unknown'}' (high risk sites: {', '.join(sorted(list(high_risk_sites)))})"
            },
            {
                "feature": "demographic_risk_score",
                "value": risk_score,
                "threshold": mod_upper,
                "direction": "<",
                "passed": risk_score < mod_upper,
                "note": f"Context cumulative risk score is {risk_score:.2f} (MODERATE >= {low_upper:.2f}, HIGH >= {mod_upper:.2f})"
            },
            {
                "feature": "demographic_completeness",
                "value": completeness_pct,
                "threshold": 100.0,
                "direction": "==",
                "passed": len(missing_flags) == 0,
                "note": f"Demographic chart completeness is {completeness_pct:.0f}% ({len(missing_flags)} missing fields flagged)"
            }
        ]

        thresholds_used = {
            "age_high_risk": age_high,
            "age_very_high_risk": age_very_high,
            "risk_moderate_from": low_upper,
            "risk_high_from": mod_upper,
            "missing_field_penalty": neutral_prior
        }

        margin_to_threshold = {
            "risk_high_from": round(mod_upper - risk_score, 3)
        }
        if parsed_age is not None:
            margin_to_threshold["age_high_risk"] = round(age_high - parsed_age, 1)

        summary = f"Patient presentation exhibits {risk_indicator} demographic risk based on age, anatomical site, and solar UV history."

        explanation = (
            f"Context & Demographics Agent evaluated clinical risk at {risk_score:.2f}/1.0 ({risk_indicator}). "
            f"Patient is {f'{parsed_age:.0f} years old' if parsed_age else 'unspecified age'} presenting with lesion at '{parsed_loc or 'unrecorded site'}'. "
            f"{evidence_points[0]}"
        )

        conf_in_agent = 1.0 if completeness_pct == 100.0 else (0.75 if completeness_pct >= 66.0 else 0.50)
        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return {
            # Explaining Agent Contract
            "agent": "A4",
            "name": "Context & Demographics",
            "status": risk_indicator,
            "score": risk_score,
            "summary": summary,
            "evidence": evidence,
            "thresholds_used": thresholds_used,
            "margin_to_threshold": margin_to_threshold,
            "flags": flags,
            "explanation": explanation,
            "confidence_in_this_agent": conf_in_agent,
            "runtime_ms": runtime_ms,
            # Preserved legacy fields
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
            }
        }

# Singleton instance
context_agent = ContextAgent()
