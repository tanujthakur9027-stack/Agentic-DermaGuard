import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "_version": "2.0.0",
    "_description": "Centralized Clinical & Operational Thresholds for Agentic DermaGuard",
    "quality": {
        "blur_threshold": 40.0,
        "lighting_threshold": 40.0,
        "glare_threshold": 70.0,
        "occlusion_threshold": 50.0,
        "quality_cutoff": 50.0,
        "caution_overall_threshold": 72.0,
        "caution_confidence_required": 0.70
    },
    "diagnosis": {
        "auto_triage_confidence": 0.70,
        "margin_minimum": 0.15,
        "max_entropy": 1.75,
        "temperature": 1.0,
        "tta_enabled": True,
        "tta_disagreement_threshold": 0.22,
        "class_probability_bias": {
            "mel": 0.0,
            "nv": 0.0,
            "bcc": 0.0,
            "akiec": 0.0,
            "bkl": 0.0,
            "df": 0.0,
            "vasc": 0.0
        }
    },
    "malignancy": {
        "malignant_prob_threshold": 0.28,
        "mel_threshold": 0.20,
        "bcc_threshold": 0.25,
        "akiec_threshold": 0.25
    },
    "lesion": {
        "suspicion_cutoffs": {
            "benign_upper": 0.35,
            "indeterminate_upper": 0.55
        },
        "tds_cutoffs": {
            "benign_upper": 4.75,
            "indeterminate_upper": 5.45
        },
        "asymmetry_weight": 0.35,
        "border_weight": 0.35,
        "color_weight": 0.30,
        "border_points_discordance": 6,
        "asymmetry_axes_discordance": 1,
        "suspicion_discordance": 0.65
    },
    "context": {
        "risk_cutoffs": {
            "low_upper": 0.45,
            "moderate_upper": 0.70
        },
        "age_cutoffs": {
            "pediatric": 25,
            "adult": 40,
            "senior": 60,
            "geriatric": 75
        },
        "chronic_uv_sites": ["face", "scalp", "ear", "neck"],
        "intermittent_uv_sites": ["back", "chest", "abdomen", "trunk"],
        "extremity_sites": ["upper extremity", "lower extremity", "hand", "foot", "genital", "acral"],
        "high_risk_triad_bonus": 0.15,
        "neutral_missing_prior": 0.20
    },
    "safety": {
        "s1_predicted_malignancy_enabled": True,
        "s2_residual_malignancy_enabled": True,
        "s3_abcd_discordance_enabled": True,
        "s4_demographic_discordance_enabled": True,
        "s5_quality_precaution_enabled": True,
        "s6_diagnostic_uncertainty_enabled": True,
        "min_diagnostic_confidence": 0.65,
        "high_context_risk_confidence_required": 0.85
    }
}

_CACHED_THRESHOLDS: Optional[Dict[str, Any]] = None

def get_thresholds_path() -> Path:
    candidates = [
        Path("thresholds.json"),
        Path(__file__).parent / "thresholds.json",
        Path(__file__).parent.parent / "thresholds.json"
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("thresholds.json")

def load_thresholds(reload: bool = False) -> Dict[str, Any]:
    global _CACHED_THRESHOLDS
    if _CACHED_THRESHOLDS is not None and not reload:
        return _CACHED_THRESHOLDS

    p = get_thresholds_path()
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Merge with defaults in case of missing keys
                merged = json.loads(json.dumps(DEFAULT_THRESHOLDS))
                for section, vals in data.items():
                    if isinstance(vals, dict) and section in merged and isinstance(merged[section], dict):
                        merged[section].update(vals)
                    else:
                        merged[section] = vals
                _CACHED_THRESHOLDS = merged
                return _CACHED_THRESHOLDS
        except Exception as e:
            print(f"[Thresholds] Warning: Could not read {p} ({e}), using defaults.")

    _CACHED_THRESHOLDS = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    return _CACHED_THRESHOLDS

def save_thresholds(new_thresholds: Dict[str, Any], path: Optional[str] = None) -> None:
    global _CACHED_THRESHOLDS
    target_path = Path(path) if path else get_thresholds_path()
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(new_thresholds, f, indent=2)
    _CACHED_THRESHOLDS = new_thresholds
    print(f"[Thresholds] Saved updated thresholds to {target_path}")

def get_flat_thresholds(custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns a unified flat dictionary of thresholds with runtime overrides applied.
    Enables backward compatibility with existing agent kwargs and slider inputs.
    """
    th = load_thresholds()
    flat = {
        # Quality
        "blur_threshold": th["quality"]["blur_threshold"],
        "lighting_threshold": th["quality"]["lighting_threshold"],
        "glare_threshold": th["quality"]["glare_threshold"],
        "occlusion_threshold": th["quality"]["occlusion_threshold"],
        "quality_cutoff": th["quality"]["quality_cutoff"],
        "caution_overall_threshold": th["quality"]["caution_overall_threshold"],
        "caution_confidence_required": th["quality"]["caution_confidence_required"],
        # Diagnosis
        "auto_triage_confidence": th["diagnosis"]["auto_triage_confidence"],
        "margin_minimum": th["diagnosis"]["margin_minimum"],
        "max_entropy": th["diagnosis"]["max_entropy"],
        "temperature": th["diagnosis"]["temperature"],
        "tta_enabled": th["diagnosis"]["tta_enabled"],
        "tta_disagreement_threshold": th["diagnosis"]["tta_disagreement_threshold"],
        "class_probability_bias": th["diagnosis"]["class_probability_bias"],
        # Malignancy
        "malignant_prob_threshold": th["malignancy"]["malignant_prob_threshold"],
        "mel_threshold": th["malignancy"]["mel_threshold"],
        "bcc_threshold": th["malignancy"]["bcc_threshold"],
        "akiec_threshold": th["malignancy"]["akiec_threshold"],
        # Lesion
        "suspicion_benign_upper": th["lesion"]["suspicion_cutoffs"]["benign_upper"],
        "suspicion_indeterminate_upper": th["lesion"]["suspicion_cutoffs"]["indeterminate_upper"],
        "tds_benign_upper": th["lesion"]["tds_cutoffs"]["benign_upper"],
        "tds_indeterminate_upper": th["lesion"]["tds_cutoffs"]["indeterminate_upper"],
        "asymmetry_weight": th["lesion"]["asymmetry_weight"],
        "border_weight": th["lesion"]["border_weight"],
        "color_weight": th["lesion"]["color_weight"],
        "border_points_discordance": th["lesion"]["border_points_discordance"],
        "asymmetry_axes_discordance": th["lesion"]["asymmetry_axes_discordance"],
        "suspicion_discordance": th["lesion"]["suspicion_discordance"],
        # Context
        "risk_low_upper": th["context"]["risk_cutoffs"]["low_upper"],
        "risk_moderate_upper": th["context"]["risk_cutoffs"]["moderate_upper"],
        "high_risk_triad_bonus": th["context"]["high_risk_triad_bonus"],
        "neutral_missing_prior": th["context"]["neutral_missing_prior"],
        # Safety
        "s1_enabled": th["safety"]["s1_predicted_malignancy_enabled"],
        "s2_enabled": th["safety"]["s2_residual_malignancy_enabled"],
        "s3_enabled": th["safety"]["s3_abcd_discordance_enabled"],
        "s4_enabled": th["safety"]["s4_demographic_discordance_enabled"],
        "s5_enabled": th["safety"]["s5_quality_precaution_enabled"],
        "s6_enabled": th["safety"]["s6_diagnostic_uncertainty_enabled"],
        "min_diagnostic_confidence": th["safety"]["min_diagnostic_confidence"],
        "high_context_risk_confidence_required": th["safety"]["high_context_risk_confidence_required"],
    }
    if custom_overrides:
        for k, v in custom_overrides.items():
            if v is not None and k in flat:
                flat[k] = v
            elif v is not None:
                flat[k] = v
    return flat
