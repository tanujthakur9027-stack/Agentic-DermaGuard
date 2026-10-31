import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# --- Complete Master Threshold Registry ---
# Each entry defines: key, agent, label, default, min, max, step, unit,
# description, effect_up, effect_down.
REGISTRY: List[Dict[str, Any]] = [
    # --- Agent 1: Image Quality ---
    {
        "key": "blur_min",
        "agent": "A1",
        "label": "Min Sharpness (Laplacian / Tenengrad)",
        "default": 40.0,
        "min": 10.0,
        "max": 90.0,
        "step": 1.0,
        "unit": "score",
        "description": "Minimum composite sharpness score (Laplacian variance & Tenengrad gradient) required for clinical capture usability.",
        "effect_up": "Stricter sharpness criteria; flags more images as blurry and increases RE_IMAGE triggers.",
        "effect_down": "Permits softer or slightly out-of-focus images to proceed to diagnostic classification."
    },
    {
        "key": "tenengrad_min",
        "agent": "A1",
        "label": "Tenengrad Gradient Cutoff",
        "default": 220.0,
        "min": 50.0,
        "max": 800.0,
        "step": 10.0,
        "unit": "gradient",
        "description": "Threshold for Sobel squared gradient mean, measuring fine micro-structural edge sharpness.",
        "effect_up": "Requires higher edge contrast; triggers quality caution on subtle pigment reticular networks.",
        "effect_down": "Tolerates lower edge contrast and diffuse lesion borders."
    },
    {
        "key": "brightness_min",
        "agent": "A1",
        "label": "Minimum Illumination Intensity",
        "default": 35.0,
        "min": 10.0,
        "max": 80.0,
        "step": 1.0,
        "unit": "intensity",
        "description": "Lower bound of mean grayscale intensity to detect underexposed captures.",
        "effect_up": "Demands brighter dermoscopic illumination; rejects dark or shadowed captures.",
        "effect_down": "Allows darker exposures to be processed, risking masked melanin pigmentation."
    },
    {
        "key": "brightness_max",
        "agent": "A1",
        "label": "Maximum Illumination Intensity",
        "default": 225.0,
        "min": 180.0,
        "max": 250.0,
        "step": 1.0,
        "unit": "intensity",
        "description": "Upper bound of mean grayscale intensity to detect severe overexposure washouts.",
        "effect_up": "Allows brighter captures before triggering overexposure defect warnings.",
        "effect_down": "Flags moderately bright images earlier as overexposed."
    },
    {
        "key": "entropy_min",
        "agent": "A1",
        "label": "Histogram Entropy Lower Bound",
        "default": 4.5,
        "min": 2.0,
        "max": 7.5,
        "step": 0.1,
        "unit": "bits",
        "description": "Minimum Shannon information entropy of the image pixel histogram, ensuring sufficient dynamic range.",
        "effect_up": "Demands wider tonal dynamic range; rejects low-contrast or monochromatic lighting.",
        "effect_down": "Accepts narrower dynamic range and flatter lighting."
    },
    {
        "key": "glare_max_pct",
        "agent": "A1",
        "label": "Maximum Specular Glare Coverage",
        "default": 2.0,
        "min": 0.2,
        "max": 10.0,
        "step": 0.1,
        "unit": "%",
        "description": "Maximum allowable percentage of specular saturated pixels (RGB >= 246) before triggering a glare defect.",
        "effect_up": "Tolerates larger flash reflection white spots before mandating RE_IMAGE.",
        "effect_down": "Stricter flash control; rejects images with small reflection artifacts."
    },
    {
        "key": "obstruction_max_pct",
        "agent": "A1",
        "label": "Maximum Peripheral Obstruction",
        "default": 18.0,
        "min": 5.0,
        "max": 40.0,
        "step": 1.0,
        "unit": "%",
        "description": "Maximum allowed edge concentration along the outer perimeter (detecting dark lens rings and hair occlusion).",
        "effect_up": "Tolerates stronger lens vignetting or corner dark rings.",
        "effect_down": "Demands clean borders free from hair, adhesive tape, or dermatoscope rim shadow."
    },
    {
        "key": "quality_score_reimage_below",
        "agent": "A1",
        "label": "Mandatory RE_IMAGE Cutoff",
        "default": 50.0,
        "min": 20.0,
        "max": 75.0,
        "step": 1.0,
        "unit": "score",
        "description": "Overall composite image quality score below which the image is clinically unreadable and mandates RE_IMAGE.",
        "effect_up": "Forces more borderline captures into RE_IMAGE, raising referral and re-capture rates.",
        "effect_down": "Decreases RE_IMAGE requests, forcing low-quality captures into downstream analysis."
    },
    {
        "key": "quality_score_caution_below",
        "agent": "A1",
        "label": "Quality Caution Threshold",
        "default": 72.0,
        "min": 45.0,
        "max": 85.0,
        "step": 1.0,
        "unit": "score",
        "description": "Overall quality score below which status is ACCEPTABLE_WITH_CAUTION, triggering heightened safety checks.",
        "effect_up": "Expands cautionary band; subjects more captures to safety scrutiny and higher confidence hurdles.",
        "effect_down": "Classifies more images as fully clear PASS, lowering safety scrutiny."
    },

    # --- Agent 2: Diagnosis ---
    {
        "key": "auto_triage_confidence",
        "agent": "A2",
        "label": "Auto-Triage Confidence Hurdle",
        "default": 0.80,
        "min": 0.50,
        "max": 0.98,
        "step": 0.01,
        "unit": "prob",
        "description": "Minimum classifier probability required to approve autonomous discharge (AUTO_TRIAGE) for benign lesions.",
        "effect_up": "Fewer autonomous triage discharges; safer, more conservative referrals for dermatologist evaluation.",
        "effect_down": "Increases autonomous throughput; higher risk of discharging an atypical or borderline lesion."
    },
    {
        "key": "top2_margin_min",
        "agent": "A2",
        "label": "Minimum Differential Margin",
        "default": 0.10,
        "min": 0.01,
        "max": 0.40,
        "step": 0.01,
        "unit": "prob",
        "description": "Minimum separation between candidate top-1 and top-2 class probabilities to prevent ambiguous decisions.",
        "effect_up": "Requires clear separation between diagnoses; refers close differential calls to clinical specialists.",
        "effect_down": "Permits auto-triage even when a close competitor diagnosis exists."
    },
    {
        "key": "max_entropy",
        "agent": "A2",
        "label": "Maximum Shannon Entropy",
        "default": 1.65,
        "min": 0.50,
        "max": 2.50,
        "step": 0.05,
        "unit": "bits",
        "description": "Maximum Shannon entropy across the 7 class probabilities. High entropy signals diagnostic confusion.",
        "effect_up": "Tolerates more diffused differential distributions before vetoing auto-triage.",
        "effect_down": "Demands sharp peak probabilities for auto-triage; refers diffuse probability spreads."
    },
    {
        "key": "malignant_prob_threshold",
        "agent": "A2",
        "label": "Cumulative Malignancy Risk Cutoff",
        "default": 0.22,
        "min": 0.05,
        "max": 0.50,
        "step": 0.01,
        "unit": "prob",
        "description": "Maximum tolerated cumulative probability mass of malignant classes (p_mel + p_bcc + p_akiec).",
        "effect_up": "Permits higher residual malignant probability before vetoing; raises AUTO_TRIAGE count.",
        "effect_down": "Ultra-sensitive malignancy sentinel; vetoes auto-triage at slightest malignant probability signal."
    },
    {
        "key": "mel_prob_threshold",
        "agent": "A2",
        "label": "Melanoma Probability Cutoff",
        "default": 0.18,
        "min": 0.05,
        "max": 0.40,
        "step": 0.01,
        "unit": "prob",
        "description": "Dedicated strict cutoff for invasive melanoma probability (p_mel), regardless of the leading predicted class.",
        "effect_up": "Tolerates higher melanoma residual probability before forcing referral.",
        "effect_down": "Near-zero tolerance for missed melanoma; escalates to REFER on faint melanoma suspicion."
    },
    {
        "key": "bias_mel",
        "agent": "A2",
        "label": "Logit Bias: Melanoma (mel)",
        "default": 0.30,
        "min": -2.0,
        "max": 2.0,
        "step": 0.05,
        "unit": "logit",
        "description": "Probability logit bias for Melanoma class to optimize multi-class Macro-F1.",
        "effect_up": "Increases sensitivity to Melanoma; lowers specificity.",
        "effect_down": "Decreases melanoma predictions; increases specificity."
    },
    {
        "key": "bias_nv",
        "agent": "A2",
        "label": "Logit Bias: Nevus (nv)",
        "default": 0.75,
        "min": -2.0,
        "max": 2.0,
        "step": 0.05,
        "unit": "logit",
        "description": "Probability logit bias for Melanocytic Nevus class.",
        "effect_up": "Favors nevus predictions across borderline lesions.",
        "effect_down": "Discourages overpredicting nevus on rare classes."
    },
    {
        "key": "bias_bcc",
        "agent": "A2",
        "label": "Logit Bias: Basal Cell Carcinoma (bcc)",
        "default": -0.40,
        "min": -2.0,
        "max": 2.0,
        "step": 0.05,
        "unit": "logit",
        "description": "Probability logit bias for Basal Cell Carcinoma.",
        "effect_up": "Promotes BCC recall across translucent nodular lesions.",
        "effect_down": "Reduces false positive BCC diagnoses."
    },
    {
        "key": "bias_akiec",
        "agent": "A2",
        "label": "Logit Bias: Actinic Keratoses (akiec)",
        "default": -0.85,
        "min": -2.0,
        "max": 2.0,
        "step": 0.05,
        "unit": "logit",
        "description": "Probability logit bias for Actinic Keratoses / Bowen's Disease.",
        "effect_up": "Increases detection of scaly keratotic pre-malignancies.",
        "effect_down": "Avoids confusing benign keratoses with AKIEC."
    },
    {
        "key": "bias_bkl",
        "agent": "A2",
        "label": "Logit Bias: Benign Keratosis (bkl)",
        "default": 0.00,
        "min": -2.0,
        "max": 2.0,
        "step": 0.05,
        "unit": "logit",
        "description": "Probability logit bias for Benign Keratosis-like lesions.",
        "effect_up": "Favors seborrheic keratosis classification.",
        "effect_down": "Suppresses BKL predictions in favor of competing classes."
    },
    {
        "key": "bias_df",
        "agent": "A2",
        "label": "Logit Bias: Dermatofibroma (df)",
        "default": 0.80,
        "min": -2.0,
        "max": 2.0,
        "step": 0.05,
        "unit": "logit",
        "description": "Probability logit bias for Dermatofibroma to overcome minority class imbalance.",
        "effect_up": "Significantly lifts Dermatofibroma sensitivity and Macro-F1.",
        "effect_down": "Reduces DF predictions; minority class under-represented."
    },
    {
        "key": "bias_vasc",
        "agent": "A2",
        "label": "Logit Bias: Vascular Lesions (vasc)",
        "default": 0.00,
        "min": -2.0,
        "max": 2.0,
        "step": 0.05,
        "unit": "logit",
        "description": "Probability logit bias for Vascular lesions (hemangiomas).",
        "effect_up": "Increases detection of red/purple vascular anomalies.",
        "effect_down": "Reduces vascular false positives."
    },

    # --- Agent 3: Lesion Analysis (ABCD) ---
    {
        "key": "asymmetry_high",
        "agent": "A3",
        "label": "Asymmetry Ratio Cutoff",
        "default": 0.35,
        "min": 0.10,
        "max": 0.70,
        "step": 0.01,
        "unit": "ratio",
        "description": "Non-overlapping area fraction across horizontal/vertical centroid folds defining geometric asymmetry.",
        "effect_up": "Requires higher geometric deformation before flagging asymmetry.",
        "effect_down": "Stricter symmetry criteria; flags mildly elongated lesions as asymmetrical."
    },
    {
        "key": "border_compactness_high",
        "agent": "A3",
        "label": "Border Compactness (P²/4πA)",
        "default": 1.60,
        "min": 1.10,
        "max": 3.00,
        "step": 0.05,
        "unit": "ratio",
        "description": "Isoperimetric compactness quotient. Circle = 1.0; jagged/notched borders exceed 1.60.",
        "effect_up": "Tolerates more notched or irregular borders before flagging irregularity.",
        "effect_down": "Flags slightly scalloped or undulating borders as jagged."
    },
    {
        "key": "color_variation_high",
        "agent": "A3",
        "label": "Color Variation Standard Deviation",
        "default": 32.0,
        "min": 15.0,
        "max": 60.0,
        "step": 1.0,
        "unit": "std_dev",
        "description": "Standard deviation of pixel color channels across the lesion, indicating variegation.",
        "effect_up": "Allows richer color variegation before flagging suspicion.",
        "effect_down": "Flags lesions with mild color gradients as suspicious."
    },
    {
        "key": "suspicion_indeterminate_from",
        "agent": "A3",
        "label": "ABCD Indeterminate Lower Bound",
        "default": 0.35,
        "min": 0.15,
        "max": 0.55,
        "step": 0.01,
        "unit": "score",
        "description": "ABCD composite suspicion score transition from BENIGN_CUES to INDETERMINATE.",
        "effect_up": "Broadens benign category; fewer lesions labeled indeterminate.",
        "effect_down": "Shifts mildly irregular lesions into indeterminate band."
    },
    {
        "key": "suspicion_suspicious_from",
        "agent": "A3",
        "label": "ABCD Suspicious Lower Bound",
        "default": 0.55,
        "min": 0.40,
        "max": 0.80,
        "step": 0.01,
        "unit": "score",
        "description": "ABCD composite suspicion score transition from INDETERMINATE to SUSPICIOUS.",
        "effect_up": "Requires more extreme ABCD structural cues to trigger SUSPICIOUS status.",
        "effect_down": "Easily triggers SUSPICIOUS status on moderate ABCD features, prompting safety vetoes."
    },
    {
        "key": "min_lesion_area_pct",
        "agent": "A3",
        "label": "Minimum Lesion Area (% of field)",
        "default": 2.0,
        "min": 0.2,
        "max": 10.0,
        "step": 0.1,
        "unit": "%",
        "description": "Minimum segmented lesion area percentage required for reliable ABCD calculation.",
        "effect_up": "Flags tiny lesions as LOW_RELIABILITY; protects against noise artifacts.",
        "effect_down": "Attempts ABCD calculations on extremely small dots, risking unstable ratios."
    },
    {
        "key": "segmentation_fail",
        "agent": "A3",
        "label": "Segmentation Failure Safeguard Rule",
        "default": True,
        "unit": "boolean",
        "description": "Enables fallback ellipse and attaches LOW_RELIABILITY flag when border-touch or low-contrast occurs.",
        "effect_up": "Always warns when segmentation is imperfect.",
        "effect_down": "Suppresses segmentation failure warnings."
    },

    # --- Agent 4: Context & Reasoning ---
    {
        "key": "age_high_risk",
        "agent": "A4",
        "label": "High-Risk Senior Age Cutoff",
        "default": 60,
        "min": 40,
        "max": 75,
        "step": 1,
        "unit": "years",
        "description": "Age threshold at which cumulative lifetime UV exposure raises baseline skin cancer incidence.",
        "effect_up": "Classifies fewer patients into senior high-risk cohort.",
        "effect_down": "Broader group marked high risk; enforces higher confidence hurdle for auto-triage."
    },
    {
        "key": "age_very_high_risk",
        "agent": "A4",
        "label": "Very High-Risk Geriatric Age Cutoff",
        "default": 75,
        "min": 65,
        "max": 90,
        "step": 1,
        "unit": "years",
        "description": "Age threshold at which marked solar elastosis and high lentigo maligna / NMSC incidence occurs.",
        "effect_up": "Restricts geriatric risk tier to older patients.",
        "effect_down": "Expands geriatric tier; heavily restricts autonomous discharge."
    },
    {
        "key": "high_risk_sites",
        "agent": "A4",
        "label": "High-Risk Anatomical Sites",
        "default": ["face", "scalp", "ear", "neck", "acral"],
        "unit": "list",
        "description": "Anatomical locations with chronic solar exposure or poor clinical prognosis (e.g. acral lentiginous).",
        "effect_up": "Adding sites expands sentinel scrutiny to more body regions.",
        "effect_down": "Removing sites relaxes referral policy for those body areas."
    },
    {
        "key": "sex_weight",
        "agent": "A4",
        "label": "Male Demographic Mortality Weight",
        "default": 0.10,
        "min": 0.0,
        "max": 0.30,
        "step": 0.01,
        "unit": "weight",
        "description": "Risk score increment reflecting higher male melanoma mortality rates.",
        "effect_up": "Increases male patient context risk score.",
        "effect_down": "Treats male and female demographics more equally."
    },
    {
        "key": "risk_moderate_from",
        "agent": "A4",
        "label": "Moderate Context Risk Threshold",
        "default": 0.45,
        "min": 0.20,
        "max": 0.60,
        "step": 0.01,
        "unit": "score",
        "description": "Context risk score threshold for transitioning from LOW to MODERATE.",
        "effect_up": "Keeps more patients in LOW context risk category.",
        "effect_down": "More easily labels cases as MODERATE risk."
    },
    {
        "key": "risk_high_from",
        "agent": "A4",
        "label": "High Context Risk Threshold",
        "default": 0.70,
        "min": 0.50,
        "max": 0.85,
        "step": 0.01,
        "unit": "score",
        "description": "Context risk score threshold for transitioning to HIGH risk, triggering safety veto checks.",
        "effect_up": "Restricts HIGH risk classification to severe demographic triads.",
        "effect_down": "More patients labeled HIGH risk; blocks auto-triage for mild risk presentations."
    },
    {
        "key": "risk_very_high_from",
        "agent": "A4",
        "label": "Very High Context Risk Threshold",
        "default": 0.85,
        "min": 0.70,
        "max": 0.98,
        "step": 0.01,
        "unit": "score",
        "description": "Context risk score threshold for VERY_HIGH risk tier.",
        "effect_up": "Higher threshold for highest demographic risk category.",
        "effect_down": "More cases designated VERY_HIGH risk."
    },
    {
        "key": "missing_field_penalty",
        "agent": "A4",
        "label": "Missing Demographic Field Prior",
        "default": 0.15,
        "min": 0.0,
        "max": 0.40,
        "step": 0.01,
        "unit": "penalty",
        "description": "Neutral epidemiological prior added per missing clinical field (age, sex, localization) during hidden test.",
        "effect_up": "Penalizes missing data more aggressively, leaning toward specialist referral.",
        "effect_down": "More permissive with missing clinical metadata."
    },

    # --- Agent 5: Safety Sentinel ---
    {
        "key": "safety_s1_enabled",
        "agent": "A5",
        "label": "Rule S1: Residual / Predicted Malignancy Veto",
        "default": True,
        "unit": "boolean",
        "description": "Vetoes AUTO_TRIAGE if predicted class is malignant or residual malignant probability mass >= threshold.",
        "effect_up": "Protects against missed cancers; mandatorily enabled in clinical production.",
        "effect_down": "Disables malignancy veto (unsafe; allows potential cancers to auto-triage)."
    },
    {
        "key": "safety_s2_enabled",
        "agent": "A5",
        "label": "Rule S2: Benign vs Suspicious ABCD Lesion Cues",
        "default": True,
        "unit": "boolean",
        "description": "Vetoes AUTO_TRIAGE if benign prediction conflicts with jagged borders, severe asymmetry, or high TDS score.",
        "effect_up": "Prevents discharging atypical dysplastic nevi with alarming visual morphology.",
        "effect_down": "Disables visual discordance veto; risks missing early-stage melanomas."
    },
    {
        "key": "safety_s3_enabled",
        "agent": "A5",
        "label": "Rule S3: Diagnostic Ambiguity (Margin & Entropy)",
        "default": True,
        "unit": "boolean",
        "description": "Vetoes AUTO_TRIAGE if candidate margin < top2_margin_min or Shannon entropy > max_entropy.",
        "effect_up": "Forces specialist evaluation whenever the model's differential diagnosis is split.",
        "effect_down": "Disables ambiguity checks, permitting auto-triage of contested classifications."
    },
    {
        "key": "safety_s4_enabled",
        "agent": "A5",
        "label": "Rule S4: Demographic Discordance (Elderly / High UV)",
        "default": True,
        "unit": "boolean",
        "description": "Vetoes AUTO_TRIAGE when benign diagnosis occurs in high-risk demographic without very high confidence.",
        "effect_up": "Protects high-risk populations (e.g. scalp/facial lesions in elderly) from false reassurance.",
        "effect_down": "Disables demographic conflict safeguards."
    },
    {
        "key": "safety_s5_enabled",
        "agent": "A5",
        "label": "Rule S5: Capture Degradation Discharge Block",
        "default": True,
        "unit": "boolean",
        "description": "Blocks AUTO_TRIAGE when capture quality has CAUTION or RE_IMAGE defects.",
        "effect_up": "Ensures no patient is autonomously discharged on a degraded or borderline image.",
        "effect_down": "Allows autonomous discharge despite capture quality concerns."
    },
    {
        "key": "safety_s6_enabled",
        "agent": "A5",
        "label": "Rule S6: Test-Time Augmentation Disagreement",
        "default": True,
        "unit": "boolean",
        "description": "Vetoes AUTO_TRIAGE if model probability swings significantly under geometric rotation/flip views.",
        "effect_up": "Guards against orientation artifacts and unstable morphology.",
        "effect_down": "Disables test-time rotation stability verification."
    },
    {
        "key": "caution_confidence_required",
        "agent": "A5",
        "label": "Required Confidence for Borderline Quality",
        "default": 0.70,
        "min": 0.50,
        "max": 0.95,
        "step": 0.01,
        "unit": "prob",
        "description": "Higher confidence hurdle required when image quality is ACCEPTABLE_WITH_CAUTION.",
        "effect_up": "Requires near-certainty to auto-triage a borderline quality capture.",
        "effect_down": "Allows auto-triage on lower confidence despite borderline quality."
    },
    {
        "key": "high_context_risk_confidence_required",
        "agent": "A5",
        "label": "Required Confidence for High-Risk Context",
        "default": 0.85,
        "min": 0.60,
        "max": 0.98,
        "step": 0.01,
        "unit": "prob",
        "description": "Stricter confidence hurdle required to auto-triage a benign lesion in high-risk patient demographics.",
        "effect_up": "Stricter standard for elderly/sun-damaged presentations.",
        "effect_down": "Lowers threshold for auto-triaging high-risk demographics."
    },
    {
        "key": "tta_disagreement_threshold",
        "agent": "A5",
        "label": "TTA Maximum Permissible Std-Dev",
        "default": 0.22,
        "min": 0.05,
        "max": 0.40,
        "step": 0.01,
        "unit": "std_dev",
        "description": "Maximum standard deviation across 4 test-time augmentation views before vetoing.",
        "effect_up": "Tolerates higher view variance across image rotations.",
        "effect_down": "Demands strict rotational invariance from the diagnostic model."
    },

    # --- Agent 6: Coordinator Policy ---
    {
        "key": "policy_quality_gate_enabled",
        "agent": "A6",
        "label": "Policy Rule 1: Quality Failure -> RE_IMAGE",
        "default": True,
        "unit": "boolean",
        "description": "Mandates RE_IMAGE if Agent 1 status is RE_IMAGE (while preserving diagnostic prediction).",
        "effect_up": "Enforces quality gatekeeper rule.",
        "effect_down": "Allows degraded images to bypass re-imaging to referral/triage."
    },
    {
        "key": "policy_malignancy_refer_enabled",
        "agent": "A6",
        "label": "Policy Rule 2: Malignancy Presentation -> REFER",
        "default": True,
        "unit": "boolean",
        "description": "Mandates specialist REFER if predicted disease code is mel, bcc, or akiec.",
        "effect_up": "Guarantees specialist referral for all suspected malignancies.",
        "effect_down": "Violates safety guidelines (unsafe)."
    },
    {
        "key": "policy_safety_veto_enabled",
        "agent": "A6",
        "label": "Policy Rule 3: Safety Sentinel Veto -> REFER",
        "default": True,
        "unit": "boolean",
        "description": "Mandates specialist REFER if Agent 5 issues a SAFETY_FLAG or SAFETY_VETO.",
        "effect_up": "Honors multi-agent clinical cross-checks and vetoes.",
        "effect_down": "Overrides safety vetoes (unsafe)."
    },
    {
        "key": "policy_confidence_gate_enabled",
        "agent": "A6",
        "label": "Policy Rule 4: Diagnostic Uncertainty -> REFER",
        "default": True,
        "unit": "boolean",
        "description": "Mandates specialist REFER if benign candidate confidence is below auto_triage_confidence or ambiguous.",
        "effect_up": "Ensures uncertain benign lesions receive human expert inspection.",
        "effect_down": "Auto-triages low-confidence cases (unsafe)."
    },
    {
        "key": "policy_benign_autotriage_enabled",
        "agent": "A6",
        "label": "Policy Rule 5: Verified Benign -> AUTO_TRIAGE",
        "default": True,
        "unit": "boolean",
        "description": "Permits AUTO_TRIAGE only when lesion is verified benign, high confidence, and all safety checks PASS.",
        "effect_up": "Standard clinical workflow for benign clearance.",
        "effect_down": "Disables all autonomous triage (refers 100% of cases)."
    },
    {
        "key": "policy_rule_order",
        "agent": "A6",
        "label": "Coordinator Decision Policy Evaluation Order",
        "default": [
            "quality_gate",
            "malignancy_refer",
            "safety_veto",
            "confidence_gate",
            "benign_autotriage"
        ],
        "unit": "order",
        "description": "Hierarchical priority order for resolving clinical decision policy branches.",
        "effect_up": "Reorders precedence of quality, safety, and diagnostic branches.",
        "effect_down": "Maintains documented summit clinical hierarchy."
    }
]

# --- Clinical Operating Presets ---
PRESETS: Dict[str, Dict[str, Any]] = {
    "Balanced": {
        "blur_min": 40.0,
        "tenengrad_min": 220.0,
        "brightness_min": 35.0,
        "brightness_max": 225.0,
        "entropy_min": 4.5,
        "glare_max_pct": 2.0,
        "obstruction_max_pct": 18.0,
        "quality_score_reimage_below": 50.0,
        "quality_score_caution_below": 72.0,
        "auto_triage_confidence": 0.80,
        "top2_margin_min": 0.10,
        "max_entropy": 1.65,
        "malignant_prob_threshold": 0.22,
        "mel_prob_threshold": 0.18,
        "bias_mel": 0.30,
        "bias_nv": 0.75,
        "bias_bcc": -0.40,
        "bias_akiec": -0.85,
        "bias_bkl": 0.00,
        "bias_df": 0.80,
        "bias_vasc": 0.00,
        "asymmetry_high": 0.35,
        "border_compactness_high": 1.60,
        "color_variation_high": 32.0,
        "suspicion_indeterminate_from": 0.35,
        "suspicion_suspicious_from": 0.55,
        "min_lesion_area_pct": 2.0,
        "segmentation_fail": True,
        "age_high_risk": 60,
        "age_very_high_risk": 75,
        "high_risk_sites": ["face", "scalp", "ear", "neck", "acral"],
        "sex_weight": 0.10,
        "risk_moderate_from": 0.45,
        "risk_high_from": 0.70,
        "risk_very_high_from": 0.85,
        "missing_field_penalty": 0.15,
        "safety_s1_enabled": True,
        "safety_s2_enabled": True,
        "safety_s3_enabled": True,
        "safety_s4_enabled": True,
        "safety_s5_enabled": True,
        "safety_s6_enabled": True,
        "caution_confidence_required": 0.70,
        "high_context_risk_confidence_required": 0.85,
        "tta_disagreement_threshold": 0.22,
        "policy_quality_gate_enabled": True,
        "policy_malignancy_refer_enabled": True,
        "policy_safety_veto_enabled": True,
        "policy_confidence_gate_enabled": True,
        "policy_benign_autotriage_enabled": True,
        "policy_rule_order": [
            "quality_gate",
            "malignancy_refer",
            "safety_veto",
            "confidence_gate",
            "benign_autotriage"
        ]
    },
    "Conservative": {
        "blur_min": 50.0,
        "tenengrad_min": 260.0,
        "brightness_min": 40.0,
        "brightness_max": 220.0,
        "entropy_min": 4.8,
        "glare_max_pct": 1.5,
        "obstruction_max_pct": 15.0,
        "quality_score_reimage_below": 58.0,
        "quality_score_caution_below": 78.0,
        "auto_triage_confidence": 0.90,
        "top2_margin_min": 0.18,
        "max_entropy": 1.45,
        "malignant_prob_threshold": 0.15,
        "mel_prob_threshold": 0.12,
        "bias_mel": 0.50,
        "bias_nv": 0.30,
        "bias_bcc": -0.20,
        "bias_akiec": -0.50,
        "bias_bkl": 0.00,
        "bias_df": 0.60,
        "bias_vasc": 0.00,
        "asymmetry_high": 0.30,
        "border_compactness_high": 1.50,
        "color_variation_high": 28.0,
        "suspicion_indeterminate_from": 0.30,
        "suspicion_suspicious_from": 0.50,
        "min_lesion_area_pct": 3.0,
        "segmentation_fail": True,
        "age_high_risk": 55,
        "age_very_high_risk": 70,
        "high_risk_sites": ["face", "scalp", "ear", "neck", "acral", "chest", "back"],
        "sex_weight": 0.15,
        "risk_moderate_from": 0.40,
        "risk_high_from": 0.65,
        "risk_very_high_from": 0.80,
        "missing_field_penalty": 0.20,
        "safety_s1_enabled": True,
        "safety_s2_enabled": True,
        "safety_s3_enabled": True,
        "safety_s4_enabled": True,
        "safety_s5_enabled": True,
        "safety_s6_enabled": True,
        "caution_confidence_required": 0.80,
        "high_context_risk_confidence_required": 0.92,
        "tta_disagreement_threshold": 0.18,
        "policy_quality_gate_enabled": True,
        "policy_malignancy_refer_enabled": True,
        "policy_safety_veto_enabled": True,
        "policy_confidence_gate_enabled": True,
        "policy_benign_autotriage_enabled": True,
        "policy_rule_order": [
            "quality_gate",
            "malignancy_refer",
            "safety_veto",
            "confidence_gate",
            "benign_autotriage"
        ]
    },
    "Aggressive": {
        "blur_min": 32.0,
        "tenengrad_min": 180.0,
        "brightness_min": 30.0,
        "brightness_max": 235.0,
        "entropy_min": 4.0,
        "glare_max_pct": 3.0,
        "obstruction_max_pct": 22.0,
        "quality_score_reimage_below": 42.0,
        "quality_score_caution_below": 65.0,
        "auto_triage_confidence": 0.70,
        "top2_margin_min": 0.08,
        "max_entropy": 1.85,
        "malignant_prob_threshold": 0.30,
        "mel_prob_threshold": 0.24,
        "bias_mel": 0.20,
        "bias_nv": 1.00,
        "bias_bcc": -0.60,
        "bias_akiec": -1.00,
        "bias_bkl": 0.10,
        "bias_df": 0.90,
        "bias_vasc": 0.00,
        "asymmetry_high": 0.42,
        "border_compactness_high": 1.75,
        "color_variation_high": 36.0,
        "suspicion_indeterminate_from": 0.40,
        "suspicion_suspicious_from": 0.62,
        "min_lesion_area_pct": 1.5,
        "segmentation_fail": True,
        "age_high_risk": 65,
        "age_very_high_risk": 80,
        "high_risk_sites": ["face", "scalp", "ear", "neck", "acral"],
        "sex_weight": 0.08,
        "risk_moderate_from": 0.50,
        "risk_high_from": 0.75,
        "risk_very_high_from": 0.90,
        "missing_field_penalty": 0.10,
        "safety_s1_enabled": True,
        "safety_s2_enabled": True,
        "safety_s3_enabled": True,
        "safety_s4_enabled": True,
        "safety_s5_enabled": True,
        "safety_s6_enabled": True,
        "caution_confidence_required": 0.65,
        "high_context_risk_confidence_required": 0.80,
        "tta_disagreement_threshold": 0.26,
        "policy_quality_gate_enabled": True,
        "policy_malignancy_refer_enabled": True,
        "policy_safety_veto_enabled": True,
        "policy_confidence_gate_enabled": True,
        "policy_benign_autotriage_enabled": True,
        "policy_rule_order": [
            "quality_gate",
            "malignancy_refer",
            "safety_veto",
            "confidence_gate",
            "benign_autotriage"
        ]
    }
}

# In-memory cached active thresholds
_ACTIVE_THRESHOLDS: Optional[Dict[str, Any]] = None

def get_thresholds_file_path() -> Path:
    candidates = [
        Path("thresholds.json"),
        Path(__file__).parent / "thresholds.json",
        Path(__file__).parent.parent / "thresholds.json"
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("thresholds.json")

def get_registry() -> List[Dict[str, Any]]:
    """Returns the full master registry metadata for all thresholds."""
    return REGISTRY

def get_presets() -> Dict[str, Dict[str, Any]]:
    """Returns the predefined clinical operating presets."""
    return PRESETS

def load_thresholds(reload: bool = False) -> Dict[str, Any]:
    """
    Loads active thresholds from thresholds.json, falling back to Balanced preset.
    """
    global _ACTIVE_THRESHOLDS
    if _ACTIVE_THRESHOLDS is not None and not reload:
        return _ACTIVE_THRESHOLDS

    p = get_thresholds_file_path()
    base = dict(PRESETS["Balanced"])

    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # If stored as nested sections, flatten appropriately
            if "quality" in data or "diagnosis" in data:
                # Legacy nested format conversion
                flat = {}
                q = data.get("quality", {})
                flat["blur_min"] = q.get("blur_threshold", base["blur_min"])
                flat["tenengrad_min"] = base["tenengrad_min"]
                flat["brightness_min"] = base["brightness_min"]
                flat["brightness_max"] = base["brightness_max"]
                flat["entropy_min"] = q.get("lighting_threshold", base["entropy_min"])
                flat["glare_max_pct"] = base["glare_max_pct"]
                flat["obstruction_max_pct"] = base["obstruction_max_pct"]
                flat["quality_score_reimage_below"] = q.get("quality_cutoff", base["quality_score_reimage_below"])
                flat["quality_score_caution_below"] = q.get("caution_overall_threshold", base["quality_score_caution_below"])
                
                d = data.get("diagnosis", {})
                flat["auto_triage_confidence"] = d.get("auto_triage_confidence", base["auto_triage_confidence"])
                flat["top2_margin_min"] = d.get("margin_minimum", base["top2_margin_min"])
                flat["max_entropy"] = d.get("max_entropy", base["max_entropy"])
                
                biases = d.get("class_probability_bias", {})
                for c in ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]:
                    flat[f"bias_{c}"] = biases.get(c, base[f"bias_{c}"])
                    
                m = data.get("malignancy", {})
                flat["malignant_prob_threshold"] = m.get("malignant_prob_threshold", base["malignant_prob_threshold"])
                flat["mel_prob_threshold"] = m.get("mel_threshold", base["mel_prob_threshold"])
                
                # Merge rest from base
                for k, v in base.items():
                    if k not in flat:
                        flat[k] = v
                base.update(flat)
            else:
                # Direct flat dictionary
                for k, v in data.items():
                    if not k.startswith("_"):
                        base[k] = v
        except Exception as e:
            print(f"[Thresholds] Warning reading {p}: {e}, using Balanced preset.")

    _ACTIVE_THRESHOLDS = base
    return _ACTIVE_THRESHOLDS

def save_thresholds(new_thresholds: Dict[str, Any], path: Optional[str] = None) -> Dict[str, Any]:
    """
    Saves updated thresholds to thresholds.json and updates in-memory cache.
    """
    global _ACTIVE_THRESHOLDS
    target_path = Path(path) if path else get_thresholds_file_path()
    
    current = load_thresholds()
    # Merge updates
    for k, v in new_thresholds.items():
        if k in current and v is not None:
            current[k] = v
        elif not k.startswith("_") and v is not None:
            current[k] = v

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)

    _ACTIVE_THRESHOLDS = current
    print(f"[Thresholds] Saved updated thresholds to {target_path}")
    return current

def reset_thresholds(preset: str = "Balanced", path: Optional[str] = None) -> Dict[str, Any]:
    """
    Resets thresholds to a specified preset ('Balanced', 'Conservative', 'Aggressive').
    """
    if preset not in PRESETS:
        preset = "Balanced"
    target_preset = dict(PRESETS[preset])
    return save_thresholds(target_preset, path=path)

def get_flat_thresholds(custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns active flat thresholds with optional per-request overrides applied.
    Also creates aliases for backward compatibility with legacy keys.
    """
    th = dict(load_thresholds())

    # Map legacy keys if provided in overrides or missing
    if custom_overrides:
        # Handle overrides formatted in either new or legacy keys
        legacy_to_new = {
            "blur_threshold": "blur_min",
            "lighting_threshold": "entropy_min",
            "glare_threshold": "glare_max_pct",
            "occlusion_threshold": "obstruction_max_pct",
            "quality_cutoff": "quality_score_reimage_below",
            "caution_overall_threshold": "quality_score_caution_below",
            "margin_minimum": "top2_margin_min",
            "mel_threshold": "mel_prob_threshold",
            "suspicion_benign_upper": "suspicion_indeterminate_from",
            "suspicion_indeterminate_upper": "suspicion_suspicious_from",
            "risk_low_upper": "risk_moderate_from",
            "risk_moderate_upper": "risk_high_from",
            "s1_enabled": "safety_s1_enabled",
            "s2_enabled": "safety_s2_enabled",
            "s3_enabled": "safety_s3_enabled",
            "s4_enabled": "safety_s4_enabled",
            "s5_enabled": "safety_s5_enabled",
            "s6_enabled": "safety_s6_enabled",
        }
        for k, v in custom_overrides.items():
            if v is not None:
                if k in legacy_to_new:
                    th[legacy_to_new[k]] = v
                th[k] = v

    # Add legacy backward-compatible aliases to the returned dictionary
    th["blur_threshold"] = th.get("blur_min", 40.0)
    th["lighting_threshold"] = th.get("entropy_min", 40.0)
    th["glare_threshold"] = 70.0 # legacy score
    th["occlusion_threshold"] = 50.0 # legacy score
    th["quality_cutoff"] = th.get("quality_score_reimage_below", 50.0)
    th["caution_overall_threshold"] = th.get("quality_score_caution_below", 72.0)
    th["margin_minimum"] = th.get("top2_margin_min", 0.10)
    th["mel_threshold"] = th.get("mel_prob_threshold", 0.18)
    th["suspicion_benign_upper"] = th.get("suspicion_indeterminate_from", 0.35)
    th["suspicion_indeterminate_upper"] = th.get("suspicion_suspicious_from", 0.55)
    th["risk_low_upper"] = th.get("risk_moderate_from", 0.45)
    th["risk_moderate_upper"] = th.get("risk_high_from", 0.70)
    th["s1_enabled"] = th.get("safety_s1_enabled", True)
    th["s2_enabled"] = th.get("safety_s2_enabled", True)
    th["s3_enabled"] = th.get("safety_s3_enabled", True)
    th["s4_enabled"] = th.get("safety_s4_enabled", True)
    th["s5_enabled"] = th.get("safety_s5_enabled", True)
    th["s6_enabled"] = th.get("safety_s6_enabled", True)

    # Class probability bias dict
    th["class_probability_bias"] = {
        "mel": th.get("bias_mel", 0.30),
        "nv": th.get("bias_nv", 0.75),
        "bcc": th.get("bias_bcc", -0.40),
        "akiec": th.get("bias_akiec", -0.85),
        "bkl": th.get("bias_bkl", 0.00),
        "df": th.get("bias_df", 0.80),
        "vasc": th.get("bias_vasc", 0.00)
    }

    return th
