import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass, field
import io
import base64
import time
from thresholds import load_thresholds, get_flat_thresholds

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

@dataclass
class QualityMetrics:
    blur_score: float
    lighting_score: float
    glare_score: float
    occlusion_score: float
    overall_quality: float
    status: str
    reasoning: list
    triggered_criteria: list
    edge_map_b64: str = ""
    # Standard Explaining Agent Contract fields
    agent: str = "A1"
    name: str = "Image Quality"
    score: float = 0.0
    summary: str = ""
    evidence: list = field(default_factory=list)
    thresholds_used: dict = field(default_factory=dict)
    margin_to_threshold: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    explanation: str = ""
    confidence_in_this_agent: float = 1.0
    runtime_ms: float = 0.0

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "name": self.name,
            "status": self.status,
            "score": self.score,
            "summary": self.summary,
            "evidence": self.evidence,
            "thresholds_used": self.thresholds_used,
            "margin_to_threshold": self.margin_to_threshold,
            "flags": self.flags,
            "explanation": self.explanation,
            "confidence_in_this_agent": self.confidence_in_this_agent,
            "runtime_ms": self.runtime_ms,
            # Preserved legacy fields
            "blur_score": self.blur_score,
            "lighting_score": self.lighting_score,
            "glare_score": self.glare_score,
            "occlusion_score": self.occlusion_score,
            "overall_quality": self.overall_quality,
            "reasoning": self.reasoning,
            "reasons": self.reasoning,
            "triggered_criteria": self.triggered_criteria,
            "edge_map_b64": self.edge_map_b64
        }

class ImageQualityAgent:
    """
    Agent 1: Image Quality Agent
    Evaluates skin lesion image quality across multiple dimensions:
      - Blur detection (Combined Laplacian variance & Tenengrad gradient)
      - Lighting assessment (Histogram entropy & dynamic range)
      - Specular glare detection (Saturated pixels in V-channel)
      - Occlusion / Obstruction detection (Centering & border vignette)
    Outputs:
      - Explaining Agent contract with full evidence, margins, and explanation
      - Quality status (PASS, ACCEPTABLE_WITH_CAUTION, RE_IMAGE)
      - Quality scores (0-100)
      - Defect trigger reasons
      - Edge/sharpness visualization map
    """

    def __init__(
        self,
        blur_threshold: Optional[float] = None,
        lighting_threshold: Optional[float] = None,
        glare_threshold: Optional[float] = None,
        occlusion_threshold: Optional[float] = None,
        quality_cutoff: Optional[float] = None,
        caution_overall_threshold: Optional[float] = None
    ):
        defaults = get_flat_thresholds()
        self.blur_threshold = blur_threshold if blur_threshold is not None else defaults.get("blur_min", 40.0)
        self.lighting_threshold = lighting_threshold if lighting_threshold is not None else defaults.get("entropy_min", 40.0)
        self.glare_threshold = glare_threshold if glare_threshold is not None else defaults.get("glare_threshold", 70.0)
        self.occlusion_threshold = occlusion_threshold if occlusion_threshold is not None else defaults.get("occlusion_threshold", 50.0)
        self.quality_cutoff = quality_cutoff if quality_cutoff is not None else defaults.get("quality_score_reimage_below", 50.0)
        self.caution_overall_threshold = caution_overall_threshold if caution_overall_threshold is not None else defaults.get("quality_score_caution_below", 72.0)

    def assess_blur(self, arr_gray: np.ndarray) -> Tuple[float, str, np.ndarray, float, float]:
        """
        Calculates blur score using Laplacian variance and edge gradient sharpness.
        Higher score = sharper image.
        Returns:
          blur_score (0-100), status, grad_mag, lap_var, tenengrad
        """
        pad = np.pad(arr_gray, 1, mode='reflect')
        lap = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4.0 * arr_gray)
        lap_var = float(lap.var())

        gx = (pad[:-2, 2:] + 2*pad[1:-1, 2:] + pad[2:, 2:]) - (pad[:-2, :-2] + 2*pad[1:-1, :-2] + pad[2:, :-2])
        gy = (pad[2:, :-2] + 2*pad[2:, 1:-1] + pad[2:, 2:]) - (pad[:-2, :-2] + 2*pad[:-2, 1:-1] + pad[:-2, 2:])
        grad_mag = np.sqrt(gx**2 + gy**2)
        tenengrad = float(np.mean(grad_mag**2))

        lap_component = min(100.0, (lap_var / 55.0) * 100.0)
        grad_component = min(100.0, (tenengrad / 800.0) * 100.0)
        blur_score = round(0.55 * lap_component + 0.45 * grad_component, 1)

        if blur_score >= 65:
            status = "SHARP"
        elif blur_score >= self.blur_threshold:
            status = "ACCEPTABLE"
        else:
            status = "BLURRY"

        return blur_score, status, grad_mag, lap_var, tenengrad

    def assess_lighting(self, arr_gray: np.ndarray) -> Tuple[float, str, float, float]:
        """
        Evaluates illumination distribution and dynamic range using histogram entropy.
        Returns lighting_score, status, entropy, mean_brightness
        """
        mean_brightness = float(np.mean(arr_gray))
        counts, _ = np.histogram(arr_gray, bins=256, range=(0, 256))
        p = counts / float(counts.sum() + 1e-8)
        entropy = -float(np.sum(p * np.log2(p + 1e-8)))

        lighting_score = round(min(100.0, (entropy / 7.6) * 100.0), 1)

        if lighting_score >= 60:
            status = "OPTIMAL"
        elif lighting_score >= self.lighting_threshold:
            status = "ACCEPTABLE"
        else:
            status = "POOR_LIGHTING"

        return lighting_score, status, entropy, mean_brightness

    def assess_glare(self, arr_rgb: np.ndarray) -> Tuple[float, str, float]:
        """
        Detects specular glare / flash reflection voids.
        Specular glare in dermoscopy occurs when all 3 color channels saturate (R, G, B >= 246).
        Returns glare_score, status, specular_pct
        """
        min_channel = np.min(arr_rgb, axis=2)
        specular_mask = min_channel >= 246.0
        specular_ratio = float(np.sum(specular_mask) / specular_mask.size)
        specular_pct = round(specular_ratio * 100.0, 2)

        glare_score = round(max(0.0, 100.0 - (specular_ratio * 100.0 * 18.0)), 1)

        if glare_score >= 85:
            status = "NO_GLARE"
        elif glare_score >= self.glare_threshold:
            status = "MINIMAL_GLARE"
        else:
            status = "GLARE_DEFECT"

        return glare_score, status, specular_pct

    def assess_occlusion(self, grad_mag: np.ndarray) -> Tuple[float, str, float]:
        """
        Detects border obstruction, lens vignetting, and off-center placement.
        Compares edge density in central active field vs extreme outer border band.
        Returns occlusion_score, status, outer_edge_pct
        """
        edges = grad_mag > 28.0
        h, w = edges.shape

        b_margin = max(6, int(min(h, w) * 0.06))
        mask_outer = np.zeros((h, w), dtype=bool)
        mask_outer[:b_margin, :] = True
        mask_outer[-b_margin:, :] = True
        mask_outer[:, :b_margin] = True
        mask_outer[:, -b_margin:] = True

        c_margin_y = int(h * 0.125)
        c_margin_x = int(w * 0.125)
        mask_center = np.zeros((h, w), dtype=bool)
        mask_center[c_margin_y:h - c_margin_y, c_margin_x:w - c_margin_x] = True

        center_edges = float(np.sum(edges & mask_center))
        outer_edges = float(np.sum(edges & mask_outer))
        total_edges = float(np.sum(edges)) + 1e-8

        outer_ratio = outer_edges / total_edges
        center_ratio = center_edges / total_edges
        outer_edge_pct = round(outer_ratio * 100.0, 1)

        occlusion_score = round(max(0.0, min(100.0, (center_ratio * 90.0) + (1.0 - outer_ratio * 2.5) * 20.0)), 1)

        if occlusion_score >= 65:
            status = "UNOBSTRUCTED"
        elif occlusion_score >= self.occlusion_threshold:
            status = "ACCEPTABLE_CENTERING"
        else:
            status = "OCCLUDED_OR_OFFCENTER"

        return occlusion_score, status, outer_edge_pct

    def evaluate(self, image_input: Any, custom_thresholds: Dict[str, Any] = None) -> QualityMetrics:
        """
        Evaluates image quality from either a file path, PIL Image, or numpy array.
        Returns QualityMetrics complying with the Explaining Agent contract.
        """
        t0 = time.perf_counter()
        th = get_flat_thresholds(custom_thresholds)

        b_th = float(th.get("blur_min", th.get("blur_threshold", self.blur_threshold)))
        l_th = float(th.get("entropy_min", th.get("lighting_threshold", self.lighting_threshold)))
        g_max_pct = float(th.get("glare_max_pct", 2.0))
        o_max_pct = float(th.get("obstruction_max_pct", 18.0))
        g_th = float(th.get("glare_threshold", self.glare_threshold))
        o_th = float(th.get("occlusion_threshold", self.occlusion_threshold))
        q_cutoff = float(th.get("quality_score_reimage_below", th.get("quality_cutoff", self.quality_cutoff)))
        caution_overall = float(th.get("quality_score_caution_below", th.get("caution_overall_threshold", self.caution_overall_threshold)))

        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Resize to standard analysis resolution
        pil_img = pil_img.resize((224, 224))
        arr_rgb = np.array(pil_img, dtype=np.float32)
        arr_gray = 0.299 * arr_rgb[:, :, 0] + 0.587 * arr_rgb[:, :, 1] + 0.114 * arr_rgb[:, :, 2]

        blur_score, blur_status, grad_mag, lap_var, tenengrad = self.assess_blur(arr_gray)
        lighting_score, lighting_status, entropy, mean_brightness = self.assess_lighting(arr_gray)
        glare_score, glare_status, specular_pct = self.assess_glare(arr_rgb)
        occlusion_score, occlusion_status, outer_edge_pct = self.assess_occlusion(grad_mag)

        overall_quality = round((blur_score + lighting_score + glare_score + occlusion_score) / 4.0, 1)

        reasoning = []
        triggers = []
        flags = []

        is_blur_defect = blur_score < b_th
        is_light_defect = lighting_score < l_th
        is_glare_defect = glare_score < g_th
        is_occl_defect = occlusion_score < o_th
        is_low_overall = overall_quality < q_cutoff

        if is_blur_defect:
            triggers.append(f"Blur ({blur_score:.1f} < {b_th:.1f})")
            flags.append("BLUR_DEFECT")
            reasoning.append(f"Image is too blurry ({blur_score:.1f}/100, minimum {b_th:.1f}); fine skin details cannot be clearly seen.")
        if is_light_defect:
            triggers.append(f"Lighting ({lighting_score:.1f} < {l_th:.1f})")
            flags.append("POOR_LIGHTING")
            reasoning.append(f"Lighting is too dark or uneven ({lighting_score:.1f}/100, minimum {l_th:.1f}); skin colors cannot be reliably analyzed.")
        if is_glare_defect:
            triggers.append(f"Glare ({glare_score:.1f} < {g_th:.1f})")
            flags.append("SPECULAR_GLARE")
            reasoning.append(f"Flash glare detected ({specular_pct:.1f}% of image covered by shiny reflections); part of the spot is washed out.")
        if is_occl_defect:
            triggers.append(f"Occlusion ({occlusion_score:.1f} < {o_th:.1f})")
            flags.append("BORDER_OCCLUSION")
            reasoning.append(f"The spot is too close to the photo edge or partially cut off ({occlusion_score:.1f}/100, minimum {o_th:.1f}).")

        if is_blur_defect or is_light_defect or is_glare_defect or is_occl_defect or is_low_overall or len(triggers) >= 2:
            status = "RE_IMAGE"
            flags.append("CRITICAL_QUALITY_FAILURE")
            if not reasoning:
                reasoning.append(f"Overall photo quality ({overall_quality:.1f}/100) is below the minimum required standard ({q_cutoff:.1f}) for safe screening.")
            summary = "Photo is not clear enough for a reliable assessment; please take a new photo."
        elif overall_quality < caution_overall or len(triggers) == 1:
            status = "ACCEPTABLE_WITH_CAUTION"
            flags.append("BORDERLINE_QUALITY")
            summary = "Photo quality is fair; usable, but minor blur or lighting imperfections were noticed."
        else:
            status = "PASS"
            summary = "Photo quality is excellent: clear, sharp, and well-lit."

        if not reasoning:
            reasoning.append("The photo is sharp, well-lit, free of glare, and the spot is clearly centered.")

        # Build formal Explaining Agent contract evidence list
        evidence = [
            {
                "feature": "blur_sharpness",
                "value": blur_score,
                "threshold": b_th,
                "direction": ">=",
                "passed": not is_blur_defect,
                "note": f"Sharpness score is {blur_score:.1f}/100 (needs at least {b_th:.1f} to pass; clear focus)"
            },
            {
                "feature": "lighting_entropy",
                "value": lighting_score,
                "threshold": l_th,
                "direction": ">=",
                "passed": not is_light_defect,
                "note": f"Lighting score is {lighting_score:.1f}/100 (needs at least {l_th:.1f} to pass; balanced brightness)"
            },
            {
                "feature": "specular_glare",
                "value": glare_score,
                "threshold": g_th,
                "direction": ">=",
                "passed": not is_glare_defect,
                "note": f"Glare-free score is {glare_score:.1f}/100 (shiny flash covers {specular_pct:.1f}%, limit {g_max_pct:.1f}%)"
            },
            {
                "feature": "centering_occlusion",
                "value": occlusion_score,
                "threshold": o_th,
                "direction": ">=",
                "passed": not is_occl_defect,
                "note": f"Centering score is {occlusion_score:.1f}/100 (spot is well positioned away from image edges)"
            },
            {
                "feature": "composite_quality",
                "value": overall_quality,
                "threshold": q_cutoff,
                "direction": ">=",
                "passed": not is_low_overall,
                "note": f"Overall quality score is {overall_quality:.1f}/100 (minimum {q_cutoff:.1f} needed for automated analysis)"
            }
        ]

        thresholds_used = {
            "blur_min": b_th,
            "entropy_min": l_th,
            "glare_max_pct": g_max_pct,
            "obstruction_max_pct": o_max_pct,
            "quality_score_reimage_below": q_cutoff,
            "quality_score_caution_below": caution_overall
        }

        margin_to_threshold = {
            "blur_min": round(blur_score - b_th, 2),
            "entropy_min": round(lighting_score - l_th, 2),
            "glare_max_pct": round(g_max_pct - specular_pct, 2),
            "quality_score_reimage_below": round(overall_quality - q_cutoff, 2)
        }

        # 2-4 sentence plain-English reasoning
        explanation = (
            f"The Image Quality Agent evaluated this photo at {overall_quality:.1f} out of 100 ({status}). "
            f"Sharpness was measured at {blur_score:.1f}/100 and lighting at {lighting_score:.1f}/100. "
            f"{'The photo is clear, sharp, and easy to inspect.' if status == 'PASS' else 'Issues found: ' + '; '.join(reasoning)}"
        )

        # Generate high-contrast edge & sharpness visualization map
        p96 = float(np.percentile(grad_mag, 96) + 1e-5)
        norm_edge = np.clip(grad_mag / p96, 0.0, 1.0)

        vis_map = np.zeros((224, 224, 3), dtype=np.uint8)
        vis_map[:, :, 0] = np.clip(14 + norm_edge * 120 + np.where(norm_edge > 0.6, (norm_edge - 0.6) * 200, 0), 0, 255).astype(np.uint8)
        vis_map[:, :, 1] = np.clip(18 + norm_edge * 230, 0, 255).astype(np.uint8)
        vis_map[:, :, 2] = np.clip(32 + norm_edge * 220, 0, 255).astype(np.uint8)

        orig_arr = np.clip(arr_rgb, 0, 255).astype(np.uint8)
        blended = np.clip(0.75 * vis_map.astype(np.float32) + 0.25 * orig_arr.astype(np.float32), 0, 255).astype(np.uint8)

        grad_img = Image.fromarray(blended)
        buf = io.BytesIO()
        grad_img.save(buf, format="PNG")
        edge_map_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        conf_in_agent = 0.95 if status == "PASS" else (0.80 if status == "ACCEPTABLE_WITH_CAUTION" else 0.90)

        return QualityMetrics(
            blur_score=blur_score,
            lighting_score=lighting_score,
            glare_score=glare_score,
            occlusion_score=occlusion_score,
            overall_quality=overall_quality,
            status=status,
            reasoning=reasoning,
            triggered_criteria=triggers,
            edge_map_b64=edge_map_b64,
            agent="A1",
            name="Image Quality",
            score=overall_quality,
            summary=summary,
            evidence=evidence,
            thresholds_used=thresholds_used,
            margin_to_threshold=margin_to_threshold,
            flags=flags,
            explanation=explanation,
            confidence_in_this_agent=conf_in_agent,
            runtime_ms=runtime_ms
        )

# Singleton instance
quality_agent = ImageQualityAgent()
