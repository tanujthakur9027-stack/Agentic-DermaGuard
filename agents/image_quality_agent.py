import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
import io
import base64

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

class ImageQualityAgent:
    """
    Agent 1: Image Quality Agent
    Evaluates skin lesion image quality across multiple dimensions:
      - Blur detection (Combined Laplacian variance & Tenengrad gradient)
      - Lighting assessment (Histogram entropy & dynamic range)
      - Specular glare detection (Saturated pixels in V-channel)
      - Occlusion / Obstruction detection (Centering & border vignette)
    Outputs:
      - Quality status (PASS, ACCEPTABLE_WITH_CAUTION, RE_IMAGE)
      - Quality scores (0-100)
      - Defect trigger reasons
      - Edge/sharpness visualization map
    """

    def __init__(
        self,
        blur_threshold: float = 40.0,
        lighting_threshold: float = 40.0,
        glare_threshold: float = 70.0,
        occlusion_threshold: float = 50.0,
        quality_cutoff: float = 50.0
    ):
        self.blur_threshold = blur_threshold
        self.lighting_threshold = lighting_threshold
        self.glare_threshold = glare_threshold
        self.occlusion_threshold = occlusion_threshold
        self.quality_cutoff = quality_cutoff

    def assess_blur(self, arr_gray: np.ndarray) -> Tuple[float, str, np.ndarray]:
        """
        Calculates blur score using Laplacian variance and edge gradient sharpness.
        Higher score = sharper image.
        Returns:
          blur_score (0-100), status, and edge gradient map.
        """
        # Laplacian calculation via 2D convolution kernel
        pad = np.pad(arr_gray, 1, mode='reflect')
        lap = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4.0 * arr_gray)
        lap_var = float(lap.var())

        # Tenengrad gradient sharpness (Sobel operator)
        gx = (pad[:-2, 2:] + 2*pad[1:-1, 2:] + pad[2:, 2:]) - (pad[:-2, :-2] + 2*pad[1:-1, :-2] + pad[2:, :-2])
        gy = (pad[2:, :-2] + 2*pad[2:, 1:-1] + pad[2:, 2:]) - (pad[:-2, :-2] + 2*pad[:-2, 1:-1] + pad[:-2, 2:])
        grad_mag = np.sqrt(gx**2 + gy**2)
        tenengrad = float(np.mean(grad_mag**2))

        # Composite sharpness metric: calibrated for dermoscopy
        # In 224x224 scaled dermoscopy:
        # Sharp microstructures give lap_var >= 55, tenengrad >= 800
        # Severely blurred captures give lap_var < 5, tenengrad < 220
        lap_component = min(100.0, (lap_var / 55.0) * 100.0)
        grad_component = min(100.0, (tenengrad / 800.0) * 100.0)
        blur_score = round(0.55 * lap_component + 0.45 * grad_component, 1)

        if blur_score >= 65:
            status = "SHARP"
        elif blur_score >= self.blur_threshold:
            status = "ACCEPTABLE"
        else:
            status = "BLURRY"

        return blur_score, status, grad_mag

    def assess_lighting(self, arr_gray: np.ndarray) -> Tuple[float, str]:
        """
        Evaluates illumination distribution and dynamic range using histogram entropy.
        """
        counts, _ = np.histogram(arr_gray, bins=256, range=(0, 256))
        p = counts / float(counts.sum() + 1e-8)
        entropy = -float(np.sum(p * np.log2(p + 1e-8)))
        
        # Max entropy for 256 bins is 8.0 bits
        lighting_score = round(min(100.0, (entropy / 7.6) * 100.0), 1)

        if lighting_score >= 60:
            status = "OPTIMAL"
        elif lighting_score >= self.lighting_threshold:
            status = "ACCEPTABLE"
        else:
            status = "POOR_LIGHTING"

        return lighting_score, status

    def assess_glare(self, arr_rgb: np.ndarray) -> Tuple[float, str]:
        """
        Detects specular glare / flash reflection voids.
        Specular glare in dermoscopy occurs when all 3 color channels saturate (R, G, B >= 246),
        causing pure sensor clip white voids that obliterate pigment networks.
        Normal fair skin, in contrast, absorbs blue light (B < 220).
        """
        min_channel = np.min(arr_rgb, axis=2)
        specular_mask = min_channel >= 246.0
        specular_ratio = float(np.sum(specular_mask) / specular_mask.size)

        # Calibrated clinical penalty:
        # 1.5% specular coverage drops score to 73 (caution); > 2% drops below 64 (defect & RE_IMAGE)
        glare_score = round(max(0.0, 100.0 - (specular_ratio * 100.0 * 18.0)), 1)

        if glare_score >= 85:
            status = "NO_GLARE"
        elif glare_score >= self.glare_threshold:
            status = "MINIMAL_GLARE"
        else:
            status = "GLARE_DEFECT"

        return glare_score, status

    def assess_occlusion(self, grad_mag: np.ndarray) -> Tuple[float, str]:
        """
        Detects border obstruction, lens vignetting, and off-center placement.
        Compares edge density in central active field vs extreme outer border band.
        """
        edges = grad_mag > 28.0
        h, w = edges.shape

        # Outer border band (outer 5% perimeter pixels)
        b_margin = max(6, int(min(h, w) * 0.06))
        mask_outer = np.zeros((h, w), dtype=bool)
        mask_outer[:b_margin, :] = True
        mask_outer[-b_margin:, :] = True
        mask_outer[:, :b_margin] = True
        mask_outer[:, -b_margin:] = True

        # Central 75% field of view
        c_margin_y = int(h * 0.125)
        c_margin_x = int(w * 0.125)
        mask_center = np.zeros((h, w), dtype=bool)
        mask_center[c_margin_y:h - c_margin_y, c_margin_x:w - c_margin_x] = True

        center_edges = float(np.sum(edges & mask_center))
        outer_edges = float(np.sum(edges & mask_outer))
        total_edges = float(np.sum(edges)) + 1e-8

        # Penalty for heavy edge density on the extreme outer border (clipping/vignette)
        outer_ratio = outer_edges / total_edges
        center_ratio = center_edges / total_edges

        # Good centering score: high center ratio, low outer ratio
        occlusion_score = round(max(0.0, min(100.0, (center_ratio * 90.0) + (1.0 - outer_ratio * 2.5) * 20.0)), 1)

        if occlusion_score >= 65:
            status = "UNOBSTRUCTED"
        elif occlusion_score >= self.occlusion_threshold:
            status = "ACCEPTABLE_CENTERING"
        else:
            status = "OCCLUDED_OR_OFFCENTER"

        return occlusion_score, status

    def evaluate(self, image_input: Any, custom_thresholds: Dict[str, float] = None) -> QualityMetrics:
        """
        Evaluates image quality from either a file path, PIL Image, or numpy array.
        Returns QualityMetrics with scores, status, defect triggers, and visual edge overlay.
        """
        th = custom_thresholds or {}
        b_th = th.get("blur_threshold", self.blur_threshold)
        l_th = th.get("lighting_threshold", self.lighting_threshold)
        g_th = th.get("glare_threshold", self.glare_threshold)
        o_th = th.get("occlusion_threshold", self.occlusion_threshold)
        q_cutoff = th.get("quality_cutoff", self.quality_cutoff)

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

        blur_score, blur_status, grad_mag = self.assess_blur(arr_gray)
        lighting_score, lighting_status = self.assess_lighting(arr_gray)
        glare_score, glare_status = self.assess_glare(arr_rgb)
        occlusion_score, occlusion_status = self.assess_occlusion(grad_mag)

        overall_quality = round((blur_score + lighting_score + glare_score + occlusion_score) / 4.0, 1)

        reasoning = []
        triggers = []

        if blur_score < b_th:
            triggers.append(f"Blur ({blur_score:.1f} < {b_th:.1f})")
            reasoning.append(f"Image sharpness below threshold ({blur_score:.1f} < {b_th:.1f}); edge blur may obscure pigment network.")
        if lighting_score < l_th:
            triggers.append(f"Lighting ({lighting_score:.1f} < {l_th:.1f})")
            reasoning.append(f"Low illumination entropy ({lighting_score:.1f} < {l_th:.1f}); poor dynamic range.")
        if glare_score < g_th:
            triggers.append(f"Glare ({glare_score:.1f} < {g_th:.1f})")
            reasoning.append(f"Specular reflection/glare detected ({glare_score:.1f} < {g_th:.1f}); white reflection voids.")
        if occlusion_score < o_th:
            triggers.append(f"Occlusion ({occlusion_score:.1f} < {o_th:.1f})")
            reasoning.append(f"Lesion off-center or peripheral artifact ({occlusion_score:.1f} < {o_th:.1f}).")

        # Track Rule: Any critical defect requires re-capture (RE_IMAGE)
        is_blur_defect = blur_score < b_th
        is_light_defect = lighting_score < l_th
        is_glare_defect = glare_score < g_th
        is_occl_defect = occlusion_score < o_th
        is_low_overall = overall_quality < q_cutoff

        if is_blur_defect or is_light_defect or is_glare_defect or is_occl_defect or is_low_overall or len(triggers) >= 2:
            status = "RE_IMAGE"
            if not reasoning:
                reasoning.append(f"Composite quality below threshold ({overall_quality:.1f} < {q_cutoff:.1f}).")
        elif overall_quality < 72.0 or len(triggers) == 1:
            status = "ACCEPTABLE_WITH_CAUTION"
        else:
            status = "PASS"

        if not reasoning:
            reasoning.append("Image quality fully acceptable for dermoscopic evaluation.")

        # Generate high-contrast edge & sharpness visualization map
        p96 = float(np.percentile(grad_mag, 96) + 1e-5)
        norm_edge = np.clip(grad_mag / p96, 0.0, 1.0)

        # High-contrast edge heatmap:
        # Dark obsidian navy for smooth background -> gold for mid edges -> electric cyan for sharp microstructures
        vis_map = np.zeros((224, 224, 3), dtype=np.uint8)
        # Red channel: highlights strong textures
        vis_map[:, :, 0] = np.clip(14 + norm_edge * 120 + np.where(norm_edge > 0.6, (norm_edge - 0.6) * 200, 0), 0, 255).astype(np.uint8)
        # Green channel: peaks on strong edge gradient
        vis_map[:, :, 1] = np.clip(18 + norm_edge * 230, 0, 255).astype(np.uint8)
        # Blue channel: high cyan component
        vis_map[:, :, 2] = np.clip(32 + norm_edge * 220, 0, 255).astype(np.uint8)

        # Blend subtle 20% original image texture into edge map for anatomical context
        orig_arr = np.clip(arr_rgb, 0, 255).astype(np.uint8)
        blended = np.clip(0.75 * vis_map.astype(np.float32) + 0.25 * orig_arr.astype(np.float32), 0, 255).astype(np.uint8)

        grad_img = Image.fromarray(blended)
        buf = io.BytesIO()
        grad_img.save(buf, format="PNG")
        edge_map_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        return QualityMetrics(
            blur_score=blur_score,
            lighting_score=lighting_score,
            glare_score=glare_score,
            occlusion_score=occlusion_score,
            overall_quality=overall_quality,
            status=status,
            reasoning=reasoning,
            triggered_criteria=triggers,
            edge_map_b64=edge_map_b64
        )

# Singleton instance
quality_agent = ImageQualityAgent()
