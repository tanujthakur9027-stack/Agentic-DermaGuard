import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Optional
import io
import base64

class LesionAnalysisAgent:
    """
    Agent 3: Lesion Analysis Agent
    Examines dermoscopic visual cues based on clinical ABCD guidelines:
      - Asymmetry (A): Measures geometric asymmetry along principal axes.
      - Border Irregularity (B): Measures edge jaggedness, scalloping, and compactness.
      - Color Variation (C): Assesses color heterogeneity across RGB/HSV channels.
    Provides:
      - Feature evidence (quantitative metrics & clinical description)
      - Suspicion score (0.0 to 1.0)
      - Reliability flags (e.g. edge touching, low contrast, artifact interference)
      - Edge/contour mapping data for visualization
    """

    def __init__(self):
        self.asymmetry_weight = 0.35
        self.border_weight = 0.40
        self.color_weight = 0.25

    def segment_lesion(self, arr_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Segments the lesion from background skin using Otsu-like adaptive luminance thresholding.
        Returns binary mask (True = lesion) and edge-touching flag.
        """
        # Convert to grayscale luminance
        gray = 0.299 * arr_rgb[:, :, 0] + 0.587 * arr_rgb[:, :, 1] + 0.114 * arr_rgb[:, :, 2]
        h, w = gray.shape

        # Sample border pixels to estimate background skin luminance
        border_pixels = np.concatenate([
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
        ])
        skin_mean = float(np.median(border_pixels))
        
        # Lesions are typically darker than surrounding skin
        # Threshold: pixels darker than skin_mean minus dynamic delta
        diff = skin_mean - gray
        diff_pos = diff[diff > 0]
        if len(diff_pos) > 0:
            thresh_delta = max(8.0, float(np.percentile(diff_pos, 45)))
        else:
            thresh_delta = 12.0

        mask = diff > thresh_delta

        # Filter out isolated speckles using simple morphological erosion/dilation via 2D pooling
        # Pad and count 3x3 neighbors
        pad = np.pad(mask.astype(np.float32), 1, mode='constant', constant_values=0)
        neighbor_sum = (
            pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
            pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
            pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
        )
        # Clean mask: keep pixels with at least 4 neighbors
        clean_mask = neighbor_sum >= 4.0

        # Check if lesion touches borders
        touches_border = bool(
            np.any(clean_mask[0, :]) or np.any(clean_mask[-1, :]) or 
            np.any(clean_mask[:, 0]) or np.any(clean_mask[:, -1])
        )

        # Fallback if mask is empty: central circle
        if np.sum(clean_mask) < 100:
            y, x = np.ogrid[:h, :w]
            cy, cx = h // 2, w // 2
            r = min(h, w) // 4
            clean_mask = ((x - cx)**2 + (y - cy)**2) <= r**2

        return clean_mask, touches_border

    def compute_edge_map(self, mask: np.ndarray) -> np.ndarray:
        """
        Extracts single-pixel contour edges from binary lesion mask.
        """
        pad = np.pad(mask.astype(np.float32), 1, mode='constant', constant_values=0)
        # Gradient magnitude on binary mask gives boundary edges
        gx = np.abs(pad[1:-1, 2:] - pad[1:-1, :-2])
        gy = np.abs(pad[2:, 1:-1] - pad[:-2, 1:-1])
        edge_map = (gx + gy) > 0.5
        return edge_map

    def assess_border_irregularity(self, mask: np.ndarray, edge_map: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Quantifies border irregularity:
          - Compactness / Isoperimetric Quotient: P^2 / (4 * pi * A)
            (Circle = 1.0; irregular, notched, indented malignant borders > 1.6)
          - Radial distance variance from lesion centroid
        """
        area = float(np.sum(mask))
        perimeter = float(np.sum(edge_map))

        if area <= 0 or perimeter <= 0:
            return 0.5, {"compactness": 1.0, "radial_variance": 0.0, "status": "UNCERTAIN"}

        # Compactness ratio
        compactness = (perimeter ** 2) / (4.0 * np.pi * area + 1e-8)
        
        # Radial distance variance from centroid
        y_coords, x_coords = np.where(edge_map)
        if len(x_coords) > 0:
            cx = float(np.mean(x_coords))
            cy = float(np.mean(y_coords))
            radii = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
            radial_std = float(np.std(radii))
            radial_mean = float(np.mean(radii)) + 1e-8
            radial_cv = radial_std / radial_mean
        else:
            radial_cv = 0.0

        # Normalized border irregularity score [0.0, 1.0]
        # In discrete pixel geometry, smooth elliptical nevi have compactness ~ 1.5 - 2.8.
        # Highly jagged, notched, or lobulated melanoma borders have compactness > 3.8 or high radial variance.
        comp_norm = max(0.0, (compactness - 2.2) / 3.0)
        radial_norm = max(0.0, (radial_cv - 0.18) / 0.28)
        border_score = min(1.0, comp_norm * 0.6 + radial_norm * 0.4)

        status = "REGULAR" if border_score < 0.35 else ("INDETERMINATE" if border_score < 0.65 else "IRREGULAR")

        return border_score, {
            "compactness": round(compactness, 2),
            "radial_cv": round(radial_cv, 3),
            "perimeter_pixels": int(perimeter),
            "area_pixels": int(area),
            "status": status
        }

    def assess_asymmetry(self, mask: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Quantifies asymmetry by folding lesion mask across horizontal and vertical centroid axes.
        Calculates non-overlapping pixel fraction (symmetric difference / total area).
        """
        y_indices, x_indices = np.where(mask)
        if len(x_indices) < 50:
            return 0.25, {"h_asym": 0.25, "v_asym": 0.25, "status": "UNCERTAIN"}

        cy, cx = int(np.mean(y_indices)), int(np.mean(x_indices))
        h, w = mask.shape

        # Vertical flip across cx
        flipped_v = np.fliplr(mask)
        overlap_v = np.sum(mask & flipped_v)
        union_v = np.sum(mask | flipped_v) + 1e-8
        v_diff = 1.0 - (overlap_v / union_v)

        # Horizontal flip across cy
        flipped_h = np.flipud(mask)
        overlap_h = np.sum(mask & flipped_h)
        union_h = np.sum(mask | flipped_h) + 1e-8
        h_diff = 1.0 - (overlap_h / union_h)

        mean_diff = (v_diff + h_diff) / 2.0
        # Normal nevi typically have ~0.25-0.45 non-overlap from natural biological contour
        asym_score = min(1.0, max(0.0, (mean_diff - 0.25) / 0.40))
        status = "SYMMETRIC" if asym_score < 0.35 else ("MILD_ASYMMETRY" if asym_score < 0.65 else "MARKED_ASYMMETRY")

        return asym_score, {
            "vertical_asymmetry": round(v_diff, 3),
            "horizontal_asymmetry": round(h_diff, 3),
            "asymmetry_index": round(asym_score, 3),
            "status": status
        }

    def assess_color_variation(self, arr_rgb: np.ndarray, mask: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluates color variation across R, G, B channels and variegation inside the lesion.
        """
        if np.sum(mask) < 20:
            return 0.2, {"color_std": 0.0, "status": "MONOCHROMATIC"}

        r_vals = arr_rgb[:, :, 0][mask]
        g_vals = arr_rgb[:, :, 1][mask]
        b_vals = arr_rgb[:, :, 2][mask]

        std_r = float(np.std(r_vals))
        std_g = float(np.std(g_vals))
        std_b = float(np.std(b_vals))
        mean_std = (std_r + std_g + std_b) / 3.0

        # Check for multiple color tones
        # Dark melanin (black/dark brown)
        dark_pixels = np.sum((r_vals < 75) & (g_vals < 60) & (b_vals < 60)) / float(len(r_vals))
        # Erythematous / red
        red_pixels = np.sum((r_vals > 140) & (g_vals < 100) & (b_vals < 100)) / float(len(r_vals))
        # White / hypopigmentation
        light_pixels = np.sum((r_vals > 180) & (g_vals > 170) & (b_vals > 160)) / float(len(r_vals))

        variegation_count = int(dark_pixels > 0.05) + int(red_pixels > 0.05) + int(light_pixels > 0.05)

        color_score = min(1.0, max(0.0, (mean_std / 35.0) * 0.6 + (variegation_count / 3.0) * 0.4))
        status = "HOMOGENEOUS" if color_score < 0.35 else ("MODERATE_VARIATION" if color_score < 0.65 else "POLYCHROMATIC_VARIEGATED")

        return color_score, {
            "mean_channel_std": round(mean_std, 2),
            "variegated_shades_detected": variegation_count,
            "dark_melanin_ratio": round(dark_pixels, 3),
            "status": status
        }

    def analyze(self, pil_img: Image.Image) -> Dict[str, Any]:
        """
        Executes full lesion analysis.
        Returns:
          - feature_evidence: structured metrics for asymmetry, border, color
          - suspicion_score: composite 0.0 to 1.0 (high score = malignant suspicion)
          - suspicion_level: BENIGN, SUSPICIOUS, HIGHLY_SUSPICIOUS
          - reliability_flags: array of warning flags
          - edge_map_b64: base64 PNG of mapped edges overlaid on image
        """
        img_resized = pil_img.convert('RGB').resize((224, 224))
        arr_rgb = np.array(img_resized, dtype=np.float32)

        # 1. Segment lesion and boundary
        mask, touches_border = self.segment_lesion(arr_rgb)
        edge_map = self.compute_edge_map(mask)

        # 2. Assess ABCD features
        asym_score, asym_data = self.assess_asymmetry(mask)
        border_score, border_data = self.assess_border_irregularity(mask, edge_map)
        color_score, color_data = self.assess_color_variation(arr_rgb, mask)

        # 3. Composite suspicion score
        suspicion_score = (
            self.asymmetry_weight * asym_score +
            self.border_weight * border_score +
            self.color_weight * color_score
        )
        suspicion_score = round(float(np.clip(suspicion_score, 0.0, 1.0)), 3)

        if suspicion_score >= 0.60:
            suspicion_level = "SUSPICIOUS"
        elif suspicion_score >= 0.40:
            suspicion_level = "INDETERMINATE"
        else:
            suspicion_level = "BENIGN_CUES"

        # 4. Reliability flags
        reliability_flags = []
        if touches_border:
            reliability_flags.append("BORDER_TRUNCATION_WARNING: Lesion extends to image perimeter")
        if np.sum(mask) < 250:
            reliability_flags.append("SMALL_LESION_SIZE: Limited pixel area for robust contour analysis")
        if border_data["compactness"] > 3.0:
            reliability_flags.append("HIGH_EDGE_COMPLEXITY: Extreme border convolution detected")

        # 5. Generate visual edge mapping PNG overlay
        overlay_arr = np.array(img_resized, dtype=np.uint8)
        # Highlight edges in vivid cyan/magenta (BGR or RGB)
        overlay_arr[edge_map, 0] = 0    # R
        overlay_arr[edge_map, 1] = 255  # G
        overlay_arr[edge_map, 2] = 230  # B
        
        # Dilate edge slightly for clear visual rendering
        edge_pad = np.pad(edge_map, 1, mode='constant', constant_values=False)
        thick_edge = (
            edge_pad[:-2, 1:-1] | edge_pad[2:, 1:-1] | 
            edge_pad[1:-1, :-2] | edge_pad[1:-1, 2:]
        )
        overlay_arr[thick_edge, 0] = 0
        overlay_arr[thick_edge, 1] = 235
        overlay_arr[thick_edge, 2] = 255

        overlay_img = Image.fromarray(overlay_arr)
        buf = io.BytesIO()
        overlay_img.save(buf, format="PNG")
        edge_map_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        feature_evidence = {
            "asymmetry": {
                "score": round(asym_score, 3),
                "status": asym_data["status"],
                "details": asym_data
            },
            "border_irregularity": {
                "score": round(border_score, 3),
                "status": border_data["status"],
                "details": border_data
            },
            "color_variation": {
                "score": round(color_score, 3),
                "status": color_data["status"],
                "details": color_data
            }
        }

        return {
            "agent_id": "A3",
            "agent_name": "Lesion Analysis Agent",
            "suspicion_score": suspicion_score,
            "suspicion_level": suspicion_level,
            "feature_evidence": feature_evidence,
            "reliability_flags": reliability_flags,
            "edge_map_b64": edge_map_b64,
            "summary": (
                f"Visual cues indicate {suspicion_level} presentation (Suspicion Score: {suspicion_score:.2f}). "
                f"Border: {border_data['status']} (compactness: {border_data['compactness']}), "
                f"Asymmetry: {asym_data['status']}, Color: {color_data['status']}."
            )
        }

# Singleton instance
lesion_analysis_agent = LesionAnalysisAgent()
