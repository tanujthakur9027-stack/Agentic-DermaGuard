import cv2
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class QualityMetrics:
    blur_score: float
    lighting_score: float
    glare_score: float
    occlusion_score: float
    overall_quality: float
    status: str
    reasoning: list

class ImageQualityAgent:
    """
    Evaluates skin lesion image quality across multiple dimensions:
    - Blur detection (Laplacian variance)
    - Lighting assessment (histogram uniformity)
    - Glare detection (overexposed regions)
    - Occlusion detection (edge coverage)
    """
    
    def __init__(self):
        self.blur_threshold = 100
        self.lighting_threshold = 0.3
        self.glare_threshold = 0.15
        self.occlusion_threshold = 0.2
    
    def assess_blur(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Calculate blur score using Laplacian variance.
        Higher score = sharper image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-100 scale
        blur_score = min(100, (laplacian_var / self.blur_threshold) * 100)
        
        if blur_score >= 70:
            status = "SHARP"
        elif blur_score >= 40:
            status = "ACCEPTABLE"
        else:
            status = "BLURRY"
        
        return blur_score, status
    
    def assess_lighting(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Assess lighting quality using histogram entropy.
        Good lighting = well-distributed histogram
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        lighting_score = min(100, (entropy / 8) * 100)  # Max entropy is ~8
        
        if lighting_score >= 60:
            status = "OPTIMAL"
        elif lighting_score >= 40:
            status = "ACCEPTABLE"
        else:
            status = "POOR_LIGHTING"
        
        return lighting_score, status
    
    def assess_glare(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Detect glare/overexposed regions.
        Glare = high intensity pixels > 240 in value channel
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        # Calculate percentage of overexposed pixels
        overexposed_ratio = np.sum(v_channel > 240) / v_channel.size
        
        glare_score = 100 - (overexposed_ratio * 100)
        
        if glare_score >= 85:
            status = "NO_GLARE"
        elif glare_score >= 70:
            status = "MINIMAL_GLARE"
        else:
            status = "GLARE_DETECTED"
        
        return glare_score, status
    
    def assess_occlusion(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Detect occlusion/obstruction using edge coverage.
        Good images have lesion centered with minimal obstruction
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check if edges are concentrated in center (not at borders)
        h, w = edges.shape
        mask_center = np.zeros((h, w), dtype=bool)
        mask_center[h//4:-h//4, w//4:-w//4] = True
        
        center_edges = np.sum((edges > 0) & mask_center)
        border_edges = np.sum((edges > 0) & (~mask_center))
        
        total_edges = center_edges + border_edges + 1e-8
        center_ratio = center_edges / total_edges
        
        occlusion_score = center_ratio * 100
        
        if occlusion_score >= 70:
            status = "UNOBSTRUCTED"
        elif occlusion_score >= 50:
            status = "MINIMAL_OCCLUSION"
        else:
            status = "OCCLUDED"
        
        return occlusion_score, status
    
    def evaluate(self, image_path: str) -> QualityMetrics:
        """
        Comprehensive image quality assessment.
        Returns QualityMetrics with all scores and triage recommendation.
        """
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize to standard size for consistent evaluation
        image = cv2.resize(image, (224, 224))
        
        # Assess all dimensions
        blur_score, blur_status = self.assess_blur(image)
        lighting_score, lighting_status = self.assess_lighting(image)
        glare_score, glare_status = self.assess_glare(image)
        occlusion_score, occlusion_status = self.assess_occlusion(image)
        
        # Calculate overall quality
        overall_quality = np.mean([blur_score, lighting_score, glare_score, occlusion_score])
        
        # Determine status and reasoning
        reasoning = []
        if blur_score < 40:
            reasoning.append(f"Image is blurry (score: {blur_score:.1f})")
        if lighting_score < 40:
            reasoning.append(f"Poor lighting conditions (score: {lighting_score:.1f})")
        if glare_score < 70:
            reasoning.append(f"Glare/overexposure detected (score: {glare_score:.1f})")
        if occlusion_score < 50:
            reasoning.append(f"Possible occlusion/obstruction (score: {occlusion_score:.1f})")
        
        if overall_quality < 50:
            status = "RE_IMAGE"
            if not reasoning:
                reasoning.append(f"Overall image quality poor (score: {overall_quality:.1f})")
        elif overall_quality < 70:
            status = "ACCEPTABLE_WITH_CAUTION"
        else:
            status = "ACCEPTABLE"
        
        return QualityMetrics(
            blur_score=round(blur_score, 2),
            lighting_score=round(lighting_score, 2),
            glare_score=round(glare_score, 2),
            occlusion_score=round(occlusion_score, 2),
            overall_quality=round(overall_quality, 2),
            status=status,
            reasoning=reasoning if reasoning else ["Image quality acceptable"]
        )

# Singleton instance
quality_agent = ImageQualityAgent()
