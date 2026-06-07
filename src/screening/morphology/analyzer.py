import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from scipy import ndimage
from scipy.spatial import distance
import cv2

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely import affinity
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("Shapely not available, some morphology features will be limited")


@dataclass
class NucleusFeatures:
    instance_id: int
    contour: np.ndarray
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    major_axis_length: float
    minor_axis_length: float
    eccentricity: float
    edge_roughness: float
    solidity: float
    convexity: float
    form_factor: float
    roundness: float
    is_abnormal: bool = False
    abnormality_score: float = 0.0
    abnormality_reasons: List[str] = field(default_factory=list)
    feature_vector: Optional[np.ndarray] = None


class MorphologyAnalyzer:
    def __init__(
        self,
        circularity_threshold: float = 0.55,
        aspect_ratio_threshold: float = 3.0,
        edge_roughness_threshold: float = 0.4,
        min_nucleus_area: float = 100.0,
        max_nucleus_area: float = 50000.0,
        abnormality_score_threshold: float = 0.6,
    ):
        self.circularity_threshold = circularity_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.edge_roughness_threshold = edge_roughness_threshold
        self.min_nucleus_area = min_nucleus_area
        self.max_nucleus_area = max_nucleus_area
        self.abnormality_score_threshold = abnormality_score_threshold

    def compute_circularity(self, area: float, perimeter: float) -> float:
        if perimeter <= 0 or area <= 0:
            return 0.0
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return min(circularity, 1.0)

    def compute_aspect_ratio(
        self, contour: np.ndarray
    ) -> Tuple[float, float, float]:
        if len(contour) < 5:
            return 1.0, 0.0, 0.0
        
        ellipse = cv2.fitEllipse(contour)
        (_, (axis_a, axis_b), _) = ellipse
        
        major_axis = max(axis_a, axis_b)
        minor_axis = min(axis_a, axis_b)
        
        if minor_axis <= 0:
            return 1.0, major_axis, minor_axis
        
        aspect_ratio = major_axis / minor_axis
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0.0
        
        return aspect_ratio, major_axis, minor_axis

    def compute_edge_roughness(self, contour: np.ndarray, perimeter: float) -> float:
        if len(contour) < 6 or perimeter <= 0:
            return 0.0
        
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        
        if hull_perimeter <= 0:
            return 0.0
        
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        roughness = len(approx) / max(len(contour), 1)
        
        convexity_defect = 1.0 - (hull_perimeter / perimeter) if perimeter > 0 else 0.0
        
        edge_roughness = 0.6 * roughness + 0.4 * max(0, convexity_defect)
        
        return min(edge_roughness, 1.0)

    def compute_solidity(self, contour: np.ndarray, area: float) -> float:
        if area <= 0:
            return 0.0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return 0.0
        return area / hull_area

    def compute_convexity(self, contour: np.ndarray, perimeter: float) -> float:
        if perimeter <= 0:
            return 0.0
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        if hull_perimeter <= 0:
            return 0.0
        return hull_perimeter / perimeter

    def compute_form_factor(self, area: float, perimeter: float) -> float:
        if perimeter <= 0 or area <= 0:
            return 0.0
        return (4 * np.pi * area) / (perimeter ** 2)

    def compute_roundness(self, area: float, major_axis: float) -> float:
        if major_axis <= 0 or area <= 0:
            return 0.0
        return (4 * area) / (np.pi * major_axis ** 2)

    def analyze_nucleus(
        self,
        mask: np.ndarray,
        contour: np.ndarray,
        instance_id: int,
        image_offset: Tuple[int, int] = (0, 0),
    ) -> Optional[NucleusFeatures]:
        area = cv2.contourArea(contour)
        
        if area < self.min_nucleus_area or area > self.max_nucleus_area:
            return None
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None
        
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (
            x + image_offset[0],
            y + image_offset[1],
            w,
            h,
        )
        
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = M["m10"] / M["m00"] + image_offset[0]
        cy = M["m01"] / M["m00"] + image_offset[1]
        centroid = (cx, cy)
        
        circularity = self.compute_circularity(area, perimeter)
        aspect_ratio, major_axis, minor_axis = self.compute_aspect_ratio(contour)
        edge_roughness = self.compute_edge_roughness(contour, perimeter)
        solidity = self.compute_solidity(contour, area)
        convexity = self.compute_convexity(contour, perimeter)
        form_factor = self.compute_form_factor(area, perimeter)
        roundness = self.compute_roundness(area, major_axis)
        
        eccentricity = 0.0
        if major_axis > 0:
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if minor_axis <= major_axis else 1.0
        
        is_abnormal, abnormality_score, abnormality_reasons = self._classify_abnormality(
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            edge_roughness=edge_roughness,
            solidity=solidity,
            eccentricity=eccentricity,
        )
        
        feature_vector = self._build_feature_vector(
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            edge_roughness=edge_roughness,
            solidity=solidity,
            convexity=convexity,
            eccentricity=eccentricity,
            form_factor=form_factor,
            roundness=roundness,
        )
        
        return NucleusFeatures(
            instance_id=instance_id,
            contour=contour,
            mask=mask,
            bbox=bbox,
            centroid=centroid,
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            major_axis_length=major_axis,
            minor_axis_length=minor_axis,
            eccentricity=eccentricity,
            edge_roughness=edge_roughness,
            solidity=solidity,
            convexity=convexity,
            form_factor=form_factor,
            roundness=roundness,
            is_abnormal=is_abnormal,
            abnormality_score=abnormality_score,
            abnormality_reasons=abnormality_reasons,
            feature_vector=feature_vector,
        )

    def _classify_abnormality(
        self,
        circularity: float,
        aspect_ratio: float,
        edge_roughness: float,
        solidity: float,
        eccentricity: float,
    ) -> Tuple[bool, float, List[str]]:
        reasons = []
        score = 0.0
        
        if circularity < self.circularity_threshold:
            severity = (self.circularity_threshold - circularity) / self.circularity_threshold
            reasons.append(f"low_circularity({circularity:.3f})")
            score += severity * 0.35
        
        if aspect_ratio > self.aspect_ratio_threshold:
            severity = (aspect_ratio - self.aspect_ratio_threshold) / self.aspect_ratio_threshold
            reasons.append(f"high_aspect_ratio({aspect_ratio:.2f})")
            score += min(severity, 1.0) * 0.25
        
        if edge_roughness > self.edge_roughness_threshold:
            severity = (edge_roughness - self.edge_roughness_threshold) / (1.0 - self.edge_roughness_threshold)
            reasons.append(f"high_edge_roughness({edge_roughness:.3f})")
            score += min(severity, 1.0) * 0.25
        
        if solidity < 0.7:
            severity = (0.7 - solidity) / 0.7
            reasons.append(f"low_solidity({solidity:.3f})")
            score += severity * 0.15
        
        is_abnormal = score >= self.abnormality_score_threshold
        
        return is_abnormal, min(score, 1.0), reasons

    def _build_feature_vector(
        self,
        area: float,
        perimeter: float,
        circularity: float,
        aspect_ratio: float,
        edge_roughness: float,
        solidity: float,
        convexity: float,
        eccentricity: float,
        form_factor: float,
        roundness: float,
    ) -> np.ndarray:
        log_area = np.log1p(area)
        log_perimeter = np.log1p(perimeter)
        
        feature_vector = np.array([
            log_area,
            log_perimeter,
            circularity,
            min(aspect_ratio, 10.0) / 10.0,
            edge_roughness,
            solidity,
            convexity,
            eccentricity,
            form_factor,
            roundness,
        ], dtype=np.float32)
        
        return feature_vector

    def analyze_instances(
        self,
        masks: np.ndarray,
        contours: List[np.ndarray],
        scores: np.ndarray,
        image_offset: Tuple[int, int] = (0, 0),
    ) -> List[NucleusFeatures]:
        results = []
        
        for i in range(len(contours)):
            if len(contours[i]) < 3:
                continue
            
            features = self.analyze_nucleus(
                mask=masks[i],
                contour=contours[i],
                instance_id=i,
                image_offset=image_offset,
            )
            
            if features is not None:
                features.instance_id = i
                results.append(features)
        
        logger.info(
            f"Analyzed {len(results)} nuclei, "
            f"{sum(1 for r in results if r.is_abnormal)} abnormal"
        )
        
        return results

    @staticmethod
    def polygon_to_tuple(contour: np.ndarray) -> List[Tuple[float, float]]:
        points = contour.reshape(-1, 2)
        return [(float(p[0]), float(p[1])) for p in points]
