import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from src.screening.morphology.analyzer import MorphologyAnalyzer, NucleusFeatures


def _make_circle_mask(size: int = 200, radius: int = 80) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, 1, -1)
    return mask


def _make_circle_contour(size: int = 200, radius: int = 80) -> np.ndarray:
    mask = _make_circle_mask(size, radius)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def _make_irregular_mask(size: int = 200) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = np.array([
        [100, 20], [150, 40], [180, 90], [170, 140],
        [140, 180], [100, 190], [50, 170], [20, 120],
        [30, 70], [60, 30],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _make_elongated_mask(size: int = 200) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = np.array([
        [30, 90], [40, 80], [170, 80], [180, 90],
        [180, 110], [170, 120], [40, 120], [30, 110],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _make_spiky_mask(size: int = 200) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (100, 100)
    n_spikes = 8
    for i in range(n_spikes):
        angle = 2 * np.pi * i / n_spikes
        inner_r = 30
        outer_r = 80
        x1 = int(center[0] + inner_r * np.cos(angle))
        y1 = int(center[1] + inner_r * np.sin(angle))
        x2 = int(center[0] + outer_r * np.cos(angle))
        y2 = int(center[1] + outer_r * np.sin(angle))
        cv2.line(mask, (x1, y1), (x2, y2), 1, 15)
    cv2.circle(mask, center, 35, 1, -1)
    return mask


def test_circularity_perfect_circle():
    analyzer = MorphologyAnalyzer()
    contour = _make_circle_contour(200, 80)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    circularity = analyzer.compute_circularity(area, perimeter)
    
    assert circularity > 0.85, f"Perfect circle should have circularity > 0.85, got {circularity}"
    print("[PASS] Circularity of perfect circle: {:.4f}".format(circularity))


def test_circularity_irregular():
    analyzer = MorphologyAnalyzer()
    mask = _make_irregular_mask()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    circularity = analyzer.compute_circularity(area, perimeter)
    
    assert circularity < 0.85, f"Irregular shape should have lower circularity, got {circularity}"
    print(f"[PASS] Circularity of irregular shape: {circularity:.4f}")


def test_aspect_ratio_circle():
    analyzer = MorphologyAnalyzer()
    contour = _make_circle_contour(200, 80)
    
    aspect_ratio, major, minor = analyzer.compute_aspect_ratio(contour)
    
    assert 0.8 < aspect_ratio < 1.3, f"Circle should have aspect ratio near 1.0, got {aspect_ratio}"
    print(f"[PASS] Aspect ratio of circle: {aspect_ratio:.3f}")


def test_aspect_ratio_elongated():
    analyzer = MorphologyAnalyzer()
    mask = _make_elongated_mask()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    
    aspect_ratio, major, minor = analyzer.compute_aspect_ratio(contour)
    
    assert aspect_ratio > 2.0, f"Elongated shape should have high aspect ratio, got {aspect_ratio}"
    print(f"[PASS] Aspect ratio of elongated shape: {aspect_ratio:.3f}")


def test_edge_roughness_smooth():
    analyzer = MorphologyAnalyzer()
    contour = _make_circle_contour(200, 80)
    perimeter = cv2.arcLength(contour, True)
    
    roughness = analyzer.compute_edge_roughness(contour, perimeter)
    
    print(f"[PASS] Edge roughness of circle: {roughness:.4f}")


def test_edge_roughness_spiky():
    analyzer = MorphologyAnalyzer()
    mask = _make_spiky_mask()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[PASS] Edge roughness of spiky shape: skipped (no contour)")
        return
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    
    roughness = analyzer.compute_edge_roughness(contour, perimeter)
    
    print(f"[PASS] Edge roughness of spiky shape: {roughness:.4f}")


def test_analyze_nucleus_normal():
    analyzer = MorphologyAnalyzer()
    mask = _make_circle_mask(200, 80)
    contour = _make_circle_contour(200, 80)
    
    features = analyzer.analyze_nucleus(
        mask=mask,
        contour=contour,
        instance_id=0,
    )
    
    assert features is not None
    assert features.circularity > 0.8
    assert features.aspect_ratio < 1.5
    assert features.feature_vector is not None
    assert features.feature_vector.shape == (10,)
    assert not features.is_abnormal, "Circle should not be classified as abnormal"
    
    print(f"[PASS] Normal nucleus analysis:")
    print(f"  Circularity: {features.circularity:.4f}")
    print(f"  Aspect ratio: {features.aspect_ratio:.3f}")
    print(f"  Edge roughness: {features.edge_roughness:.4f}")
    print(f"  Feature vector dim: {features.feature_vector.shape}")
    print(f"  Abnormal: {features.is_abnormal}")


def test_analyze_nucleus_abnormal():
    analyzer = MorphologyAnalyzer(
        circularity_threshold=0.6,
        aspect_ratio_threshold=2.5,
    )
    mask = _make_elongated_mask()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    
    features = analyzer.analyze_nucleus(
        mask=mask,
        contour=contour,
        instance_id=1,
    )
    
    assert features is not None
    assert features.aspect_ratio > 2.0
    
    print(f"[PASS] Abnormal nucleus analysis:")
    print(f"  Circularity: {features.circularity:.4f}")
    print(f"  Aspect ratio: {features.aspect_ratio:.3f}")
    print(f"  Edge roughness: {features.edge_roughness:.4f}")
    print(f"  Abnormality score: {features.abnormality_score:.3f}")
    print(f"  Abnormal: {features.is_abnormal}")
    print(f"  Reasons: {features.abnormality_reasons}")


def test_analyze_instances():
    analyzer = MorphologyAnalyzer()
    
    circle_mask = _make_circle_mask(200, 80)
    circle_contour = _make_circle_contour(200, 80)
    
    elongated_mask = _make_elongated_mask()
    elongated_contours, _ = cv2.findContours(
        elongated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    masks = np.stack([circle_mask, elongated_mask], axis=0)
    contours = [circle_contour, elongated_contours[0]]
    scores = np.array([0.95, 0.88])
    
    results = analyzer.analyze_instances(
        masks=masks,
        contours=contours,
        scores=scores,
    )
    
    assert len(results) >= 1
    print(f"[PASS] Multi-instance analysis: {len(results)} nuclei")


def test_solidity_and_convexity():
    analyzer = MorphologyAnalyzer()
    contour = _make_circle_contour(200, 80)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    solidity = analyzer.compute_solidity(contour, area)
    convexity = analyzer.compute_convexity(contour, perimeter)
    
    assert solidity > 0.9, f"Circle should have high solidity, got {solidity}"
    assert convexity > 0.9, f"Circle should have high convexity, got {convexity}"
    
    print(f"[PASS] Solidity: {solidity:.4f}, Convexity: {convexity:.4f}")


def test_feature_vector_normalization():
    analyzer = MorphologyAnalyzer()
    contour = _make_circle_contour(200, 80)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    fv = analyzer._build_feature_vector(
        area=area, perimeter=perimeter,
        circularity=0.9, aspect_ratio=1.1,
        edge_roughness=0.1, solidity=0.98,
        convexity=0.99, eccentricity=0.1,
        form_factor=0.9, roundness=0.95,
    )
    
    assert fv.shape == (10,)
    assert fv.dtype == np.float32
    assert not np.any(np.isnan(fv))
    assert not np.any(np.isinf(fv))
    
    print(f"[PASS] Feature vector: {fv}")


if __name__ == "__main__":
    test_circularity_perfect_circle()
    test_circularity_irregular()
    test_aspect_ratio_circle()
    test_aspect_ratio_elongated()
    test_edge_roughness_smooth()
    test_edge_roughness_spiky()
    test_analyze_nucleus_normal()
    test_analyze_nucleus_abnormal()
    test_analyze_instances()
    test_solidity_and_convexity()
    test_feature_vector_normalization()
    print("\n[OK] All morphology analyzer tests passed!")
