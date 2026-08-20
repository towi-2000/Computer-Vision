"""Typisierte Version von marker_lib.py - fuer mypy-Check."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2 as cv
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

@dataclass
class Marker:
    """Ein erkannter verschachtelter Quadrat-Marker."""
    center: tuple[float, float]
    area: float

def find_markers(gray_img: MatLike, min_area: int = 20) -> list[Marker]:
    """Findet verschachtelte Quadrat-Marker (schwarz-weiss-schwarz, 3 Ebenen)."""
    _, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    h = hierarchy[0]

    markers: list[Marker] = []
    for i, cnt in enumerate(contours):
        if h[i][3] != -1:
            continue
        child = h[i][2]
        if child == -1:
            continue
        grandchild = h[child][2]
        if grandchild == -1:
            continue

        area = cv.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4 or not cv.isContourConvex(approx):
            continue

        M = cv.moments(cnt)
        if M['m00'] == 0:
            continue
        cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
        markers.append(Marker(center=(cx, cy), area=area))

    return markers


def order_markers_canonically(markers: list[Marker]) -> Optional[list[Marker]]:
    """Ordnet genau 4 Marker eindeutig: [klein, dann Umlaufsinn]."""
    if len(markers) != 4:
        return None

    sorted_by_area = sorted(markers, key=lambda m: m.area)
    small = sorted_by_area[0]
    others = sorted_by_area[1:]

    areas_others = [m.area for m in others]
    if small.area > 0.7 * min(areas_others):
        return None

    cx = float(np.mean([m.center[0] for m in markers]))
    cy = float(np.mean([m.center[1] for m in markers]))

    def angle(m: Marker) -> float:
        return float(np.arctan2(m.center[1] - cy, m.center[0] - cx))

    small_angle = angle(small)

    def angle_diff(m: Marker) -> float:
        return (angle(m) - small_angle) % (2 * np.pi)

    others_sorted = sorted(others, key=angle_diff)
    return [small] + others_sorted


def get_reference_marker_layout(target_gray: MatLike) -> npt.NDArray[np.float32]:
    """Positionen der 4 Marker im Referenzbild, kanonisch geordnet."""
    markers = find_markers(target_gray)
    ordered = order_markers_canonically(markers)
    if ordered is None:
        raise RuntimeError(
            f"Referenzbild: {len(markers)} statt 4 Marker gefunden oder Zuordnung unsicher"
        )
    return np.array([m.center for m in ordered], dtype=np.float32)


def find_target_homography(
    frame_gray: MatLike, reference_pts: npt.NDArray[np.float32]
) -> Optional[MatLike]:
    """Sucht die 4 Marker im Frame, berechnet Homographie Referenz -> Frame."""
    markers = find_markers(frame_gray)
    ordered = order_markers_canonically(markers)
    if ordered is None:
        return None

    frame_pts = np.array([m.center for m in ordered], dtype=np.float32)
    H = cv.getPerspectiveTransform(reference_pts, frame_pts)
    return H


class TargetTracker:
    """Glaettet die Zielposition per Kalman-Filter, ueberbrueckt Aussetzer."""

    def __init__(
        self,
        max_lost_frames: int = 5,
        process_noise: float = 0.03,
        measurement_noise: float = 1.0,
    ) -> None:
        self.max_lost_frames: int = max_lost_frames
        self.lost_count: int = 0
        self.initialized: bool = False

        self.kf: cv.KalmanFilter = cv.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

    def update(
        self, measured_center: Optional[tuple[float, float]]
    ) -> Optional[tuple[float, float]]:
        if measured_center is not None:
            self.lost_count = 0
            mx, my = measured_center

            if not self.initialized:
                self.kf.statePost = np.array([[mx], [my], [0], [0]], dtype=np.float32)
                self.kf.errorCovPost = np.eye(4, dtype=np.float32)
                self.initialized = True
                return (mx, my)

            self.kf.predict()
            corrected = self.kf.correct(np.array([[mx], [my]], dtype=np.float32))
            return (float(corrected[0, 0]), float(corrected[1, 0]))

        else:
            if not self.initialized:
                return None

            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                self.initialized = False
                return None

            pred = self.kf.predict()
            return (float(pred[0, 0]), float(pred[1, 0]))
