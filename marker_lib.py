import cv2 as cv
import numpy as np

def find_markers(gray_img, min_area=20):
    """Findet verschachtelte Quadrat-Marker (schwarz-weiss-schwarz, 3 Ebenen).
    Gibt Liste von dicts mit 'center' (x,y) und 'area' zurueck."""
    _, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    h = hierarchy[0]

    markers = []
    for i, cnt in enumerate(contours):
        if h[i][3] != -1:
            continue  # nur oberste Ebene (aeusseres schwarzes Quadrat)
        child = h[i][2]
        if child == -1:
            continue
        grandchild = h[child][2]
        if grandchild == -1:
            continue  # braucht 3 Ebenen: schwarz -> weiss -> schwarz

        area = cv.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4 or not cv.isContourConvex(approx):
            continue  # muss ein Viereck sein (perspektivinvariant: immer 4 Ecken)

        M = cv.moments(cnt)
        if M['m00'] == 0:
            continue
        cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
        markers.append({'center': (cx, cy), 'area': area})

    return markers


def order_markers_canonically(markers):
    """Ordnet genau 4 Marker eindeutig: [klein, dann die restlichen 3 im
    Winkel-Umlaufsinn ab dem kleinen Marker]. Funktioniert unabhaengig von
    Rotation/Perspektive, solange der kleine Marker eindeutig identifizierbar ist."""
    if len(markers) != 4:
        return None

    # kleinster Marker per Flaeche identifizieren
    sorted_by_area = sorted(markers, key=lambda m: m['area'])
    small = sorted_by_area[0]
    others = sorted_by_area[1:]

    # Plausibilitaetscheck: die anderen 3 sollten sich in der Flaeche aehneln,
    # und der kleine sollte sich deutlich abheben
    areas_others = [m['area'] for m in others]
    if small['area'] > 0.7 * min(areas_others):
        return None  # kein klarer Groessenunterschied -> unsichere Zuordnung

    # Zentroid aller 4 Punkte als Bezugspunkt fuer Winkelberechnung
    cx = np.mean([m['center'][0] for m in markers])
    cy = np.mean([m['center'][1] for m in markers])

    def angle(m):
        return np.arctan2(m['center'][1] - cy, m['center'][0] - cx)

    small_angle = angle(small)
    # die restlichen 3 nach Winkel-Differenz zum kleinen Marker sortieren
    # (im mathematisch positiven Umlaufsinn, beginnend direkt nach 'small')
    def angle_diff(m):
        d = angle(m) - small_angle
        return d % (2 * np.pi)

    others_sorted = sorted(others, key=angle_diff)
    return [small] + others_sorted

def get_reference_marker_layout(target_gray):
    """Einmalig beim Programmstart: Positionen der 4 Marker im Referenzbild
    (target.png) in kanonischer Reihenfolge ermitteln."""
    markers = find_markers(target_gray)
    ordered = order_markers_canonically(markers)
    if ordered is None:
        raise RuntimeError(f"Referenzbild: {len(markers)} statt 4 Marker gefunden oder Zuordnung unsicher")
    return np.float32([m['center'] for m in ordered])

def find_target_homography(frame_gray, reference_pts):
    """Sucht die 4 Marker im aktuellen Frame und berechnet die Homographie
    vom Referenzbild-Koordinatensystem in das Kamerabild."""
    markers = find_markers(frame_gray)
    ordered = order_markers_canonically(markers)
    if ordered is None:
        return None

    frame_pts = np.float32([m['center'] for m in ordered])
    H = cv.getPerspectiveTransform(reference_pts, frame_pts)
    return H

class TargetTracker:
    """Glaettet die per Marker-Erkennung gemessene Zielposition und
    ueberbrueckt kurze Aussetzer (z.B. 1-2 Frames mit Bewegungsunschaerfe),
    ohne selbst Feature-Matching oder Optical Flow zu benoetigen - die
    eigentliche Erkennung (find_target_homography) laeuft weiterhin jeden
    Frame neu."""

    def __init__(self, max_lost_frames=5, process_noise=0.03, measurement_noise=1.0):
        self.max_lost_frames = max_lost_frames
        self.lost_count = 0
        self.initialized = False

        self.kf = cv.KalmanFilter(4, 2)  # Zustand: [x, y, vx, vy], Messung: [x, y]
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

    def update(self, measured_center):
        """measured_center: (x, y) aus find_target_homography, oder None
        wenn in diesem Frame kein Marker erkannt wurde.
        Gibt (x, y) oder None zurueck (None nur wenn endgueltig verloren)."""

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
                return None  # noch nie etwas erkannt

            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                self.initialized = False
                return None  # endgueltig verloren -> Aufrufer muss neu suchen/warten

            # kurzzeitig verloren: reine Vorhersage ohne Messkorrektur
            pred = self.kf.predict()
            return (float(pred[0, 0]), float(pred[1, 0]))
