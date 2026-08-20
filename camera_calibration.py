"""
Einfache Kamerakalibrierung mit einem Schachbrettmuster.
Ausdrucken: ein Schachbrettmuster mit bekannter Anzahl innerer Ecken
(z.B. 9x6) auf steifes Material kleben, damit es nicht wellig ist.
"""
import cv2 as cv
import numpy as np
import glob

def calibrate_camera(image_folder, checkerboard_size=(9, 6), square_size_mm=25.0):
    """Kalibriert die Kamera anhand mehrerer Fotos eines Schachbrettmusters.

    Args:
        image_folder: Ordner mit Kalibrierfotos (z.B. "calib_images/*.jpg")
        checkerboard_size: (Spalten, Zeilen) an INNEREN Ecken - nicht die
            Anzahl der Felder! Ein 10x7-Felder-Schachbrett hat 9x6 innere Ecken.
        square_size_mm: Kantenlaenge eines einzelnen Feldes in mm (ausmessen!)

    Returns:
        dict mit K, dist_coeffs, new_K (optimiert), roi, rms_error
        oder None, falls keine gueltigen Schachbrett-Erkennungen gefunden wurden.
    """
    # 3D-Referenzpunkte des Schachbretts: liegt flach in der Ebene Z=0,
    # X/Y in echten mm entsprechend der Feldgroesse
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints = []  # 3D-Punkte (gleich fuer jedes Bild)
    imgpoints = []  # 2D-Punkte (pro Bild unterschiedlich, wo die Ecken tatsaechlich liegen)
    img_size = None

    image_paths = glob.glob(image_folder)
    print(f"{len(image_paths)} Bilder gefunden")

    for path in image_paths:
        img = cv.imread(path)
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (Breite, Hoehe)

        found, corners = cv.findChessboardCorners(gray, checkerboard_size, None)
        if not found:
            print(f"  {path}: kein Schachbrett gefunden, uebersprungen")
            continue

        # Ecken auf Sub-Pixel-Genauigkeit verfeinern -> praeziseres Ergebnis
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners_refined)
        print(f"  {path}: OK")

    if len(objpoints) < 10:
        print(f"WARNUNG: nur {len(objpoints)} gueltige Bilder - fuer eine stabile "
              f"Kalibrierung werden mindestens 10-15 empfohlen (aus unterschiedlichen "
              f"Winkeln/Positionen/Distanzen)")
    if len(objpoints) == 0:
        print("Keine gueltigen Bilder - Kalibrierung nicht moeglich")
        return None

    # Kernschritt: aus allen Korrespondenzen K, Verzerrungskoeffizienten und
    # die Pose jedes einzelnen Kalibrierbilds berechnen
    rms_error, K, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # optimierte Kameramatrix fuer die Entzerrung berechnen:
    # alpha=0 -> alle ungueltigen (schwarzen) Randpixel nach dem Entzerren croppen
    # alpha=1 -> ganzes Originalbild behalten, inkl. schwarzer Raender
    new_K, roi = cv.getOptimalNewCameraMatrix(
        K, dist_coeffs, img_size, alpha=0, newImgSize=img_size
    )

    print(f"\nKalibrierung abgeschlossen mit {len(objpoints)} Bildern")
    print(f"RMS Reprojection Error: {rms_error:.4f} px  (unter ~0.5 ist gut, "
          f"deutlich darueber -> Bilder/Eckenerkennung pruefen)")

    return {
        "K": K,
        "dist_coeffs": dist_coeffs,
        "new_K": new_K,
        "roi": roi,
        "rms_error": rms_error,
        "image_size": img_size,
    }

def undistort_frame(frame, calib):
    """Entzerrt ein Kamerabild mit den Ergebnissen von calibrate_camera().
    Auf diesem entzerrten Bild sind gerade Linien wieder gerade -
    wichtig fuer praezise Eckenerkennung deiner Marker."""
    undistorted = cv.undistort(frame, calib["K"], calib["dist_coeffs"], None, calib["new_K"])
    x, y, w, h = calib["roi"]
    if w > 0 and h > 0:
        undistorted = undistorted[y:y+h, x:x+w]
    return undistorted

if __name__ == "__main__":
    # Beispielaufruf: Kalibrierfotos vorher mit der Kamera aufnehmen und in
    # einen Ordner legen, z.B. calib_images/img01.jpg, img02.jpg, ...
    result = calibrate_camera("calib_images/*.jpg", checkerboard_size=(9, 6), square_size_mm=25.0)

    if result is not None:
        np.savez("camera_calib.npz",
                 K=result["K"],
                 dist_coeffs=result["dist_coeffs"],
                 new_K=result["new_K"],
                 roi=result["roi"])
        print("\nGespeichert in camera_calib.npz")
        print("K:\n", result["K"])
        print("dist_coeffs:", result["dist_coeffs"].ravel())