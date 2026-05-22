import cv2
import numpy as np
from collections import Counter
from fpdf import FPDF
from datetime import datetime

WHEAT_URS_NORMS = {
    'Foreign Matter': 0.75,
    'Other Foodgrains': 2.0,
    'Shrivelled & Broken': 15.00,
    'Damage & Slightly Damage': 6.00,
    'Lustre Loss': 70.00
}

CLASS_MAP = {
    0: 'Broken',
    1: 'Damage',
    2: 'Foreign Matter',
    3: 'Shrivelled',
    4: 'Sound Grain'
}
# =========================================================
# IMAGE SEGMENTATION
# =========================================================

def segment_grains(image):

    # STEP 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # STEP 2
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )

    # STEP 3
    kernel = np.ones((3,3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # STEP 4
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    grain_boxes = []

    img_h, img_w = image.shape[:2]

    # STEP 5
    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 40:
            continue

        if area > 5000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Reject border objects
        if x <= 5 or y <= 5 or (x+w) >= img_w-5 or (y+h) >= img_h-5:
            continue

        aspect_ratio = h / float(w + 1e-6)

        if aspect_ratio < 1.2:
            continue

        grain_boxes.append((x, y, w, h))

    return grain_boxes

# =========================================================
# GRAIN CLASSIFICATION
# =========================================================

def classify_grain(cnt, roi_bgr, roi_gray):

    area = cv2.contourArea(cnt)

    if area < 25:
        return None

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

    hull = cv2.convexHull(cnt)

    hull_area = cv2.contourArea(hull)

    solidity = area / (hull_area + 1e-6)

    # ---------------------------
    # COLOR FEATURES
    # ---------------------------

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    mean_h = np.mean(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])

    # ---------------------------
    # TEXTURE FEATURES
    # ---------------------------

    lap_var = cv2.Laplacian(
        roi_gray,
        cv2.CV_64F
    ).var()

    std_intensity = np.std(roi_gray)

    # ---------------------------
    # EDGE FEATURES
    # ---------------------------

    edges = cv2.Canny(roi_gray, 50, 150)

    edge_density = np.sum(edges > 0) / (roi_gray.size + 1e-6)

    # =====================================================
    # FOREIGN MATTER
    # =====================================================

    if (
        mean_v < 85
        and mean_s < 25
    ):
        return 'Foreign Matter'

    # =====================================================
    # BROKEN
    # =====================================================

    if area < 95:
        return 'Broken'

    # =====================================================
    # SHRIVELLED
    # =====================================================

    shrivel_score = 0

    if aspect_ratio > 3.0:
        shrivel_score += 1

    if solidity < 0.82:
        shrivel_score += 1

    if mean_v < 125:
        shrivel_score += 1

    if shrivel_score >= 2:
        return 'Shrivelled'

    # =====================================================
    # DAMAGE
    # =====================================================

    damage_score = 0

    if lap_var > 450:
        damage_score += 1

    if edge_density > 0.18:
        damage_score += 1

    if std_intensity > 42:
        damage_score += 1

    if mean_h < 18:
        damage_score += 1

    if damage_score >= 3:
        return 'Damage'

    # =====================================================
    # LUSTRE LOSS
    # =====================================================

    lustre_score = 0

    if mean_s < 38:
        lustre_score += 1

    if std_intensity < 26:
        lustre_score += 1

    if mean_v > 145:
        lustre_score += 1

    if lustre_score >= 2:
        return 'Lustre Loss'

    return 'Sound Grain'
# =========================================================
# =========================================================
# MAIN ANALYSIS FUNCTION
# =========================================================

def analyze_sample(cv_img, model=None):

    annotated = cv_img.copy()

    contours = segment_grains(cv_img)

    labels = []

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        roi_gray = gray[y:y+h, x:x+w]

        
        roi_bgr = cv_img[y:y+h, x:x+w]

        label = classify_grain(
            cnt,
            roi_bgr,
            roi_gray
        )

        if label is None:
            continue

        labels.append(label)

        # Draw box
        color = (0, 255, 0)

        if label == 'Broken':
            color = (255, 0, 255)

        elif label == 'Shrivelled':
            color = (0, 255, 255)

        elif label == 'Lustre Loss':
            color = (255, 255, 255)

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            annotated,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    return labels, annotated

# =========================================================
# PDF REPORT
# =========================================================

def generate_pdf(total, counts, final_status):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Digital Parkhi - URS QC Report", ln=True, align='C')

    pdf.ln(10)

    pdf.set_font("Arial", '', 11)

    pdf.cell(0, 10, f"Total Grains: {total}", ln=True)

    pdf.ln(5)

    for cat, limit in WHEAT_URS_NORMS.items():

        if cat == 'Shrivelled & Broken':
            val = (
                counts.get('Shrivelled', 0) +
                counts.get('Broken', 0)
            ) / total * 100 if total > 0 else 0

        elif cat == 'Damage & Slightly Damage':
            val = (
                counts.get('Damage', 0) +
                counts.get('Slightly Damage', 0)
            ) / total * 100 if total > 0 else 0

        else:
            val = counts.get(cat, 0) / total * 100 if total > 0 else 0

        status = "OK" if val <= limit else "FAIL"

        pdf.cell(
            0,
            10,
            f"{cat}: {val:.2f}% | Limit {limit}% | {status}",
            ln=True
        )

    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"FINAL STATUS: {final_status}", ln=True)

    return pdf.output(dest='S').encode('latin-1')
