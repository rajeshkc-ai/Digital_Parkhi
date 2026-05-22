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
    """Detect grains using classical computer vision."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary inverse threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphology cleanup
    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours

# =========================================================
# GRAIN CLASSIFICATION
# =========================================================

def classify_grain(cnt, roi_gray):
    """Classify grain using morphology."""

    area = cv2.contourArea(cnt)

    if area < 20:
        return None

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    solidity = area / (hull_area + 1e-6)

    mean_intensity = np.mean(roi_gray)

    # Broken grain
    if area < 120:
        return 'Broken'

    # Shrivelled grain
    if solidity < 0.82 or aspect_ratio > 2.8:
        return 'Shrivelled'

    # Lustre loss
    if mean_intensity > 170:
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

        label = classify_grain(cnt, roi_gray)

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
