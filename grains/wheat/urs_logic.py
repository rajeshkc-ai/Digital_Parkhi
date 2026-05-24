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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # PREPROCESSING
    # ==========================================

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # ==========================================
    # DISTANCE TRANSFORM
    # ==========================================

    dist = cv2.distanceTransform(
        thresh,
        cv2.DIST_L2,
        5
    )

    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(
        dist,
        0.32,
        1.0,
        cv2.THRESH_BINARY
    )

    sure_fg = np.uint8(sure_fg * 255)

    sure_bg = cv2.dilate(
        thresh,
        kernel,
        iterations=2
    )

    unknown = cv2.subtract(sure_bg, sure_fg)

    # ==========================================
    # WATERSHED
    # ==========================================

    num_markers, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1

    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)

    grain_boxes = []

    for marker in np.unique(markers):

        if marker <= 1:
            continue

        mask = np.uint8(markers == marker)

        kernel = np.ones((2,2), np.uint8)

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )
        
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        cnt = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(cnt)

        # Reject tiny dust
        if area < 70:
            continue

        # Reject merged huge regions
        if area > 1400:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

        # Reject non-grain shapes
        if aspect_ratio < 1.3:
            continue

        # Wheat grains are elongated
        aspect_ratio = max(h, w) / (min(h, w) + 1e-6)

        if aspect_ratio < 1.4:
            continue
        
        # Ignore extremely thin noise
        if w < 8 or h < 18:
            continue

        # Reject merged grains
        if w > 70 or h > 70:
            continue

        grain_boxes.append((x, y, w, h))

    return grain_boxes

# =========================================================
# GRAIN CLASSIFICATION
# =========================================================

def classify_grain(cnt, roi_bgr, roi_gray):

    import cv2
    import numpy as np

    # -------------------------------------------------
    # BASIC FEATURES
    # -------------------------------------------------

    area = cv2.contourArea(cnt)

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = h / (w + 1e-6)

    perimeter = cv2.arcLength(cnt, True)

    circularity = (4 * np.pi * area) / ((perimeter * perimeter) + 1e-6)

    # -------------------------------------------------
    # HSV FEATURES
    # -------------------------------------------------

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    # -------------------------------------------------
    # GRAY FEATURES
    # -------------------------------------------------

    gray_mean = np.mean(roi_gray)

    gray_std = np.std(roi_gray)

    # -------------------------------------------------
    # EDGE FEATURES
    # -------------------------------------------------

    edges = cv2.Canny(roi_gray, 60, 160)

    edge_density = np.sum(edges > 0) / (roi_gray.size + 1e-6)

    # -------------------------------------------------
    # TEXTURE FEATURES
    # -------------------------------------------------

    lap_var = cv2.Laplacian(roi_gray, cv2.CV_64F).var()

    # =================================================
    # CLASSIFICATION RULES
    # =================================================

    # -------------------------------------------------
    # SHRIVELLED / BROKEN
    # -------------------------------------------------

    if (
        area < 110
        or aspect_ratio < 1.25
        or circularity > 0.58
    ):
        return "Shrivelled"

    # -------------------------------------------------
    # LUSTRE LOSS
    # -------------------------------------------------

    if (
        s_mean < 52
        and v_mean > 145
        and gray_std < 32
    ):
        return "Lustre Loss"

    # -------------------------------------------------
    # DAMAGE
    # -------------------------------------------------

    if (
        edge_density > 0.12
        and lap_var > 210
        and gray_std > 26
    ):
        return "Damage"

    # -------------------------------------------------
    # SOUND GRAIN
    # -------------------------------------------------

    return "Sound Grain"
    
# =========================================================
# =========================================================
# MAIN ANALYSIS FUNCTION
# =========================================================

def analyze_sample(cv_img, model=None):

    annotated = cv_img.copy()

    contours = segment_grains(cv_img)

    labels = []

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in contours:

        roi_gray = gray[y:y+h, x:x+w]

        roi_bgr = cv_img[y:y+h, x:x+w]

        # Create fake contour from box
        cnt = np.array([
            [[x, y]],
            [[x+w, y]],
            [[x+w, y+h]],
            [[x, y+h]]
        ])

        label = classify_grain(
            cnt,
            roi_bgr,
            roi_gray
            )

        roi_gray = gray[y:y+h, x:x+w]

        
        roi_bgr = cv_img[y:y+h, x:x+w]


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
