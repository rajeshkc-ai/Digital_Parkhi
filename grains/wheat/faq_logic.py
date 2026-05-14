import cv2
import numpy as np
from fpdf import FPDF
from datetime import datetime

# FCI Standards for RMS 2025-26
WHEAT_NORMS = {
    'Foreign Matter': 0.75,
    'Other Foodgrains': 2.0,
    'Damage': 2.0,
    'Slightly Damage': 4.0,
    'Ergoty Damage': 0.05,
    'Shrivelled & Broken': 6.00
}

CLASS_MAP = {0: 'Broken', 1: 'Damage', 2: 'Ergoty Damage', 3: 'Foreign Matter',
             4: 'Shrivelled', 5: 'Slightly Damage', 6: 'Sound Grain'}

def analyze_sample(cv_img, model):
    """Performs CLAHE enhancement and Sliced Inference"""
    # 1. Enhance
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    # 2. Slice & Predict
    h, w, _ = img.shape
    slice_size = 640
    step = int(slice_size * 0.75) # 25% overlap
    results_list = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            y2, x2 = min(y + slice_size, h), min(x + slice_size, w)
            tile = img[y:y2, x:x2]
            preds = model.predict(tile, conf=0.25, verbose=False)
            for r in preds:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = CLASS_MAP.get(cls)
                    bw, bh = float(box.xywh[0][2]), float(box.xywh[0][3])
                    
                    # Vigilance Filters
                    if label == "Ergoty Damage":
                        if conf < 0.96 or (bw*bh) < 80 or (max(bw,bh)/(min(bw,bh)+1e-6)) < 1.6:
                            cls = 6
                    elif label == "Damage" and conf < 0.50:
                        cls = 6
                    results_list.append(cls)
    return results_list

def generate_faq_pdf(total, counts, final_status):
    """Generates the bytes for the PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Digital Parkhi 2.0: Official FAQ QC Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"TOTAL GRAINS SCANNED: {total}", ln=True)
    pdf.ln(5)

    # Table
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(60, 10, " Category", 1, 0, 'L', True)
    pdf.cell(40, 10, " Found %", 1, 0, 'C', True)
    pdf.cell(40, 10, " Limit %", 1, 0, 'C', True)
    pdf.cell(40, 10, " Status", 1, 1, 'C', True)

    pdf.set_font("Arial", '', 10)
    for cat, limit in WHEAT_NORMS.items():
        if cat == 'Shrivelled & Broken':
            val = ((counts.get('Shrivelled', 0) + counts.get('Broken', 0)) / total * 100)
        else:
            val = (counts.get(cat, 0) / total * 100)
        
        status = "OK" if val <= limit else "FAIL"
        pdf.cell(60, 10, f" {cat}", 1)
        pdf.cell(40, 10, f"{val:.2f}%", 1, 0, 'C')
        pdf.cell(40, 10, f"{limit}%", 1, 0, 'C')
        pdf.cell(40, 10, status, 1, 1, 'C')

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 15, f"FINAL RESULT: {final_status}", border=1, ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1')