import cv2
import numpy as np
from fpdf import FPDF
from datetime import datetime
import tempfile

def validate_and_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)

def analyze_faq(images, model):
    """
    Main analysis function for Wheat FAQ.
    """
    # 1. Initialize all variables immediately
    class_map = {0: 'Broken', 1: 'Damage', 2: 'Ergoty Damage', 3: 'Foreign Matter',
                 4: 'Shrivelled', 5: 'Slightly Damage', 6: 'Sound Grain'}

    norms = {'Foreign Matter': 0.75, 'Other Foodgrains': 2.0, 'Damage': 2.0,
             'Slightly Damage': 4.0, 'Ergoty Damage': 0.05, 'Shrivelled & Broken': 6.00}

    master_counts = {name: 0 for name in class_map.values()}
    grand_total = 0
    slice_size = 640
    overlap = 0.25

    # 2. Run the processing loop
    for original_img in images:
        if original_img is None: continue
        
        img = validate_and_enhance(original_img)
        h, w, _ = img.shape
        step = int(slice_size * (1 - overlap))

        for y in range(0, h, step):
            for x in range(0, w, step):
                y2, x2 = min(y + slice_size, h), min(x + slice_size, w)
                tile = img[y:y2, x:x2]
                results = model.predict(tile, conf=0.25, verbose=False)

                for r in results:
                    for box in r.boxes:
                        cls_idx = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = class_map.get(cls_idx, 'Sound Grain')

                        # Shape filtering
                        bw, bh = float(box.xywh[0][2]), float(box.xywh[0][3])
                        pixel_area = bw * bh
                        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)

                        # Vigilance Filters
                        if label == "Ergoty Damage":
                            if conf < 0.96 or pixel_area < 80 or aspect_ratio < 1.6:
                                label = 'Sound Grain'
                        elif label == "Damage" and conf < 0.50:
                            label = 'Sound Grain'

                        master_counts[label] += 1
                        grand_total += 1

    # 3. Calculate final logic
    is_rejected = False
    if grand_total > 0:
        for cat in ['Foreign Matter', 'Other Foodgrains', 'Damage', 'Slightly Damage', 'Ergoty Damage']:
            if (master_counts.get(cat, 0) / grand_total) * 100 > norms[cat]:
                is_rejected = True
        
        sb_p = ((master_counts.get('Shrivelled', 0) + master_counts.get('Broken', 0)) / grand_total) * 100
        if sb_p > 6.0: 
            is_rejected = True

    final_status = "REJECTED" if is_rejected else "ACCEPTED (FAQ)"
    
    # Return 4 items to match your app.py: counts, total, norms, status
    return master_counts, grand_total, norms, final_status
    
def generate_faq_pdf(total, counts, norms, final_status):
    """
    Generates the PDF report and returns the path to the temp file.
    """
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(26, 95, 122)
    pdf.cell(0, 10, "Digital Parkhi 2.0: Official QC Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)

    # Summary Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"TOTAL GRAINS SCANNED: {total}", ln=True)
    pdf.ln(5)

    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(60, 10, " Category", 1, 0, 'L', True)
    pdf.cell(40, 10, " Found %", 1, 0, 'C', True)
    pdf.cell(40, 10, " Limit %", 1, 0, 'C', True)
    pdf.cell(40, 10, " Status", 1, 1, 'C', True)

    # Table Rows
    pdf.set_font("Arial", '', 10)
    categories = ['Foreign Matter', 'Other Foodgrains', 'Damage', 'Slightly Damage', 'Ergoty Damage']
    for cat in categories:
        val = (counts.get(cat, 0) / total) * 100 if total > 0 else 0
        limit = norms[cat]
        status = "OK" if val <= limit else "FAIL"

        pdf.cell(60, 10, f" {cat}", 1)
        pdf.cell(40, 10, f"{val:.2f}%", 1, 0, 'C')
        pdf.cell(40, 10, f"{limit}%", 1, 0, 'C')

        if status == "FAIL": pdf.set_text_color(200, 0, 0)
        else: pdf.set_text_color(0, 128, 0)
        pdf.cell(40, 10, status, 1, 1, 'C')
        pdf.set_text_color(0, 0, 0)

    # S&B Grouping
    sb_val = ((counts.get('Shrivelled', 0) + counts.get('Broken', 0)) / total) * 100 if total > 0 else 0
    sb_status = "OK" if sb_val <= 6.0 else "FAIL"
    pdf.cell(60, 10, " Shrivelled & Broken", 1)
    pdf.cell(40, 10, f"{sb_val:.2f}%", 1, 0, 'C')
    pdf.cell(40, 10, " 6.00%", 1, 0, 'C')
    pdf.cell(40, 10, sb_status, 1, 1, 'C')

    pdf.ln(15)
    pdf.set_font("Arial", 'B', 14)
    if final_status == "ACCEPTED (FAQ)": pdf.set_text_color(0, 128, 0)
    else: pdf.set_text_color(200, 0, 0)
    pdf.cell(0, 15, f"RESULT: {final_status}", border=1, ln=True, align='C')

    # Save to a temporary file for Streamlit download
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name
