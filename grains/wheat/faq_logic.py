import cv2
import numpy as np
import torch  # Required for handling NMS coordinates efficiently
from fpdf import FPDF
from datetime import datetime

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
    """Performs CLAHE enhancement and Sliced Inference with global NMS tracking"""
    # 1. Enhance Contrast safely
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    h, w, _ = img.shape
    slice_size = 640
    step = int(slice_size * 0.75) # 25% overlap
    results_list = []
    
    global_boxes = []
    global_confs = []
    global_classes = []

    # 2. Slice Processing Loop
    for y in range(0, h, step):
        for x in range(0, w, step):
            y2, x2 = min(y + slice_size, h), min(x + slice_size, w)
            tile = img[y:y2, x:x2]
            
            # Keep confidence functional baseline at 0.20 to capture small/obscured variants
            preds = model.predict(tile, conf=0.15, verbose=False)
            
            for r in preds:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = CLASS_MAP.get(cls)
                    bw, bh = float(box.xywh[0][2]), float(box.xywh[0][3])
                    # Convert tile-relative coordinates back to global canvas scale coordinates
                    bx_c, by_c, bw, bh = map(float, box.xywh[0])
                    global_x = x + (bx_c - bw/2)
                    global_y = y + (by_c - bh/2)
                    
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    global_boxes.append([global_x, global_y, global_x + bw, global_y + bh])
                    global_confs.append(conf)
                    global_classes.append(cls)

    if not global_boxes:
        return []

    # 3. Apply Global Non-Maximum Suppression to completely wipe out overlap duplicates
    boxes_t = torch.tensor(global_boxes)
    confs_t = torch.tensor(global_confs)
    keep_indices = torch.ops.torchvision.nms(boxes_t, confs_t, iou_threshold=0.3)

    final_labels_list = []
    
    # 4. Process deduplicated predictions through vigilance filters
    for idx in keep_indices:
        cls = global_classes[idx]
        conf = global_confs[idx]
        label = CLASS_MAP.get(cls)
        
        # Calculate bounding dimensions from saved global coordinates
        x1, y1, x2, y2 = global_boxes[idx]
        bw, bh = x2 - x1, y2 - y1

        # Apply corrections directly to the class labels
        if label == "Ergoty Damage":
            if conf < 0.96 or (bw * bh) < 80 or (max(bw, bh) / (min(bw, bh) + 1e-6)) < 1.6:
                label = "Sound Grain"
        elif label == "Damage" and conf < 0.90:
            label = "Sound Grain"
        elif label == "Slightly Damage" and conf < 0.55:
            label = "Sound Grain"
        elif label == "Broken" and conf < 0.20:
            # If the model is unsure about a broken piece, only discard if it's below 20%
            label = "Sound Grain"
        elif label == "Shrivelled" and conf < 0.22:
            label = "Sound Grain"
        # Append string representation directly to avoid dictionary lookup gaps
        final_labels_list.append(label)

    return final_labels_list

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

    # Table Layout Setup
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(60, 10, " Category", 1, 0, 'L', True)
    pdf.cell(40, 10, " Found %", 1, 0, 'C', True)
    pdf.cell(40, 10, " Limit %", 1, 0, 'C', True)
    pdf.cell(40, 10, " Status", 1, 1, 'C', True)

    pdf.set_font("Arial", '', 10)
    for cat, limit in WHEAT_NORMS.items():
        if cat == 'Shrivelled & Broken':
            val = ((counts.get('Shrivelled', 0) + counts.get('Broken', 0)) / total * 100) if total > 0 else 0
        else:
            val = (counts.get(cat, 0) / total * 100) if total > 0 else 0
        
        status = "OK" if val <= limit else "FAIL"
        pdf.cell(60, 10, f" {cat}", 1)
        pdf.cell(40, 10, f"{val:.2f}%", 1, 0, 'C')
        pdf.cell(40, 10, f"{limit}%", 1, 0, 'C')
        pdf.cell(40, 10, status, 1, 1, 'C')

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 15, f"FINAL RESULT: {final_status}", border=1, ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1')
