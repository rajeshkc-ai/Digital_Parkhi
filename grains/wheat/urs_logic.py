import cv2
import numpy as np
import torch
from fpdf import FPDF
from datetime import datetime

# 🔴 URS SPECIFICATIONS RMS 2026-27 (Keys perfectly aligned with standard lookups)
WHEAT_URS_NORMS = {
    'Foreign Matter': 0.75,
    'Other Foodgrains': 2.0,
    'Shrivelled & Broken': 15.00,
    'Lustre Loss': 70.00    
}

CLASS_MAP = {0: 'Broken', 1: 'Damage', 2: 'Ergoty Damage', 3: 'Foreign Matter',
             4: 'Shrivelled', 5: 'Slightly Damage', 6: 'Sound Grain', 7: 'Lustre Loss'}

def analyze_sample(cv_img, model):
    """Performs standardized image slicing and accurate defect parsing"""
    target_h = 1920
    h, w, _ = cv_img.shape
    scale = target_h / h
    target_w = int(w * scale)
    img = cv2.resize(cv_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    annotated_img = img.copy()
    
    slice_size = 640
    step = int(slice_size * 0.50) # 50% overlap prevents boundary blindspots
    
    global_boxes = []
    global_confs = []
    global_classes = []

    for y in range(0, target_h - slice_size + 1, step):
        for x in range(0, target_w - slice_size + 1, step):
            tile = img[y:y + slice_size, x:x + slice_size]
            
            # Use 0.15 to allow model predictions to surface safely before filtering
            preds = model.predict(tile, conf=0.15, verbose=False)
            
            for r in preds:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    bx_c, by_c, bw, bh = map(float, box.xywh[0])
                    global_x = x + (bx_c - bw/2)
                    global_y = y + (by_c - bh/2)
                    
                    global_boxes.append([global_x, global_y, global_x + bw, global_y + bh])
                    global_confs.append(conf)
                    global_classes.append(cls)

    if not global_boxes:
        return [], annotated_img

    boxes_t = torch.tensor(global_boxes)
    confs_t = torch.tensor(global_confs)
    keep_indices = torch.ops.torchvision.nms(boxes_t, confs_t, iou_threshold=0.35)

    final_labels_list = []

    COLOR_MAP = {
        'Sound Grain': (0, 255, 0),        # Green
        'Damage': (0, 0, 255),             # Red
        'Slightly Damage': (0, 165, 255),  # Orange
        'Shrivelled': (255, 255, 0),       # Cyan
        'Broken': (255, 0, 255),           # Magenta
        'Foreign Matter': (0, 255, 255),   # Yellow
        'Lustre Loss': (255, 255, 255)     # White
    }
    
    for idx in keep_indices:
        cls = global_classes[idx]
        conf = global_confs[idx]
        label = CLASS_MAP.get(cls, 'Sound Grain')
        
        x1, y1, x2, y2 = global_boxes[idx]
        bw, bh = x2 - x1, y2 - y1
        box_area = bw * bh
        
        # Balanced confidence gates to keep classes honest without aggressive overrides
        if label == "Shrivelled" and conf < 0.45:
            label = "Sound Grain"
        elif label == "Broken" and (conf < 0.40 or box_area > 200):
            label = "Sound Grain"
        elif label == "Slightly Damage" and conf < 0.35:
            label = "Sound Grain"
        elif label == "Damage" and conf < 0.35:
            label = "Sound Grain"
        elif label == "Lustre Loss" and conf < 0.30:
            label = "Sound Grain"
        elif label == "Foreign Matter" and conf < 0.35:
            label = "Sound Grain"          

        final_labels_list.append(label)

        # Render bounding boxes cleanly
        color = COLOR_MAP.get(label, (255, 255, 255))
        ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
        
        cv2.rectangle(annotated_img, (ix1, iy1), (ix2, iy2), color, 2)
        text_str = f"{label} {conf:.2f}"
        cv2.putText(annotated_img, text_str, (ix1, max(iy1 - 5, 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return final_labels_list, annotated_img

def generate_faq_pdf(total, counts, final_status):
    """Generates clean FPDF report matching terminal metrics"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Digital Parkhi: Official URS QC Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"TOTAL GRAINS SCANNED: {total}", ln=True)
    pdf.ln(5)

    pdf.set_fill_color(200, 220, 255)
    pdf.cell(60, 10, " Category", 1, 0, 'L', True)
    pdf.cell(40, 10, " Found %", 1, 0, 'C', True)
    pdf.cell(40, 10, " Limit %", 1, 0, 'C', True)
    pdf.cell(40, 10, " Status", 1, 1, 'C', True)

    pdf.set_font("Arial", '', 10)
    
    damage_pct = (counts.get('Damage', 0) / total * 100) if total > 0 else 0.0
    slightly_damage_pct = (counts.get('Slightly Damage', 0) / total * 100) if total > 0 else 0.0
    combined_damage_pct = damage_pct + slightly_damage_pct

    for cat, limit in WHEAT_URS_NORMS.items():
        if cat == 'Shrivelled & Broken':
            val = ((counts.get('Shrivelled', 0) + counts.get('Broken', 0)) / total * 100) if total > 0 else 0
        else:
            val = (counts.get(cat, 0) / total * 100) if total > 0 else 0
        
        status = "OK" if val <= limit else "FAIL"
        
        pdf.cell(60, 10, f" {cat}", 1)
        pdf.cell(40, 10, f"{val:.2f}%", 1, 0, 'C')
        pdf.cell(40, 10, f"{limit:.2f}%", 1, 0, 'C')
        pdf.cell(40, 10, status, 1, 1, 'C')

    # Combined matrix validation check row
    joint_row_status = "FAIL" if combined_damage_pct > 6.0 else "OK"
    pdf.cell(60, 10, " Damage & Slightly Damage", 1)
    pdf.cell(40, 10, f"{combined_damage_pct:.2f}%", 1, 0, 'C')
    pdf.cell(40, 10, "6.00%", 1, 0, 'C')
    pdf.cell(40, 10, joint_row_status, 1, 1, 'C')

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 15, f"FINAL RESULT: {final_status}", border=1, ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1')