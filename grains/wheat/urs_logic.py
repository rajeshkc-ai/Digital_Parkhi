import cv2
import numpy as np
import torch
from fpdf import FPDF
from datetime import datetime

# 🔴 URS SPECIFICATIONS RMS 2026-27
WHEAT_URS_NORMS = {
    'Foreign Matter': 0.75,
    'Other Foodgrains': 2.0,
    'Shrivelled & Broken': 15.00,     # Relaxed from 6.0% to 15%
    'Lustre Loss': 70.00    # New relaxed criteria up to 70%
}

CLASS_MAP = {0: 'Broken', 1: 'Damage', 2: 'Ergoty Damage', 3: 'Foreign Matter',
             4: 'Shrivelled', 5: 'Slightly Damage', 6: 'Sound Grain', 7: 'Lustre Loss'}


def analyze_sample(cv_img, model):
    """Performs CLAHE enhancement and Sliced Inference with global NMS tracking"""
    """Performs inference and draws bounding boxes on a copy of the original image"""
    # Create a clean copy of the original image to draw bounding boxes on
    annotated_img = cv_img.copy()
    
    # 1. Enhance Contrast safely
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    # 2. Slice Setup
    h, w, _ = img.shape
    slice_size = 640
    step = int(slice_size * 0.95) # 10% overlap
    
    global_boxes = []
    global_confs = []
    global_classes = []

    # Slicing Processing Loop
    for y in range(0, h, step):
        for x in range(0, w, step):
            y2, x2 = min(y + slice_size, h), min(x + slice_size, w)
            tile = img[y:y2, x:x2]
            
            # Using 0.15 baseline ensures smaller fragments are registered
            preds = model.predict(tile, conf=0.20, iou=0.45, agnostic_nms=False, verbose=False)
            
            for r in preds:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Convert tile-relative coordinates back to global canvas scale coordinates
                    bx_c, by_c, bw, bh = map(float, box.xywh[0])
                    global_x = x + (bx_c - bw/2)
                    global_y = y + (by_c - bh/2)
                    
                    global_boxes.append([global_x, global_y, global_x + bw, global_y + bh])
                    global_confs.append(conf)
                    global_classes.append(cls)

    if not global_boxes:
        return [], annotated_img

    # 3. Apply Global Non-Maximum Suppression to wipe out boundary duplicate counts
    boxes_t = torch.tensor(global_boxes)
    confs_t = torch.tensor(global_confs)
    keep_indices = torch.ops.torchvision.nms(boxes_t, confs_t, iou_threshold=0.15)

    final_labels_list = []

    # Color map for beautiful boxes (B, G, R format for OpenCV)
    COLOR_MAP = {
        'Sound Grain': (0, 255, 0),        # Green
        'Damage': (0, 0, 255),             # Red
        'Slightly Damage': (0, 165, 255),  # Orange
        'Shrivelled': (255, 255, 0),       # Cyan / Yellow-Blue
        'Broken': (255, 0, 255),           # Magenta
        'Foreign Matter': (0, 255, 255),   # Yellow
        'Ergoty Damage': (0, 0, 0),         # Black
        'Lustre Loss': (255,255,255)       # White
    }
    
    # 4. Process deduplicated predictions through validation filters
    for idx in keep_indices:
        cls = global_classes[idx]
        conf = global_confs[idx]
        label = CLASS_MAP.get(cls)
        
        # Calculate bounding dimensions from global coordinates
        x1, y1, x2, y2 = global_boxes[idx]
        bw, bh = x2 - x1, y2 - y1
        box_area = bw * bh
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
        
        ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
        
        # =========================
        # PURE AI VALIDATION
        # =========================

        CLASS_THRESHOLDS = {
            'Sound Grain': 0.30,
            'Damage': 0.35,
            'Slightly Damage': 0.30,
            'Shrivelled': 0.35,
            'Broken': 0.35,
            'Foreign Matter': 0.40,
            'Ergoty Damage': 0.45,
            'Lustre Loss': 0.30
        }

        # Reject only very weak predictions
        if conf < CLASS_THRESHOLDS.get(label, 0.30):
            continue

        # Accept AI prediction directly
        final_labels_list.append(label)

        # 🎨 DRAW BOXES ON THE FULL-SIZE IMAGE
        color = COLOR_MAP.get(label, (255, 255, 255)) # Default white if fallback
        
        
        # Draw bounding box rectangle
        cv2.rectangle(annotated_img, (ix1, iy1), (ix2, iy2), color, 2)
        
        # Add label text right above the box
        text_str = f"{label} {conf:.2f}"
        cv2.putText(annotated_img, text_str, (ix1, max(iy1 - 5, 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return final_labels_list, annotated_img

def generate_faq_pdf(total, counts, final_status):
    """Generates the bytes for the PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Digital Parkhi: Official URS QC Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}", ln=True, align='C')
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
    
    # Pre-calculate joint metrics for row evaluations
    damage_pct = (counts.get('Damage', 0) / total * 100) if total > 0 else 0.0
    slightly_damage_pct = (counts.get('Slightly Damage', 0) / total * 100) if total > 0 else 0.0
    combined_damage_pct = damage_pct + slightly_damage_pct
    # Loop through specifications sequentially
    for cat, limit in WHEAT_URS_NORMS.items():
        if cat == 'Shrivelled & Broken':
            val = ((counts.get('Shrivelled', 0) + counts.get('Broken', 0)) / total * 100) if total > 0 else 0
        else:
            val = (counts.get(cat, 0) / total * 100) if total > 0 else 0
        # Determine status flag cleanly based on joint URS parameters
        if cat in ['Damage', 'Slightly Damage']:
            status = "FAIL" if combined_damage_pct > 6.0 else "OK"
        else:
            status = "OK" if val <= limit else "FAIL"
        
        pdf.cell(60, 10, f" {cat}", 1)
        pdf.cell(40, 10, f"{val:.2f}%", 1, 0, 'C')
        pdf.cell(40, 10, f"{limit}%", 1, 0, 'C')
        pdf.cell(40, 10, status, 1, 1, 'C')

    # Add the structural breakdown line for the joint limit row matching the app screen
    joint_row_status = "FAIL" if combined_damage_pct > 6.0 else "OK"
    pdf.cell(60, 10, "Damage & Slightly Damage", 1)
    pdf.cell(40, 10, f"{combined_damage_pct:.2f}%", 1, 0, 'C')
    pdf.cell(40, 10, "6.00%", 1, 0, 'C')
    pdf.cell(40, 10, joint_row_status, 1, 1, 'C')
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 15, f"FINAL RESULT: {final_status}", border=1, ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1')
