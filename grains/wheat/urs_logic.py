import cv2
import numpy as np

def analyze_wheat_urs(images, model):
    """
    Logic for Wheat - Under Relaxed Specifications (URS).
    Typically used when weather conditions affect the crop.
    """
    class_map = {0: 'Broken', 1: 'Damage', 2: 'Ergoty Damage', 3: 'Foreign Matter',
                 4: 'Shrivelled', 5: 'Slightly Damage', 6: 'Sound Grain'}
    
    counts = {name: 0 for name in class_map.values()}
    total_detected = 0

    # URS Specific Norms (Example: higher limits for Shrivelled/Broken)
    urs_norms = {
        'Foreign Matter': 1.00,       # FAQ is 0.75
        'Other Foodgrains': 2.0,
        'Damage': 3.0,               # FAQ is 2.0
        'Slightly Damage': 6.0,      # FAQ is 4.0
        'Ergoty Damage': 0.05,       # Remains strict
        'Shrivelled & Broken': 8.00  # FAQ is 6.0
    }

    for img in images:
        # Standardize image size for the model
        results = model.predict(img, conf=0.25, verbose=False)
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = class_map.get(cls)
                
                # --- URS Specific Vigilance ---
                # In URS, we might be slightly more lenient on 'Slightly Damage'
                # but we keep the 'Ergoty Damage' filter very strict.
                if label == "Ergoty Damage":
                    bw, bh = float(box.xywh[0][2]), float(box.xywh[0][3])
                    pixel_area = bw * bh
                    aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
                    
                    if conf < 0.96 or pixel_area < 80 or aspect_ratio < 1.6:
                        label = 'Sound Grain'
                
                counts[label] += 1
                total_detected += 1
                
    return counts, total_detected, urs_norms