import cv2

def analyze_rra(images, model):
    """
    Logic for Rice Raw Grade A (RRA).
    Stricter norms than Common Rice (RRC).
    """
    class_map = {0: 'Broken', 1: 'Damage', 2: 'Discoloured', 3: 'Foreign Matter', 
                 4: 'Chalky', 5: 'Red Grains', 6: 'Sound Grain'}
    
    counts = {name: 0 for name in class_map.values()}
    total = 0

    # RRA Specific Norms
    norms = {
        'Broken': 20.0,        # RRC is usually 25%
        'Foreign Matter': 0.5,
        'Damage': 2.0,         # Stricter than RRC
        'Chalky': 3.0
    }

    for img in images:
        results = model.predict(img, conf=0.25, verbose=False)
        for r in results:
            for box in r.boxes:
                label = class_map.get(int(box.cls[0]), 'Sound Grain')
                counts[label] += 1
                total += 1
                
    return counts, total, norms