RICE_FRK_NORMS = { ... }
RICE_RBA_NORMS = { ... }
RICE_RRA_NORMS = { ... }
RICE_RRC_NORMS = { ... }

def analyze_rrc(images, model):
    norms = {'Broken': 25.0, 'Chalky': 5.0, 'Foreign Matter': 0.5}
    # ... YOLO detection logic ...
    return counts, total, norms
