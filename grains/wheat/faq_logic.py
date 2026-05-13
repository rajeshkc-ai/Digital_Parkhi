def analyze_faq(images, model):
    norms = {'Foreign Matter': 0.75, 'Damage': 2.0, 'Ergot': 0.05}
    # ... YOLO detection logic + Vigilance filters ...
    return counts, total, norms