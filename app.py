import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from fpdf import FPDF
import io
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Digital Parkhi 2.0", page_icon="🌾", layout="wide")

# --- SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = 'welcome'
if 'grain' not in st.session_state: st.session_state.grain = None
if 'cat' not in st.session_state: st.session_state.cat = None

@st.cache_resource
def load_global_model():
    return YOLO("best.pt")

model = load_global_model()

# --- SAHI & ENHANCEMENT LOGIC ---

def validate_and_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)

def get_sliced_predictions(cv_img, model, slice_size=640, overlap=0.25):
    class_map = {0: 'Broken', 1: 'Damage', 2: 'Ergoty Damage', 3: 'Foreign Matter',
                 4: 'Shrivelled', 5: 'Slightly Damage', 6: 'Sound Grain'}
    
    img = validate_and_enhance(cv_img)
    h, w, _ = img.shape
    predictions = []
    step = int(slice_size * (1 - overlap))

    for y in range(0, h, step):
        for x in range(0, w, step):
            y2, x2 = min(y + slice_size, h), min(x + slice_size, w)
            tile = img[y:y2, x:x2]
            results = model.predict(tile, conf=0.25, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = class_map.get(cls)
                    bw, bh = float(box.xywh[0][2]), float(box.xywh[0][3])
                    pixel_area = bw * bh
                    aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)

                    # --- VIGILANCE FILTERS ---
                    if label == "Ergoty Damage":
                        if conf < 0.96 or pixel_area < 80 or aspect_ratio < 1.6:
                            cls = 6 
                    elif label == "Damage" and conf < 0.50:
                        cls = 6
                    
                    predictions.append(cls)
    return predictions

# --- PDF GENERATOR (STABLE VERSION) ---
def create_pdf_report(total, counts, norms, final_status, grain, cat):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Digital Parkhi 2.0: Official QC Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"TOTAL GRAINS SCANNED: {total}", ln=True)
    pdf.ln(5)

    # Table Header
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(60, 10, " Category", 1, 0, 'L', True)
    pdf.cell(40, 10, " Found %", 1, 0, 'C', True)
    pdf.cell(40, 10, " Limit %", 1, 0, 'C', True)
    pdf.cell(40, 10, " Status", 1, 1, 'C', True)

    pdf.set_font("Arial", '', 10)
    for c, limit in norms.items():
        val = (counts.get(c, 0) / total * 100) if total > 0 else 0
        status = "OK" if val <= limit else "FAIL"
        pdf.cell(60, 10, f" {c}", 1)
        pdf.cell(40, 10, f"{val:.2f}%", 1, 0, 'C')
        pdf.cell(40, 10, f"{limit}%", 1, 0, 'C')
        pdf.cell(40, 10, status, 1, 1, 'C')

    # Shrivelled & Broken
    sb_val = ((counts.get('Shrivelled', 0) + counts.get('Broken', 0)) / total * 100) if total > 0 else 0
    pdf.cell(60, 10, " Shrivelled & Broken", 1)
    pdf.cell(40, 10, f"{sb_val:.2f}%", 1, 0, 'C')
    pdf.cell(40, 10, " 6.00%", 1, 0, 'C')
    pdf.cell(40, 10, "OK" if sb_val <= 6.0 else "FAIL", 1, 1, 'C')

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 15, f"RESULT: {final_status}", border=1, ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- NAVIGATION ---
if st.session_state.page == 'welcome':
    st.title("🌾 Digital Parkhi 2.0")
    if st.button("Start Analysis"):
        st.session_state.page = 'select_grain'
        st.rerun()

elif st.session_state.page == 'select_grain':
    st.header("Select Grain Type")
    col1, col2 = st.columns(2)
    if col1.button("Wheat"):
        st.session_state.grain = "Wheat"
        st.session_state.page = 'select_cat'
        st.rerun()
    if col2.button("Rice"):
        st.session_state.grain = "Rice"
        st.session_state.page = 'select_cat'
        st.rerun()

elif st.session_state.page == 'select_cat':
    st.header(f"Select Category for {st.session_state.grain}")
    opts = ["FAQ", "URS"] if st.session_state.grain == "Wheat" else ["RRC", "RBC"]
    cat = st.selectbox("Grade", opts)
    if st.button("Proceed"):
        st.session_state.cat = cat
        st.session_state.page = 'upload'
        st.rerun()

elif st.session_state.page == 'upload':
    st.header(f"Deep Scanning: {st.session_state.grain} ({st.session_state.cat})")
    files = st.file_uploader("Upload Samples", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    
    if st.button("Run Analysis") and files:
        class_names = ['Broken', 'Damage', 'Ergoty Damage', 'Foreign Matter', 'Shrivelled', 'Slightly Damage', 'Sound Grain']
        master_counts = {name: 0 for name in class_names}
        file_stats = []
        grand_total = 0

        with st.spinner("Processing Slices..."):
            for f in files:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                preds = get_sliced_predictions(img, model)
                grand_total += len(preds)
                for p in preds: master_counts[class_names[p]] += 1
                file_stats.append(f"Processed {f.name}: {len(preds)} grains.")

        # FCI Standards
        norms = {'Foreign Matter': 0.75, 'Other Foodgrains': 2.0, 'Damage': 2.0, 
                 'Slightly Damage': 4.0, 'Ergoty Damage': 0.05}
        
        rej_reasons = []
        report_lines = []
        
        for name, limit in norms.items():
            val = (master_counts.get(name, 0) / grand_total * 100) if grand_total > 0 else 0
            status = "OK"
            if val > limit:
                status = "!! EXCEEDS LIMIT !!"
                rej_reasons.append(name)
            report_lines.append(f"{name.ljust(18)} : {val:5.2f}% | Limit: {limit:4}% | {status}")

        sb_val = ((master_counts['Shrivelled'] + master_counts['Broken']) / grand_total * 100) if grand_total > 0 else 0
        sb_status = "OK"
        if sb_val > 6.0:
            sb_status = "!! EXCEEDS LIMIT !!"
            rej_reasons.append("Shrivelled & Broken")
        report_lines.append(f"{'Shrivelled & Broken'.ljust(18)} : {sb_val:5.2f}% | Limit: 6.00% | {sb_status}")

        final_status = "REJECTED" if rej_reasons else f"ACCEPTED ({st.session_state.cat})"

        # --- FORMATTED OUTPUT DISPLAY ---
        output_txt = "--- STARTING ANALYSIS ---\n"
        output_txt += "\n".join(file_stats)
        output_txt += "\n\n" + "="*50
        output_txt += f"\nFCI AGGREGATED QC REPORT (RMS 2025-26)\n" + "="*50
        output_txt += f"\nTOTAL GRAINS SCANNED : {grand_total}\n" + "-"*50 + "\n"
        output_txt += "\n".join(report_lines)
        output_txt += "\n" + "-"*50
        output_txt += f"\nFINAL STATUS: {final_status}\n" + "="*50
        
        st.code(output_txt, language="text")

        # --- DOWNLOAD BUTTON ---
        pdf_bytes = create_pdf_report(grand_total, master_counts, norms, final_status, st.session_state.grain, st.session_state.cat)
        st.download_button(label="📄 Download Official PDF Report", data=pdf_bytes, file_name=f"FCI_Report_{datetime.now().strftime('%d%m%y')}.pdf", mime="application/pdf")

    if st.button("Reset"):
        st.session_state.page = 'welcome'
        st.rerun()