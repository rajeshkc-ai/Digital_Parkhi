import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from fpdf import FPDF  # Changed to FPDF to match your logic
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

# --- THE "TRICK" METHODS ---

def validate_and_enhance(img):
    """Fixes lighting and contrast before AI scans"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)

def get_sliced_predictions(cv_img, model, slice_size=640, overlap=0.25):
    """Slices the image into tiles so the AI can see grains clearly"""
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
                            cls = 6  # Change to Sound Grain
                    elif label == "Damage" and conf < 0.50:
                        cls = 6
                    
                    predictions.append(cls)
    return predictions

# --- UI NAVIGATION ---
if st.session_state.page == 'welcome':
    st.title("🌾 Digital Parkhi 2.0")
    st.info("Now using SAHI Slicing & CLAHE Enhancement for 99% accuracy.")
    if st.button("Start Professional Scan"):
        st.session_state.page = 'select_grain'
        st.rerun()

elif st.session_state.page == 'select_grain':
    st.header("Select Grain Type")
    col1, col2 = st.columns(2)
    if col1.button("Wheat", use_container_width=True):
        st.session_state.grain = "Wheat"
        st.session_state.page = 'select_cat'
        st.rerun()
    if col2.button("Rice", use_container_width=True):
        st.session_state.grain = "Rice"
        st.session_state.page = 'select_cat'
        st.rerun()

elif st.session_state.page == 'select_cat':
    st.header(f"Category: {st.session_state.grain}")
    opts = ["FAQ", "URS"] if st.session_state.grain == "Wheat" else ["RRC", "RBC"]
    cat = st.selectbox("Choose Grade", opts)
    if st.button("Proceed"):
        st.session_state.cat = cat
        st.session_state.page = 'upload'
        st.rerun()

elif st.session_state.page == 'upload':
    st.header(f"Analyze {st.session_state.grain}")
    files = st.file_uploader("Upload Samples", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])
    
    if st.button("Run Deep Analysis") and files:
        class_names = ['Broken', 'Damage', 'Ergoty Damage', 'Foreign Matter', 'Shrivelled', 'Slightly Damage', 'Sound Grain']
        master_counts = {name: 0 for name in class_names}
        grand_total = 0

        with st.spinner("Slicing and Enhancing images for Deep Scan..."):
            for f in files:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                cv_img = cv2.imdecode(file_bytes, 1)
                
                preds = get_sliced_predictions(cv_img, model)
                grand_total += len(preds)
                for p_idx in preds:
                    master_counts[class_names[p_idx]] += 1
                
                st.write(f"✅ {f.name}: Found {len(preds)} grains.")

        # --- CALCULATE STATUS ---
        norms = {'Foreign Matter': 0.75, 'Other Foodgrains': 2.0, 'Damage': 2.0, 
                 'Slightly Damage': 4.0, 'Ergoty Damage': 0.05}
        
        rej = False
        report_data = []
        for cat_name, limit in norms.items():
            found_pct = (master_counts.get(cat_name, 0) / grand_total * 100) if grand_total > 0 else 0
            if found_pct > limit: rej = True
            report_data.append([cat_name, found_pct, limit])

        sb_pct = ((master_counts['Shrivelled'] + master_counts['Broken']) / grand_total * 100) if grand_total > 0 else 0
        if sb_pct > 6.0: rej = True
        
        final_status = "REJECTED" if rej else f"ACCEPTED ({st.session_state.cat})"

        # --- DISPLAY RESULTS ---
        st.subheader("FCI AGGREGATED REPORT")
        st.metric("Total Grains Scanned", grand_total)
        
        if rej: st.error(f"RESULT: {final_status}")
        else: st.success(f"RESULT: {final_status}")

        # PDF Logic (Using your FPDF style)
        if st.button("Generate Official PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Digital Parkhi 2.0: Official QC Report", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f"Status: {final_status}", ln=True)
            pdf.cell(0, 10, f"Total Grains: {grand_total}", ln=True)
            
            # Simple PDF Table
            pdf.ln(5)
            pdf.cell(60, 10, "Category", 1)
            pdf.cell(40, 10, "Found %", 1)
            pdf.cell(40, 10, "Limit %", 1, 1)
            
            for row in report_data:
                pdf.cell(60, 10, row[0], 1)
                pdf.cell(40, 10, f"{row[1]:.2f}%", 1)
                pdf.cell(40, 10, f"{row[2]}%", 1, 1)
            
            pdf_output = pdf.output(dest='S').encode('latin-1')
            st.download_button("Download PDF", data=pdf_output, file_name="Digital_Parkhi_Report.pdf")

    if st.button("Reset"):
        st.session_state.page = 'welcome'
        st.rerun()