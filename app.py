import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

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

def generate_pdf_report(total, counts, norms, status, grain_type, category):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, f"FCI QC REPORT: {grain_type} ({category})")
    p.setFont("Helvetica", 12)
    p.drawString(100, 730, f"Total Grains Scanned: {total}")
    p.line(100, 720, 500, 720)
    
    y = 700
    for key, limit in norms.items():
        val = (counts.get(key, 0) / total * 100) if total > 0 else 0
        p.drawString(100, y, f"{key}: {val:.2f}% (Limit: {limit}%)")
        y -= 20
    
    p.line(100, y, 500, y)
    y -= 30
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, y, f"FINAL STATUS: {status}")
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# --- UI NAVIGATION ---
if st.session_state.page == 'welcome':
    st.title("🌾 Digital Parkhi 2.0")
    if st.button("Start Analysis", use_container_width=True):
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
    st.header(f"Select Category: {st.session_state.grain}")
    opts = ["FAQ", "URS"] if st.session_state.grain == "Wheat" else ["RRC", "RBC"]
    cat = st.selectbox("Choose Grade", opts)
    if st.button("Proceed"):
        st.session_state.cat = cat
        st.session_state.page = 'upload'
        st.rerun()

elif st.session_state.page == 'upload':
    st.header(f"Upload {st.session_state.grain} - {st.session_state.cat}")
    files = st.file_uploader("Select images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    
    if st.button("Run Analysis") and files:
        cv_imgs = [cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), 1) for f in files]
        
        individual_counts = []
        aggregated_results = {}
        total_grains = 0
        
        try:
            with st.spinner("AI is performing Deep Scan..."):
                # imgsz=640 and conf=0.10 ensure small grains are not missed
                results = model.predict(cv_imgs, conf=0.10, imgsz=640)
                for res in results:
                    cnt = len(res.boxes)
                    individual_counts.append(cnt)
                    total_grains += cnt
                    for box in res.boxes:
                        cls_name = model.names[int(box.cls)]
                        aggregated_results[cls_name] = aggregated_results.get(cls_name, 0) + 1
        except Exception as e:
            st.error(f"Analysis Error: {e}")
            st.stop()

        # FCI Standards for RMS 2025-26
        norms = {
            'Foreign Matter': 0.75, 
            'Other Foodgrains': 2.0, 
            'Damage': 2.0, 
            'Slightly Damage': 4.0, 
            'Ergoty Damage': 0.05, 
            'Shrivelled & Broken': 6.0
        }
        
        reasons_for_rejection = []
        report_lines = []
        
        for c in ['Foreign Matter', 'Other Foodgrains', 'Damage', 'Slightly Damage', 'Ergoty Damage']:
            val = (aggregated_results.get(c, 0) / total_grains * 100) if total_grains > 0 else 0
            limit = norms[c]
            status_label = "OK"
            if val > limit:
                status_label = "!! EXCEEDS LIMIT !!"
                reasons_for_rejection.append(c)
            report_lines.append(f"{c.ljust(18)} : {val:5.2f}% | Limit: {limit:4}% | {status_label}")

        # Combined count for Shrivelled & Broken
        sb_val = ((aggregated_results.get('Shrivelled', 0) + aggregated_results.get('Broken', 0)) / total_grains * 100) if total_grains > 0 else 0
        sb_status = "OK"
        if sb_val > 6.0:
            sb_status = "!! EXCEEDS LIMIT !!"
            reasons_for_rejection.append("Shrivelled & Broken")
        report_lines.append(f"{'Shrivelled & Broken'.ljust(18)} : {sb_val:5.2f}% | Limit: 6.00% | {sb_status}")

        final_status = "REJECTED" if reasons_for_rejection else f"ACCEPTED ({st.session_state.cat})"

        # --- OUTPUT DISPLAY ---
        output_txt = "--- STARTING ANALYSIS ---\n"
        for i, f in enumerate(files):
            output_txt += f"Processed {f.name}: {individual_counts[i]} grains.\n"

        output_txt += f"\nTOTAL GRAINS SCANNED : {total_grains}\n" + "-"*50 + "\n"
        output_txt += "\n".join(report_lines) + f"\n\nFINAL STATUS: {final_status}\n"
        st.code(output_txt, language="text")

        # PDF Download Section
        pdf_file = generate_pdf_report(total_grains, aggregated_results, norms, final_status, st.session_state.grain, st.session_state.cat)
        st.download_button(label="📄 Download Official PDF Report", data=pdf_file, file_name="FCI_Quality_Report.pdf", mime="application/pdf")

    if st.button("Reset"):
        st.session_state.page = 'welcome'
        st.rerun()