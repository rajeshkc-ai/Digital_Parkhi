import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
# import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Digital Parkhi", page_icon="🌾", layout="wide")

# --- SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = 'welcome'
if 'grain' not in st.session_state: st.session_state.grain = None
if 'cat' not in st.session_state: st.session_state.cat = None

@st.cache_resource
def load_global_model():
    # Ensure best.pt is in the same directory as app.py on GitHub
    return YOLO("best.pt")

model = load_global_model()

# --- NAVIGATION & UI ---

if st.session_state.page == 'welcome':
    st.title("🌾 Digital Parkhi")
    st.subheader("AI-Powered Grain Quality Analysis System")
    st.markdown("---")
    st.write("Welcome to the next generation of automated grain quality control.")
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
    opts = ["FAQ", "URS"] if st.session_state.grain == "Wheat" else ["RRC", "RBC", "RRA", "RBA", "FRK RBC", "FRK RBA"]
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
        
        with st.spinner("AI is counting grains..."):
            # Increased sensitivity (conf) and image size (imgsz) to catch all grains
            results = model.predict(cv_imgs, conf=0.10, imgsz=640)
            
            individual_counts = []
            aggregated_results = {}
            total_grains = 0
            
            for res in results:
                cnt = len(res.boxes)
                individual_counts.append(cnt)
                total_grains += cnt
                for box in res.boxes:
                    cls_name = model.names[int(box.cls)]
                    aggregated_results[cls_name] = aggregated_results.get(cls_name, 0) + 1

        # --- FCI STANDARDS (RMS 2025-26) ---
        norms = {
            'Foreign Matter': 0.75,
            'Other Foodgrains': 2.0,
            'Damage': 2.0,
            'Slightly Damage': 4.0,
            'Ergoty Damage': 0.05,
            'Shrivelled & Broken': 6.0
        }

        # Logic to determine status
        reasons_for_rejection = []
        # Check for Rejection
        is_rejected = False
        report_lines = []
        
        categories = ['Foreign Matter', 'Other Foodgrains', 'Damage', 'Slightly Damage', 'Ergoty Damage']
        for c in categories:
            val = (aggregated_results.get(c, 0) / total_grains * 100) if total_grains > 0 else 0
            limit = norms[c]
            status_label = "OK"
            if val > limit:
                status_label = "!! EXCEEDS LIMIT !!"
                is_rejected = True
                #reasons_for_rejection.append(c)
            report_lines.append(f"{c.ljust(18)} : {val:5.2f}% | Limit: {limit:4}% | {status_label}")

        # Shrivelled & Broken combined
        sb_val = ((aggregated_results.get('Shrivelled', 0) + aggregated_results.get('Broken', 0)) / total_grains * 100) if total_grains > 0 else 0
        sb_status = "OK"
        if sb_val > 6.0:
            sb_status = "!! EXCEEDS LIMIT !!"
            is_rejected = True
            #reasons_for_rejection.append("Shrivelled & Broken")
        report_lines.append(f"{'Shrivelled & Broken'.ljust(18)} : {sb_val:5.2f}% | Limit: 6.00% | {sb_status}")

        final_status = "REJECTED" if is_rejected else f"ACCEPTED ({st.session_state.cat})"

        # --- START OUTPUT DISPLAY ---
        output = "--- STARTING ANALYSIS ---\n"
        for i, f in enumerate(files):
        # Now showing the REAL count from individual_counts list
            output += f"Processed {f.name}: {individual_counts[i]} grains.\n"

        output += "\n" + "="*50 + "\n"
        output += "FCI AGGREGATED QC REPORT (RMS 2025-26)\n"
        output += "="*50 + "\n"
        output += f"TOTAL GRAINS SCANNED : {total_grains}\n"
        output += "-"*50 + "\n"
        output += "\n".join(report_lines) + "\n"
        output += "-"*50 + "\n"
        output += f"FINAL STATUS: {final_status}\n"
        output += "="*50 + "\n"

        categories = ['Foreign Matter', 'Other Foodgrains', 'Damage', 'Slightly Damage', 'Ergoty Damage']
        for cat in categories:
            count = aggregated_results.get(cat, 0)
            perc = (count / total_grains * 100) if total_grains > 0 else 0
            limit = norms.get(cat, 0)
            msg = "!! EXCEEDS LIMIT !!" if perc > limit else "OK"
            output += f"{cat.ljust(18)} : {perc:5.2f}% | Limit: {limit:4}% | {msg}\n"

        # Shrivelled & Broken (Combined)
        sb_count = aggregated_results.get('Shrivelled', 0) + aggregated_results.get('Broken', 0)
        sb_perc = (sb_count / total_grains * 100) if total_grains > 0 else 0
        sb_msg = "!! EXCEEDS LIMIT !!" if sb_perc > 6.0 else "OK"
        output += f"{'Shrivelled & Broken'.ljust(18)} : {sb_perc:5.2f}% | Limit: 6.00% | {sb_msg}\n"
            
        output += "-"*50 + "\n"
        output += f"FINAL STATUS: {status}\n"
        output += "="*50 + "\n"

        st.code(output, language="text")

        # PDF Section
        pdf_path = generate_faq_pdf(total_grains, aggregated_results, norms, status)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Official PDF Report", f, file_name="FCI_Report.pdf")

    except Exception as e:
        st.error(f"Error: {e}")

if st.button("Reset"):
    st.session_state.page = 'welcome'
    st.rerun()
