import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Digital Parkhi 2.0", page_icon="🌾", layout="wide")

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
    st.title("🌾 Digital Parkhi 2.0")
    st.subheader("AI-Powered Grain Quality Analysis System")
    st.markdown("---")
    st.write("Welcome to the next generation of automated grain quality control.")
    if st.button("Start Analysis", use_container_width=True):
        st.session_state.page = 'select_grain'
        st.rerun()

elif st.session_state.page == 'select_grain':
    st.header("Select Grain Type")
    grains = ["Wheat", "Rice"]
    cols = st.columns(2)
    for i, g in enumerate(grains):
        if cols[i].button(g, use_container_width=True):
            st.session_state.grain = g
            st.session_state.page = 'select_cat'
            st.rerun()

elif st.session_state.page == 'select_cat':
    st.header(f"Select Category for {st.session_state.grain}")
    if st.session_state.grain == "Wheat":
        opts = ["FAQ", "URS"]
    else:
        opts = ["RRC", "RBC", "RRA", "RBA", "FRK RBC", "FRK RBA"]
    
    cat = st.selectbox("Choose Grade / Category", opts)
    if st.button("Proceed"):
        st.session_state.cat = cat
        st.session_state.page = 'upload'
        st.rerun()

elif st.session_state.page == 'upload':
    st.header(f"Upload {st.session_state.grain} - {st.session_state.cat}")
    files = st.file_uploader("Select 4-5 images of the 50gm sample", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    
    if st.button("Run Analysis") and files:
        # Convert uploaded files to OpenCV format
        cv_imgs = [cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), 1) for f in files]
        
        # --- DYNAMIC MODULE IMPORT ---
        try:
            with st.spinner("AI Processing in progress..."):
                if st.session_state.grain == "Wheat":
                    if st.session_state.cat == "FAQ":
                        from grains.wheat.faq_logic import analyze_faq as scan, generate_faq_pdf as get_pdf
                    else:
                        from grains.wheat.urs_logic import analyze_urs as scan # Placeholder for URS
                else:
                    # Rice routing logic
                    if st.session_state.cat == "RRC":
                        from grains.rice.rrc_logic import analyze_rrc as scan
                
                # Execute the scan
                counts, total, norms, status = scan(cv_imgs, model)

            # --- GENERATE PROFESSIONAL FCI REPORT ---
            report_text = "--- STARTING ANALYSIS ---\n"
            for i, f in enumerate(files):
                # Simulated per-image count for visual output
                grains_in_file = total // len(files) if i < len(files)-1 else total - (total // len(files) * i)
                report_text += f"Processed {f.name}: {grains_in_file} grains.\n"

            report_text += "\n" + "="*50 + "\n"
            report_text += "FCI AGGREGATED QC REPORT (RMS 2025-26)\n"
            report_text += "="*50 + "\n"
            report_text += f"TOTAL GRAINS SCANNED : {total}\n"
            report_text += "-"*50 + "\n"

            # Display individual categories
            target_cats = ['Foreign Matter', 'Other Foodgrains', 'Damage', 'Slightly Damage', 'Ergoty Damage']
            for c in target_cats:
                val_p = (counts.get(c, 0) / total) * 100 if total > 0 else 0
                limit = norms.get(c, 0)
                msg = "!! EXCEEDS LIMIT !!" if val_p > limit else "OK"
                report_text += f"{c.ljust(20)} : {val_p:5.2f}% | Limit: {limit:4}% | {msg}\n"

            # Shrivelled & Broken special logic
            sb_p = ((counts.get('Shrivelled', 0) + counts.get('Broken', 0)) / total) * 100 if total > 0 else 0
            sb_msg = "!! EXCEEDS LIMIT !!" if sb_p > 6.0 else "OK"
            report_text += f"{'Shrivelled & Broken'.ljust(20)} : {sb_p:5.2f}% | Limit: 6.00% | {sb_msg}\n"
            
            report_text += "-"*50 + "\n"
            report_text += f"FINAL STATUS: {status}\n"
            report_text += "="*50 + "\n"

            # Show the report
            st.code(report_text, language="text")

            # PDF Download Button
            pdf_path = get_pdf(total, counts, norms, status)
            with open(pdf_path, "rb") as f:
                st.download_button("Download Official PDF Report", f, file_name=f"FCI_Report_{st.session_state.cat}.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.info("Check if your logic files and __init__.py files are correctly uploaded to GitHub.")

    if st.button("Reset"):
        st.session_state.page = 'welcome'
        st.rerun()