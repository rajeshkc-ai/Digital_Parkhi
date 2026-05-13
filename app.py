import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

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
    files = st.file_uploader("Select images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    
    if st.button("Run Analysis") and files:
        cv_imgs = [cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), 1) for f in files]
        
        try:
            # Import your modular logic
            from grains.wheat.faq_logic import analyze_faq, generate_faq_pdf
            
            with st.spinner("Analyzing samples..."):
                # Run the model on the list of images
                results = model.predict(cv_imgs, conf=0.20)
                
                # 1. Track ACTUAL counts per individual file
                individual_counts = []
                aggregated_results = {}
                total_grains = 0
                
                for res in results:
                    img_count = len(res.boxes)
                    individual_counts.append(img_count)
                    total_grains += img_count
                    
                    # Sum up classes for the final report
                    for box in res.boxes:
                        cls_name = model.names[int(box.cls)]
                        aggregated_results[cls_name] = aggregated_results.get(cls_name, 0) + 1

                # 2. Get norms and status from your logic file
                # (Assuming analyze_faq can accept pre-calculated totals)
                _, _, norms, status = analyze_faq(cv_imgs, model)

            # --- START OUTPUT DISPLAY ---
            report = "--- STARTING ANALYSIS ---\n"
            for i, f in enumerate(files):
                # Now showing the REAL count from individual_counts list
                report += f"Processed {f.name}: {individual_counts[i]} grains.\n"

            report += "\n" + "="*50 + "\n"
            report += "FCI AGGREGATED QC REPORT (RMS 2025-26)\n"
            report += "="*50 + "\n"
            report += f"TOTAL GRAINS SCANNED : {total_grains}\n"
            report += "-"*50 + "\n"

            categories = ['Foreign Matter', 'Other Foodgrains', 'Damage', 'Slightly Damage', 'Ergoty Damage']
            for cat in categories:
                count = aggregated_results.get(cat, 0)
                perc = (count / total_grains * 100) if total_grains > 0 else 0
                limit = norms.get(cat, 0)
                msg = "!! EXCEEDS LIMIT !!" if perc > limit else "OK"
                report += f"{cat.ljust(18)} : {perc:5.2f}% | Limit: {limit:4}% | {msg}\n"

            # Shrivelled & Broken (Combined)
            sb_count = aggregated_results.get('Shrivelled', 0) + aggregated_results.get('Broken', 0)
            sb_perc = (sb_count / total_grains * 100) if total_grains > 0 else 0
            sb_msg = "!! EXCEEDS LIMIT !!" if sb_perc > 6.0 else "OK"
            report += f"{'Shrivelled & Broken'.ljust(18)} : {sb_perc:5.2f}% | Limit: 6.00% | {sb_msg}\n"
            
            report += "-"*50 + "\n"
            report += f"FINAL STATUS: {status}\n"
            report += "="*50 + "\n"

            st.code(report, language="text")

            # PDF Section
            pdf_path = generate_faq_pdf(total_grains, aggregated_results, norms, status)
            with open(pdf_path, "rb") as f:
                st.download_button("Download Official PDF Report", f, file_name="FCI_Report.pdf")

        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("Reset"):
        st.session_state.page = 'welcome'
        st.rerun()
