import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from grains.wheat import faq_logic
from datetime import datetime

st.set_page_config(page_title="Digital Parkhi", page_icon="🌾", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'welcome'
if 'grain' not in st.session_state: st.session_state.grain = None
if 'cat' not in st.session_state: st.session_state.cat = None

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- NAVIGATION ---
if st.session_state.page == 'welcome':
    st.title("🌾 Digital Parkhi 2.0")
    st.subheader("AI-Powered Grain Quality Analysis")
    if st.button("Start Analysis", use_container_width=True):
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
        # Initialize string keys exactly matching the values inside CLASS_MAP
        master_counts = {name: 0 for name in faq_logic.CLASS_MAP.values()}
        file_stats = []
        grand_total = 0

        with st.spinner("Applying Deep Scan (Slicing & Enhancement)..."):
            for f in files:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
                
                # Fetch text strings list straight from your NMS filter logic
                preds = faq_logic.analyze_sample(img, model)
                
                # 🔴 TEMPORARY DEBUG LINE
                st.write(f"Raw labels found in {f.name}: {set(preds)}")
                
                grand_total += len(preds)
                
                # CORRECTED: Direct string counting without looking up CLASS_MAP again
                for label in preds:
                    if label in master_counts:
                        master_counts[label] += 1
                        
                file_stats.append(f"Processed {f.name}: {len(preds)} grains.")

        # Determine Status
        rej_reasons = []
        report_lines = []
        for cat, limit in faq_logic.WHEAT_NORMS.items():
            # Standardize string checking by stripping whitespace
            clean_cat = cat.strip()
            
            if grand_total > 0:
                if clean_cat == 'Shrivelled & Broken':
                    # Explicitly fetch individual string components from master_counts
                    shrivelled_count = master_counts.get('Shrivelled', 0)
                    broken_count = master_counts.get('Broken', 0)
                    val = ((shrivelled_count + broken_count) / grand_total) * 100
                else:
                    # Safely look up the exact category name
                    val = (master_counts.get(clean_cat, 0) / grand_total) * 100
            else:
                val = 0.0
            
            status = "OK"
            if val > limit:
                status = "!! EXCEEDS LIMIT !!"
                rej_reasons.append(cat)
            report_lines.append(f"{cat.ljust(18)} : {val:5.2f}% | Limit: {limit:4}% | {status}")

        final_status = "REJECTED" if rej_reasons else "ACCEPTED (FAQ)"

        # Display Terminal Output
        output_txt = "--- STARTING ANALYSIS ---\n" + "\n".join(file_stats)
        output_txt += "\n\n" + "="*50 + "\nFCI AGGREGATED QC REPORT (RMS 2025-26)\n" + "="*50
        output_txt += f"\nTOTAL GRAINS SCANNED : {grand_total}\n" + "-"*50 + "\n"
        output_txt += "\n".join(report_lines) + "\n" + "-"*50
        output_txt += f"\nFINAL STATUS: {final_status}\n" + "="*50
        st.code(output_txt, language="text")

        # PDF Download Button
        pdf_bytes = faq_logic.generate_faq_pdf(grand_total, master_counts, final_status)
        st.download_button(
            label="📄 Download Official PDF Report",
            data=pdf_bytes,
            file_name=f"FCI_Report_{datetime.now().strftime('%d%m%y')}.pdf",
            mime="application/pdf"
        )

    if st.button("Reset"):
        st.session_state.page = 'welcome'
        st.rerun()
