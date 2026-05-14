import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from grains import faq_logic # IMPORTING YOUR LOGIC FILE
from datetime import datetime

st.set_page_config(page_title="Digital Parkhi 2.0", page_icon="🌾", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'welcome'

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

if st.session_state.page == 'welcome':
    st.title("🌾 Digital Parkhi 2.0")
    if st.button("Start Analysis"):
        st.session_state.page = 'upload'
        st.rerun()

elif st.session_state.page == 'upload':
    st.header("Upload Wheat Samples (FAQ)")
    files = st.file_uploader("Select 3-5 images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    
    if st.button("Run Analysis") and files:
        master_counts = {name: 0 for name in faq_logic.CLASS_MAP.values()}
        file_stats = []
        grand_total = 0

        with st.spinner("Applying Deep Scan (Slicing & Enhancement)..."):
            for f in files:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
                preds = faq_logic.analyze_sample(img, model)
                
                grand_total += len(preds)
                for p_idx in preds:
                    master_counts[faq_logic.CLASS_MAP[p_idx]] += 1
                file_stats.append(f"Processed {f.name}: {len(preds)} grains.")

        # Determine Status
        rej_reasons = []
        report_lines = []
        for cat, limit in faq_logic.WHEAT_NORMS.items():
            if cat == 'Shrivelled & Broken':
                val = ((master_counts['Shrivelled'] + master_counts['Broken']) / grand_total * 100)
            else:
                val = (master_counts.get(cat, 0) / grand_total * 100)
            
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
