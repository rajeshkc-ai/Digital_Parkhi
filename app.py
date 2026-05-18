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

        # --- DYNAMIC QUALITY NORMS SELECTION ---
        rej_reasons = []
        report_lines = []
        
        grade_label = st.session_state.cat  # Stores FAQ, URS, RRC, or RBC
        active_norms = {}
        
        if st.session_state.grain == "Wheat":
            if grade_label == "URS":
                active_norms = getattr(faq_logic, 'WHEAT_URS_NORMS', faq_logic.WHEAT_NORMS)
            else:
                active_norms = faq_logic.WHEAT_NORMS
        elif st.session_state.grain == "Rice":
            if grade_label == "RBC":
                active_norms = getattr(faq_logic, 'RICE_RBC_NORMS', {})
            else:
                active_norms = getattr(faq_logic, 'RICE_RRC_NORMS', {})
        
        if not active_norms:
            st.error(f"Quality specifications dictionary for {st.session_state.grain} ({grade_label}) is missing!")
            st.stop()

        # Pre-calculate percentage values for combined checking strings
        damage_pct = (master_counts.get('Damage', 0) / grand_total * 100) if grand_total > 0 else 0.0
        slightly_damage_pct = (master_counts.get('Slightly Damage', 0) / grand_total * 100) if grand_total > 0 else 0.0
        combined_damage_pct = damage_pct + slightly_damage_pct

        # Run Evaluation Loop
        for cat, limit in active_norms.items():
            current_category = cat.strip()
            
            if grand_total > 0:
                if st.session_state.grain == "Wheat" and current_category == 'Shrivelled & Broken':
                    shrivelled_count = master_counts.get('Shrivelled', 0)
                    broken_count = master_counts.get('Broken', 0)
                    val = ((shrivelled_count + broken_count) / grand_total) * 100
                else:
                    val = (master_counts.get(current_category, 0) / grand_total) * 100
            else:
                val = 0.0
            
            status = "OK"
            
            # --- SPECIAL CONDITIONS CHECKER ---
            if grade_label == "URS" and st.session_state.grain == "Wheat" and current_category in ['Damage', 'Slightly Damage']:
                # For individual lines in URS, display the standard entry but flag if the joint group totals > 6%
                if combined_damage_pct > 6.0:
                    status = "!! COMBINED LIMIT EXCEEDED (>6%) !!"
                    if "Damage + Slightly Damage Combined" not in rej_reasons:
                        rej_reasons.append("Damage + Slightly Damage Combined")
            else:
                # Standard threshold checking rule
                if val > limit:
                    status = "!! EXCEEDS LIMIT !!"
                    rej_reasons.append(cat)
                    
            report_lines.append(f"{cat.ljust(18)} : {val:5.2f}% | Limit: {limit:4}% | {status}")

        # Add an explicit breakdown row for the combined damages rule under URS print views
        if grade_label == "URS" and st.session_state.grain == "Wheat":
            combined_status = "!! EXCEEDS LIMIT !!" if combined_damage_pct > 6.0 else "OK"
            report_lines.append(f"{'Damage Total (Joint)'.ljust(18)} : {combined_damage_pct:5.2f}% | Limit:  6.0% | {combined_status}")

        # 3. Dynamic Status Formatting
        final_status = "REJECTED" if rej_reasons else f"ACCEPTED ({grade_label})"

        # Display Dynamic Terminal Output
        output_txt = f"--- STARTING {st.session_state.grain.upper()} ANALYSIS ---\n" + "\n".join(file_stats)
        output_txt += "\n\n" + "="*50 + f"\nFCI AGGREGATED QC REPORT (RMS 2025-26) - {grade_label}\n" + "="*50
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
