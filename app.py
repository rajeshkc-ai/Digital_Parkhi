import streamlit as st
import cv2
import numpy as np
from datetime import datetime

# Import specialized logic files
from grains.wheat import faq_logic
from grains.wheat import urs_logic

st.set_page_config(page_title="Digital Parkhi", page_icon="🌾", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'welcome'
if 'grain' not in st.session_state: st.session_state.grain = None
if 'cat' not in st.session_state: st.session_state.cat = None


# --- NAVIGATION ---
if st.session_state.page == 'welcome':
    st.title("🌾 Digital Parkhi")
    st.subheader("AI-Powered Grain Quality Analysis")
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
    st.header(f"Select Category for {st.session_state.grain}")
    opts = ["FAQ", "URS"] if st.session_state.grain == "Wheat" else ["RRC", "RBC"]
    cat = st.selectbox("Grade", opts)
    if st.button("Proceed"):
        st.session_state.cat = cat
        st.session_state.page = 'upload'
        st.rerun()

elif st.session_state.page == 'upload':
    st.header(f"Deep Scanning: {st.session_state.grain} ({st.session_state.cat})")
    files = st.file_uploader("Upload at least 3 images of 50 gm Sample", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    
    if st.button("Run Analysis") and files:
        grain_type = st.session_state.grain
        grade_label = st.session_state.cat
        
        if grain_type == "Wheat":
            if grade_label == "URS":
                active_module = urs_logic
                active_norms = getattr(urs_logic, 'WHEAT_URS_NORMS', {})
            else:
                active_module = faq_logic
                active_norms = getattr(faq_logic, 'WHEAT_NORMS', {})
        else:
            st.error("Selected grain configuration module not found.")
            st.stop()

        if not active_norms:
            st.error(f"Error: Specifications dictionary for {grain_type} ({grade_label}) could not be located.")
            st.stop()

        # Initialize global counts dictionary
        master_counts = {name: 0 for name in active_module.CLASS_MAP.values()}
        file_stats = []
        grand_total = 0
        processed_images_to_show = [] 

        # --- IMAGE STREAM SCANNING LAYER ---
        with st.spinner("Applying Deep Scan (Slicing & Optimization)..."):
            for f in files:
                f.seek(0)  # Reset stream pointer position to safely fetch fresh bytes
                file_bytes = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    st.error(f"Could not read file: {f.name}")
                    continue

                preds, output_visual_img = active_module.analyze_sample(img)
                grand_total += len(preds)
                
                for label in preds:
                    if label in master_counts:
                        master_counts[label] += 1
                        
                file_stats.append(f"Processed {f.name}: {len(preds)} grains.")
                rgb_output = cv2.cvtColor(output_visual_img, cv2.COLOR_BGR2RGB)
                processed_images_to_show.append((f.name, rgb_output))

        # --- EVALUATION ENGINE ---
        rej_reasons = []
        report_lines = []
        
        for cat, limit in active_norms.items():
            current_category = cat.strip()
            
            if grand_total > 0:
                if current_category == 'Shrivelled & Broken':
                    val = ((master_counts.get('Shrivelled', 0) + master_counts.get('Broken', 0)) / grand_total) * 100
                elif current_category == 'Damage & Slightly Damage':
                    val = ((master_counts.get('Damage', 0) + master_counts.get('Slightly Damage', 0)) / grand_total) * 100
                else:
                    val = (master_counts.get(current_category, 0) / grand_total) * 100
            else:
                val = 0.0
            
            status = "OK"
            if val > limit:
                status = "!! EXCEEDS LIMIT !!"
                rej_reasons.append(cat)
                    
            report_lines.append(f"{cat.ljust(26)} : {val:5.2f}% | Limit: {limit:5.2f}% | {status}")

        final_status = "REJECTED" if rej_reasons else f"ACCEPTED ({grade_label})"

        # --- CONSOLE LAYOUT PRINT OUT ---
        output_txt = f"--- STARTING {grain_type.upper()} ANALYSIS ---\n" + "\n".join(file_stats)
        output_txt += "\n\n" + "="*60 + f"\nFCI AGGREGATED QC REPORT (RMS 2026-27) - {grade_label}\n" + "="*60
        output_txt += f"\nTOTAL GRAINS SCANNED : {grand_total}\n" + "-"*60 + "\n"
        output_txt += "\n".join(report_lines) + "\n" + "-"*60
        output_txt += f"\nFINAL STATUS: {final_status}\n" + "="*60
        st.code(output_txt, language="text")

        # --- PDF STORAGE HANDOFF ---
        if hasattr(active_module, 'generate_pdf'):
            pdf_bytes = active_module.generate_pdf(grand_total, master_counts, final_status)
        else:
            pdf_bytes = active_module.generate_faq_pdf(grand_total, master_counts, final_status)
            
        st.download_button(
            label="📥 Download Official PDF Report",
            data=pdf_bytes,
            file_name=f"FCI_{grain_type}_{grade_label}_Report_{datetime.now().strftime('%d%m%y')}.pdf",
            mime="application/pdf"
        )

        st.markdown("### 🔍 Grain Detection Visual Inspector")
        for img_name, rgb_img in processed_images_to_show:
            with st.expander(f"📦 View Bounding Boxes for {img_name}", expanded=True):
                st.image(rgb_img, caption=f"AI Layer Map - {img_name}", use_container_width=True)

    if st.button("Reset"):
        st.session_state.page = 'welcome'
        st.rerun()
