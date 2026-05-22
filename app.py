import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Import specialized logic files
from grains.wheat import faq_logic
from grains.wheat import urs_logic

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
        
        # 1. DYNAMIC LOGIC ROUTING
        if grain_type == "Wheat":
            if grade_label == "URS":
                active_module = urs_logic
                active_norms = getattr(urs_logic, 'WHEAT_URS_NORMS', {})
            else:
                active_module = faq_logic
                active_norms = getattr(faq_logic, 'WHEAT_NORMS', {})
        else:
            st.error("Rice modules are not configured in this view.")
            st.stop()

        if not active_norms:
            st.error(f"Error: Specifications dictionary for {grain_type} ({grade_label}) could not be found.")
            st.stop()

        # Initialize tracking counters safely
        master_counts = {name: 0 for name in active_module.CLASS_MAP.values()}
        file_stats = []
        grand_total = 0
        processed_images_to_show = [] 

        # 2. IMAGE SCANNING ENGINE
        with st.spinner("Applying Deep Scan (Slicing & Enhancement)..."):
            for f in files:
                # FIX: Reset buffer pointer before reading to avoid duplicate array allocation
                f.seek(0)
                file_bytes = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    st.error(f"Could not decode image: {f.name}")
                    continue

                # Run inference via the active module
                preds, output_visual_img = active_module.analyze_sample(img, model)
                grand_total += len(preds)
                
                for label in preds:
                    if label in master_counts:
                        master_counts[label] += 1
                        
                file_stats.append(f"Processed {f.name}: {len(preds)} grains.")
                
                # Convert BGR to RGB for Streamlit rendering
                rgb_output = cv2.cvtColor(output_visual_img, cv2.COLOR_BGR2RGB)
                processed_images_to_show.append((f.name, rgb_output))

        # 3. DYNAMIC EVALUATION LOOP
        rej_reasons = []
        report_lines = []
        
        damage_pct = (master_counts.get('Damage', 0) / grand_total * 100) if grand_total > 0 else 0.0
        slightly_damage_pct = (master_counts.get('Slightly Damage', 0) / grand_total * 100) if grand_total > 0 else 0.0
        combined_damage_pct = damage_pct + slightly_damage_pct

        for cat, limit in active_norms.items():
            current_category = cat.strip()
            
            if grand_total > 0:
                if grain_type == "Wheat" and current_category == 'Shrivelled & Broken':
                    val = ((master_counts.get('Shrivelled', 0) + master_counts.get('Broken', 0)) / grand_total) * 100
                elif grain_type == "Wheat" and grade_label == "URS" and current_category == 'Damage & Slightly Damage':
                    val = combined_damage_pct
                else:
                    val = (master_counts.get(current_category, 0) / grand_total) * 100
            else:
                val = 0.0
            
            status = "OK"
            
            # Validation rules
            if grain_type == "Wheat" and grade_label == "URS" and current_category == 'Damage & Slightly Damage':
                if combined_damage_pct > 6.0:
                    status = "!! EXCEEDS LIMIT !!"
                    rej_reasons.append(cat)
            else:
                if val > limit:
                    status = "!! EXCEEDS LIMIT !!"
                    rej_reasons.append(cat)
                    
            report_lines.append(f"{cat.ljust(25)} : {val:5.2f}% | Limit: {limit:5.2f}% | {status}")

        final_status = "REJECTED" if rej_reasons else f"ACCEPTED ({grade_label})"

        # 4. PRINT TERMINAL VIEW
        output_txt = f"--- STARTING {grain_type.upper()} ANALYSIS ---\n" + "\n".join(file_stats)
        output_txt += "\n\n" + "="*60 + f"\nFCI AGGREGATED QC REPORT (RMS 2026-27) - {grade_label}\n" + "="*60
        output_txt += f"\nTOTAL GRAINS SCANNED : {grand_total}\n" + "-"*60 + "\n"
        output_txt += "\n".join(report_lines) + "\n" + "-"*60
        output_txt += f"\nFINAL STATUS: {final_status}\n" + "="*60
        st.code(output_txt, language="text")

        # 5. DYNAMIC PDF GENERATION CALL
        if hasattr(active_module, 'generate_faq_pdf'):
            pdf_bytes = active_module.generate_faq_pdf(grand_total, master_counts, final_status)
        else:
            pdf_bytes = active_module.generate_pdf(grand_total, master_counts, final_status)
            
        st.download_button(
            label="📥 Download Official PDF Report",
            data=pdf_bytes,
            file_name=f"FCI_{grain_type}_{grade_label}_Report_{datetime.now().strftime('%d%m%y')}.pdf",
            mime="application/pdf"
        )

        st.markdown("### 🔍 Grain Detection Visual Inspector")
        for img_name, rgb_img in processed_images_to_show:
            with st.expander(f"📦 View Bounding Boxes for {img_name}", expanded=True):
                st.image(rgb_img, caption=f"AI Detection Output Layer - {img_name}", use_container_width=True)

    if st.button("Reset"):
        st.session_state.page = 'welcome'
        st.rerun()