import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# --- PAGE CONFIG ---
st.set_page_config(page_title="Digital Parkhi 2.0", page_icon="🌾", layout="wide")

# --- SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = 'welcome'
if 'grain' not in st.session_state: st.session_state.grain = None

@st.cache_resource
def load_global_model():
    return YOLO("best.pt")
model = load_global_model()

# --- NAVIGATION ---
if st.session_state.page == 'welcome':
    st.title("Digital Parkhi 2.0")
    st.info("AI-Powered Grain Quality Control")
    if st.button("Start Analysis"):
        st.session_state.page = 'select_grain'
        st.rerun()

elif st.session_state.page == 'select_grain':
    st.header("Select Grain Type")
    grains = ["Wheat", "Rice", "Paddy", "Maize"]
    cols = st.columns(len(grains))
    for i, g in enumerate(grains):
        if cols[i].button(g):
            st.session_state.grain = g
            st.session_state.page = 'select_cat'
            st.rerun()

elif st.session_state.page == 'select_cat':
    st.header(f"Select Category for {st.session_state.grain}")
    if st.session_state.grain == "Wheat":
        opts = ["FAQ", "URS"]
    else:
        opts = ["RRC", "RBC", "RRA", "RBA", "FRK RBC", "FRK RBA"]
    
    cat = st.selectbox("Choose Category", opts)
    if st.button("Proceed"):
        st.session_state.cat = cat
        st.session_state.page = 'upload'
        st.rerun()

elif st.session_state.page == 'upload':
    st.header(f"Upload {st.session_state.grain} - {st.session_state.cat}")
    files = st.file_uploader("Select 4-5 images", accept_multiple_files=True)
    
    if st.button("Run Analysis") and files:
        cv_imgs = [cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), 1) for f in files]
        
        # --- DYNAMIC IMPORT LOGIC ---
        if st.session_state.grain == "Wheat":
            if st.session_state.cat == "FAQ":
                from grains.wheat.faq_logic import analyze_faq as scan
            else:
                from grains.wheat.urs_logic import analyze_urs as scan
        elif st.session_state.grain == "Rice":
            if "FRK" in st.session_state.cat:
                from grains.rice.frk_logic import analyze_frk as scan
            elif st.session_state.cat == "RRC":
                from grains.rice.rrc_logic import analyze_rrc as scan
            # (Add other elifs for RBC, RRA, RBA)
            
        counts, total, norms, status = scan(cv_imgs, model)
        st.write(counts) # Displaying raw counts for now