import streamlit as st
import threading

from capture_faces import capture_person
from augment_dataset import augment_data
from build_dataset import build_embeddings
from train_model import train
from recognize_live import start_recognition

st.set_page_config(layout="wide")

st.markdown("""
<style>
button {
    height: 50px;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎭 Face Recognition Demo")

# Layout similar to your image
col1, col2 = st.columns(2)

# =========================
# LEFT PANEL (Training)
# =========================
with col1:
    st.header("📸 Face Registration")

    name = st.text_input("Person Name")

    if st.button("1️⃣ Capture Faces"):
        if name:
            capture_person(name)
            st.success("Images captured!")
        else:
            st.warning("Enter a name first")

    if st.button("2️⃣ Data Augmentation"):
        augment_data(name if name else None)
        st.success("Augmentation done!")

    if st.button("3️⃣ Generate Embeddings"):
        with st.spinner("Processing images..."):
            build_embeddings()
        st.success("Embeddings ready!")

    if st.button("4️⃣ Train Model"):
        with st.spinner("Training model..."):
            train()
        st.success("Model trained!")

from recognize_live import recognize_streamlit
from capture_faces import capture_streamlit

with col2:
    mode = st.radio("Mode", ["Recognition", "Capture"])

    name = st.text_input("Person Name (for capture)")

    if mode == "Recognition":
        if st.button("Start Recognition"):
            recognize_streamlit()

    elif mode == "Capture":
        if st.button("Start Capture"):
            if name:
                capture_streamlit(name)
            else:
                st.warning("Enter a name")