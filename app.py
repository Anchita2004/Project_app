import streamlit as st
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

# -----------------------------------
# Load CNN Model
# -----------------------------------

@st.cache_resource
def load_cnn_model():
    return load_model("ecg_1dcnn.keras")

model = load_cnn_model()

# -----------------------------------
# Load Class Labels
# -----------------------------------

class_names = np.load(
    "label_classes.npy",
    allow_pickle=True
)

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="ECG Signal Analysis",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.title("❤️ ECG Signal Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 About Dataset",
        "❤️ Heartbeat Classification",
        "📈 Model Performance"
    ]
)

# ---------------------------
# OVERVIEW PAGE
# ---------------------------

if page == "🏠 Overview":

    st.title("Machine Learning Based ECG Signal Analysis")
    st.subheader("for Heart Health Assessment")

    st.image("ecg_banner.png", use_container_width=True)

    st.write("---")

    st.markdown("""
### Project Overview

This project presents a machine learning-based system for automatic ECG heartbeat classification using a One-Dimensional Convolutional Neural Network (1D CNN). The model is trained on the MIT-BIH Arrhythmia Database and classifies six clinically significant heartbeat categories.

The developed system performs:

- ECG Beat Segmentation
- Annotation Mapping
- Dataset Balancing
- CNN-based Classification
- Normal vs Abnormal Detection
""")

    st.write("---")

    st.subheader("Project Workflow")

    st.image(
        "workflow.png",
        use_container_width=True
    )

# ---------------------------
# ABOUT DATASET
# ---------------------------

elif page == "📊 About Dataset":

    st.title("📊 About Dataset")

    st.write("---")

    # =====================================================
    # Dataset Overview
    # =====================================================

    st.subheader("📚 MIT-BIH Arrhythmia Database")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.table({
            "Parameter": [
                "Dataset",
                "Patients",
                "ECG Records",
                "Sampling Frequency",
                "Lead Used",
                "Selected Heartbeat Classes"
            ],

            "Value": [
                "MIT-BIH Arrhythmia Database",
                "47 Patients",
                "48 ECG Records",
                "360 Hz",
                "MLII (V5 when MLII unavailable)",
                "N, A, V, L, R, /"
            ]
        })

    with col2:

        st.image(
            "mitbih.png",
            use_container_width=True
        )

    st.write("---")

    # =====================================================
    # Selected Beat Classes
    # =====================================================

    st.subheader("📋 Selected Heartbeat Classes")

    annotation_df = pd.DataFrame({

        "Annotation Symbol": [
            "N",
            "A",
            "V",
            "L",
            "R",
            "/"
        ],

        "Description": [
            "Normal Beat",
            "Atrial Premature Beat",
            "Ventricular Premature Beat",
            "Left Bundle Branch Block Beat",
            "Right Bundle Branch Block Beat",
            "Paced Beat"
        ]

    })

    st.dataframe(
        annotation_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("---")

    # =====================================================
    # Dataset Balancing
    # =====================================================

    st.subheader("⚖️ Dataset Balancing")

    st.write("""
The original ECG dataset exhibited a significant class imbalance, with the normal heartbeat class containing substantially more samples than the abnormal heartbeat classes.

To address this issue:

- Downsampling was applied to the majority class.
- Upsampling was applied to the minority classes.
- Each selected heartbeat class was balanced to **5000 ECG beats**.

This produced a balanced dataset for CNN training.
""")

    st.write("---")

    # =====================================================
    # Dataset Distribution
    # =====================================================

    st.subheader("📈 ECG Beat Distribution")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Before Balancing")

        st.image(
            "pie_before.png",
            use_container_width=True
        )

    with col2:

        st.markdown("#### After Balancing")

        st.image(
            "pie_after.png",
            use_container_width=True
        )

    st.write("---")

    # =====================================================
    # Dataset Summary
    # =====================================================

    st.subheader("📌 Dataset Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Selected Classes", "6")
    c2.metric("Samples / Class", "5000")
    c3.metric("Training Split", "80%")
    c4.metric("Testing Split", "20%")
# ---------------------------
# HEARTBEAT CLASSIFICATION
# ---------------------------

elif page == "❤️ Heartbeat Classification":

    st.title("❤️ Heartbeat Classification")

    st.info("""
### ECG Heartbeat Classification

Upload a segmented ECG dataset in CSV format.

The trained 1D CNN model will automatically classify every ECG beat.

Supported heartbeat classes

• N - Normal Beat

• A - Atrial Premature Beat

• V - Ventricular Premature Beat

• L - Left Bundle Branch Block Beat

• R - Right Bundle Branch Block Beat

• / - Paced Beat

Dataset Format

Patient_ID | Start_Point | End_Point | Label | ECG Signal Columns | Label
""")

    st.write("")

    st.subheader("📤 Upload ECG Dataset")

    uploaded = st.file_uploader(
        "",
        type=["csv"]
    )

    if uploaded is not None:

        st.success(f"Uploaded File : {uploaded.name}")

        df = pd.read_csv(uploaded)

        st.write("")

        st.write("Dataset Shape :", df.shape)

        if st.button("🔍 Analyze ECG", use_container_width=True):

            try:

                # -------------------------------------
                # Extract ECG signal columns
                # -------------------------------------

                X = df.iloc[:, 4:-1].astype("float32").values

                X = X.reshape(X.shape[0], X.shape[1], 1)

                # -------------------------------------
                # CNN Prediction
                # -------------------------------------

                with st.spinner("Running 1D CNN..."):

                    predictions = model.predict(
                        X,
                        verbose=0
                    )

                predicted_index = np.argmax(
                    predictions,
                    axis=1
                )

                labels = class_names[predicted_index]

                # -------------------------------------
                # Summary
                # -------------------------------------

                total_beats = len(labels)

                normal_beats = np.sum(labels == "N")

                abnormal_beats = total_beats - normal_beats

                st.write("---")

                st.subheader("📊 Dataset Summary")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Beats",
                    total_beats
                )

                c2.metric(
                    "Normal Beats",
                    normal_beats
                )

                c3.metric(
                    "Abnormal Beats",
                    abnormal_beats
                )

                st.write("---")

                st.subheader("🏷️ Predicted Beat Types")

                st.write(", ".join(sorted(np.unique(labels))))

                st.write("---")

                if abnormal_beats == 0:

                    st.success("🟢 ECG STATUS : NORMAL")

                    st.success(
                        "No abnormal heartbeat detected."
                    )

                else:

                    st.error("🔴 ECG STATUS : ABNORMAL")

                    st.warning(
                        f"{abnormal_beats} abnormal heartbeat(s) detected."
                    )

                st.write("---")

                st.subheader("📈 Beat Distribution")

                beat_table = (
                    pd.Series(labels)
                    .value_counts()
                    .rename_axis("Beat Type")
                    .reset_index(name="Count")
                )

                st.dataframe(
                    beat_table,
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as e:

                st.error("Prediction Failed")

                st.exception(e)
# ---------------------------
# HEARTBEAT CLASSIFICATION
# ---------------------------

elif page == "❤️ Heartbeat Classification":

    st.title("❤️ Heartbeat Classification")

    st.info("""
### ECG Heartbeat Classification

Upload a segmented ECG dataset in CSV format.

The trained 1D CNN model will automatically classify every ECG beat.

Supported heartbeat classes

• N - Normal Beat

• A - Atrial Premature Beat

• V - Ventricular Premature Beat

• L - Left Bundle Branch Block Beat

• R - Right Bundle Branch Block Beat

• / - Paced Beat

Dataset Format

Patient_ID | Start_Point | End_Point | Label | ECG Signal Columns | Label
""")

    st.write("")

    st.subheader("📤 Upload ECG Dataset")

    uploaded = st.file_uploader(
        "",
        type=["csv"]
    )

    if uploaded is not None:

        st.success(f"Uploaded File : {uploaded.name}")

        df = pd.read_csv(uploaded)

        st.write("")

        st.write("Dataset Shape :", df.shape)

        if st.button("🔍 Analyze ECG", use_container_width=True):

            try:

                # -------------------------------------
                # Extract ECG signal columns
                # -------------------------------------

                X = df.iloc[:, 4:-1].astype("float32").values

                X = X.reshape(X.shape[0], X.shape[1], 1)

                # -------------------------------------
                # CNN Prediction
                # -------------------------------------

                with st.spinner("Running 1D CNN..."):

                    predictions = model.predict(
                        X,
                        verbose=0
                    )

                predicted_index = np.argmax(
                    predictions,
                    axis=1
                )

                labels = class_names[predicted_index]

                # -------------------------------------
                # Summary
                # -------------------------------------

                total_beats = len(labels)

                normal_beats = np.sum(labels == "N")

                abnormal_beats = total_beats - normal_beats

                st.write("---")

                st.subheader("📊 Dataset Summary")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Beats",
                    total_beats
                )

                c2.metric(
                    "Normal Beats",
                    normal_beats
                )

                c3.metric(
                    "Abnormal Beats",
                    abnormal_beats
                )

                st.write("---")

                st.subheader("🏷️ Predicted Beat Types")

                st.write(", ".join(sorted(np.unique(labels))))

                st.write("---")

                if abnormal_beats == 0:

                    st.success("🟢 ECG STATUS : NORMAL")

                    st.success(
                        "No abnormal heartbeat detected."
                    )

                else:

                    st.error("🔴 ECG STATUS : ABNORMAL")

                    st.warning(
                        f"{abnormal_beats} abnormal heartbeat(s) detected."
                    )

                st.write("---")

                st.subheader("📈 Beat Distribution")

                beat_table = (
                    pd.Series(labels)
                    .value_counts()
                    .rename_axis("Beat Type")
                    .reset_index(name="Count")
                )

                st.dataframe(
                    beat_table,
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as e:

                st.error("Prediction Failed")

                st.exception(e)