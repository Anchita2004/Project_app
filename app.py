import streamlit as st

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="ECG Signal Analysis",
    page_icon="❤️",
    layout="wide"
)

# ---------------------------
# Sidebar
# ---------------------------
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
# HOME PAGE
# ---------------------------
if page == "🏠 Overview":

    st.title("Machine Learning Based ECG Signal Analysis")
    st.subheader("for Heart Health Assessment")

    st.image("images/ecg_banner.png", use_container_width=True)

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

    st.image("images/workflow.png", use_container_width=True)

# ---------------------------
# ABOUT DATASET
# ---------------------------
elif page == "📊 About Dataset":

    import pandas as pd

    st.title("📊 About Dataset")

    st.write("---")

    # =====================================================
    # Dataset Overview
    # =====================================================

    st.subheader("📚 MIT-BIH Arrhythmia Database")

    col1, col2 = st.columns([2,1])

    with col1:

        st.table({
            "Parameter":[
                "Dataset",
                "Patients",
                "ECG Records",
                "Sampling Frequency",
                "Lead Used",
                "Selected Heartbeat Classes"
            ],

            "Value":[
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
            "images/mitbih.png",
            
            use_container_width=True
        )

    st.write("---")

    # =====================================================
    # Selected Beat Classes
    # =====================================================

    st.subheader("📋 Selected Heartbeat Classes")

    annotation_df = pd.DataFrame({

        "Annotation Symbol":[
            "N",
            "A",
            "V",
            "L",
            "R",
            "/"
        ],

        "Description":[
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
The original ECG dataset exhibited a significant class imbalance, with the normal heartbeat class containing substantially more samples than the abnormal heartbeat classes. To address this issue, a hybrid balancing strategy was employed.

- Downsampling was applied to the majority class.
- Upsampling was applied to the minority classes.
- Each selected heartbeat class was balanced to **5000 ECG beats**, resulting in a balanced dataset suitable for CNN training.
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
            "images/pie_before.png",
            use_container_width=True
        )

    with col2:

        st.markdown("#### After Balancing")

        st.image(
            "images/pie_after.png",
            use_container_width=True
        )

    st.write("---")

    

    # =====================================================
    # Dataset Summary
    # =====================================================

    st.subheader("📌 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Selected Classes", "6")

    with col2:
        st.metric("Samples / Class", "5000")

    with col3:
        st.metric("Training Split", "80%")

    with col4:
        st.metric("Testing Split", "20%")

# ---------------------------
# HEARTBEAT CLASSIFICATION
# ---------------------------
elif page == "❤️ Heartbeat Classification":

    import pandas as pd

    st.title("❤️ Heartbeat Classification")

    # ----------------------------------------
    # Information Box
    # ----------------------------------------
    st.info("""
### ECG Dataset Analysis

Upload an ECG heartbeat dataset in **CSV format** for automatic heartbeat analysis.

The uploaded dataset will be analyzed to determine whether it contains only **Normal (N)** heartbeats or any **Abnormal** heartbeat categories.

Supported heartbeat classes:

• N – Normal Beat

• A – Atrial Premature Beat

• V – Ventricular Premature Beat

• L – Left Bundle Branch Block Beat

• R – Right Bundle Branch Block Beat

• / – Paced Beat
""")

    st.write("")

    

    # ----------------------------------------
    # Upload Section
    # ----------------------------------------

    st.subheader("📤 Upload ECG Dataset")

    st.caption("Supported format: CSV | Maximum file size: 200 MB")

    uploaded = st.file_uploader(
        "",
        type=["csv"],
        help="Upload the ECG heartbeat dataset."
    )

    # ----------------------------------------
    # After Upload
    # ----------------------------------------

    if uploaded is not None:

        st.success(f"✅ File Uploaded Successfully : {uploaded.name}")

        # Read CSV
        df = pd.read_csv(uploaded, header=None)

        # Remove header row
        df = df.iloc[1:].reset_index(drop=True)

        LABEL_COLUMN = 3

        labels = df[LABEL_COLUMN].astype(str).str.strip()

        total_beats = len(labels)
        normal_beats = (labels == "N").sum()
        abnormal_beats = total_beats - normal_beats

        st.write("")

        if st.button("🔍 Analyze ECG", use_container_width=True):

            st.subheader("📊 Dataset Summary")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Total Beats", total_beats)

            with c2:
                st.metric("Normal Beats", normal_beats)

            with c3:
                st.metric("Abnormal Beats", abnormal_beats)

            st.write("---")

            st.subheader("🏷️ Detected Beat Types")

            st.write(", ".join(sorted(labels.unique())))

            st.write("---")

            if abnormal_beats == 0:

                st.success("🟢 ECG STATUS : NORMAL")

                st.success("No abnormal heartbeat detected in the uploaded dataset.")

            else:

                st.error("🔴 ECG STATUS : ABNORMAL")

                st.warning(
                    f"{abnormal_beats} abnormal heartbeat(s) detected in the uploaded dataset."
                )

            st.write("---")

            st.subheader("📈 Beat Distribution")

            beat_table = (
                labels.value_counts()
                .rename_axis("Beat Type")
                .reset_index(name="Count")
            )

            st.dataframe(
                beat_table,
                use_container_width=True,
                hide_index=True
            )
# ---------------------------
# MODEL PERFORMANCE
# ---------------------------
elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.write("---")

    # Create two equal columns
    col1, col2 = st.columns(2)

    # ==========================
    # 1D CNN
    # ==========================
    with col1:

        st.subheader("1D Convolutional Neural Network")

        st.metric(
            label="Test Accuracy",
            value="97.25%"
        )

        st.markdown("### Accuracy Plot")
        st.image(
            "images/accuracy_plot_1d.png",
            use_container_width=True
        )

        st.markdown("### Loss Plot")
        st.image(
            "images/loss_plot_1d.png",
            use_container_width=True
        )

        st.markdown("### Confusion Matrix")
        st.image(
            "images/confusion_matrix_1d.png",
            use_container_width=True
        )

    # ==========================
    # 2D CNN
    # ==========================
    with col2:

        st.subheader("2D Convolutional Neural Network")

        st.metric(
            label="Test Accuracy",
            value="96.40%"
        )

        st.markdown("### Accuracy Plot")
        st.image(
            "images/accuracy_plot_2d.png",
            use_container_width=True
        )

        st.markdown("### Loss Plot")
        st.image(
            "images/loss_plot_2d.png",
            use_container_width=True
        )

        st.markdown("### Confusion Matrix")
        st.image(
            "images/confusion_matrix_2d.png",
            use_container_width=True
        )

