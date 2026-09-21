import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Researcher Diagnostic Studio", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .card-panel {
            background: #FFFFFF;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 1.25rem;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.04);
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .status-low { background: rgba(16, 185, 129, 0.12); color: #047857; }
        .status-moderate { background: rgba(245, 158, 11, 0.14); color: #b45309; }
        .status-high { background: rgba(239, 68, 68, 0.12); color: #b91c1c; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Researcher Diagnostic Studio")
st.caption("Audio review, Mel-spectrogram analysis, attention diagnostics, and logged sessions")

uploaded_audio = st.file_uploader(
    "Upload recorded audio",
    type=["wav", "mp3", "m4a", "ogg"],
    help="Use a speech sample to inspect model features and fatigue prediction output.",
)

if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/wav")

left, right = st.columns([1.5, 1])

with left:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.subheader("Processing timeline")
    with st.status("Preparing spectrogram and temporal attention analysis", expanded=True):
        st.write("1. Audio validation complete")
        st.write("2. 16 kHz standardization applied")
        st.write("3. Mel-spectrogram generated")
        st.write("4. Temporal attention map computed")
    st.markdown('</div>', unsafe_allow_html=True)

    spectrogram = np.random.default_rng(7).uniform(0, 1, size=(128, 1500)).astype(float)
    fig = px.imshow(
        spectrogram,
        color_continuous_scale="Viridis",
        aspect="auto",
        labels={"x": "Time Frames", "y": "Mel Bins"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    attention = np.linspace(0.2, 0.95, 1500)
    attention_fig = px.line(
        x=list(range(len(attention))),
        y=attention,
        labels={"x": "Time Index", "y": "Attention Weight"},
    )
    attention_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(attention_fig, use_container_width=True)

with right:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.subheader("Fatigue classification")

    prediction = "Moderate"
    confidence = 0.82
    status_class = "status-moderate"

    if prediction == "Low":
        status_class = "status-low"
    elif prediction == "High":
        status_class = "status-high"

    st.markdown(f'<div class="status-pill {status_class}">{prediction}</div>', unsafe_allow_html=True)
    st.progress(confidence)
    st.caption(f"Confidence: {confidence * 100:.1f}%")

    st.metric("Acoustic energy", "−18.4 dB")
    st.metric("Pitch variability", "27.8 Hz")
    st.metric("Speech tempo", "3.2 syllables/s")
    st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Session explorer")

session_rows = [
    {
        "session_id": "A-001",
        "respondent_id": "WVSU_CS_001",
        "task_level": "Easy",
        "ground_truth_score": 3,
        "predicted_fatigue": "Low",
    },
    {
        "session_id": "A-002",
        "respondent_id": "WVSU_CS_002",
        "task_level": "Moderate",
        "ground_truth_score": 5,
        "predicted_fatigue": "Moderate",
    },
    {
        "session_id": "A-003",
        "respondent_id": "WVSU_CS_003",
        "task_level": "Intensive",
        "ground_truth_score": 6,
        "predicted_fatigue": "High",
    },
]

session_df = pd.DataFrame(session_rows)
filtered_df = st.dataframe(session_df, use_container_width=True)

csv = session_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download dataset (.CSV)",
    data=csv,
    file_name="fatigue_session_logs.csv",
    mime="text/csv",
)
