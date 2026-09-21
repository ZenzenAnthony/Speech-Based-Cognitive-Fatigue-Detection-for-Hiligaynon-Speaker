import uuid

import streamlit as st

st.set_page_config(page_title="Participant Data Collection", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .wizard-shell {
            background: #FFFFFF;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 1.5rem;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.04);
        }
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid rgba(37, 99, 235, 0.08);
            border-radius: 18px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
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
        div[data-testid="stFileUploaderDropzone"] { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULTS = {
    "session_id": str(uuid.uuid4()),
    "respondent_id": "",
    "language": "Hiligaynon",
    "current_step": 1,
    "current_task_level": "Easy",
    "recorded_audio_bytes": None,
    "samn_perelli_rating": None,
    "inference_results": {},
    "consent_accepted": False,
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


TASK_LEVELS = ["Easy", "Moderate", "Intensive"]
SAMN_SCALE = [1, 2, 3, 4, 5, 6, 7]


def set_step(step):
    st.session_state.current_step = step


def render_consent_step():
    st.header("Step 1 · Informed Consent")
    st.markdown(
        """
        <div class="metric-card">
            <p><strong>Study purpose:</strong> This study examines how speech patterns relate to cognitive fatigue in Hiligaynon speakers.</p>
            <p><strong>Participation:</strong> You will complete a short speech task and provide a self-rated fatigue score.</p>
            <p><strong>Confidentiality:</strong> Your response is de-identified and will be stored securely for research analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    consent = st.checkbox("I have read the consent form and agree to participate.")
    st.session_state.consent_accepted = consent

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Continue", type="primary", use_container_width=True, disabled=not consent):
            set_step(2)
            st.rerun()


def render_screening_step():
    st.header("Step 2 · Screening")
    st.write("Please provide your demographics and language information before the speech task begins.")

    with st.form("screening_form"):
        st.session_state.respondent_id = st.text_input(
            "Respondent ID",
            value=st.session_state.respondent_id,
            placeholder="WVSU_CS_001",
        )
        birthplace = st.text_input("Birthplace", placeholder="Iloilo City")
        native_language = st.text_input("Native Language", placeholder="Hiligaynon")
        hiligaynon_frequency_score = st.slider(
            "How often do you speak Hiligaynon daily?",
            min_value=1,
            max_value=5,
            value=3,
        )
        st.session_state.language = st.radio(
            "Preferred language for instructions",
            options=["Hiligaynon", "English"],
            horizontal=True,
        )

        if st.form_submit_button("Save and continue", type="primary"):
            st.session_state.current_step = 3
            st.rerun()


def render_task_step():
    st.header("Step 3 · Task Wizard")
    task_level = st.selectbox(
        "Select task level",
        TASK_LEVELS,
        index=TASK_LEVELS.index(st.session_state.current_task_level),
    )
    st.session_state.current_task_level = task_level

    if task_level == "Easy":
        st.info("Task 1: Read the short passage in your natural voice. Speak clearly and at a comfortable pace.")
        passage = "Ang mga tawo may pagtuo nga ang edukasyon amo ang susi sang maayong kinabuhi. Magbasa kag magsulti sang tinuod kag mahambal nga mapalit ang kahulugan sang kalibutan."
        st.code(passage)
    elif task_level == "Moderate":
        st.warning("Task 2: Solve the arithmetic prompt silently, then say the final result aloud. Keep the answer concise.")
        st.code("(28 + 13) × 2 = ?")
    else:
        st.error("Task 3: Maintain focus through a multi-domain challenge: counting, problem-solving, and verbal response under time pressure.")
        st.code("Count from 1 to 30, then answer: 7 × 6 + 11 = ?")

    uploaded = st.file_uploader(
        "Record or upload speech audio",
        type=["wav", "mp3", "m4a", "ogg"],
        help="Accepted formats: WAV, MP3, M4A, and OGG.",
    )

    if uploaded is not None:
        st.session_state.recorded_audio_bytes = uploaded.read()
        st.audio(st.session_state.recorded_audio_bytes, format="audio/wav")
        st.success("Audio captured successfully. Proceed to the fatigue rating step.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Previous", use_container_width=True):
            set_step(2)
            st.rerun()
    with col2:
        if st.button("Continue to rating", type="primary", use_container_width=True, disabled=st.session_state.recorded_audio_bytes is None):
            set_step(4)
            st.rerun()


def render_rating_step():
    st.header("Step 4 · Samn-Perelli Fatigue Rating")
    st.write("Rate your current fatigue level immediately after the task.")

    rating = st.radio(
        "Choose the number that best matches your fatigue level",
        options=SAMN_SCALE,
        horizontal=True,
        index=3 if st.session_state.samn_perelli_rating is None else SAMN_SCALE.index(st.session_state.samn_perelli_rating),
    )
    st.session_state.samn_perelli_rating = rating

    st.markdown(
        """
        <div class="metric-card">
            <strong>Scale guide:</strong> 1 = very low fatigue, 7 = extreme fatigue.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back to task", use_container_width=True):
            set_step(3)
            st.rerun()
    with col2:
        if st.button("Finish and debrief", type="primary", use_container_width=True):
            set_step(5)
            st.rerun()


def render_debriefing_step():
    st.header("Step 5 · Debriefing")

    st.success("Thank you for completing the cognitive fatigue task.")
    st.markdown(
        """
        <div class="wizard-shell">
            <p><strong>Summary:</strong> Your session has been recorded. A researcher will review the audio and classification results.</p>
            <p><strong>Respondent ID:</strong> {respondent_id}</p>
            <p><strong>Task Level:</strong> {task_level}</p>
            <p><strong>Samn-Perelli Rating:</strong> {rating}</p>
        </div>
        """.format(
            respondent_id=st.session_state.respondent_id or "Not provided",
            task_level=st.session_state.current_task_level,
            rating=st.session_state.samn_perelli_rating,
        ),
        unsafe_allow_html=True,
    )

    if st.button("Reset session", type="secondary"):
        for key in DEFAULTS:
            st.session_state[key] = DEFAULTS[key]
        st.rerun()


st.title("Participant Data Collection")
st.caption("Study workflow for recording speech, rating fatigue, and completing the research task.")

if not st.session_state.consent_accepted and st.session_state.current_step == 1:
    render_consent_step()
elif st.session_state.current_step == 2:
    render_screening_step()
elif st.session_state.current_step == 3:
    render_task_step()
elif st.session_state.current_step == 4:
    render_rating_step()
else:
    render_debriefing_step()
