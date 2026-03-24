"""
app.py — Streamlit UI for AI Fake Confidence Detector.
"""

import streamlit as st
import speech_recognition as sr
import pandas as pd
from backend import extract_features, explain_text, SAMPLE_INPUTS

# ──────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Fake Confidence Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────

if "speech_text" not in st.session_state:
    st.session_state.speech_text = ""

# ──────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:          #0d0f14;
    --surface:     #13161e;
    --surface2:    #1a1e2a;
    --border:      #252a38;
    --accent-blue: #4f8ef7;
    --accent-cyan: #00d4c8;
    --green:       #22c55e;
    --green-dim:   #14532d;
    --red:         #ef4444;
    --red-dim:     #7f1d1d;
    --amber:       #f59e0b;
    --text:        #e2e8f0;
    --muted:       #64748b;
    --font:        'Space Grotesk', sans-serif;
    --mono:        'JetBrains Mono', monospace;
}

/* ── App shell ── */
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Main content padding ── */
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1100px; }

/* ── Headings ── */
h1,h2,h3 { font-family: var(--font); letter-spacing: -0.5px; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Text area ── */
textarea {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: var(--font) !important;
    font-size: 15px !important;
}
textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: all .2s ease !important;
}
.stButton > button:hover {
    background: var(--accent-blue) !important;
    border-color: var(--accent-blue) !important;
    color: #fff !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(79,142,247,.3) !important;
}

/* ── Progress bar ── */
.stProgress > div > div { border-radius: 99px !important; }

/* ── Custom card helper ── */
.card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* ── Verdict banners ── */
.verdict-genuine {
    background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
    border: 1px solid #22c55e44;
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: #86efac;
    letter-spacing: -.3px;
}
.verdict-fake {
    background: linear-gradient(135deg, #1c0707 0%, #7f1d1d 100%);
    border: 1px solid #ef444444;
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: #fca5a5;
    letter-spacing: -.3px;
}

/* ── Tag pills ── */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 3px;
    font-family: var(--mono);
}
.pill-red   { background: #7f1d1d55; color: #fca5a5; border: 1px solid #ef444455; }
.pill-amber { background: #78350f55; color: #fde68a; border: 1px solid #f59e0b55; }
.pill-green { background: #052e1655; color: #86efac; border: 1px solid #22c55e55; }

/* ── Metric grid ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-box {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: var(--mono);
}
.metric-label {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: .5px;
}

/* ── Signal bar ── */
.signal-label {
    font-size: .85rem;
    color: var(--muted);
    margin-bottom: 3px;
    font-weight: 600;
}
.bar-track {
    background: var(--surface);
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
    margin-bottom: .75rem;
}
.bar-fill-red   { height:100%; border-radius:99px; background: linear-gradient(90deg,#7f1d1d,#ef4444); transition: width .6s ease; }
.bar-fill-amber { height:100%; border-radius:99px; background: linear-gradient(90deg,#78350f,#f59e0b); transition: width .6s ease; }
.bar-fill-green { height:100%; border-radius:99px; background: linear-gradient(90deg,#052e16,#22c55e); transition: width .6s ease; }

/* ── Section title ── */
.section-title {
    font-size: .75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--muted);
    margin: 1.5rem 0 .6rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper rendering functions
# ──────────────────────────────────────────────
def render_pills(words: list, pill_class: str) -> str:
    if not words:
        return "<span style='color:#64748b;font-size:.85rem'>None detected</span>"
    return "".join(f'<span class="pill {pill_class}">{w}</span>' for w in words)


def render_signal_bars(signal_scores: dict) -> str:
    color_map = {
        "Hesitation Score":   "bar-fill-red",
        "Exaggeration Score": "bar-fill-amber",
        "Genuine Signal":     "bar-fill-green",
    }

    html = ""
    for label, cls in color_map.items():
        val = signal_scores.get(label, 0)

        try:
            val = float(val)
        except:
            val = 0

        val = max(0, min(val, 100))  # clamp between 0–100

        html += f"""
        <div class="signal-label">
            {label} — <span style="font-family:var(--mono);font-size:.85rem">{int(val)}%</span>
        </div>
        <div class="bar-track">
            <div class="{cls}" style="width:{val}%"></div>
        </div>
        """

    return html


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Fake Confidence AI")
    st.markdown("<hr>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Analyzer", "📖 About"],
        label_visibility="collapsed",
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#64748b'>AI · NLP · Scikit-learn<br>Built with Streamlit</small>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# HOME
# ──────────────────────────────────────────────

if page == "🏠 Home":

    st.markdown("# 🧠 AI Fake Confidence Detector")
    st.markdown(
        "<p style='color:#64748b;font-size:1.1rem;margin-top:-8px'>"
        "Detect whether confidence in speech is genuine or fabricated — "
        "using NLP, TF-IDF, and Logistic Regression.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card" style="border-left:4px solid #22c55e">
            <div style="font-size:1.5rem">✅</div>
            <div style="font-weight:700;margin:8px 0 4px">Genuine Confidence</div>
            <div style="color:#64748b;font-size:.9rem">Grounded in preparation, experience, and clear reasoning.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="border-left:4px solid #ef4444">
            <div style="font-size:1.5rem">❌</div>
            <div style="font-weight:700;margin:8px 0 4px">Fake (Hesitant)</div>
            <div style="color:#64748b;font-size:.9rem">Filled with uncertainty words, guessing, and unclear phrasing.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card" style="border-left:4px solid #f59e0b">
            <div style="font-size:1.5rem">⚠️</div>
            <div style="font-weight:700;margin:8px 0 4px">Fake (Over-confident)</div>
            <div style="color:#64748b;font-size:.9rem">Exaggerated claims, absolute language, or hollow reassurances.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Head to **🔍 Analyzer** in the sidebar to test the system.")


# ──────────────────────────────────────────────
# ANALYZER
# ──────────────────────────────────────────────

elif page == "🔍 Analyzer":

    st.markdown("## 🔍 Speech Confidence Analyzer")
    st.markdown(
        "<p style='color:#64748b;margin-top:-8px'>Enter or record speech text to detect fake vs genuine confidence.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Sample inputs ──
    st.markdown('<div class="section-title">Try a sample</div>', unsafe_allow_html=True)
    sample_cols = st.columns(len(SAMPLE_INPUTS))
    for i, (label, sample) in enumerate(SAMPLE_INPUTS.items()):
        if sample_cols[i].button(label, use_container_width=True):
            st.session_state.speech_text = sample
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Text input ──
    speech_text = st.text_area(
        "Speech / Text Input",
        value=st.session_state.speech_text,
        height=160,
        placeholder="Type or paste speech text here… e.g. 'I practiced this topic and I can explain each step clearly.'",
    )

    # ── Action buttons ──
    b1, b2, b3 = st.columns([1, 1, 1])
    analyze_btn = b1.button("⚡ Analyze", use_container_width=True)
    mic_btn     = b2.button("🎙 Record Mic", use_container_width=True)
    clear_btn   = b3.button("🗑 Clear", use_container_width=True)

    if clear_btn:
        st.session_state.speech_text = ""
        st.rerun()

    # ── Microphone ──
    if mic_btn:
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                with st.spinner("🎤 Listening for up to 15 seconds…"):
                    audio = recognizer.listen(source, timeout=15)
            text = recognizer.recognize_google(audio)
            st.session_state.speech_text = text
            st.success(f"Transcribed: **{text}**")
            st.rerun()
        except sr.WaitTimeoutError:
            st.error("No speech detected — try again.")
        except sr.UnknownValueError:
            st.error("Could not understand audio — please speak clearly.")
        except Exception as e:
            st.error(f"Microphone error: {e}")

    # ── Analysis ──
    if analyze_btn:
        if not speech_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing speech patterns…"):
                prediction, confidence_score = extract_features(speech_text)
                explanation = explain_text(speech_text)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Verdict banner ──
            if prediction == "genuine":
                st.markdown(
                    '<div class="verdict-genuine">✅ &nbsp; Genuine Confidence Detected</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="verdict-fake">❌ &nbsp; Fake Confidence Detected</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Metrics row ──
            color = "#22c55e" if prediction == "genuine" else "#ef4444"
            label = "GENUINE" if prediction == "genuine" else "FAKE"
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-value" style="color:{color}">{label}</div>
                    <div class="metric-label">Prediction</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color:#4f8ef7">{confidence_score}%</div>
                    <div class="metric-label">Model Confidence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color:#f59e0b">{len(speech_text.split())}</div>
                    <div class="metric-label">Word Count</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence meter ──
            st.markdown('<div class="section-title">Model Confidence Meter</div>', unsafe_allow_html=True)
            st.progress(int(confidence_score) / 100)

           # ── Signal bars ──
            st.markdown('<div class="section-title">Linguistic Signal Analysis</div>', unsafe_allow_html=True)

            html_content = render_signal_bars(explanation["signal_scores"])

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(html_content, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Word detection ──
            st.markdown('<div class="section-title">Detected Patterns</div>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)

            with d1:
                st.markdown(
                    f"""<div class="card">
                        <div style="font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#ef4444;margin-bottom:8px">🔴 Hesitation Words</div>
                        {render_pills(explanation['hesitation_words'], 'pill-red')}
                    </div>""",
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown(
                    f"""<div class="card">
                        <div style="font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#f59e0b;margin-bottom:8px">🟡 Exaggeration Words</div>
                        {render_pills(explanation['exaggeration_words'], 'pill-amber')}
                    </div>""",
                    unsafe_allow_html=True,
                )
            with d3:
                st.markdown(
                    f"""<div class="card">
                        <div style="font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#22c55e;margin-bottom:8px">🟢 Genuine Markers</div>
                        {render_pills(explanation['genuine_markers'], 'pill-green')}
                    </div>""",
                    unsafe_allow_html=True,
                )

            # ── Reasoning ──
            st.markdown('<div class="section-title">Why this prediction?</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="card" style="border-left:4px solid #4f8ef7;color:#cbd5e1;line-height:1.7">'
                f'🧠 &nbsp;{explanation["reasoning"]}</div>',
                unsafe_allow_html=True,
            )

            # ── Download ──
            st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
            df_result = pd.DataFrame({
                "Speech Text":      [speech_text],
                "Prediction":       [prediction],
                "Confidence Score": [confidence_score],
                "Hesitation Words": [", ".join(explanation["hesitation_words"])],
                "Exaggeration Words": [", ".join(explanation["exaggeration_words"])],
                "Genuine Markers":  [", ".join(explanation["genuine_markers"])],
                "Reasoning":        [explanation["reasoning"]],
            })
            st.download_button(
                "⬇ Download Result as CSV",
                df_result.to_csv(index=False),
                file_name="confidence_detection_result.csv",
                mime="text/csv",
            )


# ──────────────────────────────────────────────
# ABOUT
# ──────────────────────────────────────────────

elif page == "📖 About":

    st.markdown("## 📖 About This Project")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3 style="margin-top:0">🧠 AI Fake Confidence Detection in Speech</h3>
        <p style="color:#94a3b8;line-height:1.8">
        This capstone project detects whether confidence expressed in speech is
        <strong>genuine</strong> or <strong>fake</strong> using Natural Language Processing
        and Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
            <div style="font-weight:700;margin-bottom:.6rem">⚙️ Tech Stack</div>
            <div style="color:#94a3b8;font-size:.9rem;line-height:2">
            • Python 3.10+<br>
            • Streamlit — UI framework<br>
            • Scikit-learn — ML pipeline<br>
            • TF-IDF (1–3 ngrams) — feature extraction<br>
            • Logistic Regression — classifier<br>
            • SpeechRecognition — voice input
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <div style="font-weight:700;margin-bottom:.6rem">🔍 Detection Signals</div>
            <div style="color:#94a3b8;font-size:.9rem;line-height:2">
            • Hesitation language (um, uh, maybe…)<br>
            • Exaggeration phrases (always, never…)<br>
            • Genuine preparation markers<br>
            • Sentence structure & length<br>
            • Uncertainty ratio in text<br>
            • TF-IDF vocabulary patterns
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="border-left:4px solid #4f8ef7">
        <div style="font-weight:700;margin-bottom:.4rem">📂 Project Structure</div>
        <pre style="color:#94a3b8;font-size:.85rem;margin:0;font-family:var(--mono)">
AI-FAKE-CONFIDENCE-DETECTION/
├── app.py            ← Streamlit UI (this file)
├── backend.py        ← Feature extraction & explanation
├── train_model.py    ← Model training script
├── dataset.csv       ← Labelled training data
├── requirements.txt  ← Dependencies
└── models/
    └── confidence_model.pkl
        </pre>
    </div>
    """, unsafe_allow_html=True)
