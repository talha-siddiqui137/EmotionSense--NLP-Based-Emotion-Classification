import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.predict import predict_emotion

st.set_page_config(page_title="EmotionSense", page_icon="🙂", layout="centered")

EMOTION_EMOJI = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😄",
    "love": "❤️",
    "sadness": "😢",
    "surprise": "😲",
}

EMOTION_COLOR = {
    "anger": "#e74c3c",
    "fear": "#8e44ad",
    "joy": "#f1c40f",
    "love": "#e84393",
    "sadness": "#3498db",
    "surprise": "#e67e22",
}

EXAMPLES = [
    "I am feeling really happy today!",
    "I am so scared right now.",
    "I can't believe you did that, I'm furious.",
    "I miss you so much, you mean everything to me.",
    "I just found out and I am totally shocked.",
    "I feel like nothing will ever get better.",
]

# --- Background + button styling ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e1b2e 0%, #2b1e3d 40%, #1e2a3d 100%);
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #e84393, #8e44ad);
        color: white;
        border: none;
        border-radius: 10px;
        transition: transform 0.15s ease;
    }
    div[data-testid="stButton"] button:hover {
        transform: scale(1.03);
        border: none;
        color: white;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b2e 0%, #2b1e3d 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🙂 EmotionSense")
    st.caption("NLP-Based Emotion Classification")
    st.divider()

    st.markdown("**Model**")
    st.write("TF-IDF + Linear SVM")

    st.markdown("**Accuracy**")
    st.progress(0.90, text="90% on test data")

    st.divider()
    st.markdown("**Detects 6 emotions:**")
    st.write("😠 Anger &nbsp; 😨 Fear &nbsp; 😄 Joy", unsafe_allow_html=True)
    st.write("❤️ Love &nbsp; 😢 Sadness &nbsp; 😲 Surprise", unsafe_allow_html=True)

    st.divider()
    st.caption("Made by Talha Siddiqui")

# --- Main page ---
st.markdown(
    """
    <h1 style='text-align: center; background: linear-gradient(90deg, #e84393, #8e44ad, #3498db);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 48px;'>
        🙂 EmotionSense
    </h1>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #aaa;'>NLP-Based Emotion Classification</p>",
    unsafe_allow_html=True,
)
st.write("")

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

st.write("**Try an example:**")
cols = st.columns(3)
for i, example in enumerate(EXAMPLES):
    if cols[i % 3].button(example, use_container_width=True, key=f"btn_{i}"):
        st.session_state.user_text = example

user_text = st.text_area("Your sentence", key="user_text", height=100)

if st.button("Predict", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Please type something first.")
    else:
        emotion = predict_emotion(user_text)
        emoji = EMOTION_EMOJI.get(emotion, "")
        color = EMOTION_COLOR.get(emotion, "#2ecc71")

        st.markdown(
            f"""
            <div style='background-color:{color}22; border:2px solid {color};
                        border-radius:12px; padding:24px; text-align:center; margin-top:20px;'>
                <p style='font-size:18px; color:gray; margin:0;'>Predicted Emotion</p>
                <p style='font-size:40px; font-weight:bold; margin:8px 0;'>{emoji} {emotion.capitalize()}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.caption("👈 Check the sidebar for Project Details and About Me pages.")