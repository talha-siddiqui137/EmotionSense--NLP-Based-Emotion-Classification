import streamlit as st

from src.predict import predict_emotion

st.set_page_config(page_title="EmotionSense", page_icon="🙂", layout="centered")

# Emoji for each emotion, used to make the result look nicer
EMOTION_EMOJI = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😄",
    "love": "❤️",
    "sadness": "😢",
    "surprise": "😲",
}

# Example sentences the user can click instead of typing
EXAMPLES = [
    "I am feeling really happy today!",
    "I am so scared right now.",
    "I can't believe you did that, I'm furious.",
    "I miss you so much, you mean everything to me.",
    "I just found out and I am totally shocked.",
    "I feel like nothing will ever get better.",
]

st.title("🙂 EmotionSense")
st.caption("NLP-Based Emotion Classification")

st.write("Type a sentence, or click one of the examples below.")

# Keep the typed text in memory across clicks
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

st.write("**Try an example:**")
cols = st.columns(3)
for i, example in enumerate(EXAMPLES):
    if cols[i % 3].button(example, use_container_width=True):
        st.session_state.user_text = example

user_text = st.text_area("Your sentence", key="user_text", height=100)

if st.button("Predict", type="primary"):
    if not user_text.strip():
        st.warning("Please type something first.")
    else:
        emotion = predict_emotion(user_text)
        emoji = EMOTION_EMOJI.get(emotion, "")
        st.success(f"Predicted Emotion: **{emotion.capitalize()}** {emoji}")

with st.expander("About this project"):
    st.write(
        "EmotionSense classifies text into one of six emotions: "
        "anger, fear, joy, love, sadness, or surprise. "
        "It uses TF-IDF features with a Linear SVM model, trained on "
        "16,000 labeled sentences, reaching about 90% accuracy on unseen test data."
    )