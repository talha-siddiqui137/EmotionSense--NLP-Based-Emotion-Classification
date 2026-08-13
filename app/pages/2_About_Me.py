import streamlit as st

st.set_page_config(page_title="About Me", page_icon="👤", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e1b2e 0%, #2b1e3d 40%, #1e2a3d 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b2e 0%, #2b1e3d 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align:center;'>👤 About Me</h1>",
    unsafe_allow_html=True,
)
st.write("")

st.subheader("Talha Siddiqui")
st.write(
    "I'm a Software Engineering student at NED University, focused on AI/ML and "
    "Data Science. EmotionSense is one of my portfolio projects, built to practice "
    "NLP and traditional machine learning end-to-end — from raw data to a deployed app."
)

st.subheader("Tech stack used in this project")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        "- Python\n"
        "- pandas\n"
        "- scikit-learn\n"
        "- NLTK\n"
    )
with col2:
    st.markdown(
        "- Streamlit\n"
        "- joblib\n"
        "- TF-IDF\n"
        "- Linear SVM\n"
    )

st.subheader("Links")
st.markdown(
    "🔗 [Project GitHub Repo](https://github.com/talha-siddiqui137/EmotionSense--NLP-Based-Emotion-Classification)"
)
st.markdown("💼 [LinkedIn](https://www.linkedin.com/in/talha-siddiqui137/)")
st.markdown("🐙 [My GitHub Profile](https://github.com/talha-siddiqui137)")
st.markdown("📧 talha03182301690@gmail.com")