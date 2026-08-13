import streamlit as st

st.set_page_config(page_title="Project Details", page_icon="📊", layout="centered")

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
    "<h1 style='text-align:center;'>📊 Project Details</h1>",
    unsafe_allow_html=True,
)
st.write("")

st.subheader("What this project does")
st.write(
    "EmotionSense reads a sentence of text and predicts which of 6 emotions "
    "it expresses: anger, fear, joy, love, sadness, or surprise."
)

st.subheader("Dataset")
st.write(
    "16,000 labeled sentences for training and 2,000 for testing, "
    "from the dair-ai/emotion dataset."
)

st.subheader("Pipeline")
st.code(
    "Raw text\n"
    "  -> Preprocessing (lowercase, remove punctuation/numbers/emojis, remove stopwords)\n"
    "  -> TF-IDF Vectorization\n"
    "  -> Linear SVM\n"
    "  -> Predicted emotion",
    language=None,
)

st.subheader("Models I tested")
st.write("I compared 3 models, each with 2 feature techniques:")

st.markdown(
    """
| Model | Bag of Words | TF-IDF |
|---|---|---|
| Multinomial Naive Bayes | 76.8% | 66.1% |
| Logistic Regression | 88.9% | 86.2% |
| Linear SVM | 89.0% | 89.2% |
"""
)
st.caption("Linear SVM performed best overall.")

st.subheader("N-gram experiments")
st.markdown(
    """
| N-grams | Accuracy |
|---|---|
| Unigram only | 89.19% |
| Unigram + Bigram | 90.13% |
| Unigram + Bigram + Trigram | 89.91% |
"""
)
st.caption("Unigram + Bigram gave the best result, so I used that.")

st.subheader("Hyperparameter tuning (SVM's C value)")
st.markdown(
    """
| C | Accuracy | Macro F1 |
|---|---|---|
| 0.01 | 55.0% | 0.25 |
| 0.1 | 85.7% | 0.80 |
| 1 | 90.1% | 0.87 |
| 10 | 90.2% | 0.87 |
| 100 | 90.2% | 0.87 |
"""
)

st.subheader("Final model")
st.write("TF-IDF (unigram + bigram) + Linear SVM with C=100")
st.metric("Test Accuracy", "90.2%")

st.subheader("Project structure")
st.code(
    "emotion-sense-nlp/\n"
    "├── app/              (Streamlit UI)\n"
    "├── src/               (data loading, preprocessing, training, prediction)\n"
    "├── tests/             (automated tests)\n"
    "├── models/            (saved model files)\n"
    "├── data/              (train/test data)\n"
    "└── notebooks/         (original experiments)",
    language=None,
)