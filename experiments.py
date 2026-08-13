import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/train.txt", sep=';', header=None, names=['text', 'emotion'])
print(df.head(10))

df.isnull().sum()

df['emotion'].unique()

unique_emotions = df['emotion'].unique()

emotion_numbers = {}
i = 0
for emo in unique_emotions:
    emotion_numbers[emo] = i
    i += 1

df['emotion'] = df['emotion'].map(emotion_numbers)

print(df.head(10))

df['text'] = df['text'].apply(lambda x: x.lower()) # Convert all text to lowercase
# 1). lowering text ,

import string
def remove_punc(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# this is a punctuation removal function commonly used in NLP preprocessing.

# 2). remove punctuation from text

df['text'] = df['text'].apply(remove_punc)
print(df.head(10))

def remove_numbers(text):
    new = ""
    for i in text:
        if not i.isdigit():
            new += i
    return new

df['text'] = df['text'].apply(remove_numbers)

# remove numbers from text

def remove_emojis(text):
    new = ""
    for i in text:
        if i.isascii():
            new += i
    return new

df['text'] = df['text'].apply(remove_emojis)

#now remove stopwords from text

# from nltk , also spacy aswell

# natural language toolkit (nltk) is a library in python that provides tools for working with human language data. 

# in nlp we split the text into tokens, and then we remove the stopwords from the tokens.
# this process is called tokenization, and it is a common preprocessing step in nlp


import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
print(stop_words)
print(len(stop_words))

df.loc[1, 'text']

# .loc = Location → selects data by labels
# .iloc = Integer Location → selects data by integer positions/index
#df.loc[2, "Name"]     # label-based
# df.iloc[2, 0]         # position-based

def remove(text):
    words = word_tokenize(text)
    Cleaned = []
    for i in words:
        if i not in stop_words:
            Cleaned.append(i)
    return " ".join(Cleaned)

df['text'] = df['text'].apply(remove)

print(df.head(10))

df.loc[1, 'text']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2,random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

bow_vect = CountVectorizer()
X_train_bow = bow_vect.fit_transform(X_train)
X_test_bow = bow_vect.transform(X_test)

# The reason is that Multinomial Naive Bayes is naturally suited to word-count or word-frequency features.

# 1. What does "Multinomial" mean?

# In NLP, your document can be represented by how many times each word occurs.
# Multinomial Naive Bayes is designed to work particularly well with this kind of data.


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report, f1_score


nb_model = MultinomialNB()
nb_model.fit(X_train_bow, y_train)

pred_bow = nb_model.predict(X_test_bow)

print("accuracy_score NB_bow : ", accuracy_score(y_test, pred_bow))
print(classification_report(y_test, pred_bow))

tfidfVectorizer  = TfidfVectorizer()
X_train_tfidf = tfidfVectorizer.fit_transform(X_train)
X_test_tfidf = tfidfVectorizer.transform(X_test)

nb2_model = MultinomialNB()
nb2_model.fit(X_train_tfidf, y_train)

y_pred = nb2_model.predict(X_test_tfidf)

print("accuracy_score NB_tfidf: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.linear_model import LogisticRegression

logistic_model1 = LogisticRegression(max_iter= 1000)

logistic_model1.fit(X_train_bow,y_train)

logistic_model2 = LogisticRegression(max_iter=1000)

log_pred_bow = logistic_model1.predict(X_test_bow)

print("accuracy_score log_bow: ", accuracy_score(y_test, log_pred_bow))
print(classification_report(y_test, log_pred_bow))


logistic_model2.fit(X_train_tfidf,y_train)

log_pred_tfidf = logistic_model2.predict(X_test_tfidf)

print("accuracy_score log_tfidf: ", accuracy_score(y_test, log_pred_tfidf))
print(classification_report(y_test, log_pred_tfidf))



from sklearn.svm import LinearSVC


svm_model1 = LinearSVC()

svm_model1.fit(X_train_bow, y_train)

y_pred_svm_bow = svm_model1.predict(X_test_bow)

print("SVM bow accuracy:", accuracy_score(y_test, y_pred_svm_bow))
print(classification_report(y_test, y_pred_svm_bow))


svm_model2 = LinearSVC()

svm_model2.fit(X_train_tfidf, y_train)

y_pred_svm_tfidf = svm_model2.predict(X_test_tfidf)

print("SVM tfidf accuracy:", accuracy_score(y_test, y_pred_svm_tfidf))
print(classification_report(y_test, y_pred_svm_tfidf))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_svm_tfidf)
print(cm)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM + TF-IDF Confusion Matrix")

plt.show()

tfidf_bigram = TfidfVectorizer(ngram_range=(1, 2))

X_train_tfidf_bigram = tfidf_bigram.fit_transform(X_train)
X_test_tfidf_bigram = tfidf_bigram.transform(X_test)

svm_bigram = LinearSVC()

svm_bigram.fit(X_train_tfidf_bigram, y_train)

pred_bigram = svm_bigram.predict(X_test_tfidf_bigram)

print("SVM TF-IDF Bigram Accuracy:",
      accuracy_score(y_test, pred_bigram))

print(classification_report(y_test, pred_bigram))

print(confusion_matrix(y_test, pred_bigram))



tfidf_trigram = TfidfVectorizer(
    ngram_range=(1, 3)
)

X_train_tfidf_trigram = tfidf_trigram.fit_transform(X_train)
X_test_tfidf_trigram = tfidf_trigram.transform(X_test)

svm_trigram = LinearSVC()

svm_trigram.fit(
    X_train_tfidf_trigram,
    y_train
)

pred_trigram = svm_trigram.predict(
    X_test_tfidf_trigram
)

print(
    "SVM TF-IDF Trigram Accuracy:",
    accuracy_score(y_test, pred_trigram)
)

print(
    classification_report(
        y_test,
        pred_trigram
    )
)

print(confusion_matrix(y_test, pred_trigram))




tfidf = TfidfVectorizer(ngram_range=(1, 2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

C_values = [0.01, 0.1, 1, 10, 100]

results = {}

for c in C_values:

    svm_model = LinearSVC(
        C=c,
        max_iter=5000
    )

    svm_model.fit(X_train_tfidf, y_train)

    y_pred = svm_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average='macro'
    )

    results[c] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

    print(
        f"C = {c} | "
        f"Accuracy = {accuracy:.4f} | "
        f"Macro F1 = {macro_f1:.4f}"
    )


# this works brest that' why we choce this 

final_model = LinearSVC(
    C=100,
    max_iter=5000
)

final_model.fit(
    X_train_tfidf,
    y_train
)

final_pred = final_model.predict(X_test_tfidf)

print("Final Accuracy:",
      accuracy_score(y_test, final_pred))

print("\nClassification Report:")
print(classification_report(y_test, final_pred)) 

cm = confusion_matrix(y_test, final_pred)

print(cm)