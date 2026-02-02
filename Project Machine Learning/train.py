import pandas as pd
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
df = pd.read_csv("dataset.csv")

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

X = df["Pesan"].apply(clean_text)
y = df["Label"].map({"Ham": 0, "Spam": 1})

# Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_vect = vectorizer.fit_transform(X)

# Model
model = MultinomialNB()
model.fit(X_vect, y)

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model & Vectorizer berhasil disimpan")