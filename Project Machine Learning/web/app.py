from flask import Flask, render_template, request
import pickle
import string
import os

app = Flask(__name__)

# Load model dari folder atas
model = pickle.load(open("../model.pkl", "rb"))
vectorizer = pickle.load(open("../vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]
        clean_msg = clean_text(message)
        vect_msg = vectorizer.transform([clean_msg])

        pred = model.predict(vect_msg)
        prob = model.predict_proba(vect_msg)

        prediction = "Spam" if pred[0] == 1 else "Ham"
        probability = round(prob[0][1] * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)