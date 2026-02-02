# 📧 Spam Email Detection using Machine Learning

A simple end-to-end Machine Learning project to detect whether an email is **Spam** or **Ham (Not Spam)**.
This project covers the full ML pipeline: data preprocessing, model training, inference, and web deployment.

## 🚀 Features
- Text preprocessing (lowercase & punctuation removal)
- TF-IDF vectorization
- Naive Bayes classification
- Model inference for new email input
- Web-based interface using Flask
- Modern UI with animations

## 🧠 Machine Learning Workflow
1. Load and preprocess dataset
2. Split data into training and testing sets
3. Convert text to numerical features using TF-IDF
4. Train model using Multinomial Naive Bayes
5. Evaluate model accuracy
6. Save trained model and vectorizer
7. Deploy model into a Flask web application

## 🛠 Tech Stack
- Python
- Pandas
- Scikit-learn
- Flask
- HTML, CSS

## 📂 Project Structure
PROJECT MACHINE LEARNING/
│
├── dataset.csv               # Dataset email (Spam / Ham)
├── train.py                  # Script training model ML
├── model.pkl                 # Model hasil training
├── vectorizer.pkl            # TF-IDF vectorizer
├── README.md                 # Dokumentasi project (WAJIB)
├── requirements.txt          # Dependency project
│
└── web/
    ├── app.py                # Flask app (backend)
    │
    ├── templates/
    │   └── index.html        # Frontend (HTML)
    │
    └── static/
        └── style.css         # Styling & animasi UI

## ▶️ How to Run
```bash
# Train model
python train.py

# Run web app
cd web
python app.py
