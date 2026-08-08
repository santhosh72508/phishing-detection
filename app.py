
from flask import Flask, render_template, request
import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------------------------------------------
# CREATE FLASK APP
# ---------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------
# FILE NAMES
# ---------------------------------------------------

DATASET = "phishing_site_urls.csv"
MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"


# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

def train_model():

    print("Loading dataset...")

    df = pd.read_csv(DATASET)

    # Remove empty rows
    df = df.dropna(subset=["URL", "Label"])

    print("Dataset size:", len(df))

    # Convert labels to lowercase
    df["Label"] = df["Label"].str.lower()

    # bad = phishing
    # good = legitimate

    X = df["URL"].astype(str)

    y = df["Label"].map({
        "bad": 1,
        "good": 0
    })

    # Remove unknown labels
    valid = y.notna()

    X = X[valid]
    y = y[valid]

    print("Training model...")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Convert URLs into TF-IDF features
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    X_test_vec = vectorizer.transform(X_test)

    # Create ML model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    # Train
    model.fit(X_train_vec, y_train)

    # Test
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Model Accuracy: {accuracy * 100:.2f}%"
    )

    # Save model
    joblib.dump(model, MODEL_FILE)

    # Save vectorizer
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print("Model saved successfully.")


# ---------------------------------------------------
# LOAD OR TRAIN MODEL
# ---------------------------------------------------

if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):

    train_model()


# Load saved model
model = joblib.load(MODEL_FILE)

# Load saved vectorizer
vectorizer = joblib.load(VECTORIZER_FILE)


# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        result=None,
        message=None,
        url=None
    )


# ---------------------------------------------------
# PREDICT URL
# ---------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Get URL from HTML form
    url = request.form.get("url", "").strip()

    # Check empty URL
    if not url:

        return render_template(
            "index.html",
            result="Please enter a URL.",
            message=None,
            url=None
        )

    # Add http:// if user did not enter it
    if not url.startswith(("http://", "https://")):

        url = "http://" + url

    # Convert URL into TF-IDF vector
    url_vector = vectorizer.transform([url])

    # Make prediction
    prediction = model.predict(url_vector)[0]

    # Get prediction probability
    probability = model.predict_proba(url_vector)[0]

    # Probability values
    legitimate_probability = probability[0] * 100
    phishing_probability = probability[1] * 100


    # ------------------------------------------------
    # PHISHING RESULT
    # ------------------------------------------------

    if prediction == 1:

        result = "⚠️ PHISHING WEBSITE"

        message = (
            f"The website appears suspicious. "
            f"Phishing probability: "
            f"{phishing_probability:.2f}%"
        )


    # ------------------------------------------------
    # LEGITIMATE RESULT
    # ------------------------------------------------

    else:

        result = "✅ LEGITIMATE WEBSITE"

        message = (
            f"The website appears legitimate. "
            f"Legitimate probability: "
            f"{legitimate_probability:.2f}%"
        )


    # Send result back to HTML
    return render_template(
        "index.html",
        result=result,
        message=message,
        url=url
    )


# ---------------------------------------------------
# START FLASK SERVER
# ---------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )

