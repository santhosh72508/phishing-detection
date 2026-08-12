import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# LOAD SMS DATASET
# ============================================================

print("Loading SMS dataset...")

df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)

# Keep only the required columns
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Remove empty rows
df = df.dropna()

# Remove duplicate messages
df = df.drop_duplicates()

print("Dataset size:", len(df))
print("Labels:")
print(df["label"].value_counts())


# ============================================================
# CONVERT LABELS
# ============================================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df = df.dropna()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# TF-IDF
# ============================================================

print("Creating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("Training SMS spam model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# ============================================================
# ACCURACY
# ============================================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print(f"SMS Model Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "spam_vectorizer.pkl")

print()
print("SMS model saved successfully!")
print("Created:")
print("  spam_model.pkl")
print("  spam_vectorizer.pkl")