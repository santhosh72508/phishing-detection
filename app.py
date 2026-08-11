
import streamlit as st
import pandas as pd
import os
import re
import joblib

from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# FILE NAMES
# ============================================================

DATASET = "raw_data.csv"

MODEL_FILE = "phishing_model.pkl"
VECTORIZER_FILE = "phishing_vectorizer.pkl"


# ============================================================
# TRUSTED DOMAINS
# ============================================================

TRUSTED_DOMAINS = {
    "flipkart.com",
    "google.com",
    "amazon.in",
    "amazon.com",
    "microsoft.com",
    "apple.com",
    "walmart.com",
    "ebay.com",
    "linkedin.com",
    "github.com"
}


# ============================================================
# SUSPICIOUS WORDS
# ============================================================

SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "secure",
    "account",
    "update",
    "confirm",
    "password",
    "credential",
    "banking",
    "wallet",
    "payment",
    "invoice",
    "recover",
    "unlock",
    "bonus",
    "free",
    "prize",
    "winner",
    "claim",
    "urgent",
    "alert",
    "suspend"
]


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

def extract_url_features(url):

    parsed = urlparse(url)

    hostname = parsed.netloc.lower()

    hostname_without_port = hostname.split(":")[0]

    path = parsed.path.lower()

    full_url = url.lower()

    features = []

    # --------------------------------------------------------
    # Basic URL features
    # --------------------------------------------------------

    features.append(len(url))                       # 0
    features.append(len(hostname_without_port))     # 1
    features.append(len(path))                      # 2
    features.append(url.count("."))                 # 3
    features.append(url.count("-"))                 # 4
    features.append(url.count("_"))                 # 5
    features.append(url.count("/"))                 # 6
    features.append(url.count("?"))                 # 7
    features.append(url.count("="))                 # 8
    features.append(url.count("@"))                 # 9
    features.append(url.count("&"))                 # 10
    features.append(url.count("%"))                 # 11
    features.append(url.count("#"))                 # 12
    features.append(url.count(":"))                 # 13

    # --------------------------------------------------------
    # Digit count
    # --------------------------------------------------------

    digit_count = sum(
        character.isdigit()
        for character in url
    )

    features.append(digit_count)                    # 14

    # --------------------------------------------------------
    # Special characters
    # --------------------------------------------------------

    special_count = len(
        re.findall(
            r"[^a-zA-Z0-9]",
            url
        )
    )

    features.append(special_count)                  # 15

    # --------------------------------------------------------
    # HTTPS
    # --------------------------------------------------------

    features.append(
        1 if parsed.scheme.lower() == "https" else 0
    )                                                # 16

    # --------------------------------------------------------
    # IP address detection
    # --------------------------------------------------------

    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"

    features.append(
        1 if re.match(
            ip_pattern,
            hostname_without_port
        ) else 0
    )                                                # 17

    # --------------------------------------------------------
    # Subdomain count
    # --------------------------------------------------------

    domain_parts = [
        part
        for part in hostname_without_port.split(".")
        if part
    ]

    if len(domain_parts) > 2:

        subdomain_count = len(domain_parts) - 2

    else:

        subdomain_count = 0

    features.append(
        subdomain_count
    )                                                # 18

    # --------------------------------------------------------
    # Suspicious words
    # --------------------------------------------------------

    suspicious_count = 0

    for word in SUSPICIOUS_WORDS:

        if word in full_url:

            suspicious_count += 1

    features.append(
        suspicious_count
    )                                                # 19

    # --------------------------------------------------------
    # URL shortener
    # --------------------------------------------------------

    shortening_domains = [
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "rebrand.ly"
    ]

    is_shortened = any(
        hostname_without_port == short_domain
        or hostname_without_port.endswith(
            "." + short_domain
        )
        for short_domain in shortening_domains
    )

    features.append(
        1 if is_shortened else 0
    )                                                # 20

    # --------------------------------------------------------
    # Excessive hyphens
    # --------------------------------------------------------

    features.append(
        1 if url.count("-") >= 3 else 0
    )                                                # 21

    # --------------------------------------------------------
    # Excessive dots
    # --------------------------------------------------------

    features.append(
        1 if url.count(".") >= 5 else 0
    )                                                # 22

    # --------------------------------------------------------
    # @ symbol
    # --------------------------------------------------------

    features.append(
        1 if "@" in url else 0
    )                                                # 23

    # --------------------------------------------------------
    # Double slash inside path
    # --------------------------------------------------------

    features.append(
        1 if "//" in parsed.path else 0
    )                                                # 24

    return features


# ============================================================
# GET DOMAIN
# ============================================================

def get_domain(url):

    try:

        parsed = urlparse(url)

        domain = parsed.netloc.lower()

        # Remove port
        domain = domain.split(":")[0]

        # Remove www.
        if domain.startswith("www."):

            domain = domain[4:]

        return domain

    except Exception:

        return ""


# ============================================================
# TRUSTED DOMAIN CHECK
# ============================================================

def is_trusted_domain(url):

    domain = get_domain(url)

    if not domain:

        return False

    # Exact match

    if domain in TRUSTED_DOMAINS:

        return True

    # Subdomain match

    for trusted_domain in TRUSTED_DOMAINS:

        if domain.endswith(
            "." + trusted_domain
        ):

            return True

    return False


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(DATASET):

        st.error(
            f"Dataset '{DATASET}' was not found."
        )

        st.stop()

    try:

        df = pd.read_csv(
            DATASET
        )

    except Exception as error:

        st.error(
            f"Could not read dataset: {error}"
        )

        st.stop()

    # --------------------------------------------------------
    # Find URL column
    # --------------------------------------------------------

    possible_url_columns = [
        "URL",
        "url",
        "Url",
        "domain",
        "Domain",
        "website",
        "Website",
        "link",
        "Link"
    ]

    url_column = None

    for column in possible_url_columns:

        if column in df.columns:

            url_column = column

            break

    if url_column is None:

        st.error(
            "URL column was not found."
        )

        st.write(
            "Available columns:",
            list(df.columns)
        )

        st.stop()

    # --------------------------------------------------------
    # Find label column
    # --------------------------------------------------------

    possible_label_columns = [
        "Label",
        "label",
        "type",
        "Type",
        "class",
        "Class",
        "status",
        "Status",
        "target",
        "Target"
    ]

    label_column = None

    for column in possible_label_columns:

        if column in df.columns:

            label_column = column

            break

    if label_column is None:

        st.error(
            "Label column was not found."
        )

        st.write(
            "Available columns:",
            list(df.columns)
        )

        st.stop()

    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    df = df.rename(
        columns={
            url_column: "URL",
            label_column: "Label"
        }
    )

    # --------------------------------------------------------
    # Remove missing data
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "URL",
            "Label"
        ]
    )

    df["URL"] = (
        df["URL"]
        .astype(str)
        .str.strip()
    )

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # ========================================================
    # CONVERT LABELS
    # ========================================================

    def convert_label(label):

        # Legitimate / benign

        if label in [
            "good",
            "legitimate",
            "benign",
            "safe",
            "normal",
            "legit",
            "0"
        ]:

            return 0

        # Phishing / malicious

        if label in [
            "bad",
            "phishing",
            "malicious",
            "unsafe",
            "malware",
            "1"
        ]:

            return 1

        # Numeric labels

        try:

            number = float(label)

            if number == 0:

                return 0

            if number == 1:

                return 1

        except Exception:

            pass

        return None

    df["Target"] = df["Label"].apply(
        convert_label
    )

    # Remove unknown labels

    df = df.dropna(
        subset=[
            "Target"
        ]
    )

    df["Target"] = (
        df["Target"]
        .astype(int)
    )

    # Remove empty URLs

    df = df[
        df["URL"].str.len() > 3
    ]

    # Remove duplicate URLs

    df = df.drop_duplicates(
        subset=[
            "URL"
        ]
    )

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    st.info(
        "Loading raw_data.csv..."
    )

    df = load_dataset()

    st.info(
        f"Dataset size: {len(df):,}"
    )

    legitimate_count = int(
        (df["Target"] == 0).sum()
    )

    phishing_count = int(
        (df["Target"] == 1).sum()
    )

    st.info(
        f"Legitimate URLs: {legitimate_count:,}"
    )

    st.info(
        f"Phishing URLs: {phishing_count:,}"
    )

    # Need both classes

    if legitimate_count == 0:

        st.error(
            "Dataset contains no legitimate URLs."
        )

        st.stop()

    if phishing_count == 0:

        st.error(
            "Dataset contains no phishing URLs."
        )

        st.stop()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    X = df["URL"]

    y = df["Target"]

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ========================================================
    # TF-IDF
    # ========================================================

    st.info(
        "Creating TF-IDF URL features..."
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=30000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    # ========================================================
    # URL SECURITY FEATURES
    # ========================================================

    st.info(
        "Extracting URL security features..."
    )

    X_train_features = [
        extract_url_features(url)
        for url in X_train
    ]

    X_test_features = [
        extract_url_features(url)
        for url in X_test
    ]

    X_train_features = csr_matrix(
        X_train_features
    )

    X_test_features = csr_matrix(
        X_test_features
    )

    # ========================================================
    # COMBINE
    # ========================================================

    X_train_combined = hstack([
        X_train_tfidf,
        X_train_features
    ])

    X_test_combined = hstack([
        X_test_tfidf,
        X_test_features
    ])

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    st.info(
        "Training machine-learning model..."
    )

    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="liblinear"
    )

    model.fit(
        X_train_combined,
        y_train
    )

    # ========================================================
    # TEST
    # ========================================================

    predictions = model.predict(
        X_test_combined
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    return (
        model,
        vectorizer,
        accuracy
    )


# ============================================================
# LOAD OR TRAIN MODEL
# ============================================================

if (
    os.path.exists(MODEL_FILE)
    and
    os.path.exists(VECTORIZER_FILE)
):

    try:

        model = joblib.load(
            MODEL_FILE
        )

        vectorizer = joblib.load(
            VECTORIZER_FILE
        )

        accuracy = None

    except Exception:

        st.warning(
            "Existing model files are incompatible. "
            "Retraining model..."
        )

        try:

            os.remove(
                MODEL_FILE
            )

        except Exception:

            pass

        try:

            os.remove(
                VECTORIZER_FILE
            )

        except Exception:

            pass

        with st.spinner(
            "Training phishing detection model..."
        ):

            (
                model,
                vectorizer,
                accuracy
            ) = train_model()

else:

    with st.spinner(
        "Training phishing detection model..."
    ):

        (
            model,
            vectorizer,
            accuracy
        ) = train_model()

    st.success(
        "Model trained and saved successfully!"
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🛡️ Phishing Website Detection"
)

st.write(
    "Enter a website URL below to check "
    "whether it appears to be phishing "
    "or legitimate."
)


# ============================================================
# MODEL ACCURACY
# ============================================================

if accuracy is not None:

    st.success(
        f"Model Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# URL INPUT
# ============================================================

url = st.text_input(
    "Enter Website URL",
    placeholder="example.com"
)


# ============================================================
# CHECK WEBSITE
# ============================================================

if st.button(
    "🔍 Check Website",
    use_container_width=True
):

    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if not url.strip():

        st.warning(
            "Please enter a URL."
        )

        st.stop()

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    url = url.strip()

    # --------------------------------------------------------
    # Handle accidentally pasted Markdown link
    # --------------------------------------------------------

    markdown_match = re.match(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        url
    )

    if markdown_match:

        url = markdown_match.group(2)

    # --------------------------------------------------------
    # Add protocol
    # --------------------------------------------------------

    if not url.lower().startswith(
        (
            "http://",
            "https://"
        )
    ):

        url = "https://" + url

    # --------------------------------------------------------
    # Get domain
    # --------------------------------------------------------

    domain = get_domain(
        url
    )

    if not domain:

        st.error(
            "❌ Invalid URL. "
            "Please enter a valid website."
        )

        st.stop()

    # ========================================================
    # TRUSTED DOMAIN CHECK
    # ========================================================

    if is_trusted_domain(
        url
    ):

        st.success(
            "✅ LEGITIMATE WEBSITE"
        )

        st.write(
            f"**URL:** {url}"
        )

        st.write(
            f"**Domain:** {domain}"
        )

        st.info(
            "This domain is included in the "
            "trusted-domain list."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Legitimate Probability",
                "99.00%"
            )

        with col2:

            st.metric(
                "Phishing Probability",
                "1.00%"
            )

        st.caption(
            "Trusted-domain result. This does not "
            "guarantee that every page on the domain "
            "is safe."
        )

        st.stop()

    # ========================================================
    # MACHINE LEARNING PREDICTION
    # ========================================================

    # TF-IDF

    url_tfidf = vectorizer.transform(
        [url]
    )

    # URL security features

    url_features = csr_matrix([
        extract_url_features(url)
    ])

    # Combine

    url_combined = hstack([
        url_tfidf,
        url_features
    ])

    # Prediction

    prediction = model.predict(
        url_combined
    )[0]

    # Probability

    probability = model.predict_proba(
        url_combined
    )[0]

    legitimate_probability = (
        probability[0] * 100
    )

    phishing_probability = (
        probability[1] * 100
    )

    # ========================================================
    # RESULT
    # ========================================================

    if prediction == 1:

        st.error(
            "⚠️ PHISHING WEBSITE"
        )

    else:

        st.success(
            "✅ LEGITIMATE WEBSITE"
        )

    st.write(
        f"**URL:** {url}"
    )

    st.write(
        f"**Domain:** {domain}"
    )

    # ========================================================
    # PROBABILITY
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Legitimate Probability",
            f"{legitimate_probability:.2f}%"
        )

    with col2:

        st.metric(
            "Phishing Probability",
            f"{phishing_probability:.2f}%"
        )

    # ========================================================
    # URL SECURITY ANALYSIS
    # ========================================================

    features = extract_url_features(
        url
    )

    st.subheader(
        "🔎 URL Security Analysis"
    )

    feature_data = {

        "URL Length":
            features[0],

        "Domain Length":
            features[1],

        "Path Length":
            features[2],

        "Number of Dots":
            features[3],

        "Number of Hyphens":
            features[4],

        "Number of Underscores":
            features[5],

        "Number of Slashes":
            features[6],

        "Number of Question Marks":
            features[7],

        "Number of Equal Signs":
            features[8],

        "Number of @ Symbols":
            features[9],

        "Number of Digits":
            features[14],

        "Special Characters":
            features[15],

        "Uses HTTPS":
            "Yes" if features[16] else "No",

        "IP Address":
            "Yes" if features[17] else "No",

        "Subdomains":
            features[18],

        "Suspicious Words":
            features[19],

        "URL Shortener":
            "Yes" if features[20] else "No",

        "Excessive Hyphens":
            "Yes" if features[21] else "No",

        "Excessive Dots":
            "Yes" if features[22] else "No",

        "Contains @":
            "Yes" if features[23] else "No",

        "Double Slash in Path":
            "Yes" if features[24] else "No"
    }

    st.dataframe(
        pd.DataFrame(
            feature_data.items(),
            columns=[
                "Feature",
                "Value"
            ]
        ),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # WARNING
    # ========================================================

    st.warning(
        "ML predictions are estimates. "
        "Do not enter passwords, banking information, "
        "or other sensitive information into a website "
        "solely because this application marks it "
        "as legitimate."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Phishing Detection using Machine Learning | "
    "TF-IDF + URL Security Features + Logistic Regression"
)

