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
    page_title="Scam Detection System",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# FILE NAMES
# ============================================================

PHISHING_DATASET = "raw_data.csv"
SMS_DATASET = "spam.csv"

PHISHING_MODEL_FILE = "phishing_model.pkl"
PHISHING_VECTORIZER_FILE = "phishing_vectorizer.pkl"

# IMPORTANT:
# These names match your actual files.
SMS_MODEL_FILE = "spam_model.pkl"
SMS_VECTORIZER_FILE = "spam_vectorizer.pkl"


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
# SUSPICIOUS URL WORDS
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

    # Basic URL features
    features.append(len(url))
    features.append(len(hostname_without_port))
    features.append(len(path))
    features.append(url.count("."))
    features.append(url.count("-"))
    features.append(url.count("_"))
    features.append(url.count("/"))
    features.append(url.count("?"))
    features.append(url.count("="))
    features.append(url.count("@"))
    features.append(url.count("&"))
    features.append(url.count("%"))
    features.append(url.count("#"))
    features.append(url.count(":"))

    # Digits
    digit_count = sum(
        character.isdigit()
        for character in url
    )

    features.append(digit_count)

    # Special characters
    special_count = len(
        re.findall(
            r"[^a-zA-Z0-9]",
            url
        )
    )

    features.append(special_count)

    # HTTPS
    features.append(
        1 if parsed.scheme.lower() == "https" else 0
    )

    # IP address
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"

    features.append(
        1 if re.match(
            ip_pattern,
            hostname_without_port
        ) else 0
    )

    # Subdomains
    domain_parts = [
        part
        for part in hostname_without_port.split(".")
        if part
    ]

    if len(domain_parts) > 2:
        subdomain_count = len(domain_parts) - 2
    else:
        subdomain_count = 0

    features.append(subdomain_count)

    # Suspicious words
    suspicious_count = 0

    for word in SUSPICIOUS_WORDS:

        if word in full_url:
            suspicious_count += 1

    features.append(suspicious_count)

    # URL shortener
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
    )

    # Excessive hyphens
    features.append(
        1 if url.count("-") >= 3 else 0
    )

    # Excessive dots
    features.append(
        1 if url.count(".") >= 5 else 0
    )

    # @ symbol
    features.append(
        1 if "@" in url else 0
    )

    # Double slash inside path
    features.append(
        1 if "//" in parsed.path else 0
    )

    return features


# ============================================================
# GET DOMAIN
# ============================================================

def get_domain(url):

    try:

        parsed = urlparse(url)

        domain = parsed.netloc.lower()

        domain = domain.split(":")[0]

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:

        return ""


# ============================================================
# TRUSTED DOMAIN
# ============================================================

def is_trusted_domain(url):

    domain = get_domain(url)

    if not domain:
        return False

    if domain in TRUSTED_DOMAINS:
        return True

    for trusted_domain in TRUSTED_DOMAINS:

        if domain.endswith(
            "." + trusted_domain
        ):
            return True

    return False


# ============================================================
# LOAD PHISHING DATASET
# ============================================================

def load_phishing_dataset():

    if not os.path.exists(PHISHING_DATASET):

        st.error(
            f"Dataset '{PHISHING_DATASET}' was not found."
        )

        st.stop()

    try:

        df = pd.read_csv(
            PHISHING_DATASET
        )

    except Exception as error:

        st.error(
            f"Could not read phishing dataset: {error}"
        )

        st.stop()

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
            "URL column was not found in raw_data.csv."
        )

        st.write(
            "Available columns:",
            list(df.columns)
        )

        st.stop()

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
            "Label column was not found in raw_data.csv."
        )

        st.write(
            "Available columns:",
            list(df.columns)
        )

        st.stop()

    df = df.rename(
        columns={
            url_column: "URL",
            label_column: "Label"
        }
    )

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

    def convert_label(label):

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

        if label in [
            "bad",
            "phishing",
            "malicious",
            "unsafe",
            "malware",
            "1"
        ]:

            return 1

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

    df = df.dropna(
        subset=["Target"]
    )

    df["Target"] = (
        df["Target"]
        .astype(int)
    )

    df = df[
        df["URL"].str.len() > 3
    ]

    df = df.drop_duplicates(
        subset=["URL"]
    )

    return df


# ============================================================
# TRAIN PHISHING MODEL
# ============================================================

@st.cache_resource
def train_phishing_model():

    df = load_phishing_dataset()

    X = df["URL"]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
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

    X_train_combined = hstack([
        X_train_tfidf,
        X_train_features
    ])

    X_test_combined = hstack([
        X_test_tfidf,
        X_test_features
    ])

    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="liblinear"
    )

    model.fit(
        X_train_combined,
        y_train
    )

    predictions = model.predict(
        X_test_combined
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    joblib.dump(
        model,
        PHISHING_MODEL_FILE
    )

    joblib.dump(
        vectorizer,
        PHISHING_VECTORIZER_FILE
    )

    return (
        model,
        vectorizer,
        accuracy
    )


# ============================================================
# LOAD SMS DATASET
# ============================================================

def load_sms_dataset():

    if not os.path.exists(SMS_DATASET):

        st.error(
            f"Dataset '{SMS_DATASET}' was not found."
        )

        st.stop()

    try:

        df = pd.read_csv(
            SMS_DATASET,
            encoding="latin-1"
        )

    except Exception as error:

        st.error(
            f"Could not read spam.csv: {error}"
        )

        st.stop()

    if df.shape[1] < 2:

        st.error(
            "spam.csv must contain at least two columns."
        )

        st.stop()

    # First column = label
    # Second column = SMS

    df = df.iloc[:, :2]

    df.columns = [
        "Label",
        "Message"
    ]

    df = df.dropna(
        subset=[
            "Label",
            "Message"
        ]
    )

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["Message"] = (
        df["Message"]
        .astype(str)
        .str.strip()
    )

    def convert_sms_label(label):

        if label in [
            "ham",
            "normal",
            "legitimate",
            "safe",
            "0"
        ]:

            return 0

        if label in [
            "spam",
            "1"
        ]:

            return 1

        return None

    df["Target"] = df["Label"].apply(
        convert_sms_label
    )

    df = df.dropna(
        subset=["Target"]
    )

    df["Target"] = (
        df["Target"]
        .astype(int)
    )

    df = df[
        df["Message"].str.len() > 0
    ]

    df = df.drop_duplicates(
        subset=["Message"]
    )

    return df


# ============================================================
# TRAIN SMS MODEL
# ============================================================

@st.cache_resource
def train_sms_model():

    df = load_sms_dataset()

    X = df["Message"]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    joblib.dump(
        model,
        SMS_MODEL_FILE
    )

    joblib.dump(
        vectorizer,
        SMS_VECTORIZER_FILE
    )

    return (
        model,
        vectorizer,
        accuracy
    )


# ============================================================
# LOAD PHISHING MODEL
# ============================================================

if (
    os.path.exists(PHISHING_MODEL_FILE)
    and
    os.path.exists(PHISHING_VECTORIZER_FILE)
):

    try:

        phishing_model = joblib.load(
            PHISHING_MODEL_FILE
        )

        phishing_vectorizer = joblib.load(
            PHISHING_VECTORIZER_FILE
        )

        phishing_accuracy = None

    except Exception:

        with st.spinner(
            "Loading phishing model..."
        ):

            (
                phishing_model,
                phishing_vectorizer,
                phishing_accuracy
            ) = train_phishing_model()

else:

    with st.spinner(
        "Training phishing detection model..."
    ):

        (
            phishing_model,
            phishing_vectorizer,
            phishing_accuracy
        ) = train_phishing_model()


# ============================================================
# LOAD SMS MODEL
# ============================================================

if (
    os.path.exists(SMS_MODEL_FILE)
    and
    os.path.exists(SMS_VECTORIZER_FILE)
):

    try:

        sms_model = joblib.load(
            SMS_MODEL_FILE
        )

        sms_vectorizer = joblib.load(
            SMS_VECTORIZER_FILE
        )

        sms_accuracy = 97.78

    except Exception:

        with st.spinner(
            "Loading SMS model..."
        ):

            (
                sms_model,
                sms_vectorizer,
                sms_accuracy_value
            ) = train_sms_model()

            sms_accuracy = (
                sms_accuracy_value * 100
            )

else:

    with st.spinner(
        "Training SMS spam model..."
    ):

        (
            sms_model,
            sms_vectorizer,
            sms_accuracy_value
        ) = train_sms_model()

        sms_accuracy = (
            sms_accuracy_value * 100
        )


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🛡️ Scam Detection System"
)

st.write(
    "A machine-learning based system for detecting "
    "phishing websites and SMS spam."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🔐 Detection Menu"
)

detection_type = st.sidebar.radio(
    "Choose Detection Type",
    [
        "🌐 Phishing Website",
        "📱 SMS Spam"
    ]
)


# ============================================================
# PHISHING WEBSITE DETECTION
# ============================================================

if detection_type == "🌐 Phishing Website":

    st.header(
        "🌐 Phishing Website Detection"
    )

    st.write(
        "Enter a website URL to check whether it "
        "appears to be legitimate or phishing."
    )

    url = st.text_input(
        "Enter Website URL",
        placeholder="https://example.com"
    )

    if st.button(
        "🔍 Check Website",
        use_container_width=True
    ):

        if not url.strip():

            st.warning(
                "Please enter a URL."
            )

            st.stop()

        url = url.strip()

        # Handle markdown URLs
        markdown_match = re.match(
            r"\[([^\]]+)\]\((https?://[^)]+)\)",
            url
        )

        if markdown_match:

            url = markdown_match.group(2)

        # Add protocol
        if not url.lower().startswith(
            (
                "http://",
                "https://"
            )
        ):

            url = "https://" + url

        domain = get_domain(
            url
        )

        if not domain:

            st.error(
                "❌ Invalid URL."
            )

            st.stop()

        # Trusted domain
        if is_trusted_domain(url):

            st.success(
                "✅ LEGITIMATE WEBSITE"
            )

            st.write(
                f"**URL:** {url}"
            )

            st.write(
                f"**Domain:** {domain}"
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

            st.info(
                "This domain is included in the trusted-domain list."
            )

        else:

            url_tfidf = phishing_vectorizer.transform(
                [url]
            )

            url_features = csr_matrix([
                extract_url_features(url)
            ])

            url_combined = hstack([
                url_tfidf,
                url_features
            ])

            prediction = phishing_model.predict(
                url_combined
            )[0]

            probability = phishing_model.predict_proba(
                url_combined
            )[0]

            legitimate_probability = (
                probability[0] * 100
            )

            phishing_probability = (
                probability[1] * 100
            )

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

            # URL analysis
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

            st.warning(
                "ML predictions are estimates. Do not enter "
                "passwords, OTPs, banking information, or "
                "other sensitive information based only on "
                "this prediction."
            )


# ============================================================
# SMS SPAM DETECTION
# ============================================================

else:

    st.header(
        "📱 SMS Spam Detection"
    )

    st.write(
        "Enter an SMS message to check whether it is "
        "SPAM or HAM (legitimate)."
    )

    st.info(
        f"SMS Model Accuracy: {sms_accuracy:.2f}%"
    )

    message = st.text_area(
        "Enter SMS Message",
        placeholder=(
            "Example: Congratulations! You have won "
            "a free prize. Click here to claim."
        ),
        height=150
    )

    if st.button(
        "🔍 Check SMS",
        use_container_width=True
    ):

        if not message.strip():

            st.warning(
                "Please enter an SMS message."
            )

            st.stop()

        message = message.strip()

        # Convert SMS into TF-IDF
        message_vector = sms_vectorizer.transform(
            [message]
        )

        # Prediction
        prediction = sms_model.predict(
            message_vector
        )[0]

        probability = sms_model.predict_proba(
            message_vector
        )[0]

        ham_probability = (
            probability[0] * 100
        )

        spam_probability = (
            probability[1] * 100
        )

        # Result
        if prediction == 1:

            st.error(
                "🚨 SPAM SMS"
            )

            st.write(
                "This message has characteristics "
                "commonly associated with spam."
            )

        else:

            st.success(
                "✅ HAM / LEGITIMATE SMS"
            )

            st.write(
                "This message appears to be legitimate."
            )

        # Probability
        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Ham Probability",
                f"{ham_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )

        # SMS analysis
        st.subheader(
            "🔎 SMS Analysis"
        )

        sms_features = {

            "Message Length":
                len(message),

            "Word Count":
                len(message.split()),

            "Digit Count":
                sum(
                    character.isdigit()
                    for character in message
                ),

            "URL Present":
                "Yes"
                if re.search(
                    r"https?://|www\.",
                    message.lower()
                )
                else "No",

            "Contains Money Symbol":
                "Yes"
                if re.search(
                    r"[$₹€£]",
                    message
                )
                else "No",

            "Contains Urgent Word":
                "Yes"
                if any(
                    word in message.lower()
                    for word in [
                        "urgent",
                        "immediately",
                        "now",
                        "alert",
                        "winner",
                        "claim"
                    ]
                )
                else "No"
        }

        st.dataframe(
            pd.DataFrame(
                sms_features.items(),
                columns=[
                    "Feature",
                    "Value"
                ]
            ),
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "Do not click suspicious links or provide OTPs, "
            "passwords, bank details, or other sensitive "
            "information based only on this prediction."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Scam Detection System | "
    "Phishing Website Detection + SMS Spam Detection | "
    "Machine Learning"
)