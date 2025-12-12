import json
import unicodedata
import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

try:
    _ = stopwords.words('english')
    _ = WordNetLemmatizer()
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

MAX_LEN = 100
MODEL_PATH = 'final_model_2.h5'
TOKENIZER_PATH = 'final_tokenizer_2.json'

URL_RE = re.compile(
    r"""(
    (?:h[tＴ][tＴ][pＰsＳ]?    # http / https with homoglyphs
    |hxxp                      # obfuscated protocol
    |www)                      # www start
    [\s:/]*                    # allow obfuscation spaces, colons, slashes
    (?:\[?\.\]?|\(dot\)|-dot-|\.|\s*\.\s*)*  # dot obfuscations
    [a-z0-9\-]+                # domain part
    (?:\.(?:[a-z]{2,}|[a-z]{2,}\.[a-z]{2,}))+ # tld
    (?:[/?#][^\s]*)?           # optional path/query
    |                           # OR standard clean URLs
    (?:http|https|ftp|file)s?://\S+|www\.\S+
    )""",
    re.IGNORECASE | re.VERBOSE
)

EMAIL_RE = re.compile(
    r"""
    [a-z0-9._%+-]+             # username
    (?:@|\s*\[at\]\s*|\s*\(at\)\s*)  # @ or obfuscated versions
    [a-z0-9.-]+                # domain
    (?:\.(?:[a-z]{2,}|[a-z]{2,}\.[a-z]{2,}))  # TLD
    """,
    re.IGNORECASE | re.VERBOSE
)
#PHONE_RE = re.compile(r'(?:(?:\+|0{0,2})[1-9]\d{0,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4,9}')
PHONE_RE = re.compile(
    r"""
    # Optional country code: +91, 001, 0, etc.
    (?:\+?\d{1,3}[\s.-]?)?

    # Optional area code: (123), 123
    (?:\(?\d{2,5}\)?[\s.-]?)?

    # Main number: allow 2–4 digits blocks separated by space/dot/dash
    (?:\d{2,4}[\s.-]?){2,5}\d{2,4}
    """,
    re.VERBOSE
)

STOPWORDS = set(stopwords.words('english'))
IMPORTANT_WORDS = {'free', 'win', 'claim', 'urgent', 'prize', 'txt', 'csh',"no", "not", "free", "call","won"}
CUSTOM_STOPWORDS = STOPWORDS - IMPORTANT_WORDS

lemmatizer = WordNetLemmatizer()

def clean_text(s: str) -> str:
    s = str(s)

    # Replace sensitive patterns with placeholders
    s = URL_RE.sub(' <URL> ', s)
    s = EMAIL_RE.sub(' <EMAIL> ', s)
    s = PHONE_RE.sub(' <PHONE> ', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s<>]', ' ', s)
    s = s.strip().lower()
    s = re.sub(r'\b\d+\b', ' ', s)
    s = re.sub(r'\b(?=\w*\d)(?=\w*[a-z])[a-z0-9]{10,}\b', '<ALPHANUM>', s)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    s = re.sub(r'\s+', ' ', s).strip()

    tokens = s.split()
    tokens = [word for word in tokens if word not in CUSTOM_STOPWORDS]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)




def predict_message(msg: str, model, tokenizer, threshold: float = 0.5):
    cleaned_msg = clean_text(msg)
    seq = tokenizer.texts_to_sequences([cleaned_msg])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prob = float(model.predict(padded)[0][0])

    # Determine binary label based on threshold
    label = 'spam' if prob >= threshold else 'not spam'
    return {
        "spam_probability": float(prob),
        "predicted_label": label
    }

# -------------------------
# Load model + tokenizer (cached)
# -------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("final_model_2.h5", compile=False)

    with open("final_tokenizer_2.json") as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()



# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Spam SMS Classifier", layout="centered")
st.title("Spam SMS Classifier (BiLSTM)")
st.write("Type or paste an SMS/email text below and click **Predict**.")

text = st.text_area("Enter message", height=200)
threshold = st.slider("Spam threshold", 0.0, 1.0, 0.5, 0.01)


if st.button("Predict"):
    if not text.strip():
        st.warning("Please type an SMS message.")
    else:
        res = predict_message(text, model, tokenizer, threshold=threshold)
        st.markdown(f"**Prediction:** {res['predicted_label'].upper()}")
        st.markdown(f"**Spam probability:** {res['spam_probability']:.4f}")
       
@st.cache_data
def clean_text_cached(s: str) -> str:
    return clean_text(s)

