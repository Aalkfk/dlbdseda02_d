import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Lade Stopwörter (Deutsch)
def _ensure_nltk():
    try:
        _ = stopwords.words("german")
    except LookupError:
        nltk.download("stopwords")

_ensure_nltk()
GERMAN_SW = set(stopwords.words("german")) | {
    # Zusatz-Stopwörter, Subreddit-spezifisch erweiterbar
    "stuttgart", "stgt", "https", "http"
}

URL_RE = re.compile(r"https?://\S+")
HASHTAG_RE = re.compile(r"(?i)(?<!\w)#([A-Za-z0-9_äöüÄÖÜß]+)")

def clean_text(s: str) -> str:
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = re.sub(r"[^\wäöüÄÖÜß\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def tokenize(text: str):
    # simpler, robuste Tokenisierung (siehe Kurs: nltk.tokenize & Vektorisierung)
    return [w for w in re.findall(r"\b[\wäöüÄÖÜß]{2,}\b", text) if w not in GERMAN_SW]

def add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_all"] = (df["text"].fillna("") + " " + df["comments_text"].fillna("")).str.strip()
    df["clean"] = df["text_all"].map(clean_text)
    df["tokens"] = df["clean"].map(tokenize)
    return df

def extract_hashtags(df: pd.DataFrame):
    tags = []
    for s in (df["title"].fillna("") + " " + df["selftext"].fillna("")).tolist():
        tags += [m.lower() for m in HASHTAG_RE.findall(s)]
    return pd.Series(tags).value_counts()

