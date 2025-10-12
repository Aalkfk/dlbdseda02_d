import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer

def _ensure_nltk():
    try:
        _ = stopwords.words("german")
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

_ensure_nltk()

GERMAN_SW = set(stopwords.words("german")) | {"stuttgart", "stgt", "https", "http"}
ENGLISH_SW = set(stopwords.words("english"))

# Zusätzliche Füll-/Funktionswörter, die i. d. R. nicht themenbildend sind
FILLER_DE = {
    "heute","jemand","eigentlich","wirklich","direkt","danke","bitte","gerne",
    "halt","ne","ach","wow","super","bisschen","ziemlich","total","richtig",
    "viele","vielleicht","wurde","werden","wird","waren","sein","wäre","hat",
    "hast","habt","paar","seite","alleine","menschen","leute","deutschland",
    "stadt","ganz","schön","warum","wochen","neu","neue","minuten","kommt",
    "kommen","geht","gehen","gesehen","sieht","richtig","bereits","jedoch",
    "sowie","außerdem","daher","deshalb","trotzdem","überhaupt","dabei",
    "danach","vorher","zwei","drei","vier","fünf","jahren","jahr"
}
GERMAN_SW |= FILLER_DE

URL_RE = re.compile(r"https?://\S+")
PUNCT_RE = re.compile(r"[^\wäöüÄÖÜß\s]")

STEMMER = GermanStemmer(ignore_stopwords=True)

def clean_text(s: str) -> str:
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def tokenize(text: str):
    return re.findall(r"\b[\wäöüÄÖÜß]{2,}\b", text)

def detect_language_by_stopword_ratio(tokens, min_tokens=12, margin=1.15):
    """Heuristik: vergleicht Stopwort-Anteile Deutsch vs. Englisch."""
    if not tokens:
        return "unknown"
    n = len(tokens)
    if n < min_tokens:
        return "unknown"
    de_hits = sum(1 for t in tokens if t in GERMAN_SW)
    en_hits = sum(1 for t in tokens if t in ENGLISH_SW)
    de_ratio = de_hits / n
    en_ratio = en_hits / n
    if de_ratio >= en_ratio * margin and de_ratio > 0:
        return "de"
    if en_ratio >= de_ratio * margin and en_ratio > 0:
        return "en"
    return "unknown"

def add_clean_columns(df: pd.DataFrame, use_stemming: bool=False, filter_english: bool=True) -> pd.DataFrame:
    """
    Erweitert df um:
      - text_all, clean, tokens_raw, lang, tokens, clean_for_vect
    Achtung: Standard ohne Stemming -> bessere Lesbarkeit der Topic-Terme.
    """
    df = df.copy()
    df["text_all"] = (df.get("text", "").fillna("") + " " + df.get("comments_text", "").fillna("")).str.strip()
    df["clean"] = df["text_all"].map(clean_text)
    df["tokens_raw"] = df["clean"].map(tokenize)

    # Sprachheuristik
    df["lang"] = df["tokens_raw"].map(detect_language_by_stopword_ratio)

    # optional Englisch rausfiltern
    if filter_english:
        df = df[df["lang"] != "en"].reset_index(drop=True)

    # Stopwörter entfernen + optional Stemming
    def _proc(tokens):
        kept = []
        for t in tokens:
            # raus, wenn Stopwort, engl. Stopwort oder Ziffern enthalten
            if t in GERMAN_SW or t in ENGLISH_SW:
                continue
            if any(ch.isdigit() for ch in t):
                continue
            if len(t) < 3:
                continue
            # explizit störende Tokens
            if t in {"deleted"}:
                continue
            kept.append(t)
        # kein Stemming für bessere Lesbarkeit
        return kept

    df["tokens"] = df["tokens_raw"].map(_proc)
    df["clean_for_vect"] = df["tokens"].map(lambda ts: " ".join(ts))
    return df

def extract_flairs(df: pd.DataFrame) -> pd.Series:
    """
    Flair-Statistik; fällt tolerant auf leer zurück, wenn die CSV keine passende Spalte hat.
    """
    cand = [c for c in df.columns if c.lower() in ("flair_text", "link_flair_text", "flair")]
    if not cand:
        return pd.Series(dtype="int64")
    col = cand[0]
    return (df[col]
            .fillna("")
            .str.strip()
            .replace({"": None})
            .dropna()
            .value_counts())
