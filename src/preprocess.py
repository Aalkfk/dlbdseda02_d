"""
preprocess.py
-------------
Text-Vorverarbeitung für Reddit-Posts (r/Stuttgart).

Features
- Cleaning: Links, Klammer-Tags wie [removed]/[deleted], Satzzeichen entfernen.
- Tokenisierung: robuste Regex-Tokenisierung für DE/EN inkl. Umlaute.
- Sprachheuristik: einfacher DE/EN-Check via Stopwort-Anteile.
- Stopwort-Filter: deutsche/englische + technische + Subreddit-spezifische Füllwörter.
- Optionale Comment-Zusammenführung.
- Kollokationen: häufige Bigrams (PMI) als zusätzliche Tokens (mit Unterstrich).
- Komfort-Felder: text_all, clean, tokens_raw, lang, tokens, clean_for_vect.
- Flair-Aggregation.

Die Funktionen sind „drop-in“ kompatibel mit der bestehenden Pipeline.
"""

from __future__ import annotations

import re
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer


# ---------------------------------------------------------------------------
# NLTK-Basisressourcen sicherstellen
# ---------------------------------------------------------------------------

def _ensure_nltk() -> None:
    """
    Stellt sicher, dass NLTK-Stopwortlisten verfügbar sind (idempotent).
    Lädt fehlende Ressourcen nur bei Bedarf nach.
    """
    try:
        _ = stopwords.words("german")
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


_ensure_nltk()


# ---------------------------------------------------------------------------
# Stopwort-Sets & Füllwörter
# ---------------------------------------------------------------------------

GERMAN_SW = set(stopwords.words("german")) | {"stuttgart", "stgt", "https", "http"}
ENGLISH_SW = set(stopwords.words("english"))

# Zusätzliche Füll-/Funktionswörter, die i.d.R. nicht themenbildend sind.
# (Beinhaltet auch schwäbische Varianten/Interjektionen; bewusst großzügig.)
FILLER_DE = {
    "heute", "jemand", "eigentlich", "wirklich", "direkt", "danke", "bitte", "gerne",
    "halt", "ne", "ach", "wow", "super", "bisschen", "ziemlich", "total", "richtig",
    "viele", "vielleicht", "wurde", "werden", "wird", "waren", "sein", "wäre", "hat",
    "hast", "habt", "paar", "seite", "alleine", "menschen", "leute", "deutschland",
    "stadt", "ganz", "schön", "warum", "wochen", "neu", "neue", "minuten", "kommt",
    "kommen", "geht", "gehen", "gesehen", "sieht", "bereits", "jedoch",
    "sowie", "außerdem", "daher", "deshalb", "trotzdem", "überhaupt", "dabei",
    "danach", "vorher", "zwei", "drei", "vier", "fünf", "jahren", "jahr",
    "sowas", "meinung", "leben", "etc", "los", "weniger", "glück", "unterwegs", "damals",
    "besonders", "verstehe", "ersten", "wahrscheinlich", "bestimmt", "vermutlich",
    "endlich", "vorbei", "genug", "nein", "bild", "finden", "steht", "art", "nen", "kurz",
    "oben", "liegt", "lassen", "lange", "zurück", "komplett",
    # Subreddit-/Region-spezifisch
    "stuttgarter", "stuttgart", "stgt", "deutschland",
    # Sehr umfangreiche, generierte Funktionswörterliste (gekürzt nicht),
    # inkl. schwäbischer Dialektformen:
    "ab", "aber", "abseits", "abzüglich", "achja", "achnee", "acht", "achte", "achter",
    "achtzehn", "achtzig", "achtzigste", "achtzigster", "ah", "aha", "alle", "allein",
    "allem", "allen", "aller", "allerdings", "alles", "allesamt", "als", "also", "am",
    "an", "andere", "andererseits", "anfangs", "angesichts", "anhand", "anlässlich",
    "ans", "anschließend", "ansonsten", "anstatt", "anstelle", "aua", "auch", "auf",
    "aufgrund", "aus", "ausgenommen", "ausschliesslich", "ausschließlich", "ausser",
    "ausserhalb", "außer", "außerhalb", "bei", "beide", "beiden", "beim", "bevor",
    "beziehungsweise", "bezüglich", "bin", "binnen", "bis", "bist", "bleibe", "bleiben",
    "bleibst", "bleibt", "blieb", "blieben", "bliebst", "bloß", "boah", "bravo", "bäh",
    "da", "dabei", "dadurch", "dafür", "dagegen", "damit", "danach", "dann", "darauf",
    "darf", "darum", "das", "dass", "dazu", "dein", "deine", "deiner", "deines", "deins",
    "dem", "den", "denen", "denn", "der", "deren", "des", "deshalb", "dessen", "deswegen",
    "desto", "dich", "die", "dies", "diese", "dieselbe", "dieselben", "diesem", "diesen",
    "dieser", "dieses", "diesseits", "dir", "doch", "drei", "dreizehn", "dreißig",
    "dritte", "dritter", "du", "durch", "durfte", "durften", "dürfen", "dürft", "ebenso",
    "ehe", "ei", "ein", "eine", "einem", "einen", "einer", "einerseits", "eines",
    "einige", "einigen", "einiger", "eins", "einschließlich", "elf", "entgegen", "entlang",
    "entsprechend", "entweder", "er", "erste", "erster", "es", "etliche", "etwas", "euch",
    "euer", "eure", "eurem", "euren", "eures", "exklusive", "ey", "falls", "fern", "ferner",
    "folglich", "fünf", "fünfte", "fünfter", "fünfzehn", "fünfzig", "für", "gegen",
    "gehabt", "gekonnt", "gemocht", "genau", "genauso", "geschweige", "gewesen",
    "gewollt", "geworden", "gleich", "gleichwie", "habe", "haben", "habt", "hach", "haha",
    "halber", "hallo", "hast", "hat", "hatte", "hatten", "hattest", "hattet", "he",
    "heda", "hehe", "hey", "hinsichtlich", "hinter", "hm", "hoho", "holla", "hopp",
    "hoppla", "huch", "hui", "hundert", "hundertste", "hurra", "husch", "hä", "hätte",
    "hätten", "hättest", "hättet", "höh", "ich", "igitt", "ihm", "ihn", "ihnen", "ihr",
    "ihre", "ihrem", "ihren", "ihrer", "ihres", "im", "immerhin", "in", "indem", "indes",
    "infolge", "inklusive", "inmitten", "innerhalb", "ins", "insofern", "inzwischen",
    "irgendein", "irgendeine", "irgendeinem", "irgendeinen", "irgendeines", "irgendetwas",
    "irgendjemand", "irgendwelche", "ist", "je", "jeder", "jedoch", "jene", "jenem", "jenen",
    "jener", "jenes", "jenseits", "kann", "kannst", "kaum", "kein", "keine", "keinem",
    "keinen", "keiner", "keines", "keins", "konnte", "konnten", "konntest", "konntet",
    "kontra", "kraft", "können", "könnt", "könnte", "könnten", "lang", "laut", "links",
    "lol", "längs", "mag", "magst", "man", "manch", "manche", "manchem", "manchen",
    "mancher", "manches", "mangels", "mehrere", "mehrerer", "mein", "meine", "meiner",
    "meines", "mich", "millionen", "mir", "mit", "mitsamt", "mittels", "mmmh", "mochte",
    "mochten", "musst", "musste", "mussten", "muß", "möchte", "möchten", "möchtest",
    "mögt", "müssen", "müsst", "müsste", "nach", "nachdem", "nah", "nahe", "naja", "neben",
    "nebst", "neun", "neunte", "neunzehn", "neunzig", "nichts", "niemand", "noch", "nun",
    "nur", "nächst", "nämlich", "nördlich", "ob", "oberhalb", "obgleich", "obwohl",
    "oder", "oh", "ohne", "oho", "pah", "peng", "per", "pfui", "pro", "psst", "puh",
    "rechts", "samt", "schließlich", "schwupp", "sechs", "sechste", "sechzehn", "sechzig",
    "sei", "seid", "seien", "seine", "seiner", "seit", "seitdem", "seitens", "seitlich",
    "sich", "sie", "sieben", "siebte", "siebzig", "so", "sobald", "sodass", "sofern",
    "solange", "solche", "soll", "sollen", "sollst", "sollt", "sollte", "sondern", "sonst",
    "sooft", "soviel", "soweit", "sowie", "sowohl", "statt", "sämtlich", "südlich",
    "tada", "tausend", "tja", "trara", "trotz", "ts", "uff", "um", "umso", "und",
    "ungeachtet", "uns", "unser", "unsere", "unseres", "unter", "unterhalb", "unweit",
    "ups", "viel", "viele", "vieler", "vier", "vierte", "vierzehn", "vierzig", "vom",
    "von", "vor", "vorbehaltlich", "vorher", "wann", "war", "waren", "warst", "wart",
    "was", "weder", "wegen", "weil", "welche", "welchem", "welchen", "welcher",
    "welches", "wem", "wen", "wenig", "wenige", "wenigen", "wenn", "wer", "werde",
    "werden", "werdet", "wessen", "westlich", "wie", "wieweit", "will", "wir", "wird",
    "wirst", "wo", "wofern", "wohingegen", "wollen", "wollt", "wollte", "wurden",
    "während", "währenddessen", "wäre", "würde", "zehn", "zehnte", "zu", "zudem",
    "zugunsten", "zum", "zumal", "zur", "zusätzlich", "zuvor", "zwanzig", "zwölf",
    "östlich", "über",
    # Schwäbisch
    "ade", "alladweil", "allweil", "au", "auweia", "ben", "bisch", "bissla", "bissle",
    "bissl", "de", "derf", "derfsch", "des", "di", "do", "dr", "ebbas", "ebber", "eich",
    "eier", "em", "emol", "en", "etz", "fei", "gel", "gell", "gella", "gelle", "glei",
    "grad", "gradle", "gscheid", "gscheit", "gschwind", "haett", "hald", "han", "hasch",
    "hesch", "hend", "het", "heut", "jo", "kannsch", "ko", "koennt", "koi", "koina",
    "koine", "la", "mei", "meim", "mer", "mol", "muessa", "muesst", "na", "nauf",
    "ned", "nei", "net", "nix", "no", "nomol", "nuff", "nunder", "odr", "scho", "sel",
    "sell", "selle", "send", "sodele", "soich", "sollsch", "wella", "werd", "werds",
    "werdsch", "zwoi",
}
GERMAN_SW |= FILLER_DE


# Zusätzliche technische „Stopwörter“ / Markup-Artefakte
TECHNICAL_SW = {
    "removed", "deleted", "gelöscht", "automoderator", "moderator", "mods",
    "edit", "bearbeitet", "update", "amp", "x200b", "nbsp",
}

# Optionaler Stemmer (derzeit nicht in Gebrauch; behalten für Optionen)
STEMMER = GermanStemmer(ignore_stopwords=True)


# ---------------------------------------------------------------------------
# Regexe für Cleaning & Tokenisierung
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+")
PUNCT_RE = re.compile(r"[^\wäöüÄÖÜß\s]")
BRACKET_TAG_RE = re.compile(r"\[(?:removed|deleted|edit|bearbeitet)\]", re.I)


# ---------------------------------------------------------------------------
# Basis-Utilities
# ---------------------------------------------------------------------------

def clean_text(s: str) -> str:
    """
    Entfernt Links, Klammer-Tags ([removed] etc.), Sonderzeichen und normalisiert Whitespace.

    Args:
        s: Eingabetext (kann None/leer sein).

    Returns:
        Gesäuberte, kleingeschriebene Zeichenkette.
    """
    s = s or ""
    s = BRACKET_TAG_RE.sub(" ", s)       # [removed]/[deleted]/[edit]/[bearbeitet]
    s = URL_RE.sub(" ", s)               # URLs
    s = PUNCT_RE.sub(" ", s)             # Satzzeichen/Emoji etc.
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def tokenize(text: str) -> List[str]:
    """
    Sehr robuste Tokenisierung für DE/EN inkl. Umlaute.

    Args:
        text: bereits gereinigter Text (clean_text).

    Returns:
        Liste von Tokens (>=2 Zeichen).
    """
    return re.findall(r"\b[\wäöüÄÖÜß]{2,}\b", text)


def detect_language_by_stopword_ratio(tokens: List[str],
                                      min_tokens: int = 12,
                                      margin: float = 1.15) -> str:
    """
    Heuristik: vergleicht den Anteil deutscher vs. englischer Stopwörter.

    Args:
        tokens: Tokenliste.
        min_tokens: Mindestgröße, sonst „unknown“.
        margin: Quotient-Schwelle, um „de“ bzw. „en“ zu entscheiden.

    Returns:
        "de", "en" oder "unknown".
    """
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


# ---------------------------------------------------------------------------
# Haupt-Pipeline
# ---------------------------------------------------------------------------

def add_clean_columns(df: pd.DataFrame,
                      use_stemming: bool = False,
                      filter_english: bool = True,
                      include_comments: bool = False) -> pd.DataFrame:
    """
    Fügt dem DataFrame Standard-Spalten für die Themenmodellierung hinzu.

    Erzeugte Spalten:
      - text_all:    Post-Text (+ optional Comments) kombiniert.
      - clean:       gesäuberter Gesamttext (clean_text).
      - tokens_raw:  Tokenliste von clean (vor Stopwortfilter).
      - lang:        grobe Sprachheuristik ("de"/"en"/"unknown").
      - tokens:      gefilterte Tokens (Stopwörter/Zahlen/Länge), inkl. Collocations.
      - clean_for_vect: 'tokens' als Leerzeichen-join (für Vektorisierung).

    Args:
        df: Eingangs-DataFrame (erwartet mind. Spalten "text" und optional "comments_text").
        use_stemming: aktuell nicht genutzt; Flag bleibt für API-Kompatibilität.
        filter_english: True → EN-Beiträge werden (per Heuristik) entfernt.
        include_comments: True → Comments in text_all einbeziehen.

    Returns:
        Neuer DataFrame mit o.g. Spalten; ggf. Zeilen ohne englische Beiträge.
    """
    df = df.copy()

    # --- Textgrundlage aufbauen -------------------------------------------------
    text_series = df["text"] if "text" in df.columns else pd.Series([""] * len(df))
    comments_series = (
        df["comments_text"].fillna("") if include_comments and "comments_text" in df.columns
        else pd.Series([""] * len(df))
    )

    df["text_all"] = (text_series.fillna("") + " " + comments_series).str.strip()

    # --- Cleaning & Tokenisierung (roh) ----------------------------------------
    df["clean"] = df["text_all"].map(clean_text)
    df["tokens_raw"] = df["clean"].map(tokenize)

    # --- Sprachheuristik & optionaler EN-Filter --------------------------------
    df["lang"] = df["tokens_raw"].map(detect_language_by_stopword_ratio)
    if filter_english:
        df = df[df["lang"] != "en"].reset_index(drop=True)

    # --- Stopwort-/Zahlen-/Längenfilter → tokens --------------------------------
    def _proc(tokens: List[str]) -> List[str]:
        kept: List[str] = []
        for t in tokens:
            if t in GERMAN_SW or t in ENGLISH_SW:
                continue
            if any(ch.isdigit() for ch in t):
                continue
            if len(t) < 3:
                continue
            if t in {"deleted"}:
                continue
            kept.append(t)
        # (use_stemming wird derzeit nicht genutzt, Lesbarkeit der Terme steht im Fokus)
        return kept

    # Wichtig: Erst „tokens“ erzeugen, dann Collocations ergänzen.
    df["tokens"] = df["tokens_raw"].map(_proc)

    # --- Collocations (PMI-Bigrams) als Zusatz-Tokens ---------------------------
    # Wird bewusst lokal importiert (Lazy-Import), um Importkosten zu sparen.
    from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

    def _add_collocation_tokens(tokens_list: List[List[str]],
                                top_n: int = 200,
                                min_freq: int = 5) -> List[List[str]]:
        """
        Ergänzt PMI-starke Bigrams (mit Unterstrich) als weitere Tokens, falls
        die Bigramme im Korpus hinreichend häufig vorkommen.

        Args:
            tokens_list: Liste von Tokenlisten (Dokumente).
            top_n: Anzahl top-bewerteter Bigrams (global).
            min_freq: Mindesthäufigkeit eines Bigramms im Gesamtkorpus.

        Returns:
            Neue Liste von Tokenlisten, in denen passende Bigramme zusätzlich
            als „w1_w2“-Token angehängt sind.
        """
        if not tokens_list:
            return tokens_list

        all_tokens: List[str] = [t for ts in tokens_list for t in ts]
        finder = BigramCollocationFinder.from_words(all_tokens)
        finder.apply_freq_filter(min_freq)

        scored = finder.score_ngrams(BigramAssocMeasures.pmi)
        top_bigrams = set(tuple(bg) for bg, _ in sorted(scored, key=lambda x: -x[1])[:top_n])

        augmented_docs: List[List[str]] = []
        for ts in tokens_list:
            augmented = ts[:]
            for i in range(len(ts) - 1):
                pair = (ts[i], ts[i + 1])
                if pair in top_bigrams:
                    augmented.append(f"{ts[i]}_{ts[i + 1]}")
            augmented_docs.append(augmented)
        return augmented_docs

    df["tokens"] = _add_collocation_tokens(df["tokens"].tolist(), top_n=200, min_freq=5)

    # --- Vektor-String ----------------------------------------------------------
    df["clean_for_vect"] = df["tokens"].map(lambda ts: " ".join(ts))
    return df


# ---------------------------------------------------------------------------
# Sonstige Utilities
# ---------------------------------------------------------------------------

def extract_flairs(df: pd.DataFrame) -> pd.Series:
    """
    Extrahiert und zählt Flairs (robust gegenüber verschiedenen Spaltennamen).

    Unterstützte Spaltennamen (case-insensitive):
      - flair_text, link_flair_text, flair

    Args:
        df: DataFrame mit potentieller Flair-Spalte.

    Returns:
        Series mit Flair-Zählungen (absteigend), leer falls nicht vorhanden.
    """
    cand = [c for c in df.columns if c.lower() in ("flair_text", "link_flair_text", "flair")]
    if not cand:
        return pd.Series(dtype="int64")
    col = cand[0]
    return (
        df[col]
        .fillna("")
        .str.strip()
        .replace({"": None})
        .dropna()
        .value_counts()
    )


# Für gezielten Import von Konstanten/Funktionen in anderen Modulen
__all__ = [
    "GERMAN_SW",
    "ENGLISH_SW",
    "TECHNICAL_SW",
    "STEMMER",
    "clean_text",
    "tokenize",
    "detect_language_by_stopword_ratio",
    "add_clean_columns",
    "extract_flairs",
]
