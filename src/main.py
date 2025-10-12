# main.py
# -*- coding: utf-8 -*-
"""
r/Stuttgart – Themenreport (LDA) mit HTML-Report

Pipeline (high level)
---------------------
1) Daten laden (CSV-Cache oder Reddit-API), Grundstatistiken (Top Flairs, Top User).
2) Für jede vordefinierte "Flair-Run"-Konfiguration:
   - Preprocessing (Tokenisierung, optionale Kommentar-Einbeziehung, Sprachfilter).
   - Auto-Stopwörter + manuelle Zusatz-Stopwörter.
   - Count-Vektorisierung mit robuster Parameterglättung (min_df/max_df-Fenster).
   - Fallback auf TF-IDF-Keywords bei sehr kleinen Korpora.
   - LDA für Kandidaten-K und Seeds, Auswahl des besten Modells via Kohärenz.
   - Lesbare Topic-Labels (Bigramme bevorzugt, Domänen-Bonus) + Label-Deduplikation.
   - Ausgabe Beispiel-Posts pro Topic (CSV) + präziser Konsolen-Output.

3) Am Ende: Aus dem *Konsolen-Output* wird ein hübscher HTML-Report gebaut
   (DE/EN-Blöcke nebeneinander, Statistik, vollständiger Roh-Log am Ende).

WICHTIGER KONTRAKT (für den Report-Parser)
------------------------------------------
Die HTML-Generierung unten parst **genau** folgende Konsolenzeilen:
- Abschnittsüberschrift:          "# ==== Themen für Flair: {Name} ===="
- Vectorizer-Protokoll:           "[Info] Try vectorize: n_docs=... -> features=..."
- Modell-Zeile (K/Coh/Features):  "[Info] {Name}: K=... Seed=... Coh=... (features=..., min_df=..., ...)"
- Topic-Zeilen:                   "Topic {rank} (ID {tid}) [Label optional]: {term1}, {term2}, ..."

Diese Strings **nicht** verändern (keine anderen Klammern, keine extra Doppelpunkte usw.),
sonst versteht der Parser den Log nicht mehr.

Artefakte/Outputs
-----------------
- DATA_DIR/top_flairs.csv            – Top-Flairs
- DATA_DIR/top_users.csv             – Aktive Nutzer
- DATA_DIR/samples_..._per_topic.csv – Beispielposts pro Topic und Flair-Block
- DATA_DIR/report.html               – HTML-Report
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Dict, Tuple, Optional

import pandas as pd

from .fetch import fetch_subreddit_posts
from .preprocess import (
    add_clean_columns, extract_flairs, GERMAN_SW, ENGLISH_SW,
    clean_text, tokenize, TECHNICAL_SW
)
from .topics import (
    vectorize_counts, top_terms_per_topic, pick_top_k_topics, fit_best_lda,
)
from .utils import DATA_DIR


# ============================================================================
# Konfiguration
# ============================================================================

# Seeds: für Reproduzierbarkeit und robuste Auswahl des LDA-Modells.
SEEDS = (13, 21, 42, 77)

# Default-Kandidaten für K (Anzahl Topics), wenn pro-Block nichts erzwungen wird.
DEFAULT_CANDIDATE_K = [6, 8, 10, 12]

# (Derzeit nicht aktiv benutzt – historischer Tuning-Anker für Feature-Bandbreiten.)
FEAT_TARGET_LOW  = 300
FEAT_TARGET_HIGH = 2000

# Harte Untergrenze an Features; darunter LDA i. d. R. nicht sinnvoll.
FEAT_MIN_HARD    = 100

# Pro-Flair Modelle (Block-Konfigurationen).
# Tipp: Für jeden Block lassen sich Stopwort-Logik, Titelgewichtung, min_tokens,
#       K-Kandidaten usw. getrennt justieren.
FLAIR_RUNS = [
    # ----------------------------
    # Diskussion (Deutsch)
    # ----------------------------
    {"name": "Diskussion - DE",
     "flairs": ["Diskussion"],
     "lang": "de",
     "filter_english": True,   # EN wird schon im Preprocess entfernt
     "include_comments": True,
     "min_tokens": 20,
     "title_boost": 1,         # Titel leicht berücksichtigen
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"mal","mehr","einfach","echt","leider","besser","immer","schon",
                         "klar","weiß","gesagt","gemacht","findet","find","sagen"},
     "feat_min_hard": 80,
     "candidate_k": [4, 5, 6]},

    # ----------------------------
    # Diskussion (Englisch)
    # ----------------------------
    {"name": "Diskussion - EN",
     "flairs": ["Diskussion"],
     "lang": "en",
     "filter_english": False,  # explizit EN-Doks zulassen
     "include_comments": True,
     "min_tokens": 10,
     "title_boost": 2,         # Headlines etwas stärker
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"get","know","like","make","one","people","really","thing","things","would","could","also","still"},
     "feat_min_hard": 40,
     "candidate_k": [2]},

    # ----------------------------
    # News (Deutsch)
    # ----------------------------
    {"name": "News - DE",
     "flairs": ["News"],
     "lang": "de",
     "filter_english": True,
     "include_comments": False, # Headlines dominieren, Kommentare oft Rauschen
     "min_tokens": 5,
     "title_boost": 5,          # Headlines sehr stark gewichten
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"mal","mehr","letzte","letzten","schon"},
     "feat_min_hard": 40,
     "candidate_k": [3],
     # Enges (min_df, max_df)-Fenster für kleine, headline-lastige Korpora
     "pairs": [(3, 0.60), (3, 1.00), (2, 0.60)]},

    # ----------------------------
    # News (Englisch)
    # ----------------------------
    {"name": "News - EN",
     "flairs": ["News"],
     "lang": "en",
     "filter_english": False,
     "include_comments": False,
     "min_tokens": 8,
     "title_boost": 5,
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"get","know","like","make","one","people","really","thing","things","would","could","also","still"},
     "feat_min_hard": 30,
     "candidate_k": [2, 3]},

    # ----------------------------
    # Events (Deutsch)
    # ----------------------------
    {"name": "Events - DE",
     "flairs": ["Events"],
     "lang": "de",
     "filter_english": True,
     "include_comments": True,
     "min_tokens": 15,
     "title_boost": 2,
     "df_ratio_auto_sw": 0.60,
     "extra_stopwords": {"post", "infos", "weitere", "weitere informationen"},
     "feat_min_hard": 80,
     "candidate_k": [3, 4, 5]},

    # ----------------------------
    # Events (Englisch)
    # ----------------------------
    {"name": "Events - EN",
     "flairs": ["Events"],
     "lang": "en",
     "filter_english": False,
     "include_comments": True,
     "min_tokens": 10,
     "title_boost": 2,
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"get","know","like","make","one","people","really","thing","things","would","could","also","still"},
     "feat_min_hard": 30,
     "candidate_k": [2, 3]},

    # ----------------------------
    # Frage/Advice (Deutsch)
    # ----------------------------
    {"name": "Frage/Advice - DE",
     "flairs": ["Frage / Advice", "Looking for..."],
     "lang": "de",
     "filter_english": True,
     "include_comments": True,
     "min_tokens": 15,
     "title_boost": 1,
     "df_ratio_auto_sw": 0.60,
     "extra_stopwords": {"mal","mehr","einfach","schon","leider","eigentlich"},
     "feat_min_hard": 80,
     "candidate_k": [5, 6]},

    # ----------------------------
    # Frage/Advice (Englisch)
    # ----------------------------
    {"name": "Frage/Advice - EN",
     "flairs": ["Frage / Advice", "Looking for..."],
     "lang": "en",
     "filter_english": False,
     "include_comments": True,
     "min_tokens": 10,
     "title_boost": 1,
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"get","know","like","make","one","people","really","thing","things","would","could","also","still"},
     "feat_min_hard": 40,
     "candidate_k": [2, 3]},
]


# ============================================================================
# Hilfsfunktionen (Vektorisierung, Heuristiken, Labeling)
# ============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer


def _fallback_keywords(texts: List[str], stop_words_norm, top_n: int = 12):
    """
    Fallback-Schritt für sehr kleine Korpora:
    Ermittelt einfache TF-IDF-Bigramm-Keywords, wenn LDA keinen Sinn ergibt.

    Args:
        texts: Liste der vektorisierten Texte (Strings).
        stop_words_norm: normalisierte Stopwortliste (so wie an Vectorizer übergeben).
        top_n: Anzahl zu zeigender Keywords.

    Side effects:
        Druckt eine Zeile "Fallback-Keywords: kw1, kw2, ..." in die Konsole.
    """
    try:
        vec = TfidfVectorizer(ngram_range=(2, 2), min_df=1, max_df=0.95, stop_words=stop_words_norm)
        X = vec.fit_transform(texts)
        if X.shape[1] == 0:
            print("[Fallback] Keine Keywords ermittelbar.")
            return
        scores = X.mean(axis=0).A1
        idx = scores.argsort()[::-1][:top_n]
        terms = vec.get_feature_names_out()[idx]
        print("Fallback-Keywords:", ", ".join(terms))
    except Exception as e:
        print(f"[Fallback] TF-IDF fehlgeschlagen: {e}")


def _infer_lang_from_tokens(tokens):
    """
    Sehr einfache Sprachheuristik auf Token-Basis (DE/EN/unk),
    verwendet Treffermengen der jeweils anderen Stopwortliste.

    Returns:
        "de" | "en" | "unk"
    """
    if not tokens:
        return "unk"
    de_hits = sum((t in GERMAN_SW) for t in tokens)
    en_hits = sum((t in ENGLISH_SW) for t in tokens)
    if en_hits >= max(2, de_hits + 1):
        return "en"
    if de_hits >= max(2, en_hits + 1):
        return "de"
    return "unk"


def _normalize_stopwords(stopset: set[str]) -> List[str]:
    """
    Normalisiert Stopwörter durch clean_text/tokenize (z. B. Mehrwortbegriffe -> Tokens).
    Gibt eine Liste zurück, die direkt in Vectorizer.stop_words passt.
    """
    norm = set()
    for w in stopset:
        toks = tokenize(clean_text(w))
        norm.update(toks)
    return sorted({t for t in norm if len(t) >= 2})


def _build_texts_with_title_boost(df: pd.DataFrame, title_boost: int) -> List[str]:
    """
    Erzeugt pro Dokument den Vektorisierungs-Text:
      clean_for_vect + (Titel-Tokens * title_boost)
    Entfernt zusätzlich Unterstriche in Collocations.

    Achtung: Reihenfolge und Tokenisierung muss kompatibel zur Vektorisierung bleiben.
    """
    base = df["clean_for_vect"].fillna("")
    if title_boost <= 0 or "title" not in df.columns:
        texts = base
    else:
        title_tokens = (
            df["title"].fillna("").map(clean_text).map(tokenize).map(lambda ts: " ".join(ts))
        )
        title_boost_str = title_tokens.map(lambda s: (s + " ") * title_boost)
        texts = (base + " " + title_boost_str).str.strip()

    # Collocations: "_" -> " ", Duplikat-Leerzeichen entfernen
    texts = texts.str.replace("_", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
    return texts.tolist()


def _auto_stopwords_from_tokens(tokens_list: List[List[str]],
                                df_ratio: float = 0.70,
                                top_n_cap: int = 60,
                                whitelist: Optional[set] = None) -> set:
    """
    Heuristik: Alle Tokens, die in >= df_ratio Dokumenten vorkommen (bis top_n_cap),
    werden als Auto-Stopwörter vorgeschlagen – außer sie sind whitelisted/Domänenwörter.

    Args:
        tokens_list: Liste Tokenlisten je Dokument.
        df_ratio: Dokumentfrequenz-Schwelle (z. B. 0.5 = in mind. 50% aller Doks).
        top_n_cap: Obergrenze der Kandidaten (nur die häufigsten).
        whitelist: Wörter, die *nicht* zu Stopwörtern werden sollen (z. B. Domänennamen).
    """
    whitelist = whitelist or set()
    n_docs = len(tokens_list)
    if n_docs == 0:
        return set()

    doc_freq = Counter()
    for ts in tokens_list:
        doc_freq.update(set(ts))

    thresh = max(1, int(math.ceil(df_ratio * n_docs)))
    cand = [t for t, dfc in doc_freq.items() if dfc >= thresh]
    cand.sort(key=lambda t: (-doc_freq[t], t))
    cand = cand[:top_n_cap]

    auto_sw = set()
    for t in cand:
        if t in whitelist:                # Domänenwörter nicht muten
            continue
        if len(t) < 3:                     # sehr kurze Tokens ignorieren
            continue
        if any(ch.isdigit() for ch in t):  # Zahlentokens ignorieren
            continue
        auto_sw.add(t)
    return auto_sw


def _safe_vectorize(texts, min_df, max_df, stop_words_norm, max_features):
    """
    Wrapper um vectorize_counts mit defensiver Fehlerbehandlung
    (z. B. wenn max_df < min_df resultiert).

    Returns:
        (vectorizer, X) oder (None, None) bei Fehlschlag.
    """
    try:
        vec, X = vectorize_counts(
            texts,
            ngram=(1, 2),
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            stop_words=stop_words_norm
        )
        return vec, X
    except ValueError as e:
        msg = str(e)
        # Häufiger Sonderfall: max_df < min_df -> auf max_df=1.0 hochsetzen
        if "max_df corresponds to < documents than min_df" in msg:
            adj_max_df = 1.0
            print(f"[Warn] Adjusting max_df to {adj_max_df:.2f} (min_df={min_df}) due to: {e}")
            try:
                vec, X = vectorize_counts(
                    texts, ngram=(1, 2), min_df=min_df, max_df=adj_max_df,
                    max_features=max_features, stop_words=stop_words_norm
                )
                return vec, X
            except Exception:
                return None, None
        # Nach dem Pruning keine Terme -> zurückfallen lassen
        if "After pruning, no terms remain" in msg:
            return None, None
        return None, None


def _vectorize_to_window(
    texts, stop_words_norm, feat_min_hard: int = 80,
    max_features_cap: int = 12000,
    pairs: Optional[List[Tuple[int, float]]] = None
):
    """
    Probiert mehrere (min_df, max_df)-Paare (Fenster) durch und wählt:
      - den ersten Versuch, der >= feat_min_hard Features liefert
      - sonst den Versuch mit den meisten Features (best effort)

    Args:
        texts: Liste der vektorisierten Dokumenttexte.
        stop_words_norm: normalisierte Stopwörter.
        feat_min_hard: harte Untergrenze an Features für LDA.
        max_features_cap: Vectorizer-Deckel (zur Laufzeitkontrolle).
        pairs: falls None -> default_pairs (robuste Standardfenster).
    """
    n_docs = len(texts)

    # Standardfenster für typische, mittelgroße Korpora
    default_pairs = [(15, 1.00), (10, 0.30), (5, 0.40), (2, 0.60)]
    pairs = pairs or default_pairs

    best_by_feats = None
    best_feats = -1

    for (min_df, max_df) in pairs:
        # min_df an n_docs koppeln (verhindert zu hohe absolute min_df bei kleinen n)
        min_df_eff = min(min_df, max(1, int(0.15 * n_docs)))
        vec, X = _safe_vectorize(texts, min_df_eff, max_df, stop_words_norm, max_features=max_features_cap)
        n_features = int(X.shape[1]) if X is not None else 0
        print(f"[Info] Try vectorize: n_docs={n_docs}, min_df={min_df_eff}, max_df={max_df:.2f} -> features={n_features}")

        if X is not None and n_features >= feat_min_hard:
            return vec, X, {"min_df": min_df_eff, "max_df": max_df}

        if n_features > best_feats and X is not None:
            best_by_feats = (vec, X, {"min_df": min_df_eff, "max_df": max_df}, n_features)
            best_feats = n_features

    if best_by_feats:
        vec, X, params, n_features = best_by_feats
        print(f"[Info] Using best-effort features={n_features} (min_df={params['min_df']}, max_df={params['max_df']:.2f})")
        return vec, X, params

    return None, None, None


def _lang_guess_series(df: pd.DataFrame) -> pd.Series:
    """
    Heuristischer Sprach-Guess pro Dokument ("de"/"en"/"unk") basierend
    auf Titel + Rohtext (ohne Stopwort-Removal).
    """
    parts = []
    if "title" in df.columns:
        parts.append(df["title"].fillna(""))

    for c in ("selftext", "body", "text", "content"):
        if c in df.columns:
            parts.append(df[c].fillna(""))
            break

    if not parts:
        if "clean_for_vect" in df.columns:
            parts.append(df["clean_for_vect"].fillna(""))
        else:
            return pd.Series(["unk"] * len(df), index=df.index)

    # Titel + erster Textkörper zusammenkleben
    s = parts[0]
    for p in parts[1:]:
        s = s.str.cat(p, sep=" ", na_rep="")

    # Dann clean/tokenize und auf Stopwort-Treffer prüfen
    lang_tokens = s.map(clean_text).map(tokenize)
    return lang_tokens.map(_infer_lang_from_tokens)


# Stopwort-/Label-Hilfssets ----------------------------------------------------

# Wörter, die zwar häufig sind, aber als Topic-Labels kaum Mehrwert liefern:
LABEL_FILLER = {
    # DE
    "mal","mehr","schon","einfach","leider","echt","besser","immer","klar","gut","gibt",
    "find","findet","frage","wohl","gerade","vielleicht","irgendwie","ziemlich","bisschen",
    "eher","sogar","glaube","finde","tatsächlich","natürlich","halt","eigentlich","wirklich",
    "besten","deutschen","absolut","alte","alten","neue","neuen","groß","klein",
    "teuer","nie","rum","ganzen","davon","drauf","bzw",
    # EN
    "like","get","know","make","one","people","really","thing","things","would","could",
    "also","still","maybe","just","even","well","good","bad","new","old","time"
}

# Domänenterms (Stuttgart/ÖPNV/Politik etc.), die wir als Label eher *bevorzugen*:
DOMAIN_WHITELIST = {
    "stuttgart","stuttgarter","0711","kessel","neckar",
    "hbf","hauptbahnhof","olgaeck","vaihingen","bad","cannstatt","bad_cannstatt","cannstatter",
    "schlossplatz","wilhelma","feuersee","boeblingen","ludwigsburg","esslingen",
    "vvs","ssb","u-bahn","s-bahn","stadtbahn","fernsehturm","uni","messe","killesberg",
    "polizei","demo","wahl","bundestagswahl","afd","cdu","spd","grüne","fdp",
    "olgaeck","marienplatz","stadtmitte","münchener","börsenplatz","pragfriedhof","pragsattel",
    "heslach","deggerloch","berg","ostfildern","leonberg"
}

# Anzeigeform schöner schreiben (z. B. "hbf" -> "Hbf", "bad_cannstatt" -> "Bad Cannstatt")
DISPLAY_CASE = {
    "stuttgart":"Stuttgart","stuttgarter":"Stuttgarter","neckar":"Neckar",
    "hbf":"Hbf","hauptbahnhof":"Hauptbahnhof","olgaeck":"Olgaeck",
    "vaihingen":"Vaihingen","bad_cannstatt":"Bad Cannstatt","cannstatt":"Cannstatt","cannstatter":"Cannstatter",
    "schlossplatz":"Schlossplatz","wilhelma":"Wilhelma","feuersee":"Feuersee","ludwigsburg":"Ludwigsburg","esslingen":"Esslingen",
    "vvs":"VVS","ssb":"SSB","u-bahn":"U-Bahn","s-bahn":"S-Bahn","stadtbahn":"Stadtbahn","fernsehturm":"Fernsehturm",
    "polizei":"Polizei","demo":"Demo","wahl":"Wahl","bundestagswahl":"Bundestagswahl","afd":"AfD","cdu":"CDU","spd":"SPD","grüne":"Grüne","fdp":"FDP"
}


# ============================================================================
# Label-Kandidaten & -Deduplikation
# ============================================================================

def _label_candidate_list(primary_label: str, topic_terms) -> list[str]:
    """
    Erzeugt alternative Label-Kandidaten (Varianten) für *ein* Topic:

    - Primärlabel (falls vorhanden)
    - Bigramme aus Top-Unigrammen in verschiedenen Anordnungen
    - Einwort-Fallbacks

    Args:
        primary_label: bereits gewähltes Label (z. B. durch _build_topic_labels)
        topic_terms: Liste [(term, weight), ...] der Top-Terme für dieses Topic

    Returns:
        Liste eindeutiger Kandidatenstrings (erste ist beste).
    """
    terms = [w for (w, _) in (topic_terms or [])][:4]

    def T(s: str) -> str:
        return (s or "").strip().replace("_", " ").title()

    cands: list[str] = []
    seen = set()

    def add(lbl: str):
        key = lbl.strip().lower()
        if key and key not in seen:
            seen.add(key)
            cands.append(lbl)

    # 1) Primärlabel
    if primary_label:
        add(primary_label)

    # 2) Bigramme aus Top-Unigrammen
    if len(terms) >= 2:
        a, b = T(terms[0]), T(terms[1])
        add(f"{a} - {b}")
        add(f"{b} - {a}")

    # 3) Weitere Kombinationen
    if len(terms) >= 3:
        add(f"{T(terms[0])} - {T(terms[2])}")
    if len(terms) >= 4:
        add(f"{T(terms[1])} - {T(terms[3])}")

    # 4) Unigram-Fallbacks
    for t in terms[:3]:
        add(T(t))

    return cands


def _dedup_topic_labels(primary_labels: list[str], topic_terms_list: list[list[tuple[str, float]]]) -> list[str]:
    """
    Dedupliziert Labels über *mehrere* Topics hinweg. Wenn ein Label schon
    vergeben ist, wird für das Topic die nächste Kandidatenvariante gewählt.

    Args:
        primary_labels: Labels in Topic-Reihenfolge (primär vorgeschlagen).
        topic_terms_list: Top-Terme je Topic (gleiches Indexing wie primary_labels).

    Returns:
        Liste derselben Länge mit eindeutigen (deduplizierten) Labels.
    """
    used = set()
    out: list[str] = []

    for i, prim in enumerate(primary_labels):
        cands = _label_candidate_list(prim, topic_terms_list[i])
        chosen = None
        for c in cands:
            key = c.strip().lower()
            if key and key not in used:
                chosen = c
                break
        if not chosen:
            # Harter Fallback, praktisch selten nötig
            chosen = f"Topic {i+1}"
        used.add(chosen.strip().lower())
        out.append(chosen)
    return out


# ============================================================================
# Topic-Labeling (mit Domänen-Bonus)
# ============================================================================

def _split_ngram(term: str):
    """Hilfsfunktion: 'a_b'/'a b' -> ['a', 'b']; Sonst ['a']."""
    term = term.strip()
    if " " in term:
        return term.split()
    if "_" in term:
        return term.split("_")
    return [term]


def _is_valid_token(tok: str, stop_words, min_len: int = 3) -> bool:
    """Token ist brauchbar, wenn lang genug, keine Ziffern, nicht Stopwort/Füllwort."""
    if not tok or len(tok) < min_len:
        return False
    if tok.isdigit():
        return False
    if tok in stop_words or tok in LABEL_FILLER:
        return False
    return True


def _pretty_token(tok: str) -> str:
    """Anzeigeform (Unterstriche weg, Domain-Casing anwenden, sonst Title-Case)."""
    t = tok.replace("_", " ")
    return DISPLAY_CASE.get(tok, DISPLAY_CASE.get(t, t.title()))


def _score_bigram_for_topic(bigram: str, t_id: int, vocab_index: dict, comps, stop_words) -> float:
    """
    Scoring für Bigramme pro Topic:
    - Basis: Gewicht des Bigramms + 0.5*(Gewicht Wort1 + Gewicht Wort2)
    - Domänenbonus für Stuttgart-Wörter
    - leichte Strafe für Füllwörter
    """
    toks = _split_ngram(bigram)
    if len(toks) != 2:
        return -1e9
    if not all(_is_valid_token(t, stop_words) for t in toks):
        return -1e9

    w_bigram = comps[t_id, vocab_index.get(bigram, -1)] if bigram in vocab_index else 0.0
    w_t1 = comps[t_id, vocab_index.get(toks[0], -1)] if toks[0] in vocab_index else 0.0
    w_t2 = comps[t_id, vocab_index.get(toks[1], -1)] if toks[1] in vocab_index else 0.0
    score = w_bigram + 0.5 * (w_t1 + w_t2)

    if toks[0] in DOMAIN_WHITELIST: score += 2.0
    if toks[1] in DOMAIN_WHITELIST: score += 2.0
    if toks[0] in LABEL_FILLER:     score -= 1.0
    if toks[1] in LABEL_FILLER:     score -= 1.0
    return score


def _build_topic_labels(lda_model, feat_names, stop_words, topn: int = 40, join_char: str = " - "):
    """
    Konstruiert pro Topic ein **lesbares Label**:
    - Bevorzugt das bestbewertete Bigramm (mit Domänen-Bonus),
    - sonst nimmt es zwei valide Unigramme (Domänen-Terme bevorzugt),
    - als letzter Fallback das *erste* Top-Term.

    Returns:
        Liste von Strings (ein Label pro Topic, Reihenfolge = Topic-ID).
    """
    comps = lda_model.components_
    n_topics = comps.shape[0]
    labels = [""] * n_topics
    vocab_index = {feat: idx for idx, feat in enumerate(feat_names)}

    for t_id in range(n_topics):
        order = comps[t_id].argsort()[::-1]
        top_idx = order[:topn]
        top_terms = [feat_names[j] for j in top_idx]

        # Kandidaten-Bigramme scoren
        cand_bi = [w for w in top_terms if len(_split_ngram(w)) == 2]
        scored = [(w, _score_bigram_for_topic(w, t_id, vocab_index, comps, stop_words)) for w in cand_bi]
        scored.sort(key=lambda x: x[1], reverse=True)

        label = ""
        if scored and scored[0][1] > 0:
            t1, t2 = _split_ngram(scored[0][0])
            label = _pretty_token(t1) + join_char + _pretty_token(t2)
        else:
            # Unigram-Variante
            unis = [w for w in top_terms if len(_split_ngram(w)) == 1 and _is_valid_token(w, stop_words)]
            if not unis:
                labels[t_id] = top_terms[0] if top_terms else f"Topic {t_id}"
                continue
            # Domänenwörter nach vorn
            unis.sort(key=lambda w: (w not in DOMAIN_WHITELIST, ), reverse=False)
            if len(unis) >= 2:
                label = _pretty_token(unis[0]) + join_char + _pretty_token(unis[1])
            else:
                label = _pretty_token(unis[0])

        labels[t_id] = label
    return labels


# ============================================================================
# Pro-Flair-Block: End-to-End-Lauf (Preprocess -> Vectorize -> LDA -> Output)
# ============================================================================

def _run_for_flair_block(df_raw: pd.DataFrame, run_cfg: Dict, base_stop_union: set) -> None:
    """
    Führt den gesamten Prozess für *einen* Flair-Block aus:
    Subsetting, Preprocess, Stopwort-Heuristik, Vektorisierung, LDA,
    Label-Erzeugung + Deduplikation, Beispielpost-Export, Konsolen-Output.

    Achtung:
      - Konsolenprints werden vom HTML-Parser weiter unten geparst (Format beibehalten!).
    """
    name = run_cfg["name"]
    fe = run_cfg.get("filter_english", True)
    flairs = run_cfg["flairs"]
    include_comments = run_cfg.get("include_comments", False)
    min_tokens = run_cfg.get("min_tokens", 30)
    title_boost = run_cfg.get("title_boost", 2)
    df_ratio_auto_sw = run_cfg.get("df_ratio_auto_sw", 0.70)
    extra_stopwords = set(run_cfg.get("extra_stopwords", set()))
    feat_min_hard = run_cfg.get("feat_min_hard", FEAT_MIN_HARD)
    cand_k = run_cfg.get("candidate_k", None)
    lang_target = run_cfg.get("lang")  # "de" | "en" | None

    print(f"\n# ==== Themen für Flair: {name} ====")

    # --- Subset auf gewünschten Flair
    if "flair_text" in df_raw.columns:
        sub_raw = df_raw[df_raw["flair_text"].isin(flairs)].reset_index(drop=True)
    else:
        sub_raw = df_raw.copy()
    if sub_raw.empty:
        print(f"[Skip] Keine Dokumente für {name}.")
        return

    # --- Preprocess (inkl. optionaler Kommentare)
    df = add_clean_columns(sub_raw, use_stemming=False, filter_english=fe, include_comments=include_comments)

    # Dok-länge filtern
    if "tokens" in df.columns:
        df = df[df["tokens"].map(len) >= min_tokens].reset_index(drop=True)
    if df.empty:
        print(f"[Skip] Nach Preprocess/Tokenfilter keine Dokumente für {name}.")
        return

    # --- Sprachfilter (nur wenn explizit DE/EN gefordert)
    if lang_target in ("de", "en"):
        if lang_target == "de" and fe is True:
            # DE wird implizit durch filter_english=True beibehalten
            print("[Info] Language filter 'de': skipped (filter_english=True hat EN bereits entfernt).")
        else:
            df["lang_guess"] = _lang_guess_series(df)
            before = len(df)
            df = df[df["lang_guess"] == lang_target].reset_index(drop=True)
            print(f"[Info] Language filter '{lang_target}': kept {len(df)}/{before} docs.")
            if len(df) < 5:
                print(f"[Skip] Zu wenig klare {lang_target.upper()}-Dokumente – LDA übersprungen.")
                return

    # --- Auto-Stopwörter + manuelle Ergänzungen
    auto_sw = _auto_stopwords_from_tokens(
        df["tokens"].tolist(),
        df_ratio=df_ratio_auto_sw,
        top_n_cap=(40 if len(df) < 25 else 60),
        whitelist=DOMAIN_WHITELIST
    )
    if extra_stopwords:
        auto_sw |= set(extra_stopwords)
    if auto_sw:
        show = sorted(list(auto_sw))[:12]
        print(f"[Info] Auto-Stopwörter ({len(auto_sw)}): {show}{' ...' if len(auto_sw)>12 else ''}")

    # Gesamtes Stopwort-Set (DE/EN/Technik + Auto)
    stop_union = (GERMAN_SW | ENGLISH_SW | TECHNICAL_SW | auto_sw)
    stop_union_norm = _normalize_stopwords(stop_union)

    # --- Texte mit Titel-Boost bauen (Vektorisierungsbasis)
    texts = _build_texts_with_title_boost(df, title_boost)

    # --- Param-Fenster für Vectorizer (News/Events etwas anders)
    is_en = (run_cfg.get("lang") == "en")
    is_news_or_events = any(nm in run_cfg["name"].lower() for nm in ["news", "events"])
    pairs = run_cfg.get("pairs")
    if pairs is None and is_news_or_events:
        pairs = [(5, 0.60), (10, 0.30), (15, 1.00), (2, 0.60)]

    # --- Vektorisierung (robust)
    vec_counts, Xc, used_params = _vectorize_to_window(
        texts,
        stop_words_norm=stop_union_norm,
        feat_min_hard=feat_min_hard,
        max_features_cap=(4000 if is_en else 12000),
        pairs=pairs
    )
    if vec_counts is None or Xc is None:
        print(f"[Abbruch] Zu wenig Features in {name}.")
        return

    # --- Fallback auf TF-IDF bei zu kleinem Korpus
    min_feats_for_lda = run_cfg.get("feat_min_hard", 80)
    min_docs_for_lda  = run_cfg.get("docs_min", 8)
    if Xc.shape[0] < min_docs_for_lda or Xc.shape[1] < min_feats_for_lda:
        print(
            f"[Info] {name}: kleiner Korpus – Fallback auf TF-IDF-Keywords "
            f"(docs={Xc.shape[0]} < {min_docs_for_lda} oder "
            f"features={Xc.shape[1]} < {min_feats_for_lda})."
        )
        _fallback_keywords(texts, stop_union_norm, top_n=12)
        return

    # --- K-Kandidaten begrenzen (nach Datenlage)
    candidates_k = cand_k if cand_k is not None else DEFAULT_CANDIDATE_K
    if is_en:  # kleine EN-Korpora: realistisch nur 2/3 Topics
        candidates_k = [k for k in candidates_k if k in (2, 3)]
    max_topics_allowed = max(2, min(12, Xc.shape[0] - 1, Xc.shape[1] - 1))
    candidates_k = [k for k in candidates_k if 2 <= k <= max_topics_allowed] or [2]

    # --- LDA: bestes Modell via Kohärenz (zentralisiert)
    lda, doc_topic, best_k, best_seed, best_coh = fit_best_lda(
        Xc, candidates_k, SEEDS, max_iter=60, topn=12
    )
    if lda is None:
        print(f"[Abbruch] Kein LDA-Modell für {name}.")
        return

    # --- Top-Terme und Labels bilden
    feat_names = vec_counts.get_feature_names_out()
    topics_all = top_terms_per_topic(lda, feat_names, topn=12)
    labels = _build_topic_labels(lda, feat_names, stop_union_norm, topn=40)

    # Konsistente Statuszeile (vom HTML-Report geparst)
    print(
        f"[Info] {name}: K={best_k} Seed={best_seed} Coh={best_coh:.3f} "
        f"(features={Xc.shape[1]}, min_df={used_params['min_df']}, "
        f"max_df={used_params['max_df']:.2f}, title_boost={title_boost}, comments={include_comments})"
    )

    # --- Auswahl der anzuzeigenden Topics (Top-K nach Gesamtzuweisung)
    k_report = min(5, doc_topic.shape[1])
    top_idx, _ = pick_top_k_topics(doc_topic, k=k_report)

    # Label-Dedup: pro angezeigtem Topic alternative Labels generieren und Dubs vermeiden
    ordered_terms = [topics_all[t] for t in top_idx]
    primary_labels = [labels[t] for t in top_idx]
    ordered_labels = _dedup_topic_labels(primary_labels, ordered_terms)
    label_by_topic = {t: ordered_labels[i] for i, t in enumerate(top_idx)}

    # --- Output (Konsole + Beispiele in CSV)
    rows = []
    topic_assign = doc_topic.argmax(axis=1)

    for rank, t_id in enumerate(top_idx, start=1):
        terms_str = ", ".join([w for w, _ in topics_all[t_id]])
        label = label_by_topic.get(t_id, labels[t_id])

        # WICHTIG: Diese Zeile wird vom Parser erwartet:
        if label:
            print(f"Topic {rank} (ID {t_id}) [{label}]: {terms_str}")
        else:
            print(f"Topic {rank} (ID {t_id}): {terms_str}")

        # 3 Beispielposts pro Topic exportieren (falls vorhanden)
        mask = (topic_assign == t_id)
        scores = doc_topic[mask, t_id]
        if getattr(scores, "size", 0) == 0:
            continue

        which = scores.argsort()[::-1][:3]
        candidates = df.loc[mask].iloc[which]
        for idx_row in candidates.index:
            rows.append({
                "flair_block": name,
                "reported_topic_rank": rank,
                "topic_id": int(t_id),
                "label": label,
                "title": df.at[idx_row, "title"] if "title" in df.columns else "",
                "permalink": df.at[idx_row, "permalink"] if "permalink" in df.columns else ""
            })

    if rows:
        pd.DataFrame(rows).to_csv(
            DATA_DIR / f"samples_{name.lower().replace('/','_')}_per_topic.csv", index=False
        )


# ============================================================================
# main(): Laden, Statistik, Blockläufe
# ============================================================================

def main():
    """
    Hauptfunktion:
      - lädt/holt Rohdaten
      - schreibt Top-Flairs und Top-User (CSV + Konsole)
      - iteriert über FLAIR_RUNS und führt _run_for_flair_block aus
    """
    raw_path = DATA_DIR / "raw_r_stuttgart.csv"
    if raw_path.exists():
        print("[Info] Lade RAW aus CSV …")
        df_raw = pd.read_csv(raw_path)
    else:
        print("[Info] Lade RAW über Reddit-API …")
        df_raw = fetch_subreddit_posts(
            subreddit_name="Stuttgart",
            where="top",
            time_filter="year",
            limit=600,
            with_comments=True,
            max_comments_per_post=50
        )
        df_raw.to_csv(raw_path, index=False)

    # Übersichtstabellen
    flairs = extract_flairs(df_raw).head(20)
    active_users = (df_raw["author"]
                    .dropna()
                    .loc[lambda s: s.ne("[deleted]") & s.ne("None")]
                    .value_counts()
                    .head(20))
    flairs.to_csv(DATA_DIR / "top_flairs.csv")
    active_users.to_csv(DATA_DIR / "top_users.csv")

    print("\n=== Top 20 Flairs ===")
    print(flairs.to_string() if not flairs.empty else "(keine Flairs gefunden)")

    print("\n=== Top 20 aktivste User ===")
    print(active_users.to_string() if not active_users.empty else "(keine Nutzer ermittelt)")

    # Pro-Flair die Themenmodelle bauen
    base_stop_union = GERMAN_SW | ENGLISH_SW
    for run in FLAIR_RUNS:
        _run_for_flair_block(df_raw, run, base_stop_union)


# ============================================================================
# HTML-Report – Parser & Renderer (aus Konsolen-Output)
# ============================================================================

from pathlib import Path
import re as _re
import html as _html
import pandas as _pd

# Regexe für den Parser (kontraktgebunden an die Print-Ausgaben oben!)
_SEC_HDR = _re.compile(r"^# ==== (.+?) ====\s*$")
_RE_TRYV = _re.compile(r"^\[Info\]\s*Try vectorize:\s*n_docs=(\d+).*?->\s*features=(\d+)", _re.I)
_RE_TRYV_SIMPLE = _re.compile(r"^\[Info\]\s*Try vectorize:\s*n_docs=(\d+)", _re.I)
_RE_LANG_EN = _re.compile(r"^\[Info\]\s*Language filter 'en': kept (\d+)\/(\d+) docs\.", _re.I)
_RE_KLINE = _re.compile(r"K\s*=\s*(\d+).*?Coh\s*=\s*([-\d\.]+).*?features\s*=\s*(\d+)", _re.I)
_RE_TOPIC = _re.compile(r"^Topic\s+(\d+)\s+\(ID\s+(\d+)\)\s*(\[[^\]]+\])?:\s*(.+)$")


def _split_sections(captured_text: str):
    """
    Zerlegt den Roh-Log in Abschnitte (Titel -> Zeilen),
    anhand der "# ===="-Header.
    """
    sections = {}
    current = None
    for ln in captured_text.splitlines():
        m = _SEC_HDR.match(ln)
        if m:
            current = m.group(1).strip()
            sections[current] = []
        elif current:
            sections[current].append(ln)
    return sections


def _parse_block(lines):
    """
    Extrahiert aus den Zeilen eines Abschnitts:
      - Topicliste (rank, id, label, terms)
      - Kennzahlen (K, Kohärenz, Features)
      - Doks/EN-Kept (falls vorhanden)
    """
    topics = []
    k_info = {"K": None, "Coh": None, "features": None}
    docs = None
    feats_from_try = None
    kept_en = None

    for ln in lines:
        m = _RE_TRYV.match(ln)
        if m:
            docs = int(m.group(1))
            feats_from_try = int(m.group(2))
        else:
            m2 = _RE_TRYV_SIMPLE.match(ln)
            if m2 and docs is None:
                docs = int(m2.group(1))

        m3 = _RE_LANG_EN.match(ln)
        if m3:
            kept_en = (int(m3.group(1)), int(m3.group(2)))

        if "K=" in ln and "Coh" in ln:
            mk = _RE_KLINE.search(ln)
            if mk:
                k_info["K"] = mk.group(1)
                k_info["Coh"] = mk.group(2)
                k_info["features"] = mk.group(3)

        mt = _RE_TOPIC.match(ln)
        if mt:
            rank, tid, label_raw, terms = mt.groups()
            label = (label_raw or "").strip("[] ").replace(" / ", " – ").replace(" - ", " – ")
            topics.append({
                "rank": int(rank), "topic_id": int(tid),
                "label": label if label else "",
                "terms": terms
            })

    # Fallback: Features aus Try-Vectorize, falls K-Zeile fehlte
    if k_info["features"] is None and feats_from_try is not None:
        k_info["features"] = str(feats_from_try)
    return topics, k_info, docs, kept_en


def _tbl_topics(topics, k_info, docs, kept_en):
    """
    Baut eine HTML-Tabelle für Topics + schlanke Kopfleiste mit Stats.
    """
    if not topics:
        return "<div class='muted'>Keine Topics vorhanden.</div>"

    headbits = []
    if docs is not None:
        headbits.append(f"{docs} Dok.")
    if kept_en:
        headbits.append(f"EN kept {kept_en[0]}/{kept_en[1]}")
    if k_info.get("K"):
        headbits.append(f"K={_html.escape(k_info['K'])}")
    if k_info.get("Coh"):
        headbits.append(f"Coh={_html.escape(k_info['Coh'])}")
    if k_info.get("features"):
        headbits.append(f"Features={_html.escape(k_info['features'])}")

    head = " · ".join(headbits)
    rows = []
    rows.append("<table class='tbl'><thead><tr><th>#</th><th>Label</th><th>Top-Terme</th></tr></thead><tbody>")
    for t in topics:
        label = _html.escape(t["label"]) if t["label"] else "<span class='muted'>(kein Label)</span>"
        rows.append(
            f"<tr><td>{t['rank']}</td>"
            f"<td>{label}<br><span class='muted'>ID {t['topic_id']}</span></td>"
            f"<td>{_html.escape(t['terms'])}</td></tr>"
        )
    rows.append("</tbody></table>")
    return (f"<div class='muted'>{head}</div>" if head else "") + "\n" + "\n".join(rows)


def _pair_card(title_left, block_left, title_right, block_right):
    """
    Stellt zwei Topics-Tabellen nebeneinander (DE/EN-Pair) dar.
    """
    left_html = _tbl_topics(*block_left)
    right_html = _tbl_topics(*block_right)
    return f"""
    <div class="grid2">
      <div class="card"><h3>{_html.escape(title_left)}</h3>{left_html}</div>
      <div class="card"><h3>{_html.escape(title_right)}</h3>{right_html}</div>
    </div>
    """


def _build_topic_grids(sections):
    """
    Baut die vier DE/EN-Paare (Diskussion, News, Events, Frage/Advice).
    """
    def blk(name):
        lines = sections.get(name, [])
        return _parse_block(lines)

    grids = []
    grids.append(_pair_card(
        "Diskussion – DE", blk("Themen für Flair: Diskussion - DE"),
        "Diskussion – EN", blk("Themen für Flair: Diskussion - EN"),
    ))
    grids.append(_pair_card(
        "News – DE", blk("Themen für Flair: News - DE"),
        "News – EN", blk("Themen für Flair: News - EN"),
    ))
    grids.append(_pair_card(
        "Events – DE", blk("Themen für Flair: Events - DE"),
        "Events – EN", blk("Themen für Flair: Events - EN"),
    ))
    grids.append(_pair_card(
        "Frage/Advice – DE", blk("Themen für Flair: Frage/Advice - DE"),
        "Frage/Advice – EN", blk("Themen für Flair: Frage/Advice - EN"),
    ))
    return "\n".join(grids)


def _stats_card(sections):
    """
    Kleine Übersichts-Tabelle (pro Block: Doks, EN kept, K, Coh, Features).
    """
    rows = []
    rows.append("<table class='tbl'><thead><tr><th>Block</th><th>Dok.</th><th>EN kept</th><th>K</th><th>Coh</th><th>Features</th></tr></thead><tbody>")
    wanted = [
        ("Diskussion – DE", "Themen für Flair: Diskussion - DE"),
        ("Diskussion – EN", "Themen für Flair: Diskussion - EN"),
        ("News – DE", "Themen für Flair: News - DE"),
        ("News – EN", "Themen für Flair: News - EN"),
        ("Events – DE", "Themen für Flair: Events - DE"),
        ("Events – EN", "Themen für Flair: Events - EN"),
        ("Frage/Advice – DE", "Themen für Flair: Frage/Advice - DE"),
        ("Frage/Advice – EN", "Themen für Flair: Frage/Advice - EN"),
    ]
    for label, key in wanted:
        topics, k_info, docs, kept_en = _parse_block(sections.get(key, []))
        en_txt = f"{kept_en[0]}/{kept_en[1]}" if kept_en else "–"
        rows.append(
            f"<tr><td>{_html.escape(label)}</td>"
            f"<td>{docs if docs is not None else '–'}</td>"
            f"<td>{en_txt}</td>"
            f"<td>{_html.escape(k_info.get('K') or '–')}</td>"
            f"<td>{_html.escape(k_info.get('Coh') or '–')}</td>"
            f"<td>{_html.escape(k_info.get('features') or '–')}</td></tr>"
        )
    rows.append("</tbody></table>")
    return "<div class='card'><h2>Statistik</h2>" + "\n".join(rows) + "</div>"


def _try_read_table_csv(path: Path, title: str):
    """
    Liest eine CSV (falls vorhanden) und rendert sie als HTML-Tabelle im Report.
    """
    try:
        if path.exists():
            df = _pd.read_csv(path)
            return f"<div class='card'><h2>{_html.escape(title)}</h2>{df.to_html(index=False, border=0, classes='tbl')}</div>"
    except Exception:
        pass
    return f"<div class='card'><h2>{_html.escape(title)}</h2><div class='muted'>keine Tabelle gefunden</div></div>"


def _write_html_report_from_text(captured_text: str, outpath: Path):
    """
    Schreibt den finalen HTML-Report basierend auf dem *Konsolen-Output*.

    Aufbau:
      - Top Flairs & Top User (CSV -> Tabellen)
      - Statistik-Card (pro Block)
      - Topics-Grids (DE/EN nebeneinander für alle Flairs)
      - Ganz unten: Roh-Log (als <pre>)

    Hinweis:
      - Das Styling ist bewusst schlank/generisch gehalten.
    """
    sections = _split_sections(captured_text)

    flairs_html = _try_read_table_csv(Path(DATA_DIR) / "top_flairs.csv", "Top Flairs")
    users_html  = _try_read_table_csv(Path(DATA_DIR) / "top_users.csv",  "Top User")

    topic_grids = _build_topic_grids(sections)
    stats_html  = _stats_card(sections)

    # kompletter Roh-Log wie auf der Konsole
    raw_log = _html.escape(captured_text)

    html_doc = f"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>r/Stuttgart – Themenreport</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --bg:#0b1020; --card:#121a2d; --text:#e8eefc; --muted:#93a1c8; --accent:#5aa7ff;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, "Helvetica Neue", Arial;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); margin: 0; }}
  header {{ padding: 24px; border-bottom: 1px solid #1f2942; background: #0e1530; }}
  header h1 {{ margin: 0; font-size: 22px; letter-spacing: .2px; }}
  main {{ padding: 24px; }}
  .card {{ background: var(--card); border: 1px solid #1f2942; border-radius: 14px; padding: 18px; margin-bottom: 18px; }}
  h2, h3 {{ margin: 0 0 12px 0; font-weight: 600; }}
  .muted {{ color: var(--muted); }}
  .grid2 {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }}
  .grid12 {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
  table.tbl {{ width: 100%; border-collapse: collapse; margin-top: 6px; }}
  table.tbl thead th {{ text-align: left; color: var(--muted); font-weight: 600; border-bottom: 1px solid #223056; padding: 8px; }}
  table.tbl tbody td {{ border-bottom: 1px solid #1a2544; padding: 8px; vertical-align: top; }}
  pre {{ background: #0b1329; border: 1px solid #1f2942; border-radius: 10px; padding: 12px; overflow:auto; white-space: pre-wrap; font-family: var(--mono); }}
</style>
</head>
<body>
<header>
  <h1>r/Stuttgart – Themenreport</h1>
  <div class="muted">Automatisch generiert</div>
</header>
<main>
  <div class="grid12">
    {flairs_html}
    {users_html}
  </div>

  {stats_html}

  <div class="card">
    <h2>Topics nach Flair (DE/EN nebeneinander)</h2>
    {topic_grids}
  </div>

  <div class="card">
    <h2>Roh-Log (Konsolen-Output)</h2>
    <pre>{raw_log}</pre>
  </div>
</main>
</body>
</html>"""
    outpath.write_text(html_doc, encoding="utf-8")
    print(f"[Info] HTML-Report gespeichert: {outpath}")


# ============================================================================
# Script-Entry (mit stdout-Capture für den Report)
# ============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    import io, contextlib
    buf = io.StringIO()

    # Konsole "normal" befüllen, aber parallel den Text für den Report sammeln.
    with contextlib.redirect_stdout(buf):
        main()

    captured = buf.getvalue()
    print(captured, end="")  # Konsole wie bisher

    REPORT_PATH = (DATA_DIR / "report.html")
    _write_html_report_from_text(captured, REPORT_PATH)