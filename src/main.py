from __future__ import annotations
import sys
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
    vectorize_counts, vectorize_tfidf,  # TF-IDF nur für evtl. spätere Fallbacks
    lda_from_matrix, top_terms_per_topic, pick_top_k_topics, umass_coherence
)
from .utils import DATA_DIR

# =========================
# Konfiguration
# =========================
RUN_LSA_COMPARE = False  # aus, Fokus: LDA-Stabilität

# Pro-Flair-Modelle (mit flairspezifischen Feineinstellungen)
FLAIR_RUNS = [
    # Diskussion: mehr Substanz -> Kommentare an, aggressivere Auto-Stopwörter, Extra-Stopwörter
    {"name": "Diskussion - DE",
     "flairs": ["Diskussion"],
     "lang": "de",
     "filter_english": True,
     "include_comments": True,    # <— neu: Kommentare an
     "min_tokens": 20,
     "title_boost": 1,
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"mal","mehr","einfach","echt","leider","besser","immer","schon",
                         "klar","weiß","gesagt","gemacht","findet","find","sagen"},
     "feat_min_hard": 80,         # ab hier nicht abbrechen
     "candidate_k": [6, 8, 10]},

    {"name": "Diskussion - EN",
     "flairs": ["Diskussion"],
     "lang": "en",
     "filter_english": False,
     "include_comments": True,    # <— neu: Kommentare an
     "min_tokens": 10,
     "title_boost": 2,
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"get","know","like","make","one","people","really","thing","things","would","could","also","still"},
     "feat_min_hard": 40,         # ab hier nicht abbrechen
     "candidate_k": [2,3]},

    # News: sehr kleiner Korpus -> niedrige Feature-Untergrenze, kleineres K, starker Titel-Boost
    {"name": "News - DE",
     "flairs": ["News"],
     "lang": "de",
     "filter_english": True,
     "include_comments": False,
     "min_tokens": 5,
     "title_boost": 4,            # <— Headlines stärker gewichten
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"mal","mehr","letzte","letzten","schon"},
     "feat_min_hard": 40,         # <— nicht mehr abbrechen bei ~33 Features
     "candidate_k": [3, 4],   # <— kleines K
     "pairs": [(3, 0.60), (3, 1.00), (2, 0.60)]},

    {"name": "News - EN",
     "flairs": ["News"],
     "lang": "en",
     "filter_english": False,
     "include_comments": False,
     "min_tokens": 8,
     "title_boost": 3,
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"get","know","like","make","one","people","really","thing","things","would","could","also","still"},
     "feat_min_hard": 30,         # ab hier nicht abbrechen
     "candidate_k": [2,3]},

    # Events: unverändert, aber mit parametrischem Slot
    {"name": "Events - DE",
     "flairs": ["Events"],
     "lang": "de",
     "filter_english": True,
     "include_comments": True,
     "min_tokens": 15,
     "title_boost": 2,
     "df_ratio_auto_sw": 0.60,
     "extra_stopwords": {"post", "infos", "weitere", "weitere informationen"},
     "feat_min_hard": 80,         # Standard
     "candidate_k": [6, 8, 10]},        # None => Default-Kandidaten
    
    {"name": "Events - EN",
     "flairs": ["Events"],
     "lang": "en",
     "filter_english": False,
     "include_comments": True,
     "min_tokens": 10,
     "title_boost": 2,
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"get","know","like","make","one","people","really","thing","things","would","could","also","still"},
     "feat_min_hard": 30,         # ab hier nicht abbrechen
     "candidate_k": [2,3]},

    # Frage/Advice: Kommentare an (mehr Inhalt), sonst wie gehabt
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
     "candidate_k": [8, 10]},

    {"name": "Frage/Advice - EN",
     "flairs": ["Frage / Advice", "Looking for..."],
     "lang": "en",
     "filter_english": False,
     "include_comments": True,
     "min_tokens": 10,
     "title_boost": 1,
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"get","know","like","make","one","people","really","thing","things","would","could","also","still"},
     "feat_min_hard": 40,         # ab hier nicht abbrechen
     "candidate_k": [2,3]}
]

# Default-Kandidaten für Anzahl Topics und Seeds
DEFAULT_CANDIDATE_K = [6, 8, 10, 12]
SEEDS = (13, 21, 42, 77)

# Zielbereich für Anzahl Features (soft window)
FEAT_TARGET_LOW  = 300
FEAT_TARGET_HIGH = 2000
FEAT_MIN_HARD    = 100  # globale harte Untergrenze (kann pro Flair übersteuert werden)

# =========================
# Hilfsfunktionen
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer

def _fallback_keywords(texts: List[str], stop_words_norm, top_n: int = 12):
    """Einfache Fallback-Keyword-Liste, wenn LDA wegen kleinem Korpus nicht sinnvoll ist."""
    try:
        vec = TfidfVectorizer(ngram_range=(2,2), min_df=1, max_df=0.95, stop_words=stop_words_norm)
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
    if not tokens:
        return "unk"
    de_hits = sum((t in GERMAN_SW) for t in tokens)
    en_hits = sum((t in ENGLISH_SW) for t in tokens)
    # klare Mehrheiten; kleine Mindestschwelle gegen Zufallstreffer
    if en_hits >= max(2, de_hits + 1):
        return "en"
    if de_hits >= max(2, en_hits + 1):
        return "de"
    # unklar
    return "unk"


def _normalize_stopwords(stopset: set[str]) -> List[str]:
    norm = set()
    for w in stopset:
        toks = tokenize(clean_text(w))
        norm.update(toks)
    return sorted({t for t in norm if len(t) >= 2})

def _build_texts_with_title_boost(df: pd.DataFrame, title_boost: int) -> List[str]:
    base = df["clean_for_vect"].fillna("")
    if title_boost <= 0 or "title" not in df.columns:
        texts = base
    else:
        title_tokens = (
            df["title"].fillna("")
            .map(clean_text).map(tokenize).map(lambda ts: " ".join(ts))
        )
        title_boost_str = title_tokens.map(lambda s: (s + " ") * title_boost)
        texts = (base + " " + title_boost_str).str.strip()
    # NEU: Unterstriche aus evtl. Collocations entfernen
    texts = texts.str.replace("_", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
    return texts.tolist()


def _auto_stopwords_from_tokens(tokens_list: List[List[str]],
                                df_ratio: float = 0.70,
                                top_n_cap: int = 60,
                                whitelist: Optional[set] = None) -> set:
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
        if t in whitelist:     continue
        if len(t) < 3:         continue
        if any(ch.isdigit() for ch in t): continue
        auto_sw.add(t)
    return auto_sw

def _choose_df_params(n_docs: int,
                      base_min_df_abs: int, base_min_df_frac: float,
                      base_max_df_frac: float) -> Tuple[int, float]:
    n_docs = max(int(n_docs), 1)
    min_df_val = max(1, max(base_min_df_abs, int(math.ceil(base_min_df_frac * n_docs))))
    min_df_val = min(min_df_val, max(1, n_docs - 1))
    max_df_val = min(1.0, max(base_max_df_frac, 2.0 / n_docs))
    max_docs_allowed = int((max_df_val * n_docs) // 1)
    if max_docs_allowed <= 1:
        max_df_val = 1.0
        max_docs_allowed = n_docs
    if min_df_val > max_docs_allowed - 1:
        max_df_val = 1.0
        max_docs_allowed = n_docs
        if min_df_val > max_docs_allowed - 1:
            min_df_val = max(1, n_docs - 1)
    return int(min_df_val), float(max_df_val)

# -------------------------
# Helper: robuste Vektorisierung
# -------------------------
def _safe_vectorize(texts, min_df, max_df, stop_words_norm, max_features):
    """Robuster Count-Vect mit Fehlerabfang."""
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
        if "After pruning, no terms remain" in msg:
            return None, None
        return None, None


def _vectorize_to_window(
    texts, stop_words_norm, feat_min_hard: int = 80,
    max_features_cap: int = 12000,
    pairs: Optional[List[Tuple[int, float]]] = None
):
    """
    Probiert mehrere (min_df, max_df)-Paare und nimmt den ersten 'gut genug' Treffer,
    sonst den Versuch mit den meisten Features.
    """
    n_docs = len(texts)

    # Default-Fenster (für Diskussion/Frage-DE etc.)
    default_pairs = [(15, 1.00), (10, 0.30), (5, 0.40), (2, 0.60)]
    # Quick-Win: für News/Events liefert dieses Set meist schneller brauchbare Features
    news_events_pairs = [(5, 0.60), (10, 0.30), (15, 1.00), (2, 0.60)]

    # Wenn der Aufrufer kein Set übergibt, nehmen wir default
    pairs = pairs or default_pairs

    best_by_feats = None
    best_feats = -1

    for (min_df, max_df) in pairs:
        min_df_eff = min(min_df, max(1, int(0.15 * n_docs)))  # min_df an n_docs koppeln
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

def _autolabel_topic(feature_terms: List[str]) -> str:
    domain_hits = ["bahn","vvs","haltestelle","linie","ticket","parkhaus","bahnhof",
                   "stuttgart_21","s21","schlossplatz","fernsehturm","innenstadt","demo","afd","wahl","brandmauer","polizei","öpnv"]
    for t in feature_terms:
        if " " in t:
            return t
    for t in feature_terms:
        if t in domain_hits:
            return t
    return " / ".join(feature_terms[:2])

def _lang_guess_series(df: pd.DataFrame) -> pd.Series:
    """Schätzt Sprache aus Titel + Rohtext (ohne Stopwort-Removal)."""
    # 1) Titel
    parts = []
    if "title" in df.columns:
        parts.append(df["title"].fillna(""))

    # 2) Body/Selftext – nimm die erste existierende Spalte
    for c in ("selftext", "body", "text", "content"):
        if c in df.columns:
            parts.append(df[c].fillna(""))
            break

    # 3) Fallback: wenn nichts gefunden, nimm clean_for_vect (besser als nichts)
    if not parts:
        if "clean_for_vect" in df.columns:
            parts.append(df["clean_for_vect"].fillna(""))
        else:
            return pd.Series(["unk"] * len(df), index=df.index)

    s = parts[0]
    for p in parts[1:]:
        s = s.str.cat(p, sep=" ", na_rep="")

    # Jetzt erst clean + tokenize (ohne Stopwort-Entfernung!)
    lang_tokens = s.map(clean_text).map(tokenize)
    return lang_tokens.map(_infer_lang_from_tokens)

# -------------------------
# Topic-Labeling (ohne Extra-Libs)
# -------------------------

# -------------------------
# Labeling-Konfiguration
# -------------------------

# Füllwörter, die für Labels unbrauchbar sind (nur fürs Labeling!):
LABEL_FILLER = {
    # DE
    "mal","mehr","schon","einfach","leider","echt","besser","immer","klar","gut","gibt",
    "find","findet","frage","wohl","gerade","vielleicht","irgendwie","ziemlich","bisschen",
    "eher","sogar","glaube","finde","tatsächlich","natürlich","halt","eigentlich","wirklich",
    "besten","deutschen","absolut","alte","alten","neue","neuen","groß","klein",
    "teuer","nie","rum","ganzen","davon","drauf","bzw","wohl", "nie","rum","davon",
    "drauf","bzw","wohl",
    # EN
    "like","get","know","make","one","people","really","thing","things","would","could",
    "also","still","maybe","just","even","well","good","bad","new","old","time"
}

# Domänen-Whitelist (triggert Bonus-Punkte in Labels)
DOMAIN_WHITELIST = {
    "stuttgart","stuttgarter","0711","kessel","neckar",
    "hbf","hauptbahnhof","olgaeck","vaihingen","bad","cannstatt","bad_cannstatt","cannstatter",
    "schlossplatz","wilhelma","feuersee","boeblingen","ludwigsburg","esslingen",
    "vvs","ssb","u-bahn","s-bahn","stadtbahn","fernsehturm","uni","messe","killesberg",
    "polizei","demo","wahl","bundestagswahl","afd","cdu","spd","grüne","fdp",
    "olgaeck","marienplatz","stadtmitte","münchener","börsenplatz","pragfriedhof","pragsattel",
    "heslach","deggerloch","berg","ostfildern","leonberg"
}

# Schöne Schreibweisen (Anzeigeform)
DISPLAY_CASE = {
    "stuttgart":"Stuttgart","stuttgarter":"Stuttgarter","neckar":"Neckar",
    "hbf":"Hbf","hauptbahnhof":"Hauptbahnhof","olgaeck":"Olgaeck",
    "vaihingen":"Vaihingen","bad_cannstatt":"Bad Cannstatt","cannstatt":"Cannstatt","cannstatter":"Cannstatter",
    "schlossplatz":"Schlossplatz","wilhelma":"Wilhelma","feuersee":"Feuersee","ludwigsburg":"Ludwigsburg","esslingen":"Esslingen",
    "vvs":"VVS","ssb":"SSB","u-bahn":"U-Bahn","s-bahn":"S-Bahn","stadtbahn":"Stadtbahn","fernsehturm":"Fernsehturm",
    "polizei":"Polizei","demo":"Demo","wahl":"Wahl","bundestagswahl":"Bundestagswahl","afd":"AfD","cdu":"CDU","spd":"SPD","grüne":"Grüne","fdp":"FDP"
}

def _split_ngram(term: str):
    term = term.strip()
    if " " in term:
        return term.split()
    if "_" in term:
        return term.split("_")
    return [term]

def _is_valid_token(tok: str, stop_words, min_len: int = 3) -> bool:
    if not tok or len(tok) < min_len:
        return False
    if tok.isdigit():
        return False
    if tok in stop_words or tok in LABEL_FILLER:
        return False
    return True

def _pretty_token(tok: str) -> str:
    t = tok.replace("_", " ")
    return DISPLAY_CASE.get(tok, DISPLAY_CASE.get(t, t))

def _score_bigram_for_topic(bigram: str, t_id: int, vocab_index: dict, comps, stop_words) -> float:
    toks = _split_ngram(bigram)
    if len(toks) != 2:
        return -1e9
    if not all(_is_valid_token(t, stop_words) for t in toks):
        return -1e9

    # Basis-Score aus Topic-Gewichten
    w_bigram = comps[t_id, vocab_index.get(bigram, -1)] if bigram in vocab_index else 0.0
    w_t1 = comps[t_id, vocab_index.get(toks[0], -1)] if toks[0] in vocab_index else 0.0
    w_t2 = comps[t_id, vocab_index.get(toks[1], -1)] if toks[1] in vocab_index else 0.0
    score = w_bigram + 0.5 * (w_t1 + w_t2)

    # Domänen-Bonus
    if toks[0] in DOMAIN_WHITELIST: score += 2.0
    if toks[1] in DOMAIN_WHITELIST: score += 2.0

    # leichte Strafe für „weiche“ Wörter
    if toks[0] in LABEL_FILLER: score -= 1.0
    if toks[1] in LABEL_FILLER: score -= 1.0
    return score

def _build_topic_labels(lda_model, feat_names, stop_words, topn: int = 40, join_char: str = " - "):
    """
    Wählt pro Topic das beste Bigram (mit Domänen-Bonus), sonst zwei Unigramme.
    Gibt eine Liste von display-fertigen Strings zurück (z. B. 'Hbf - Unfall').
    """
    comps = lda_model.components_
    n_topics = comps.shape[0]
    labels = [""] * n_topics

    # Index für schnelles Lookup
    vocab_index = {feat: idx for idx, feat in enumerate(feat_names)}

    for t_id in range(n_topics):
        order = comps[t_id].argsort()[::-1]
        top_idx = order[:topn]
        top_terms = [feat_names[j] for j in top_idx]

        # Kandidaten-Bigramme (nur echte 2-Tokens)
        cand_bi = [w for w in top_terms if len(_split_ngram(w)) == 2]

        # Scoring
        scored = [(w, _score_bigram_for_topic(w, t_id, vocab_index, comps, stop_words)) for w in cand_bi]
        scored.sort(key=lambda x: x[1], reverse=True)

        label = ""
        if scored and scored[0][1] > 0:
            t1, t2 = _split_ngram(scored[0][0])
            label = _pretty_token(t1) + join_char + _pretty_token(t2)
        else:
            # Fallback: zwei beste Unigramme (mit Domänen-Priorität)
            unis = [
                w for w in top_terms
                if len(_split_ngram(w)) == 1 and _is_valid_token(w, stop_words)
            ]
            if not unis:
                labels[t_id] = top_terms[0] if top_terms else f"Topic {t_id}"
                continue

            # Domänen-Tokens nach vorne ziehen
            unis.sort(key=lambda w: (w not in DOMAIN_WHITELIST, ), reverse=False)
            if len(unis) >= 2:
                label = _pretty_token(unis[0]) + join_char + _pretty_token(unis[1])
            else:
                label = _pretty_token(unis[0])

        labels[t_id] = label
    return labels


def _run_for_flair_block(df_raw: pd.DataFrame, run_cfg: Dict, base_stop_union: set) -> None:
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

    # Subset
    if "flair_text" in df_raw.columns:
        sub_raw = df_raw[df_raw["flair_text"].isin(flairs)].reset_index(drop=True)
    else:
        sub_raw = df_raw.copy()
    if sub_raw.empty:
        print(f"[Skip] Keine Dokumente für {name}.")
        return

    # Vorverarbeitung
    df = add_clean_columns(sub_raw, use_stemming=False, filter_english=fe, include_comments=include_comments)
    if "tokens" in df.columns:
        df = df[df["tokens"].map(len) >= min_tokens].reset_index(drop=True)
    if df.empty:
        print(f"[Skip] Nach Preprocess/Tokenfilter keine Dokumente für {name}.")
        return

    # Sprachfilter (nach Preprocess, aber auf Basis von Titel/Rohtext heuristisch geschätzt)
    if lang_target in ("de", "en"):
        if lang_target == "de" and fe is True:
            print("[Info] Language filter 'de': skipped (filter_english=True hat EN bereits entfernt).")
        else:
            df["lang_guess"] = _lang_guess_series(df)
            before = len(df)
            df = df[df["lang_guess"] == lang_target].reset_index(drop=True)
            print(f"[Info] Language filter '{lang_target}': kept {len(df)}/{before} docs.")
            if len(df) < 5:
                print(f"[Skip] Zu wenig klare {lang_target.upper()}-Dokumente – LDA übersprungen.")
                return

    # Auto-Stopwörter
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

    stop_union = (GERMAN_SW | ENGLISH_SW | TECHNICAL_SW | auto_sw)
    stop_union_norm = _normalize_stopwords(stop_union)

    # Texte + Titel-Boost
    texts = _build_texts_with_title_boost(df, title_boost)

    # Param-Fenster
    is_en = (run_cfg.get("lang") == "en")
    is_news_or_events = any(nm in run_cfg["name"].lower() for nm in ["news", "events"])
    pairs = run_cfg.get("pairs")
    if pairs is None and is_news_or_events:
        pairs = [(5, 0.60), (10, 0.30), (15, 1.00), (2, 0.60)]

    # Vektorisierung
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

    # Fallback bei sehr kleinen Korpora
    min_feats_for_lda = run_cfg.get("feat_min_hard", 80)          # z. B. News-DE: 40
    min_docs_for_lda  = run_cfg.get("docs_min", 8)                # Standard bleibt 8

    if Xc.shape[0] < min_docs_for_lda or Xc.shape[1] < min_feats_for_lda:
        print(
            f"[Info] {name}: kleiner Korpus – Fallback auf TF-IDF-Keywords "
            f"(docs={Xc.shape[0]} < {min_docs_for_lda} oder "
            f"features={Xc.shape[1]} < {min_feats_for_lda})."
        )
        _fallback_keywords(texts, stop_union_norm, top_n=12)
        return

    # Kandidaten-K einschränken
    candidates_k = cand_k if cand_k is not None else DEFAULT_CANDIDATE_K
    if is_en:
        candidates_k = [k for k in candidates_k if k in (2,3)]
    max_topics_allowed = max(2, min(12, Xc.shape[0]-1, Xc.shape[1]-1))
    candidates_k = [k for k in candidates_k if 2 <= k <= max_topics_allowed] or [2]

    # LDA + Kohärenz
    best = {"coh": -1e9, "k": None, "model": None, "doc_topic": None, "seed": None}
    for k in candidates_k:
        for seed in SEEDS:
            lda_k, dt_k = lda_from_matrix(Xc, n_topics=k, max_iter=60, random_state=seed)
            coh_k = umass_coherence(lda_k, Xc, topn=12)
            if coh_k > best["coh"]:
                best.update({"coh": coh_k, "k": k, "model": lda_k, "doc_topic": dt_k, "seed": seed})

    if best["model"] is None:
        print(f"[Abbruch] Kein LDA-Modell für {name}.")
        return

    lda = best["model"]
    doc_topic = best["doc_topic"]
    feat_names = vec_counts.get_feature_names_out()
    topics_all = top_terms_per_topic(lda, feat_names, topn=12)  # <- konsistente Variable
    labels = _build_topic_labels(lda, feat_names, stop_union_norm, topn=40)

    print(f"[Info] {name}: K={best['k']} Seed={best['seed']} Coh={best['coh']:.3f} "
          f"(features={Xc.shape[1]}, min_df={used_params['min_df']}, "
          f"max_df={used_params['max_df']:.2f}, title_boost={title_boost}, comments={include_comments})")

    # Ausgabe
    k_report = min(5, doc_topic.shape[1])
    top_idx, _ = pick_top_k_topics(doc_topic, k=k_report)

    rows = []
    topic_assign = doc_topic.argmax(axis=1)
    for rank, t_id in enumerate(top_idx, start=1):
        terms_str = ", ".join([w for w, _ in topics_all[t_id]])
        label = labels[t_id] if t_id < len(labels) else ""
        if label:
            print(f"Topic {rank} (ID {t_id}) [{label}]: {terms_str}")
        else:
            print(f"Topic {rank} (ID {t_id}): {terms_str}")

        # Beispiel-Posts
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

def main():
# 1) Daten abrufen (oder vorhandene RAW verwenden)
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


    # 1) Flairs & aktivste User (auf kompletter RAW-Basis)
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

    # 2) Pro Flair eigene Themenmodelle
    base_stop_union = GERMAN_SW | ENGLISH_SW
    for run in FLAIR_RUNS:
        _run_for_flair_block(df_raw, run, base_stop_union)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
