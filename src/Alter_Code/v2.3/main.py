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
     "min_tokens": 12,
     "title_boost": 1,
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"like","get","people","really","know","make","thing","things","one"},
     "feat_min_hard": 80,         # ab hier nicht abbrechen
     "candidate_k": [6, 8, 10]},

    # News: sehr kleiner Korpus -> niedrige Feature-Untergrenze, kleineres K, starker Titel-Boost
    {"name": "News - DE",
     "flairs": ["News"],
     "lang": "de",
     "filter_english": True,
     "include_comments": False,
     "min_tokens": 5,
     "title_boost": 3,            # <— Headlines stärker gewichten
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"mal","mehr","letzte","letzten","schon"},
     "feat_min_hard": 20,         # <— nicht mehr abbrechen bei ~33 Features
     "candidate_k": [3, 4]},   # <— kleines K

    {"name": "News - EN",
     "flairs": ["News"],
     "lang": "en",
     "filter_english": False,
     "include_comments": False,
     "min_tokens": 2,
     "title_boost": 3,            # <— Headlines stärker gewichten
     "df_ratio_auto_sw": 0.50,
     "extra_stopwords": {"like","get","people","really","know","make","thing","things","one"},
     "feat_min_hard": 20,         # <— nicht mehr abbrechen bei ~33 Features
     "candidate_k": [3, 4]},   # <— kleines K

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
     "candidate_k": None},        # None => Default-Kandidaten
    
    {"name": "Events - EN",
     "flairs": ["Events"],
     "lang": "en",
     "filter_english": False,
     "include_comments": True,
     "min_tokens": 10,
     "title_boost": 2,
     "df_ratio_auto_sw": 0.60,
     "extra_stopwords": {"like","get","people","really","know","make","thing","things","one"},
     "feat_min_hard": 80,         # Standard
     "candidate_k": None},        # None => Default-Kandidaten

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
     "candidate_k": None},

    {"name": "Frage/Advice - EN",
     "flairs": ["Frage / Advice", "Looking for..."],
     "lang": "en",
     "filter_english": False,
     "include_comments": True,
     "min_tokens": 10,
     "title_boost": 1,
     "df_ratio_auto_sw": 0.60,
     "extra_stopwords": {"like","get","people","really","know","make","thing","things","one"},
     "feat_min_hard": 80,
     "candidate_k": None},
]

# Default-Kandidaten für Anzahl Topics und Seeds
DEFAULT_CANDIDATE_K = [6, 8, 10, 12]
SEEDS = (13, 21, 42, 77)

# Zielbereich für Anzahl Features (soft window)
FEAT_TARGET_LOW  = 300
FEAT_TARGET_HIGH = 2000
FEAT_MIN_HARD    = 100  # globale harte Untergrenze (kann pro Flair übersteuert werden)

# Domain-Whitelist: nie automatisch zu Stopwörtern
DOMAIN_WHITELIST = {
    "bahn","vvs","sbahn","s-bahn","haltestelle","linie","ticket","parkhaus",
    "parken","bahnhof","stuttgart_21","s21","schlossplatz","fernsehturm",
    "killesberg","neckar","innenstadt","u-bahn","u","s1","s2","s3","demo","afd","wahl",
    # --- Stadtteile/Orte ---
    "cannstatt","vaihingen","feuerbach","zuffenhausen","degerloch","sillenbuch",
    "möhringen","weilimdorf","untertürkheim","hedelfingen","stammheim","botnang",
    "münster","mühlhausen","ost","west","mitte","süd","nord","olgaeck","wilhelma"
}

# =========================
# Hilfsfunktionen
# =========================
def _infer_lang_from_tokens(tokens: List[str]) -> str:
    """Mehr Treffer in GERMAN_SW oder ENGLISH_SW entscheidet.
    Nur wenn beide 0 -> 'unk'."""
    if not tokens:
        return "unk"
    de_hits = sum((t in GERMAN_SW) for t in tokens)
    en_hits = sum((t in ENGLISH_SW) for t in tokens)
    if de_hits == 0 and en_hits == 0:
        return "unk"
    return "de" if de_hits >= en_hits else "en"

    # leichte Schwelle gegen Zufallstreffer
    if de_hits == 0 and en_hits == 0:
        return "unk"
    if de_hits >= en_hits * 1.2:
        return "de"
    if en_hits >= de_hits * 1.2:
        return "en"
    # unklare Fälle: als 'unk' behandeln (wir filtern später hart)
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

def _safe_vectorize(texts: List[str], min_df_val: int, max_df_val: float, stop_words_norm: List[str]):
    # Primärversuch
    from .topics import CountVectorizer  # nur für Typprüfung bei Fehlern
    try:
        return vectorize_counts(
            texts, ngram=(1, 2),
            min_df=min_df_val, max_df=max_df_val,
            max_features=12000, stop_words=stop_words_norm
        )
    except ValueError:
        # Entschärfte Fallbacks
        fallback_params = [
            (min_df_val, 1.0, (1, 2)),
            (max(1, min_df_val // 2), 1.0, (1, 2)),
            (1, 1.0, (1, 2)),
            (1, 1.0, (1, 1)),
        ]
        for mdf, mxf, ngr in fallback_params:
            try:
                from .topics import vectorize_counts as _vc
                return _vc(texts, ngram=ngr, min_df=mdf, max_df=mxf,
                           max_features=12000, stop_words=stop_words_norm)
            except ValueError:
                continue
        raise ValueError("Vectorization failed after multiple fallbacks (no terms remain).")

def _vectorize_to_window(texts: List[str], stop_words_norm: List[str],
                         feat_min_hard: int = FEAT_MIN_HARD) -> Tuple:
    n_docs = len(texts)
    attempts = [
        (15, 0.018, 0.20),
        (10, 0.010, 0.30),
        (5,  0.005, 0.40),
        (2,  0.000, 0.60),
    ]
    best = {"delta": 10**9, "vec": None, "X": None, "params": None, "n_feat": 0}
    target_mid = (FEAT_TARGET_LOW + FEAT_TARGET_HIGH) / 2.0

    for (abs_min, frac_min, frac_max) in attempts:
        min_df_val, max_df_val = _choose_df_params(n_docs, abs_min, frac_min, frac_max)
        try:
            vec, X = _safe_vectorize(texts, min_df_val, max_df_val, stop_words_norm)
        except ValueError as e:
            print(f"[Warn] Vectorize attempt failed ({min_df_val},{max_df_val:.2f}): {e}")
            continue

        n_feat = X.shape[1]
        print(f"[Info] Try vectorize: n_docs={n_docs}, min_df={min_df_val}, max_df={max_df_val:.2f} -> features={n_feat}")

        if FEAT_TARGET_LOW <= n_feat <= FEAT_TARGET_HIGH:
            return vec, X, {"min_df": min_df_val, "max_df": max_df_val}

        delta = abs(n_feat - target_mid)
        if delta < best["delta"]:
            best.update({"delta": delta, "vec": vec, "X": X,
                         "params": {"min_df": min_df_val, "max_df": max_df_val},
                         "n_feat": n_feat})

    if best["vec"] is not None and best["n_feat"] >= feat_min_hard:
        print(f"[Info] Using best-effort features={best['n_feat']} "
              f"(min_df={best['params']['min_df']}, max_df={best['params']['max_df']:.2f})")
        return best["vec"], best["X"], best["params"]

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

    # *** HIER: Sprachfilter NACH dem Preprocessing anwenden ***
    if lang_target in ("de", "en"):
        # DE-Lauf + filter_english=True → EN wurde bereits entfernt
        if lang_target == "de" and fe is True:
            print("[Info] Language filter 'de': skipped (filter_english=True hat EN bereits entfernt).")
        else:
            # EN-Lauf (oder DE ohne vorgelagerten EN-Filter) → jetzt hart filtern auf Basis von ROHTEXT
            df["lang_guess"] = _lang_guess_series(df)
            before = len(df)
            df = df[df["lang_guess"] == lang_target].reset_index(drop=True)
            print(f"[Info] Language filter '{lang_target}': kept {len(df)}/{before} docs.")
            if df.empty:
                print(f"[Skip] Keine Dokumente nach Sprachfilter für {name}.")
                return

    # Auto-Stopwörter + flairspezifische Extras
    auto_sw = _auto_stopwords_from_tokens(df["tokens"].tolist(), df_ratio=df_ratio_auto_sw,
                                          top_n_cap=60, whitelist=DOMAIN_WHITELIST)
    if extra_stopwords:
        auto_sw |= set(extra_stopwords)

    if auto_sw:
        show = sorted(list(auto_sw))[:12]
        print(f"[Info] Auto-Stopwörter ({len(auto_sw)}): {show}{' ...' if len(auto_sw)>12 else ''}")

    stop_union = (GERMAN_SW | ENGLISH_SW | TECHNICAL_SW | auto_sw)
    stop_union_norm = _normalize_stopwords(stop_union)

    # Texte + Titel-Boost
    texts = _build_texts_with_title_boost(df, title_boost)

    # Vektorisierung (robust, mit flairspez. Feature-Untergrenze)
    vec_counts, Xc, used_params = _vectorize_to_window(texts, stop_union_norm, feat_min_hard=feat_min_hard)
    if vec_counts is None or Xc is None:
        print(f"[Abbruch] Zu wenig Features in {name}.")
        return

    # Kandidaten-K dynamisch begrenzen (kleine Korpora!)
    n_docs = Xc.shape[0]
    max_topics_allowed = max(2, min(12, n_docs - 1, Xc.shape[1] - 1))
    candidates_k = cand_k if cand_k is not None else DEFAULT_CANDIDATE_K
    candidates_k = [k for k in candidates_k if 2 <= k <= max_topics_allowed]
    if not candidates_k:
        candidates_k = [2]

    # LDA: bestes K/Seed via Kohärenz
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
    topics_all = top_terms_per_topic(lda, feat_names, topn=12)

    k_report = min(5, doc_topic.shape[1])
    top_idx, _ = pick_top_k_topics(doc_topic, k=k_report)

    print(f"[Info] {name}: K={best['k']} Seed={best['seed']} Coh={best['coh']:.3f} "
          f"(features={Xc.shape[1]}, min_df={used_params['min_df']}, max_df={used_params['max_df']:.2f}, title_boost={title_boost}, comments={include_comments})")

    # Ausgabe mit Auto-Label + Beispielposts
    rows = []
    topic_assign = doc_topic.argmax(axis=1)
    for rank, t_id in enumerate(top_idx, start=1):
        terms = [w for w, _ in topics_all[t_id]]
        label = _autolabel_topic(terms)
        print(f"Topic {rank} (ID {t_id}) [{label}]: " + ", ".join(terms))

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
    # 0) Daten laden oder fetchen
    try:
        df_raw = pd.read_csv(DATA_DIR / "raw_r_stuttgart.csv")
    except FileNotFoundError:
        df_raw = fetch_subreddit_posts(
            subreddit_name="Stuttgart",
            where="top",
            time_filter="year",
            limit=600,
            with_comments=True,
            max_comments_per_post=50
        )

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
