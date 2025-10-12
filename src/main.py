from __future__ import annotations
import sys
import math
import warnings
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import pandas as pd

from .fetch import fetch_subreddit_posts
from .preprocess import (
    add_clean_columns, extract_flairs, GERMAN_SW, ENGLISH_SW,
    clean_text, tokenize
)
from .topics import (
    vectorize_counts, vectorize_tfidf, lda_from_matrix, lsa_from_tfidf,
    top_terms_per_topic, pick_top_k_topics, umass_coherence
)
from .utils import DATA_DIR


# =========================
# Konfiguration
# =========================
RUN_LSA_COMPARE = False  # Vergleichsausgabe optional; Fokus hier auf LDA & Stabilität

# Quick Win #1: Pro Flair modellieren (Kommentare je nach Flair-Klasse)
FLAIR_RUNS = [
    {"name": "Diskussion",     "flairs": ["Diskussion"],                    "include_comments": False, "min_tokens": 30, "title_boost": 2},
    {"name": "News",           "flairs": ["News"],                          "include_comments": False, "min_tokens": 30, "title_boost": 2},
    {"name": "Events",         "flairs": ["Events"],                        "include_comments": True,  "min_tokens": 15, "title_boost": 1},
    {"name": "Frage/Advice",   "flairs": ["Frage / Advice","Looking for..."], "include_comments": True,  "min_tokens": 15, "title_boost": 1},
]

# Kandidaten für Anzahl Topics und Seeds
CANDIDATE_K = [6, 8, 10, 12]
SEEDS = (13, 21, 42, 77)

# Zielbereich für Anzahl Features (nicht hart, aber angestrebt)
FEAT_TARGET_LOW  = 300
FEAT_TARGET_HIGH = 2000
FEAT_MIN_HARD    = 100  # harte Untergrenze, sonst abbrechen

# Domain-Whitelist: diese Begriffe sollen NIE automatisch zu Stopwörtern werden
DOMAIN_WHITELIST = {
    "bahn","vvs","sbahn","s-bahn","haltestelle","linie","ticket","parkhaus",
    "parken","bahnhof","stuttgart_21","s21","schlossplatz","fernsehturm",
    "killesberg","neckar","innenstadt","u-bahn","u","s1","s2","s3"
}


# =========================
# Hilfsfunktionen
# =========================
def _normalize_stopwords(stopset: set[str]) -> List[str]:
    """
    Stopwörter wie unsere Texte normalisieren (clean_text + tokenize),
    um Inkonsistenzen wie 'hä?' vs. 'hä' zu vermeiden.
    """
    norm = set()
    for w in stopset:
        toks = tokenize(clean_text(w))
        norm.update(toks)
    norm = {t for t in norm if len(t) >= 2}
    return sorted(norm)


def _build_texts_with_title_boost(df: pd.DataFrame, title_boost: int) -> List[str]:
    base = df["clean_for_vect"].fillna("")
    if title_boost <= 0 or "title" not in df.columns:
        return base.tolist()
    title_tokens = (
        df["title"].fillna("")
        .map(clean_text)
        .map(tokenize)
        .map(lambda ts: " ".join(ts))
    )
    title_boost_str = title_tokens.map(lambda s: (s + " ") * title_boost)
    texts = (base + " " + title_boost_str).str.strip().tolist()
    return texts


# Quick Win #2: Auto-Stopwörter (Doc-Frequenz-basiert) mit Whitelist-Schutz
def _auto_stopwords_from_tokens(tokens_list: List[List[str]],
                                df_ratio: float = 0.70,
                                top_n_cap: int = 40,
                                whitelist: Optional[set] = None) -> set:
    whitelist = whitelist or set()
    n_docs = len(tokens_list)
    if n_docs == 0:
        return set()

    # Dokumentfrequenz: in wie vielen Doks kommt Token vor?
    doc_freq = Counter()
    for ts in tokens_list:
        doc_freq.update(set(ts))  # Präsenz statt Häufigkeit

    # Kandidaten: Tokens, die in >= df_ratio der Dokumente vorkommen
    thresh = max(1, int(math.ceil(df_ratio * n_docs)))
    cand = [t for t, dfc in doc_freq.items() if dfc >= thresh]

    # sortiere nach DF, deckele Anzahl
    cand.sort(key=lambda t: (-doc_freq[t], t))
    cand = cand[:top_n_cap]

    # filtere Whitelist & sehr kurze Tokens oder reine Ziffern
    auto_sw = set()
    for t in cand:
        if t in whitelist:
            continue
        if len(t) < 3:
            continue
        if any(ch.isdigit() for ch in t):
            continue
        auto_sw.add(t)
    return auto_sw


def _choose_df_params(n_docs: int,
                      base_min_df_abs: int, base_min_df_frac: float,
                      base_max_df_frac: float) -> Tuple[int, float]:
    """
    Konsistente min_df/max_df-Wahl abhängig von n_docs.
    """
    min_df_val = max(base_min_df_abs, int(math.ceil(base_min_df_frac * n_docs)))
    max_df_val = base_max_df_frac
    max_docs_allowed = max_df_val * n_docs
    if max_docs_allowed <= 1:
        max_df_val = min(0.9, max(base_max_df_frac, 2.0 / max(n_docs, 1)))
        max_docs_allowed = max_df_val * n_docs
    if min_df_val >= max_docs_allowed:
        min_df_val = max(2, int(max_docs_allowed) - 1)
    return min_df_val, max_df_val


def _vectorize_to_window(texts: List[str], stop_words_norm: List[str]) -> Tuple:
    """
    Versucht mehrere (min_df, max_df)-Kombinationen, um die Feature-Anzahl
    in den Zielbereich zu bringen. Fällt robust auf 'best effort' zurück.
    """
    n_docs = len(texts)
    attempts = [
        # (min_df_abs, min_df_frac, max_df_frac)
        (15, 0.018, 0.20),
        (10, 0.010, 0.30),
        (5,  0.005, 0.40),
        (2,  0.000, 0.60),
    ]

    best = {"delta": 10**9, "vec": None, "X": None, "params": None, "n_feat": 0}
    target_mid = (FEAT_TARGET_LOW + FEAT_TARGET_HIGH) / 2.0

    for (abs_min, frac_min, frac_max) in attempts:
        min_df_val, max_df_val = _choose_df_params(n_docs, abs_min, frac_min, frac_max)
        vec, X = vectorize_counts(
            texts,
            ngram=(1, 2),
            min_df=min_df_val,
            max_df=max_df_val,
            max_features=12000,
            stop_words=stop_words_norm
        )
        n_feat = X.shape[1]
        print(f"[Info] Try vectorize: n_docs={n_docs}, min_df={min_df_val}, max_df={max_df_val:.2f} -> features={n_feat}")

        # innerhalb Zielbereich? -> sofort nehmen
        if FEAT_TARGET_LOW <= n_feat <= FEAT_TARGET_HIGH:
            return vec, X, {"min_df": min_df_val, "max_df": max_df_val}

        # ansonsten: Lösung merken, die der Mitte am nächsten ist
        delta = abs(n_feat - target_mid)
        if delta < best["delta"]:
            best.update({"delta": delta, "vec": vec, "X": X, "params": {"min_df": min_df_val, "max_df": max_df_val}, "n_feat": n_feat})

    # Best effort zurückgeben – solange > harte Untergrenze
    if best["vec"] is not None and best["n_feat"] >= FEAT_MIN_HARD:
        print(f"[Info] Using best-effort features={best['n_feat']} (min_df={best['params']['min_df']}, max_df={best['params']['max_df']:.2f})")
        return best["vec"], best["X"], best["params"]

    # finaler Abbruch
    return None, None, None


def _autolabel_topic(feature_terms: List[str]) -> str:
    """
    Sehr einfache Heuristik:
    - erstes Bigram (enthält Leerzeichen) -> Label
    - sonst erster 'Domain'-Term aus einer groben Liste
    - sonst Top-2 Unigramme
    """
    domain_hits = ["bahn","vvs","haltestelle","linie","ticket","parkhaus","bahnhof",
                   "stuttgart_21","s21","schlossplatz","fernsehturm","innenstadt","demo","afd","wahl"]
    # 1) Bigram
    for t in feature_terms:
        if " " in t:
            return t
    # 2) Domain
    for t in feature_terms:
        if t in domain_hits:
            return t
    # 3) Fallback: Top-2
    return " / ".join(feature_terms[:2])


def _run_for_flair_block(df_raw: pd.DataFrame, run_cfg: Dict, base_stop_union: set) -> None:
    name = run_cfg["name"]
    flairs = run_cfg["flairs"]
    include_comments = run_cfg["include_comments"]
    min_tokens = run_cfg["min_tokens"]
    title_boost = run_cfg["title_boost"]

    print(f"\n# ==== Themen für Flair: {name} ====")

    if "flair_text" in df_raw.columns:
        sub_raw = df_raw[df_raw["flair_text"].isin(flairs)].reset_index(drop=True)
    else:
        sub_raw = df_raw.copy()

    if sub_raw.empty:
        print(f"[Skip] Keine Dokumente für {name}.")
        return

    # Vorverarbeitung
    df = add_clean_columns(sub_raw, use_stemming=False, filter_english=True, include_comments=include_comments)
    if "tokens" in df.columns:
        df = df[df["tokens"].map(len) >= min_tokens].reset_index(drop=True)

    if df.empty:
        print(f"[Skip] Nach Preprocess/Tokenfilter keine Dokumente für {name}.")
        return

    # Quick Win #2: Auto-Stopwörter aus DF (mit Whitelist)
    auto_sw = _auto_stopwords_from_tokens(df["tokens"].tolist(), df_ratio=0.70, top_n_cap=40, whitelist=DOMAIN_WHITELIST)
    if auto_sw:
        print(f"[Info] Auto-Stopwörter ({len(auto_sw)}): {sorted(list(auto_sw))[:12]}{' ...' if len(auto_sw)>12 else ''}")

    # Stopwort-Union normalisieren
    stop_union = (GERMAN_SW | ENGLISH_SW | auto_sw)
    stop_union_norm = _normalize_stopwords(stop_union)

    # Texte + Titel-Boost
    texts = _build_texts_with_title_boost(df, title_boost)

    # Vektorisierung in Ziel-Fenster (best effort)
    vec_counts, Xc, used_params = _vectorize_to_window(texts, stop_union_norm)
    if vec_counts is None or Xc is None:
        print(f"[Abbruch] Zu wenig Features in {name}.")
        return

    # LDA: bestes K/Seed per Kohärenz
    best = {"coh": -1e9, "k": None, "model": None, "doc_topic": None, "seed": None}
    for k in CANDIDATE_K:
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

    # Ausgabe mit Auto-Label
    rows = []
    for rank, t_id in enumerate(top_idx, start=1):
        terms = [w for w, _ in topics_all[t_id]]
        label = _autolabel_topic(terms)
        print(f"Topic {rank} (ID {t_id}) [{label}]: " + ", ".join(terms))
        # 3 Beispiel-Posts
        topic_assign = doc_topic.argmax(axis=1)
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

    # Samples speichern
    if rows:
        out = pd.DataFrame(rows)
        out_path = DATA_DIR / f"samples_{name.lower().replace('/','_')}_per_topic.csv"
        out.to_csv(out_path, index=False)


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

    # 2) Quick Win #1: pro Flair eigene Themenmodelle
    base_stop_union = GERMAN_SW | ENGLISH_SW
    for run in FLAIR_RUNS:
        _run_for_flair_block(df_raw, run, base_stop_union)

    # (Optional) LSA-Vergleich pro Flair ließe sich ähnlich in _run_for_flair_block() integrieren,
    # ist hier bewusst deaktiviert (RUN_LSA_COMPARE=False) um Warnungen/Noise zu vermeiden.


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
