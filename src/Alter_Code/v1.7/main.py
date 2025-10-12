from __future__ import annotations
import sys
import math
import warnings
from collections import Counter
from typing import List, Dict, Tuple, Optional
import pandas as pd

from .fetch import fetch_subreddit_posts
from .preprocess import (
    add_clean_columns, extract_flairs, GERMAN_SW, ENGLISH_SW,
    clean_text, tokenize
)
from .topics import (
    vectorize_counts, lda_from_matrix,
    top_terms_per_topic, pick_top_k_topics, umass_coherence
)
from .utils import DATA_DIR

# =========================
# Konfiguration
# =========================
RUN_LSA_COMPARE = False  # bewusst aus -> Fokus: Stabilität LDA

# Pro-Flair-Modelle (Quick Win #1)
FLAIR_RUNS = [
    {"name": "Diskussion",     "flairs": ["Diskussion"],                      "include_comments": False, "min_tokens": 30, "title_boost": 2},
    {"name": "News",           "flairs": ["News"],                            "include_comments": False, "min_tokens": 30, "title_boost": 2},
    {"name": "Events",         "flairs": ["Events"],                          "include_comments": True,  "min_tokens": 15, "title_boost": 1},
    {"name": "Frage/Advice",   "flairs": ["Frage / Advice", "Looking for..."],"include_comments": True,  "min_tokens": 15, "title_boost": 1},
]

# Kandidaten für Anzahl Topics und Seeds
CANDIDATE_K = [6, 8, 10, 12]
SEEDS = (13, 21, 42, 77)

# Zielbereich für Feature-Anzahl
FEAT_TARGET_LOW  = 300
FEAT_TARGET_HIGH = 2000
FEAT_MIN_HARD    = 100  # harte Untergrenze – darunter oft sinnlos

# Domain-Whitelist (niemals automatisch zu Stopwörtern)
DOMAIN_WHITELIST = {
    "bahn","vvs","sbahn","s-bahn","haltestelle","linie","ticket","parkhaus",
    "parken","bahnhof","stuttgart_21","s21","schlossplatz","fernsehturm",
    "killesberg","neckar","innenstadt","u-bahn","u","s1","s2","s3","demo","afd","wahl"
}

# =========================
# Hilfsfunktionen
# =========================
def _normalize_stopwords(stopset: set[str]) -> List[str]:
    """Stopwörter wie unsere Texte normalisieren (clean_text + tokenize)."""
    norm = set()
    for w in stopset:
        toks = tokenize(clean_text(w))
        norm.update(toks)
    return sorted({t for t in norm if len(t) >= 2})

def _build_texts_with_title_boost(df: pd.DataFrame, title_boost: int) -> List[str]:
    base = df["clean_for_vect"].fillna("")
    if title_boost <= 0 or "title" not in df.columns:
        return base.tolist()
    title_tokens = (
        df["title"].fillna("")
        .map(clean_text).map(tokenize).map(lambda ts: " ".join(ts))
    )
    title_boost_str = title_tokens.map(lambda s: (s + " ") * title_boost)
    return (base + " " + title_boost_str).str.strip().tolist()

# Quick Win #2: Auto-Stopwörter (Doc-Frequenz-basiert) mit Whitelist-Schutz
def _auto_stopwords_from_tokens(tokens_list: List[List[str]],
                                df_ratio: float = 0.70,
                                top_n_cap: int = 40,
                                whitelist: Optional[set] = None) -> set:
    whitelist = whitelist or set()
    n_docs = len(tokens_list)
    if n_docs == 0:
        return set()
    doc_freq = Counter()
    for ts in tokens_list:
        doc_freq.update(set(ts))  # Präsenz je Dokument
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
    """
    Liefert konsistente (min_df:int, max_df:float).
    Regeln:
      - min_df = max(1, abs, ceil(frac*n_docs)), dann clamp auf <= n_docs-1
      - max_df >= 2/n_docs
      - wenn min_df > floor(max_df*n_docs)-1: erst max_df -> 1.0, dann min_df <= n_docs-1
    """
    n_docs = max(int(n_docs), 1)

    # Basis
    min_df_val = max(1, max(base_min_df_abs, int(math.ceil(base_min_df_frac * n_docs))))
    # clamp min_df auf <= n_docs-1 (sonst ist per Definition alles weg)
    min_df_val = min(min_df_val, max(1, n_docs - 1))

    # max_df initial
    max_df_val = min(1.0, max(base_max_df_frac, 2.0 / n_docs))

    # Konflikt prüfen
    max_docs_allowed = int(math.floor(max_df_val * n_docs))
    if max_docs_allowed <= 1:
        max_df_val = 1.0
        max_docs_allowed = n_docs

    if min_df_val > max_docs_allowed - 1:
        # erst max_df anheben
        max_df_val = 1.0
        max_docs_allowed = n_docs
        # falls immer noch Konflikt: min_df auf n_docs-1
        if min_df_val > max_docs_allowed - 1:
            min_df_val = max(1, n_docs - 1)

    return int(min_df_val), float(max_df_val)

def _safe_vectorize(texts: List[str], min_df_val: int, max_df_val: float, stop_words_norm: List[str]):
    """
    Führt CountVectorizer aus und fängt typische sklearn-Fehler ab:
    - 'max_df corresponds to < documents than min_df'
    - 'After pruning, no terms remain' / 'empty vocabulary'
    Probiert automatisch entschärfte Parameter (max_df=1.0, min_df↓, ngram=(1,1)).
    """
    try:
        return vectorize_counts(
            texts, ngram=(1, 2),
            min_df=min_df_val, max_df=max_df_val,
            max_features=12000, stop_words=stop_words_norm
        )
    except ValueError:
        # Fallback-Kandidaten
        fallback_params = [
            (min_df_val, 1.0, (1, 2)),
            (max(1, min_df_val // 2), 1.0, (1, 2)),
            (1, 1.0, (1, 2)),
            (1, 1.0, (1, 1)),
        ]
        for mdf, mxf, ngr in fallback_params:
            try:
                return vectorize_counts(
                    texts, ngram=ngr,
                    min_df=mdf, max_df=mxf,
                    max_features=12000, stop_words=stop_words_norm
                )
            except ValueError:
                continue
        raise ValueError("Vectorization failed after multiple fallbacks (no terms remain).")

def _vectorize_to_window(texts: List[str], stop_words_norm: List[str]) -> Tuple:
    """
    Versucht mehrere (min_df, max_df)-Kombinationen und fällt auf 'best effort' zurück.
    Fängt ValueError ab und probiert nächste Kombination.
    """
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

    if best["vec"] is not None and best["n_feat"] >= FEAT_MIN_HARD:
        print(f"[Info] Using best-effort features={best['n_feat']} "
              f"(min_df={best['params']['min_df']}, max_df={best['params']['max_df']:.2f})")
        return best["vec"], best["X"], best["params"]

    return None, None, None

def _autolabel_topic(feature_terms: List[str]) -> str:
    """Sehr einfache Heuristik für ein lesbares Topic-Label."""
    domain_hits = ["bahn","vvs","haltestelle","linie","ticket","parkhaus","bahnhof",
                   "stuttgart_21","s21","schlossplatz","fernsehturm","innenstadt","demo","afd","wahl"]
    for t in feature_terms:
        if " " in t:
            return t
    for t in feature_terms:
        if t in domain_hits:
            return t
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

    # Auto-Stopwörter (Quick Win #2)
    auto_sw = _auto_stopwords_from_tokens(df["tokens"].tolist(), df_ratio=0.70, top_n_cap=40, whitelist=DOMAIN_WHITELIST)
    if auto_sw:
        print(f"[Info] Auto-Stopwörter ({len(auto_sw)}): {sorted(list(auto_sw))[:12]}{' ...' if len(auto_sw)>12 else ''}")

    stop_union = (GERMAN_SW | ENGLISH_SW | auto_sw)
    stop_union_norm = _normalize_stopwords(stop_union)

    # Texte + Titel-Boost
    texts = _build_texts_with_title_boost(df, title_boost)

    # Vektorisierung (robust)
    vec_counts, Xc, used_params = _vectorize_to_window(texts, stop_union_norm)
    if vec_counts is None or Xc is None:
        print(f"[Abbruch] Zu wenig Features in {name}.")
        return

    # LDA: bestes K/Seed via Kohärenz
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

    # Ausgabe mit Auto-Label + Beispielposts (Quick Win #3: Labels)
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
