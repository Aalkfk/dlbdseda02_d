from __future__ import annotations
import sys
import math
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

# -------------------------
# Konfiguration
# -------------------------
RUN_LSA_COMPARE = True  # LSA-Vergleich zusätzlich zu LDA

# Flairs, die in die Themenanalyse dürfen
ALLOWED_FLAIRS = [
    "Diskussion", "News", "Events", "Looking for...", "Frage / Advice", "Sonstiges"
]

# Startwerte für Mindestlänge (Tokens) pro Dokument
MIN_TOKENS_FOR_TOPIC = 30

# Kandidaten für Anzahl Topics und Seeds
CANDIDATE_K = [6, 8, 10, 12]
SEEDS = (13, 21, 42, 77)

# Titel-Booster (0 = aus)
TITLE_BOOST = 2

# Mindest-Anzahl Features, die wir für eine sinnvolle LDA wollen
MIN_FEATURES_TARGET = 50


def _build_texts_with_title_boost(df: pd.DataFrame, title_boost: int) -> list[str]:
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


def _normalize_stopwords(stopset: set[str]) -> list[str]:
    norm = set()
    for w in stopset:
        toks = tokenize(clean_text(w))
        norm.update(toks)
    norm = {t for t in norm if len(t) >= 2}
    return sorted(norm)


def _choose_df_params(n_docs: int, base_min_df_abs: int, base_min_df_frac: float, base_max_df_frac: float):
    min_df_val = max(base_min_df_abs, int(math.ceil(base_min_df_frac * n_docs)))
    max_df_val = base_max_df_frac
    max_docs_allowed = max_df_val * n_docs
    if max_docs_allowed <= 1:
        max_df_val = min(0.9, max(base_max_df_frac, 2.0 / max(n_docs, 1)))
        max_docs_allowed = max_df_val * n_docs
    if min_df_val >= max_docs_allowed:
        min_df_val = max(2, int(max_docs_allowed) - 1)
    return min_df_val, max_df_val


def _vectorize_with_fallbacks(texts, stop_words_norm):
    """
    Versucht mehrere (min_df, max_df)-Kombinationen, bis genügend Features
    vorhanden sind. Gibt (vec, X, params_dict) zurück oder None bei totalem Fehlschlag.
    """
    n_docs = len(texts)
    attempts = [
        # (min_df_abs, min_df_frac, max_df_frac)
        (15, 0.018, 0.20),
        (10, 0.010, 0.30),
        (5,  0.005, 0.40),
        (2,  0.000, 0.60),
    ]
    for (abs_min, frac_min, frac_max) in attempts:
        min_df_val, max_df_val = _choose_df_params(n_docs, abs_min, frac_min, frac_max)
        print(f"[Info] Try vectorize: n_docs={n_docs}, min_df={min_df_val}, max_df={max_df_val:.2f}")
        vec, X = vectorize_counts(
            texts,
            ngram=(1, 2),
            min_df=min_df_val,
            max_df=max_df_val,
            max_features=9000,
            stop_words=stop_words_norm
        )
        n_feat = X.shape[1]
        print(f"[Info] -> features={n_feat}")
        if n_feat >= MIN_FEATURES_TARGET:
            return vec, X, {"min_df": min_df_val, "max_df": max_df_val}
    # Falls wir hier landen, letzter Versuch: akzeptiere, was da ist (auch wenn < Target)
    # aber brich hart ab, wenn wirklich nur 0 oder 1 Feature vorhanden ist.
    min_df_val, max_df_val = _choose_df_params(n_docs, 1, 0.0, 0.80)
    vec, X = vectorize_counts(
        texts,
        ngram=(1, 2),
        min_df=min_df_val,
        max_df=max_df_val,
        max_features=12000,
        stop_words=stop_words_norm
    )
    n_feat = X.shape[1]
    print(f"[Info] Final fallback -> features={n_feat} (min_df={min_df_val}, max_df={max_df_val:.2f})")
    if n_feat <= 1:
        return None, None, None
    return vec, X, {"min_df": min_df_val, "max_df": max_df_val}


def main():
    # 1) Daten abrufen / laden
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

    # 2) Flairs & aktivste User (immer auf kompletter RAW-Basis)
    flairs = extract_flairs(df_raw).head(20)
    active_users = (df_raw["author"]
                    .dropna()
                    .loc[lambda s: s.ne("[deleted]") & s.ne("None")]
                    .value_counts()
                    .head(20))
    flairs.to_csv(DATA_DIR / "top_flairs.csv")
    active_users.to_csv(DATA_DIR / "top_users.csv")

    # 3) Flair-Subset für Topic Modeling
    if "flair_text" in df_raw.columns:
        df_model_raw = df_raw[df_raw["flair_text"].isin(ALLOWED_FLAIRS)].reset_index(drop=True)
        if df_model_raw.empty:
            print("[Hinweis] Kein Beitrag mit erlaubten Flairs – nutze gesamten Korpus.", file=sys.stderr)
            df_model_raw = df_raw.copy()
    else:
        df_model_raw = df_raw.copy()

    # 4) Vorverarbeitung (zunächst ohne Kommentare)
    df = add_clean_columns(df_model_raw, use_stemming=False, filter_english=True, include_comments=False)
    if "tokens" in df.columns:
        df = df[df["tokens"].map(len) >= MIN_TOKENS_FOR_TOPIC].reset_index(drop=True)

    # Fallbacks: wenn nach Filterung sehr wenige Doks, mit Kommentaren/geringerem Schwellwert neu versuchen
    if len(df) < 60:
        print(f"[Warn] Nur {len(df)} Doks nach Preprocess ohne Kommentare. Versuche mit Kommentaren …")
        df_alt = add_clean_columns(df_model_raw, use_stemming=False, filter_english=True, include_comments=True)
        if "tokens" in df_alt.columns:
            # senke Schwellwert moderat
            df_alt = df_alt[df_alt["tokens"].map(len) >= max(15, MIN_TOKENS_FOR_TOPIC // 2)].reset_index(drop=True)
        if len(df_alt) > len(df):
            df = df_alt

    if len(df) < 30:
        print(f"[Warn] Nur {len[df]} Doks übrig. Senke Token-Grenze auf 10.")
        df = add_clean_columns(df_model_raw, use_stemming=False, filter_english=True, include_comments=True)
        if "tokens" in df.columns:
            df = df[df["tokens"].map(len) >= 10].reset_index(drop=True)

    if df.empty:
        print("[Abbruch] Keine Dokumente nach Vorverarbeitung/Filter. Parameter (Flairs/MinTokens) anpassen.", file=sys.stderr)
        return

    df.to_csv(DATA_DIR / "clean_r_stuttgart_topics_subset.csv", index=False)

    # 5) Texte bauen (mit optionalem Titel-Boost)
    texts = _build_texts_with_title_boost(df, TITLE_BOOST)
    stop_union_norm = _normalize_stopwords(GERMAN_SW | ENGLISH_SW)

    # 6) Vektorisierung mit Fallback-Leiter
    vec_counts, Xc, used_params = _vectorize_with_fallbacks(texts, stop_union_norm)
    if vec_counts is None or Xc is None:
        print("[Abbruch] Zu wenig Features für sinnvolle Themen (<=1). Prüfe Stopwortliste/Filter.", file=sys.stderr)
        return

    # 7) LDA: beste K/Seed via Kohärenz
    best = {"coh": -1e9, "k": None, "model": None, "doc_topic": None, "seed": None}
    for k in CANDIDATE_K:
        for seed in SEEDS:
            lda_k, dt_k = lda_from_matrix(Xc, n_topics=k, max_iter=60, random_state=seed)
            coh_k = umass_coherence(lda_k, Xc, topn=12)
            if coh_k > best["coh"]:
                best.update({"coh": coh_k, "k": k, "model": lda_k, "doc_topic": dt_k, "seed": seed})

    if best["model"] is None:
        print("[Abbruch] Kein LDA-Modell gefunden. Prüfe Kandidaten/Korpusgröße.", file=sys.stderr)
        return

    lda = best["model"]
    doc_topic = best["doc_topic"]
    feat_names = vec_counts.get_feature_names_out()
    lda_topics_all = top_terms_per_topic(lda, feat_names, topn=12)
    top_idx, totals = pick_top_k_topics(doc_topic, k=5)

    # 8) Ausgabe (LDA)
    print("\n=== Top 20 Flairs ===")
    print(flairs.to_string() if not flairs.empty else "(keine Flairs gefunden)")

    print("\n=== Top 20 aktivste User ===")
    print(active_users.to_string() if not active_users.empty else "(keine Nutzer ermittelt)")

    print(f"\n=== 5 Haupt-Themen (LDA auf Counts) — Top-Terme ===")
    print(f"[Info] Gewähltes K={best['k']} (Seed {best['seed']}, Coherence {best['coh']:.3f}, "
          f"TitleBoost={TITLE_BOOST}, min_df={used_params['min_df']}, max_df={used_params['max_df']:.2f})")
    for rank, t_id in enumerate(top_idx, start=1):
        terms = ", ".join([w for w, _ in lda_topics_all[t_id]])
        print(f"Topic {rank} (ID {t_id}): {terms}")

    # Beispiel-Posts je Thema
    topic_assign = doc_topic.argmax(axis=1)
    rows = []
    for rank, t_id in enumerate(top_idx, start=1):
        mask = (topic_assign == t_id)
        scores = doc_topic[mask, t_id]
        if getattr(scores, "size", 0) == 0:
            continue
        which = scores.argsort()[::-1][:3]
        candidates = df.loc[mask].iloc[which]
        for idx_row in candidates.index:
            rows.append({
                "reported_topic_rank": rank,
                "topic_id": int(t_id),
                "title": df.at[idx_row, "title"] if "title" in df.columns else "",
                "permalink": df.at[idx_row, "permalink"] if "permalink" in df.columns else ""
            })
    pd.DataFrame(rows).to_csv(DATA_DIR / "sample_posts_per_topic.csv", index=False)

    # 9) LSA-Vergleich nur, wenn genug Features
    if RUN_LSA_COMPARE:
        try:
            min_df_val = used_params["min_df"]
            max_df_val = used_params["max_df"]
            vec_tfidf, Xt = vectorize_tfidf(
                texts,
                ngram=(1, 2),
                min_df=min_df_val,
                max_df=max_df_val,
                stop_words=stop_union_norm
            )
            if Xt.shape[1] >= 2:
                n_comp = min(best['k'], max(2, Xt.shape[1] - 1), 100)
                lsa_model, lsa_dt = lsa_from_tfidf(Xt, n_topics=int(n_comp))
                lsa_topics_all = top_terms_per_topic(lsa_model, vec_tfidf.get_feature_names_out(), topn=12)
                lsa_top_idx, _ = pick_top_k_topics(lsa_dt, k=5)
                print("\n=== (Vergleich) LSA — Top-Terme ===")
                for rank, t_id in enumerate(lsa_top_idx, start=1):
                    terms = ", ".join([w for w, _ in lsa_topics_all[t_id]])
                    print(f"LSA Topic {rank} (ID {t_id}): {terms}")
            else:
                print("[Info] LSA übersprungen (zu wenige TF-IDF-Features).")
        except Exception as e:
            print(f"[LSA-Vergleich übersprungen] Grund: {e}", file=sys.stderr)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
