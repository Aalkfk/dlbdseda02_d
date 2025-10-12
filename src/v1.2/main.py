from __future__ import annotations
import sys
import pandas as pd
from .fetch import fetch_subreddit_posts
from .preprocess import add_clean_columns, extract_flairs, GERMAN_SW, ENGLISH_SW
from .topics import (
    vectorize_counts, vectorize_tfidf, lda_from_matrix, lsa_from_tfidf,
    top_terms_per_topic, pick_top_k_topics, umass_coherence
)
from .utils import DATA_DIR

# Optionaler LSA-Check (deaktiviert um sklearn-Warnings zu vermeiden)
RUN_LSA = False

def main():
    # 1) Daten abrufen (oder vorhandene RAW verwenden)
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

    # 2) Vorverarbeitung (ohne Stemming -> lesbarere Topics)
    df = add_clean_columns(df_raw, use_stemming=False, filter_english=True)
    df.to_csv(DATA_DIR / "clean_r_stuttgart.csv", index=False)

    # 3) Flairs & aktivste User
    flairs = extract_flairs(df_raw).head(20)
    active_users = (df_raw["author"]
                    .dropna()
                    .loc[lambda s: s.ne("[deleted]") & s.ne("None")]
                    .value_counts()
                    .head(20))
    flairs.to_csv(DATA_DIR / "top_flairs.csv")
    active_users.to_csv(DATA_DIR / "top_users.csv")

    # 4) Themen: LDA auf Count-Features (besser als TF-IDF für LDA)
    texts = df["clean_for_vect"].tolist()
    stop_union = list(GERMAN_SW | ENGLISH_SW)

    vec_counts, Xc = vectorize_counts(
        texts,
        ngram=(1,2),
        min_df=max(8, int(0.005 * max(len(texts), 1))),  # mind. 8 Doks oder 0,5%
        max_df=0.35,
        max_features=15000,
        stop_words=stop_union
    )

    if Xc.shape[1] == 0:
        print("[Abbruch] Keine Features nach Vorverarbeitung. min_df/max_df reduzieren.", file=sys.stderr)
        return

    # Bestes K via einfacher Kohärenzsuche
    CANDIDATE_K = [6, 8, 10, 12]
    best = {"k": None, "coh": -1e9, "model": None, "doc_topic": None}
    for k in CANDIDATE_K:
        lda_k, dt_k = lda_from_matrix(Xc, n_topics=k, max_iter=30, learning_decay=0.7)
        coh_k = umass_coherence(lda_k, Xc, topn=12)
        if coh_k > best["coh"]:
            best.update({"k": k, "coh": coh_k, "model": lda_k, "doc_topic": dt_k})

    lda = best["model"]
    doc_topic = best["doc_topic"]
    feat_names = vec_counts.get_feature_names_out()
    lda_topics_all = top_terms_per_topic(lda, feat_names, topn=12)

    top_idx, totals = pick_top_k_topics(doc_topic, k=5)

    # 5) Ausgabe
    print("\n=== Top 20 Flairs ===")
    if not flairs.empty:
        print(flairs.to_string())
    else:
        print("(keine Flairs gefunden)")

    print("\n=== Top 20 aktivste User ===")
    if not active_users.empty:
        print(active_users.to_string())
    else:
        print("(keine Nutzer ermittelt)")

    print(f"\n=== 5 Haupt-Themen (LDA auf Counts) — Top-Terme ===")
    print(f"[Info] Gewähltes K={best['k']} (Coherence {best['coh']:.3f})")
    for rank, t_id in enumerate(top_idx, start=1):
        terms = ", ".join([w for w,_ in lda_topics_all[t_id]])
        print(f"Topic {rank} (ID {t_id}): {terms}")

    # je berichtetem Thema 3 Beispiel-Posts mit höchster Zugehörigkeit
    topic_assign = doc_topic.argmax(axis=1)
    rows = []
    for rank, t_id in enumerate(top_idx, start=1):
        mask = (topic_assign == t_id)
        scores = doc_topic[mask, t_id]
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

    # 6) Optional: LSA aus Stabilitätsgründen deaktiviert
    if RUN_LSA:
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        vec_tfidf, Xt = vectorize_tfidf(texts, ngram=(1,2), min_df=5, max_df=0.5, stop_words=stop_union)
        n_comp = min(8, max(2, Xt.shape[1]-1), 100)
        if n_comp >= 2:
            _lsa, _lsa_dt = lsa_from_tfidf(Xt, n_topics=int(n_comp))
        # keine Ausgabe notwendig

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
