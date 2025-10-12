from __future__ import annotations
import pandas as pd
from .fetch import fetch_subreddit_posts
from .preprocess import add_clean_columns, extract_flairs, GERMAN_SW, ENGLISH_SW
from .topics import vectorize_counts, vectorize_tfidf, lda_from_matrix, lsa_from_tfidf, top_terms_per_topic, pick_top_k_topics
from .utils import DATA_DIR

N_TOPICS_ALL = 8     # wir modellieren 8 und wählen unten die Top-5
N_TOPICS_REPORT = 5

def main():
    # 1) Daten abrufen (oder vorhandene CSV weiterverwenden)
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

    # 2) Vorverarbeitung (+ Sprachfilter, Stemming)
    df = add_clean_columns(df_raw, use_stemming=True, filter_english=True)
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

    # 4) Themen: LDA auf Count-Features (besser für LDA als TF-IDF)
    texts = df["clean_for_vect"].tolist()
    stop_union = list(GERMAN_SW | ENGLISH_SW)  # engl. Stoppwörter ebenfalls raus
    vec_counts, Xc = vectorize_counts(
        texts, ngram=(1,2), min_df=5, max_df=0.45, max_features=20000, stop_words=stop_union
    )
    lda, doc_topic = lda_from_matrix(Xc, n_topics=N_TOPICS_ALL, max_iter=25, learning_decay=0.7)

    feat_names = vec_counts.get_feature_names_out()
    lda_topics_all = top_terms_per_topic(lda, feat_names, topn=12)

    # Wichtig: Wir berichten die 5 stärksten Topics (Summe über alle Doks)
    top_idx, totals = pick_top_k_topics(doc_topic, k=N_TOPICS_REPORT)

    # 5) Ausgabe
    print("\n=== Top 20 Flairs ===")
    if not flairs.empty:
        print(flairs.to_string())
    else:
        print("(keine Flairs gefunden)")

    print("\n=== Top 20 aktivste User ===")
    print(active_users.to_string())

    print(f"\n=== {N_TOPICS_REPORT} Haupt-Themen (LDA auf Counts) — Top-Terme ===")
    for rank, t_id in enumerate(top_idx, start=1):
        terms = ", ".join([w for w,_ in lda_topics_all[t_id]])
        print(f"Topic {rank} (ID {t_id}): {terms}")

    # je berichtetem Thema 3 Beispiel-Posts mit höchster Zugehörigkeit
    topic_assign = doc_topic.argmax(axis=1)
    rows = []
    for rank, t_id in enumerate(top_idx, start=1):
        idx = (topic_assign == t_id)
        # Score = eigentliche Zuordnung (höchster Posterior)
        scores = doc_topic[idx, t_id]
        which = scores.argsort()[::-1][:3]
        candidates = df.loc[idx].iloc[which]
        for _, r in candidates.iterrows():
            rows.append({
                "reported_topic_rank": rank,
                "topic_id": int(t_id),
                "title": r["title"] if "title" in r else "",
                "permalink": r["permalink"] if "permalink" in r else ""
            })
    pd.DataFrame(rows).to_csv(DATA_DIR / "sample_posts_per_topic.csv", index=False)

    # 6) Optional: LSA als Gegencheck (stabiler gemacht, aber standardmäßig nur loggen)
    try:
        vec_tfidf, Xt = vectorize_tfidf(texts, ngram=(1,2), min_df=5, max_df=0.5, stop_words=stop_union)
        n_comp = min(N_TOPICS_ALL, Xt.shape[1]-1, 100) if Xt.shape[1] > 1 else 1
        if n_comp >= 2:
            lsa, lsa_doc_topic = lsa_from_tfidf(Xt, n_topics=int(n_comp))
            # keine Konsole-Ausgabe nötig; nur zur internen Validierung
    except Exception as e:
        print(f"[LSA übersprungen] Grund: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()

