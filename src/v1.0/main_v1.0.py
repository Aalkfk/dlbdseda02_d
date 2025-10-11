from __future__ import annotations
import os
import pandas as pd
from collections import Counter
from .fetch import fetch_subreddit_posts
from .preprocess import add_clean_columns, extract_hashtags, GERMAN_SW
from .topics import vectorize_tfidf, lda_from_matrix, lsa_from_tfidf, top_terms_per_topic
from .utils import DATA_DIR

def main():
    # 1) Daten abrufen
    df = fetch_subreddit_posts(
        subreddit_name="Stuttgart",
        where="top",
        time_filter="year",
        limit=500,
        with_comments=True,
        max_comments_per_post=50
    )

    # 2) Vorverarbeitung
    df = add_clean_columns(df)
    df.to_csv(DATA_DIR / "clean_r_stuttgart.csv", index=False)

    # 3) Entitätsanalyse: Hashtags & aktivste User
    hashtags = extract_hashtags(df).head(20)
    active_users = (df["author"].dropna()
                           .loc[lambda s: s.ne("[deleted]") & s.ne("None")]
                           .value_counts()
                           .head(20))
    hashtags.to_csv(DATA_DIR / "top_hashtags.csv")
    active_users.to_csv(DATA_DIR / "top_users.csv")

    # 4) Themenextraktion (LDA, wie im Skript gezeigt – auf TF-IDF)
    texts = df["clean"].tolist()
    vec, X = vectorize_tfidf(texts, ngram=(1,2), min_df=5, max_df=0.6, stop_words=list(GERMAN_SW))
    lda, doc_topic = lda_from_matrix(X, n_topics=5, max_iter=10)

    feature_names = vec.get_feature_names_out()
    lda_topics = top_terms_per_topic(lda, feature_names, topn=12)

    # Optional: LSA als Gegencheck (aus Kursbuch)
    lsa, lsa_doc_topic = lsa_from_tfidf(X, n_topics=5)

    # 5) Ergebnisse drucken
    print("\n=== Top 20 Hashtags ===")
    print(hashtags.to_string())

    print("\n=== Top 20 aktivste User ===")
    print(active_users.to_string())

    print("\n=== 5 Themen (LDA) — jeweils Top-Terme ===")
    for k, topic in enumerate(lda_topics, start=1):
        terms = ", ".join([w for w,_ in topic])
        print(f"Topic {k}: {terms}")

    # je Thema 3 beispielhafte Posts mit höchster Zuordnung
    topic_assign = doc_topic.argmax(axis=1)
    out_rows = []
    for k in range(5):
        idx = (topic_assign == k)
        top_idx = idx.nonzero()[0][:3]
        for i in top_idx:
            out_rows.append({
                "topic": k+1,
                "post_id": df.iloc[i]["id"],
                "title": df.iloc[i]["title"],
                "permalink": df.iloc[i]["permalink"]
            })
    pd.DataFrame(out_rows).to_csv(DATA_DIR / "sample_posts_per_topic.csv", index=False)

if __name__ == "__main__":
    # Optional: .env laden (REDDIT_CLIENT_ID/SECRET/AGENT)
    from dotenv import load_dotenv
    load_dotenv()
    main()

