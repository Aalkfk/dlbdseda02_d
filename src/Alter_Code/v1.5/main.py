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
# LSA-Vergleichsausgabe zusätzlich zu LDA
RUN_LSA_COMPARE = True

# Welche Flairs sollen für das Topic Modeling verwendet werden?
ALLOWED_FLAIRS = [
    "Diskussion", "News", "Events", "Looking for...", "Frage / Advice", "Sonstiges"
]

# Mindestlänge (Tokens) pro Dokument, damit es in die Topic-Analyse eingeht
MIN_TOKENS_FOR_TOPIC = 30

# Kandidaten für die Anzahl der Topics (K) und Seeds für robustere Ergebnisse
CANDIDATE_K = [6, 8, 10, 12]
SEEDS = (13, 21, 42, 77)

# Wie stark sollen Titel-Token zusätzlich gewichtet werden?
# 0 = aus, 1..3 = moderat, >3 = stark
TITLE_BOOST = 2


def _build_texts_with_title_boost(df: pd.DataFrame, title_boost: int) -> list[str]:
    """
    Nimmt df['clean_for_vect'] als Basis und hängt bereinigte Titel-Token
    'title_boost'-mal an. So werden klare Titelbegriffe stärker berücksichtigt.
    """
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
    """
    Wendet die gleiche Normalisierung an wie unsere Texte (clean_text + tokenize),
    damit es keine Inkonsistenzen wie 'hä?' vs. 'hä' gibt.
    """
    norm = set()
    for w in stopset:
        # zerlegen in evtl. mehrere Tokens (z. B. bei zusammengesetzten Einträgen)
        toks = tokenize(clean_text(w))
        norm.update(toks)
    # sehr kurze Tokens (Sicherheitsfilter)
    norm = {t for t in norm if len(t) >= 2}
    return sorted(norm)


def _choose_df_params(n_docs: int, base_min_df_abs: int, base_min_df_frac: float, base_max_df_frac: float):
    """
    Wählt konsistente min_df / max_df-Werte abhängig von n_docs.
    - min_df beginnt als max(abs, frac*n_docs)
    - wenn min_df >= max_df*n_docs, wird min_df abgesenkt, damit sklearn keinen Fehler wirft
    """
    # Ausgangswerte
    min_df_val = max(base_min_df_abs, int(math.ceil(base_min_df_frac * n_docs)))
    max_df_val = base_max_df_frac  # als Anteil belassen

    # Konfliktprüfung: min_df (int) muss < max_df*n_docs sein
    max_docs_allowed = max_df_val * n_docs
    if max_docs_allowed <= 1:
        # bei extrem kleinem Korpus: max_df etwas großzügiger machen
        max_df_val = min(0.9, max(base_max_df_frac, 2.0 / max(n_docs, 1)))
        max_docs_allowed = max_df_val * n_docs

    if min_df_val >= max_docs_allowed:
        # min_df so reduzieren, dass er sicher darunter liegt
        min_df_val = max(2, int(max_docs_allowed) - 1)

    return min_df_val, max_df_val


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

    # 2) Flairs & aktivste User (auf kompletter RAW-Basis)
    flairs = extract_flairs(df_raw).head(20)
    active_users = (df_raw["author"]
                    .dropna()
                    .loc[lambda s: s.ne("[deleted]") & s.ne("None")]
                    .value_counts()
                    .head(20))
    flairs.to_csv(DATA_DIR / "top_flairs.csv")
    active_users.to_csv(DATA_DIR / "top_users.csv")

    # 3) Facettierung nach Flair -> NUR diese Flairs für Topic Modeling verwenden
    if "flair_text" in df_raw.columns:
        df_model_raw = df_raw[df_raw["flair_text"].isin(ALLOWED_FLAIRS)].reset_index(drop=True)
        if df_model_raw.empty:
            print("[Hinweis] Kein Beitrag mit den erlaubten Flairs gefunden – verwende gesamten Korpus.", file=sys.stderr)
            df_model_raw = df_raw.copy()
    else:
        df_model_raw = df_raw.copy()

    # 4) Vorverarbeitung (nur auf dem gefilterten Korpus für Topics), Kommentare raus
    df = add_clean_columns(df_model_raw, use_stemming=False, filter_english=True, include_comments=False)

    # Optional: ultrakurze Dokumente ausschließen
    if "tokens" in df.columns:
        df = df[df["tokens"].map(len) >= MIN_TOKENS_FOR_TOPIC].reset_index(drop=True)

    df.to_csv(DATA_DIR / "clean_r_stuttgart_topics_subset.csv", index=False)

    if df.empty:
        print("[Abbruch] Keine Dokumente nach Vorverarbeitung/Filter. Parameter anpassen (Flairs/MinTokens).", file=sys.stderr)
        return

    # 5) Texte für die Vektorisierung bauen (mit optionalem Titel-Boosting)
    texts = _build_texts_with_title_boost(df, TITLE_BOOST)
    n_docs = len(texts)

    # Stopwörter normieren, damit 'hä?' -> 'hä' & Co. keine Warnung auslösen
    stop_union_norm = _normalize_stopwords(GERMAN_SW | ENGLISH_SW)  # <- fix für UserWarning

    # 6) min_df / max_df konsistent wählen (verhindert den ValueError)
    min_df_val, max_df_val = _choose_df_params(
        n_docs=n_docs,
        base_min_df_abs=15,      # deine Basis: mind. 15
        base_min_df_frac=0.018,  # oder ~1,8% der Dokumente
        base_max_df_frac=0.20    # höchstens 20% der Dokumente
    )
    print(f"[Info] n_docs={n_docs}, min_df={min_df_val}, max_df={max_df_val:.2f}")

    # 7) Themen: LDA auf Count-Features (besser als TF-IDF für LDA)
    vec_counts, Xc = vectorize_counts(
        texts,
        ngram=(1, 2),
        min_df=min_df_val,
        max_df=max_df_val,
        max_features=9000,
        stop_words=stop_union_norm
    )

    if Xc.shape[1] == 0:
        print("[Abbruch] Keine Features nach Vorverarbeitung. min_df/max_df reduzieren oder Stopwörter prüfen.", file=sys.stderr)
        return

    # 8) Bestes K via Kohärenz (und mehrere Seeds testen)
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

    # 9) Ausgabe (LDA)
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
    print(f"[Info] Gewähltes K={best['k']} (Seed {best['seed']}, Coherence {best['coh']:.3f}, TitleBoost={TITLE_BOOST})")
    for rank, t_id in enumerate(top_idx, start=1):
        terms = ", ".join([w for w, _ in lda_topics_all[t_id]])
        print(f"Topic {rank} (ID {t_id}): {terms}")

    # je berichtetem Thema 3 Beispiel-Posts mit höchster Zugehörigkeit
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

    # 10) (Bonus) LSA als Vergleichsausgabe
    if RUN_LSA_COMPARE:
        try:
            vec_tfidf, Xt = vectorize_tfidf(
                texts,
                ngram=(1, 2),
                min_df=min_df_val,   # gleiche, konsistente Schwellen nutzen
                max_df=max_df_val,
                stop_words=stop_union_norm
            )
            n_comp = min(best['k'], max(2, Xt.shape[1] - 1), 100)
            if n_comp >= 2:
                lsa_model, lsa_dt = lsa_from_tfidf(Xt, n_topics=int(n_comp))
                lsa_topics_all = top_terms_per_topic(lsa_model, vec_tfidf.get_feature_names_out(), topn=12)
                lsa_top_idx, _ = pick_top_k_topics(lsa_dt, k=5)

                print("\n=== (Vergleich) LSA — Top-Terme ===")
                for rank, t_id in enumerate(lsa_top_idx, start=1):
                    terms = ", ".join([w for w, _ in lsa_topics_all[t_id]])
                    print(f"LSA Topic {rank} (ID {t_id}): {terms}")
        except Exception as e:
            print(f"[LSA-Vergleich übersprungen] Grund: {e}", file=sys.stderr)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
