# topics.py
# =========
# Zentrale Utilities fürs Vektorisieren (Count/TF-IDF), LDA/LSA-Modellierung
# sowie für die Ableitung von Top-Termen und einer einfachen Kohärenzmetrik.
#
# Design-Ziele:
# - Schlanke, klar benannte Funktionen mit stabilen Signaturen
# - Keine Seiteneffekte (Rückgaben statt globaler Zustände)
# - Kompatibel zu main.py: fit_best_lda(...) liefert (model, doc_topic, k, seed, coh)

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorize_counts(
    texts: List[str],
    ngram: Tuple[int, int] = (1, 2),
    min_df: int | float = 5,
    max_df: float = 0.3,
    max_features: Optional[int] = 12000,
    stop_words: Optional[List[str] | set[str]] = None,
):
    """Erzeugt eine Count-Matrix mit n-Grammen (standardmäßig Uni- und Bigrams).

    Hinweise:
      - `binary=True` (Term-Präsenz statt Häufigkeit) harmoniert gut mit LDA in
        noisigen Social-Media-Texten.
      - `min_df`/`max_df` dürfen absolut (int) oder relativ (float) sein.

    Args:
      texts: Liste vorbereiteter Dokumente (bereits gereinigt/tokenisiert/zusammengeführt).
      ngram: Untere/obere Grenze für n-Gram-Länge.
      min_df: Minimale Dokumentfrequenz (int: absolut, float: Anteil).
      max_df: Maximale Dokumentfrequenz (float: Anteil).
      max_features: Optionales Feature-Cap (None = unbeschränkt).
      stop_words: Zusätzliche Stoppwortliste (bereits normalisiert).

    Returns:
      (vectorizer, X) mit CountVectorizer und CSR-Matrix (Dokument x Term).
    """
    vec = CountVectorizer(
        ngram_range=ngram,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words=stop_words,
        binary=True,             # Präsenz genügt, reduziert Rauschen
    )
    X = vec.fit_transform(texts)
    return vec, X


def vectorize_tfidf(
    texts: List[str],
    ngram: Tuple[int, int] = (1, 2),
    min_df: int | float = 5,
    max_df: float = 0.6,
    stop_words: Optional[List[str] | set[str]] = None,
):
    """TF-IDF-Vektorisierung (z. B. als Basis für LSA/TruncatedSVD).

    Args:
      texts: Liste Dokumente.
      ngram: n-Gram-Fenster.
      min_df: min. Dokumentfrequenz.
      max_df: max. Dokumentfrequenz (Anteil).
      stop_words: Stoppwortliste.

    Returns:
      (vectorizer, X) mit TfidfVectorizer und TF-IDF-Matrix (CSR).
    """
    vec = TfidfVectorizer(
        ngram_range=ngram,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
    )
    X = vec.fit_transform(texts)
    return vec, X


def lda_from_matrix(
    X,
    n_topics: int = 8,
    max_iter: int = 60,
    random_state: int = 42,
    learning_decay: float = 0.7,
):
    """Trainiert ein LDA auf einer Count-Matrix.

    Tuning:
      - `doc_topic_prior` (α) etwas >0.1 fördert „gemischte“ Dokumente.
      - `topic_word_prior` (η) klein hält Topics sparsamer.

    Args:
      X: Sparse Count-Matrix (CSR/CSC) mit Form [n_docs, n_terms].
      n_topics: Anzahl Topics.
      max_iter: Maximale Iterationen (Batch-LDA).
      random_state: Seed für Reproduzierbarkeit.
      learning_decay: Decay-Parameter (bei batch geringer Effekt).

    Returns:
      (lda_model, doc_topic) wobei doc_topic die Dokument-Topic-Verteilung ist.
    """
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        learning_decay=learning_decay,
        max_iter=max_iter,
        random_state=random_state,
        doc_topic_prior=0.5,
        topic_word_prior=0.05,
    )
    doc_topic = lda.fit_transform(X)
    return lda, doc_topic


def lsa_from_tfidf(
    X,
    n_topics: int = 8,
    random_state: int = 42,
    n_iter: int = 10,
):
    """LSA (TruncatedSVD) auf TF-IDF-Matrix (oder anderen reellen Matrizen).

    Args:
      X: TF-IDF-Matrix (CSR/CSC) [n_docs, n_terms].
      n_topics: Ziel-Rang (Anzahl LSA-„Topics“).
      random_state: Seed.
      n_iter: Power-Iterations (Stabilität/Genauigkeit vs. Laufzeit).

    Returns:
      (svd_model, doc_topic) mit SVD-Objekt und dokument-seitigen Komponenten.
    """
    svd = TruncatedSVD(n_components=n_topics, random_state=random_state, n_iter=n_iter)
    doc_topic = svd.fit_transform(X)
    return svd, doc_topic


def top_terms_per_topic(
    model,
    feature_names: List[str],
    topn: int = 12,
) -> List[List[Tuple[str, float]]]:
    """Gibt pro Topic die Top-Terme (Wort, Gewicht).

    Works for:
      - LDA: `model.components_` = Topic-Wort-Verteilungen (ungewichtet positiv).
      - LSA/SVD: `components_` sind auch positiv/negativ möglich; hier rein betragsbasiert
        sortieren **nicht**, sondern wie übergeben – für LDA ausreichend.

    Args:
      model: LDA/LSA-Modell mit `components_`.
      feature_names: Vokabular in Vektorizer-Reihenfolge.
      topn: Anzahl Wörter pro Topic.

    Returns:
      Liste pro Topic: [(term, score), ...].
    """
    comps = model.components_
    topics: List[List[Tuple[str, float]]] = []
    for comp in comps:
        idx = np.argsort(comp)[::-1][:topn]
        topics.append([(feature_names[i], float(comp[i])) for i in idx])
    return topics


def pick_top_k_topics(doc_topic, k: int = 5):
    """Wählt die `k` stärksten Topics (Summen-Gewicht über alle Dokumente).

    Args:
      doc_topic: Dokument-Topic-Matrix (z. B. aus LDA/LSA), Form [n_docs, n_topics].
      k: Anzahl gewünschter Top-Topics.

    Returns:
      (top_indices, totals) mit Indizes der Top-Topics und deren Gesamtsummen.
    """
    totals = doc_topic.sum(axis=0).ravel()
    order = np.argsort(totals)[::-1]
    return order[:k], totals


def umass_coherence(model, Xc, topn: int = 10) -> float:
    """Einfache UMass-ähnliche Kohärenz (höher = konsistenter).

    Implementationsskizze:
      - Binarisiert X (Term vorkommend ja/nein).
      - Für die Top-`topn` Terme je Topic werden alle (wi, wj)-Paare betrachtet.
      - Score ≈ Mittelwert log((D(wi,wj)+eps)/D(wj)).

    Achtung:
      - Kann negativ sein (üblich bei kurzen, noisigen Social-Media-Texten).
      - Rein intern/konsistent genutzt (Modelle vergleichbar halten).

    Args:
      model: LDA-Modell mit `components_`.
      Xc: Count-Matrix (CSR/CSC).
      topn: Anzahl Top-Terme je Topic.

    Returns:
      Skalarer Kohärenzwert (größer = besser, oft < 0 in Praxis).
    """
    Xb = Xc.copy().tocsr()
    Xb.data[:] = 1  # binäre Präsenz
    df = np.asarray(Xb.sum(axis=0)).ravel()  # D(w)
    comps = model.components_
    eps = 1.0

    topic_scores = []
    for comp in comps:
        idx = np.argsort(comp)[::-1][:topn]
        sc = 0.0
        pairs = 0
        for i in range(1, len(idx)):
            wi = idx[i]
            col_i = Xb[:, wi]
            for j in range(i):
                wj = idx[j]
                col_j = Xb[:, wj]
                co = col_i.multiply(col_j).sum()  # D(wi, wj)
                denom = max(df[wj], 1.0)
                sc += np.log((co + eps) / denom)
                pairs += 1
        topic_scores.append(sc / max(pairs, 1))
    return float(np.mean(topic_scores))


def fit_best_lda(
    X,
    candidates_k: List[int],
    seeds: List[int] | Tuple[int, ...],
    max_iter: int = 60,
    topn: int = 12,
):
    """Grid-Search über (K, Seed) nach bester UMass-Kohärenz.

    Nutzung:
      Wird aus main.py aufgerufen und liefert genau die 5 erwarteten Rückgaben.

    Args:
      X: Count-Matrix (CSR/CSC).
      candidates_k: Kandidaten für die Topic-Zahl (z. B. [4, 5, 6]).
      seeds: Seeds für Re-Runs (z. B. (13, 21, 42, 77)).
      max_iter: Iterationen für LDA.
      topn: Top-Terme je Topic für Kohärenzschätzung.

    Returns:
      (best_model, best_doc_topic, best_k, best_seed, best_coh)
    """
    best = {"model": None, "dt": None, "k": None, "seed": None, "coh": -1e9}
    for k in candidates_k:
        for seed in seeds:
            lda, dt = lda_from_matrix(X, n_topics=k, max_iter=max_iter, random_state=seed)
            coh = umass_coherence(lda, X, topn=topn)
            if coh > best["coh"]:
                best.update({"model": lda, "dt": dt, "k": k, "seed": seed, "coh": coh})
    return best["model"], best["dt"], best["k"], best["seed"], best["coh"]
