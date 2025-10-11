from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

def vectorize_counts(texts: List[str], ngram=(1,2), min_df=5, max_df=0.5,
                     max_features: Optional[int]=15000, stop_words=None):
    vec = CountVectorizer(ngram_range=ngram, min_df=min_df, max_df=max_df,
                          max_features=max_features, stop_words=stop_words)
    X = vec.fit_transform(texts)
    return vec, X

def vectorize_tfidf(texts: List[str], ngram=(1,2), min_df=5, max_df=0.6, stop_words=None):
    vec = TfidfVectorizer(ngram_range=ngram, min_df=min_df, max_df=max_df,
                          stop_words=stop_words)
    X = vec.fit_transform(texts)
    return vec, X

def lda_from_matrix(X, n_topics=8, max_iter=20, random_state=42, learning_decay=0.7):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="online",
        learning_decay=learning_decay,
        max_iter=max_iter,
        random_state=random_state,
        doc_topic_prior=None,  # sklearn wählt default 1/n_topics
        topic_word_prior=None
    )
    doc_topic = lda.fit_transform(X)
    return lda, doc_topic

def lsa_from_tfidf(X, n_topics=8, random_state=42, n_iter=10):
    svd = TruncatedSVD(n_components=n_topics, random_state=random_state, n_iter=n_iter)
    doc_topic = svd.fit_transform(X)
    return svd, doc_topic

def top_terms_per_topic(model, feature_names: List[str], topn=12) -> List[List[Tuple[str,float]]]:
    comps = model.components_
    topics = []
    for comp in comps:
        idx = np.argsort(comp)[::-1][:topn]
        topics.append([(feature_names[i], float(comp[i])) for i in idx])
    return topics

def pick_top_k_topics(doc_topic, k=5):
    """Wählt die k häufigsten Topics nach Gesamtgewicht (Summe über Dokumente)."""
    totals = doc_topic.sum(axis=0).ravel()
    order = np.argsort(totals)[::-1]
    return order[:k], totals

