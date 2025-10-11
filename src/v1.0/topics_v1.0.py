from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

def vectorize_tfidf(texts: List[str], ngram=(1,2), min_df=5, max_df=0.6, stop_words=None):
    vec = TfidfVectorizer(ngram_range=ngram, min_df=min_df, max_df=max_df,
                          stop_words=stop_words)
    X = vec.fit_transform(texts)
    return vec, X

def lda_from_matrix(X, n_topics=5, max_iter=10, random_state=42):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="online",
        max_iter=max_iter,
        random_state=random_state
    )
    doc_topic = lda.fit_transform(X)
    return lda, doc_topic

def lsa_from_tfidf(X, n_topics=5, random_state=42):
    svd = TruncatedSVD(n_components=n_topics, random_state=random_state)
    doc_topic = svd.fit_transform(X)
    return svd, doc_topic

def top_terms_per_topic(model, feature_names: List[str], topn=10) -> List[List[Tuple[str,float]]]:
    comps = getattr(model, "components_", None)
    if comps is None:
        # TruncatedSVD hat components_ ebenfalls
        comps = model.components_
    topics = []
    for comp in comps:
        idx = np.argsort(comp)[::-1][:topn]
        topics.append([(feature_names[i], float(comp[i])) for i in idx])
    return topics

