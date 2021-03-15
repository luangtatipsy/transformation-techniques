import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing import preprocess
from tokenizer import tokenize


class ThaiPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def preprocess(self, text: str) -> str:
        return preprocess(text)

    def transform(self, X) -> pd.Series:
        return pd.Series(X).apply(self.preprocess)


class ThaiTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_placeholders: bool = False, min_char: bool = 2):
        self.remove_placeholders = remove_placeholders
        self.min_char = min_char

    def fit(self, X, y=None, **fit_params):
        return self

    def tokenize(self, text):
        tokens = tokenize(
            text,
            min_char=self.min_char,
            remove_placeholder=self.remove_placeholders,
        )

        return tokens

    def transform(self, X) -> pd.Series:
        return pd.Series(X).apply(self.tokenize)


class MeanEmbeddingVectorizer:
    def __init__(self, w2v_model):
        self.word2vec = dict(zip(w2v_model.wv.index2word, w2v_model.wv.syn0))
        self.dim = w2v_model.vector_size

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X) -> np.ndarray:
        return np.array(
            [
                np.mean(
                    [self.word2vec[word] for word in words if word in self.word2vec]
                    or [np.zeros(self.dim)],
                    axis=0,
                )
                for words in X
            ]
        )
