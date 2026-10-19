"""Classificador de Bayes com densidade normal MULTIVARIADA por classe.

Para cada classe w estima P(w)=n_w/n, mu_w e Sigma_w (MLE, divisor n_w).
Devido à singularidade possível em Spambase, opcionalmente acrescenta
lambda * s_global * I à matriz Sigma_w estimada, com s_global definido APENAS
no conjunto de treino. Lambda=0 recupera o estimador ML sem regularização,
mas exige que todas as covariâncias sejam definidas positivas.

Isso NÃO é GaussianNB: as covariâncias podem ter termos fora da diagonal.
Avaliação feita no espaço de atributos pré-processados dentro de cada fold.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted


class BayesGaussianoMultivariado(ClassifierMixin, BaseEstimator):
    """Bayes gaussiano com matriz completa e regularização isotrópica relativa.

    Parameters
    ----------
    regularizacao_relativa : float, default=1e-4
        Fator não negativo. Matriz utilizada: Sigma_ML + lambda*s_global*I.
        s_global é a média das variâncias (ML) dos atributos do treino.
        lambda=0 tenta a versão estritamente ML e falha se Sigma for singular.
    """

    def __init__(self, regularizacao_relativa: float = 1e-4):
        self.regularizacao_relativa = regularizacao_relativa

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError('y deve ter uma classe por observação de X')
        if not np.isfinite(self.regularizacao_relativa) or self.regularizacao_relativa < 0:
            raise ValueError('regularizacao_relativa deve ser um número finito e não negativo')
        self.classes_, inv, contagens = np.unique(y, return_inverse=True, return_counts=True)
        if len(self.classes_) < 2:
            raise ValueError('São necessárias pelo menos duas classes')
        self.n_features_in_ = X.shape[1]
        self.priors_ = contagens / len(y)  # P(omega_i) de máxima verossimilhança
        self.media_ = []
        self.covariancia_ml_ = []
        self.covariancia_utilizada_ = []
        self.fator_cholesky_ = []
        self.logdet_ = []

        variancia_global_ml = np.mean(np.var(X, axis=0, ddof=0))
        if variancia_global_ml <= 0:
            raise ValueError('Os dados de treino não possuem variabilidade global')
        self.escala_regularizacao_ = float(variancia_global_ml)
        self.regularizacao_absoluta_ = float(self.regularizacao_relativa * variancia_global_ml)
        I = np.eye(self.n_features_in_)

        for c in range(len(self.classes_)):
            Xc = X[inv == c]
            media = Xc.mean(axis=0)
            centrado = Xc - media
            cov_ml = centrado.T @ centrado / len(Xc)  # MLE, e NÃO n_c-1
            cov = cov_ml + self.regularizacao_absoluta_ * I
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as e:
                raise ValueError(
                    f'Covariância não definida positiva na classe {self.classes_[c]!r}. '
                    'Aumente regularizacao_relativa (> 0) ou revise os atributos.'
                ) from e
            self.media_.append(media)
            self.covariancia_ml_.append(cov_ml)
            self.covariancia_utilizada_.append(cov)
            self.fator_cholesky_.append(chol)
            self.logdet_.append(2.0 * np.log(np.diag(chol)).sum())
        self.media_ = np.stack(self.media_)
        self.covariancia_ml_ = np.stack(self.covariancia_ml_)
        self.covariancia_utilizada_ = np.stack(self.covariancia_utilizada_)
        self.logdet_ = np.asarray(self.logdet_)
        return self

    def log_prob_conjunta(self, X):
        """Retorna log[p(x|omega_c)P(omega_c)] de cada classe."""
        check_is_fitted(self, ['classes_', 'media_', 'fator_cholesky_', 'priors_'])
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Esperadas {self.n_features_in_} variáveis; recebidas {X.shape[1]}')
        p = self.n_features_in_
        saida = np.empty((len(X), len(self.classes_)), dtype=float)
        for c, L in enumerate(self.fator_cholesky_):
            centrado = (X - self.media_[c]).T
            z = solve_triangular(L, centrado, lower=True, check_finite=False)
            dist2 = np.sum(z ** 2, axis=0)
            saida[:, c] = -0.5 * (p * np.log(2 * np.pi) + self.logdet_[c] + dist2) + np.log(self.priors_[c])
        return saida

    def predict_log_proba(self, X):
        scores = self.log_prob_conjunta(X)
        return scores - logsumexp(scores, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.log_prob_conjunta(X), axis=1)]
