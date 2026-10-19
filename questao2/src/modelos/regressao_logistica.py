"""Regressão logística binária/multiclasse com regularização L2.

A implementação delega a otimização convexa ao sklearn.linear_model.LogisticRegression
(solver LBFGS). Para >=3 classes, utiliza softmax multinomial, não treinamento
separado um-contra-todos. C é o inverso da força de regularização L2;
class_weight='balanced' é uma alternativa de ponderação avaliada somente por CV.

As probabilidades de classes são fornecidas por predict_proba.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


class RegressaoLogistica(ClassifierMixin, BaseEstimator):
    """Wrapper explícito, compatível com Pipeline/GridSearchCV do scikit-learn.

    Parameters
    ----------
    C : float
        Inverso da força da penalidade L2. Valores maiores = menos penalização.
    class_weight : None | 'balanced'
        Pesos uniformes ou inversamente proporcionais à frequência por classe.
    max_iter : int
        Máximo de iterações LBFGS.
    tol : float
        Tolerância de convergência do otimizador.
    """
    def __init__(self, C: float = 1.0, class_weight=None,
                 max_iter: int = 2000, tol: float = 1e-4):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError('X deve ser uma matriz e y um vetor com o mesmo número de amostras')
        if X.shape[0] == 0 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError('X deve ser não vazio e conter apenas valores finitos')
        if len(np.unique(y)) < 2:
            raise ValueError('O classificador requer ao menos duas classes')
        if isinstance(self.C, (bool, np.bool_)) or not isinstance(self.C, (int, float, np.integer, np.floating)) or not np.isfinite(self.C) or self.C <= 0:
            raise ValueError('C deve ser um número finito positivo')
        if self.class_weight not in (None, 'balanced'):
            raise ValueError("class_weight deve ser None ou 'balanced'")
        if isinstance(self.max_iter, (bool, np.bool_)) or not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError('max_iter deve ser inteiro positivo')
        if isinstance(self.tol, (bool, np.bool_)) or not isinstance(self.tol, (int, float, np.integer, np.floating)) or not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError('tol deve ser positivo e finito')
        self.estimador_ = LogisticRegression(
            C=float(self.C), class_weight=self.class_weight, solver='lbfgs',
            max_iter=int(self.max_iter), tol=float(self.tol),
            fit_intercept=True, random_state=2026,
        )
        self.estimador_.fit(X, y)
        self.classes_ = self.estimador_.classes_
        self.n_features_in_ = self.estimador_.n_features_in_
        self.n_iter_ = self.estimador_.n_iter_
        return self

    def _verificar_X(self, X):
        check_is_fitted(self, 'estimador_')
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError('X deve ter a mesma quantidade de atributos e apenas valores finitos')
        return X

    def predict(self, X):
        return self.estimador_.predict(self._verificar_X(X))

    def predict_proba(self, X):
        return self.estimador_.predict_proba(self._verificar_X(X))

    def decision_function(self, X):
        return self.estimador_.decision_function(self._verificar_X(X))
