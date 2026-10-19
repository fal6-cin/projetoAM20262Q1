"""Classificador bayesiano por janela de Parzen, kernel gaussiano produto.

Conforme Projeto-AM-2026-2.pdf (Questão 2, item iii) e slides
Tecnicas-Nao-Parametricas.pdf (páginas 10--18):

   p_hat(x | c) = (1/n_c) * sum_{i:y_i=c} prod_{j=1}^{d}
                         [phi((x_j-x_ij)/h) / h],
   P_hat(c) = n_c/n,   P(c|x) ~ p_hat(x|c) P_hat(c).

phi é a densidade normal padrão. Usa-se uma única largura h > 0 para todas
as variáveis, escolha explícita dentro da família de kernels-produto.
A escolha de h ocorre exclusivamente na CV interna, nunca sobre o teste.

A implementação avalia os escores em logaritmos (log-sum-exp) para evitar
underflow da densidade em 57 dimensões, com blocos de consultas para não
materializar a matriz de todas as distâncias n_teste x n_treino.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted


class BayesParzen(ClassifierMixin, BaseEstimator):
    """Estimador não paramétrico bayesiano com janela Gaussiana multivariada produto.

    Parameters
    ----------
    largura_janela : float
        Desvio-padrão h da gaussiana univariada de cada atributo. Mesmo h
        para todos os 57 atributos no espaço escalonado Log1p+MinMax.
    tamanho_lote : int
        Quantidade de pontos de consulta processados de cada vez.
    """

    def __init__(self, largura_janela: float = 0.2, tamanho_lote: int = 128):
        self.largura_janela = largura_janela
        self.tamanho_lote = tamanho_lote

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError('y deve conter um rótulo por linha de X')
        if not np.isscalar(self.largura_janela) or isinstance(self.largura_janela, (bool, np.bool_)):
            raise ValueError('largura_janela deve ser escalar positivo')
        try:
            largura = float(self.largura_janela)
        except (TypeError, ValueError) as exc:
            raise ValueError('largura_janela deve ser numérica') from exc
        if not np.isfinite(largura) or largura <= 0:
            raise ValueError('largura_janela deve ser finita e > 0')
        if not isinstance(self.tamanho_lote, (int, np.integer)) or isinstance(self.tamanho_lote, (bool, np.bool_)) or self.tamanho_lote < 1:
            raise ValueError('tamanho_lote deve ser inteiro positivo')
        self.classes_, cod, n_por_classe = np.unique(y, return_inverse=True, return_counts=True)
        if len(self.classes_) < 2:
            raise ValueError('O classificador requer pelo menos duas classes')
        self.X_treino_ = np.ascontiguousarray(X.copy())
        self.codigo_classe_ = cod
        self.n_por_classe_ = n_por_classe
        self.prioris_ = n_por_classe / len(y)
        self.n_features_in_ = X.shape[1]
        return self

    def _validar_consultas(self, X):
        check_is_fitted(self, ['classes_', 'X_treino_', 'prioris_'])
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Esperadas {self.n_features_in_} características, recebidas {X.shape[1]}')
        return X

    def predict_log_joint(self, X):
        """log[p_hat(x|classe)*priori(classe)]; inclui normalização gaussiana.

        A normalização da densidade não influencia o argmax ou a posteriori,
        mas é conservada para garantir a correspondência matemática com
        o estimador de Parzen definido no enunciado.
        """
        X = self._validar_consultas(X)
        h = float(self.largura_janela)
        dimensao = self.n_features_in_
        constante = -dimensao * (np.log(h) + 0.5 * np.log(2 * np.pi))
        saida = np.empty((len(X), len(self.classes_)), dtype=float)
        for inicio in range(0, len(X), self.tamanho_lote):
            fim = min(inicio + self.tamanho_lote, len(X))
            distancia2 = cdist(X[inicio:fim], self.X_treino_, metric='sqeuclidean')
            log_kernel = (-0.5 / (h*h)) * distancia2
            for c in range(len(self.classes_)):
                densidade = logsumexp(log_kernel[:, self.codigo_classe_ == c], axis=1)
                densidade += constante - np.log(self.n_por_classe_[c])
                saida[inicio:fim, c] = densidade + np.log(self.prioris_[c])
        return saida

    def predict_log_proba(self, X):
        juntas = self.predict_log_joint(X)
        return juntas - logsumexp(juntas, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_joint(X), axis=1)]
