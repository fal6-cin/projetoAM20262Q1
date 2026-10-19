"""Classificador Bayesiano por k-vizinhos (proporção de classes na vizinhança).

Para cada observação x, estima P(omega_c | x) = k_c/k, onde k_c é a
quantidade de vizinhos da classe c dentre os k mais próximos. Isso é a
regra bayesiana por k-vizinhos apresentada em Técnicas Não Paramétricas
(páginas 23--29), e NÃO utiliza densidade gaussiana nem votação ponderada.

Distâncias avaliadas: Euclidiana (L2), City-Block/Manhattan (L1),
Chebyshev (L-infinito), conforme enunciado.

Empates no máximo das proporções: remover, só para a DECISÃO, um vizinho
mais distante por vez até surgir vencedor único (regra discutida nos slides
de Instance-Based Learning). predict_proba conserva as frações originais.
Para empates exatos de distância, a ordem é por índice de treinamento.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

_METRICAS = {'euclidiana': 'euclidean', 'city_block': 'cityblock', 'chebyshev': 'chebyshev'}


class BayesKVizinhos(ClassifierMixin, BaseEstimator):
    """Bayes k-NN com priori posterior empírica por vizinhança.

    Parameters
    ----------
    n_vizinhos : int, default=5
        Número de vizinhos, <= tamanho do treino.
    distancia : {'euclidiana','city_block','chebyshev'}
        Métrica para selecionar os vizinhos; não usa distâncias para ponderação.
    tamanho_lote : int, default=128
        Número máximo de observações avaliadas por bloco, evitando a matriz
        completa N_teste x N_treino na memória.
    """
    def __init__(self, n_vizinhos: int = 5, distancia: str = 'euclidiana', tamanho_lote: int = 128):
        self.n_vizinhos = n_vizinhos
        self.distancia = distancia
        self.tamanho_lote = tamanho_lote

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        y = np.asarray(y)
        if y.ndim != 1 or y.size != len(X):
            raise ValueError('y deve ser um vetor com um rótulo por observação')
        if not isinstance(self.n_vizinhos, (int, np.integer)) or isinstance(self.n_vizinhos, (bool, np.bool_)) or not (1 <= self.n_vizinhos <= len(X)):
            raise ValueError('n_vizinhos deve ser inteiro entre 1 e n_treino')
        if self.distancia not in _METRICAS:
            raise ValueError(f'distancia inválida; use {tuple(_METRICAS)}')
        if not isinstance(self.tamanho_lote, (int, np.integer)) or isinstance(self.tamanho_lote, (bool, np.bool_)) or self.tamanho_lote < 1:
            raise ValueError('tamanho_lote deve ser inteiro positivo')
        self.classes_, y_cod, contagens = np.unique(y, return_inverse=True, return_counts=True)
        if len(self.classes_) < 2:
            raise ValueError('São necessárias pelo menos duas classes')
        self.n_features_in_ = X.shape[1]
        self.X_treino_ = np.ascontiguousarray(X.copy())
        self.y_treino_ = y.copy()
        self.y_codificado_ = y_cod
        self.prioris_ = contagens / len(y)
        return self

    def indices_vizinhos(self, X, k: int | None = None):
        """Devolve índices [n_consultas,k] em ordem de distância crescente.

        Havendo empate na distância, prioriza o menor índice no treino.
        A operação é somente leitura, sem alterar o classificador.
        """
        check_is_fitted(self, ['X_treino_', 'y_codificado_', 'classes_'])
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Esperadas {self.n_features_in_} variáveis; recebidas {X.shape[1]}')
        if k is None:
            k = self.n_vizinhos
        if not isinstance(k, (int, np.integer)) or not (1 <= k <= len(self.X_treino_)):
            raise ValueError('k deve ser inteiro entre 1 e n_treino')
        indices = np.empty((len(X), k), dtype=np.intp)
        for inicio in range(0, len(X), self.tamanho_lote):
            fim = min(inicio + self.tamanho_lote, len(X))
            d = cdist(X[inicio:fim], self.X_treino_, metric=_METRICAS[self.distancia])
            # np.argsort estável: índice menor em distâncias idênticas.
            indices[inicio:fim] = np.argsort(d, axis=1, kind='stable')[:, :k]
        return indices

    @staticmethod
    def _vencedores_com_desempate(codigos_vizinhos, n_classes: int):
        """Prediz por pluralidade, removendo o mais distante quando há empate."""
        codigos_vizinhos = np.asarray(codigos_vizinhos, dtype=np.intp)
        n, k = codigos_vizinhos.shape
        contagens = np.zeros((n, n_classes), dtype=np.int32)
        linhas = np.arange(n)
        for pos in range(k):
            np.add.at(contagens, (linhas, codigos_vizinhos[:, pos]), 1)
        vencedor = np.empty(n, dtype=np.intp)
        pendente = np.ones(n, dtype=bool)
        for pos in range(k - 1, -1, -1):
            if not np.any(pendente):
                break
            maximo = contagens.max(axis=1)
            empate = (contagens == maximo[:, None]).sum(axis=1) > 1
            resolvido = pendente & ~empate
            vencedor[resolvido] = np.argmax(contagens[resolvido], axis=1)
            pendente &= empate
            if pos > 0 and np.any(pendente):
                idx = np.flatnonzero(pendente)
                contagens[idx, codigos_vizinhos[idx, pos]] -= 1
        if np.any(pendente):
            raise RuntimeError('Desempate não terminou, mesmo com um vizinho')
        return vencedor

    def predict_proba(self, X):
        idx = self.indices_vizinhos(X)
        cod = self.y_codificado_[idx]
        proporcoes = np.zeros((len(idx), len(self.classes_)), dtype=float)
        for c in range(len(self.classes_)):
            proporcoes[:, c] = np.mean(cod == c, axis=1)
        return proporcoes

    def predict(self, X):
        idx = self.indices_vizinhos(X)
        cod = self.y_codificado_[idx]
        return self.classes_[self._vencedores_com_desempate(cod, len(self.classes_))]
