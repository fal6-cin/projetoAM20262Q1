"""Voto majoritário dos quatro classificadores da Questão 2.

No empate, prefere-se entre as classes empatadas aquela com maior
P(classe) estimada APENAS nos dados de treino do fold externo. Se as
prioris também empatarem, adota-se a menor classe, determinístico.

As frações dos votos NÃO são probabilidades posteriores calibradas.
"""
from __future__ import annotations

import numpy as np

NUMERO_MODELOS = 4


def decidir_votos(predicoes, classes, prioris, numero_modelos=NUMERO_MODELOS):
    """Recebe matriz (modelos, observações), devolve preditos, contagens, empates.

    Contagens em (n_observacoes, n_classes); empates marca qualquer empate
    na MAIOR contagem (mesmo que a priori o resolva).
    """
    p = np.asarray(predicoes)
    classes = np.asarray(classes)
    prioris = np.asarray(prioris, dtype=float)
    if p.ndim != 2 or p.shape[0] != numero_modelos or p.shape[1] < 1:
        raise ValueError(f'Predições devem ter formato ({numero_modelos}, n_amostras), n_amostras>=1')
    if classes.ndim != 1 or classes.size < 2 or len(np.unique(classes)) != len(classes):
        raise ValueError('classes deve conter ao menos duas classes distintas')
    if prioris.shape != (len(classes),) or not np.isfinite(prioris).all() or (prioris < 0).any():
        raise ValueError('Prioris inválidas ou incompatíveis com as classes')
    if not np.isclose(np.sum(prioris), 1, atol=1e-10):
        raise ValueError('As probabilidades a priori devem somar 1')
    if not np.isin(p, classes).all():
        raise ValueError('Há previsões com classes desconhecidas')
    # Índices relativos, na mesma ordem fornecida por classes.
    contagens = np.stack([(p == classe).sum(axis=0) for classe in classes], axis=1)
    maxima = contagens.max(axis=1, keepdims=True)
    candidatas = contagens == maxima
    empates = candidatas.sum(axis=1) > 1
    # Desempate: maior priori entre as candidatas. Se igual, menor rótulo.
    # np.lexsort organiza por: 1) maior prior (chave primária) 2) menor classe.
    ordem = np.lexsort((classes, -prioris))
    candidatas_ordenadas = candidatas[:, ordem]
    escolhido = ordem[np.argmax(candidatas_ordenadas, axis=1)]
    return classes[escolhido], contagens.astype(int), empates.astype(bool)


class VotoMajoritario:
    """Combina quatro vetores de rótulos; estimativa de priori é por frequência.

    ``fit`` é feito com y_treino do fold externo, sem observar y_teste.
    ``predict`` recebe (4, n_amostras), nesta ordem de modelos:
    gaussiano, knn, parzen e regressão logística.
    """
    def fit(self, y_treino):
        y = np.asarray(y_treino)
        if y.ndim != 1 or not len(y):
            raise ValueError('y_treino deve ser vetor não vazio')
        self.classes_, counts = np.unique(y, return_counts=True)
        if len(self.classes_) < 2:
            raise ValueError('É necessário pelo menos duas classes no treino')
        self.prioris_ = counts.astype(float) / counts.sum()
        return self

    def prever_detalhes(self, predicoes):
        if not hasattr(self, 'prioris_'):
            raise RuntimeError('Execute fit(y_treino) antes da predição')
        return decidir_votos(predicoes, self.classes_, self.prioris_)

    def predict(self, predicoes):
        return self.prever_detalhes(predicoes)[0]

    def fracoes_votos(self, predicoes):
        """Frações de votos por classe (NÃO probabilidades posteriores)."""
        return self.prever_detalhes(predicoes)[1] / NUMERO_MODELOS
