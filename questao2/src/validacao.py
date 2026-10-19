"""Geração reprodutível de divisões de validação cruzada.

A validação EXTERNA é estratificada 30 x 10-folds; o ajuste de
hiperparâmetros (quando necessário) usa 5-folds INTERNA ao treino externo.
Não ajustar escalonadores antes do split ou utilizar o fold de teste para tuning.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold


def gerar_folds_externos(y, repeticoes: int = 30, folds: int = 10, semente: int = 2026):
    """Produz (repeticao, fold, indices_treino, indices_teste) sem armazená-los."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError('y deve ser um vetor de classes')
    valores, frequencias = np.unique(y, return_counts=True)
    if len(valores) < 2 or min(frequencias) < folds:
        raise ValueError('Não há exemplos suficientes de cada classe para 10-folds estratificados')
    for r in range(repeticoes):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=semente+r)
        for f, (treino, teste) in enumerate(cv.split(np.zeros(len(y)), y),start=1):
            yield r+1, f, treino, teste


def gerar_folds_internos(y_treino, folds: int = 5, semente: int = 2026):
    """Produz índices RELATIVOS ao conjunto de treino externo, nunca ao global."""
    y_treino = np.asarray(y_treino)
    valores, frequencias = np.unique(y_treino, return_counts=True)
    if len(valores) < 2 or min(frequencias) < folds:
        raise ValueError('Treino externo insuficiente para a validação interna estratificada')
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=semente)
    for treino_rel, validacao_rel in cv.split(np.zeros(len(y_treino)), y_treino):
        yield treino_rel, validacao_rel


def conferir_folds(y, repeticoes: int = 30, folds: int = 10, semente: int = 2026):
    """Audita ausência de sobreposição, cobertura e contagens por repetição."""
    y=np.asarray(y)
    acumulado=[]
    anterior=0
    for rep, fold, tr, te in gerar_folds_externos(y,repeticoes,folds,semente):
        if rep!=anterior:
            if acumulado:
                esperado=np.arange(len(y))
                vistos=np.concatenate(acumulado)
                assert len(vistos)==len(y) and np.array_equal(np.sort(vistos),esperado)
            acumulado=[]
            anterior=rep
        if np.intersect1d(tr,te).size:
            raise AssertionError('Vazamento: índice presente em treino e teste')
        if len(np.unique(y[tr])) != len(np.unique(y)) or len(np.unique(y[te])) != len(np.unique(y)):
            raise AssertionError('Classe ausente no treino ou teste')
        acumulado.append(te)
    vistos=np.concatenate(acumulado)
    assert len(vistos)==len(y) and np.array_equal(np.sort(vistos),np.arange(len(y)))
    return {'repeticoes':repeticoes,'folds_por_repeticao':folds,'total_divisoes':repeticoes*folds,'indice_minimo_classe':int(np.min(np.unique(y,return_counts=True)[1]))}
