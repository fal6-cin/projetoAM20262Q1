"""CV aninhada da regressão logística, para os dois alvos da Spambase.

A escolha de C e da ponderação de classes usa exclusivamente 5 folds internos
estratificados do treino externo. Log1p e MinMax são ajustados dentro de
cada Pipeline; o fold de teste externo nunca participa do ajuste.
A regressão multiclasse otimiza diretamente a perda logarítmica softmax.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from .metricas import calcular_metricas
from .modelos.regressao_logistica import RegressaoLogistica
from .validacao import gerar_folds_externos

VALORES_C = (0.03, 0.3, 3.0, 30.0, 300.0)
PESOS_CLASSES = (None, 'balanced')


def pipeline_logistica(C: float = 1., class_weight=None) -> Pipeline:
    return Pipeline([
        ('log1p', FunctionTransformer(np.log1p, validate=True)),
        ('minmax', MinMaxScaler()),
        ('logistica', RegressaoLogistica(C=C, class_weight=class_weight)),
    ])


def ajustar_logistica(X_treino, y_treino, semente: int,
                      valores_C=VALORES_C, pesos_classes=PESOS_CLASSES, repeticao_inicial: int = 1):
    X_treino = np.asarray(X_treino, dtype=float)
    y_treino = np.asarray(y_treino)
    if X_treino.ndim != 2 or y_treino.ndim != 1 or len(X_treino) != len(y_treino):
        raise ValueError('Dimensões de X_treino/y_treino incompatíveis')
    if not np.isfinite(X_treino).all() or (X_treino < 0).any():
        raise ValueError('Log1p exige atributos não negativos e finitos')
    if np.unique(y_treino, return_counts=True)[1].min() < 5:
        raise ValueError('São necessários pelo menos 5 exemplos por classe nos 5 folds internos')
    valores_C = tuple(dict.fromkeys(valores_C))
    pesos_classes = tuple(dict.fromkeys(pesos_classes))
    if not valores_C or not pesos_classes:
        raise ValueError('Grades de parâmetros devem ser não vazias')
    for C in valores_C:
        if isinstance(C, (bool, np.bool_)) or not isinstance(C, (int, float, np.integer, np.floating)) or not np.isfinite(C) or C <= 0:
            raise ValueError('C deve ser positivo e finito')
    if any(peso not in (None, 'balanced') for peso in pesos_classes):
        raise ValueError('Pesos permitidos: None e balanced')
    cv_interna = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)
    busca = GridSearchCV(
        estimator=pipeline_logistica(),
        param_grid={'logistica__C': [float(c) for c in valores_C],
                    'logistica__class_weight': list(pesos_classes)},
        cv=cv_interna, scoring='f1_macro', n_jobs=1, refit=True,
        error_score='raise', return_train_score=False,
    )
    busca.fit(X_treino, y_treino)
    return busca


def avaliar_logistica(X, y, nome_alvo: str, repeticoes: int = 1,
                      folds_externos: int = 10, semente: int = 2026,
                      max_folds: int | None = None, guardar_predicoes: bool = False,
                      valores_C=VALORES_C, pesos_classes=PESOS_CLASSES, repeticao_inicial: int = 1):
    """Retorna linhas de métricas externas e previsões (quando solicitado)."""
    if not 1 <= repeticao_inicial <= repeticoes:
        raise ValueError("repeticao_inicial deve estar entre 1 e repeticoes")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    linhas, predicoes = [], []
    for contador, (rep, fold, tr, te) in enumerate(gerar_folds_externos(
            y, repeticoes=repeticoes, folds=folds_externos, semente=semente)):
        if rep < repeticao_inicial:
            continue
        if max_folds is not None and contador - (repeticao_inicial - 1) * folds_externos >= max_folds:
            break
        busca = ajustar_logistica(X[tr], y[tr], semente=semente + 1000*rep + fold,
                                 valores_C=valores_C, pesos_classes=pesos_classes)
        modelo = busca.best_estimator_
        prev = modelo.predict(X[te])
        prob = modelo.predict_proba(X[te])
        metricas = calcular_metricas(y[te], prev, classes=classes)
        linhas.append({
            'alvo':nome_alvo, 'repeticao':rep, 'fold':fold,
            'n_treino':len(tr), 'n_teste':len(te),
            'C':float(busca.best_params_['logistica__C']),
            'class_weight':str(busca.best_params_['logistica__class_weight']),
            'f1_macro_cv_interna':float(busca.best_score_),
            'iteracoes_modelo':int(np.max(modelo.named_steps['logistica'].n_iter_)),
            **{k:v for k,v in metricas.items() if k != 'por_classe'},
            'metricas_por_classe':metricas['por_classe'],
        })
        if guardar_predicoes:
            for idx, yr, yp, probs in zip(te, y[te], prev, prob):
                predicoes.append({
                    'alvo':nome_alvo, 'repeticao':rep, 'fold':fold,
                    'indice_original':int(idx), 'real':int(yr), 'predito':int(yp),
                    **{f'prob_classe_{int(c)}':float(p) for c,p in zip(classes, probs)},
                })
    return linhas, predicoes
