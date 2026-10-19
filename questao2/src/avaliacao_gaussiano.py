"""Validação aninhada do Bayes gaussiano, sem vazamento de pré-processamento.

Cada fold EXTERNO avalia uma única combinação definida por CV INTERNA
estratificada (5 folds). O Log1p e o MinMax são ajustados DENTRO dos folds.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from .metricas import calcular_metricas
from .modelos.bayes_gaussiano import BayesGaussianoMultivariado
from .validacao import gerar_folds_externos

REGULARIZACOES = (1e-6, 1e-4, 1e-2, 1e-1)


def pipeline_gaussiano(regularizacao_relativa: float = 1e-4):
    """Construção do modelo, deixando todas as estatísticas para fit(treino)."""
    return Pipeline([
        ('log1p', FunctionTransformer(np.log1p, validate=True)),
        ('minmax', MinMaxScaler()),
        ('gaussiano', BayesGaussianoMultivariado(regularizacao_relativa=regularizacao_relativa)),
    ])


def ajustar_gaussiano(X_treino, y_treino, semente: int, grade=REGULARIZACOES):
    """Faz a seleção somente no treino EXTERNO via CV interna estratificada.

    Usa F1 macro (e não acurácia), inclusive para a classe minoritária.
    """
    X_treino = np.asarray(X_treino, dtype=float)
    y_treino = np.asarray(y_treino)
    if (X_treino < 0).any():
        raise ValueError('Log1p exige atributos >= 0')
    if np.unique(y_treino, return_counts=True)[1].min() < 5:
        raise ValueError('São necessários >=5 exemplos por classe para CV interna')
    interno = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)
    busca = GridSearchCV(
        estimator=pipeline_gaussiano(),
        param_grid={'gaussiano__regularizacao_relativa': list(grade)},
        cv=interno,
        scoring='f1_macro',
        n_jobs=1,
        refit=True,
        error_score='raise',
        return_train_score=False,
    )
    busca.fit(X_treino, y_treino)
    return busca


def avaliar_gaussiano(X, y, nome_alvo: str, repeticoes: int = 1,
                       folds_externos: int = 10, semente: int = 2026,
                       max_folds: int | None = None,
                       guardar_predicoes: bool = False, repeticao_inicial: int = 1):
    """Devolve resultados fold a fold e, opcionalmente, predições por amostra.

    max_folds é para um smoke-test parcial, NÃO representa o experimento 30x10.
    """
    if not 1 <= repeticao_inicial <= repeticoes:
        raise ValueError("repeticao_inicial deve estar entre 1 e repeticoes")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    linhas, predicoes = [], []
    for contador, (rep, fold, tr, te) in enumerate(
        gerar_folds_externos(y, repeticoes=repeticoes, folds=folds_externos, semente=semente)
    ):
        if rep < repeticao_inicial:
            continue
        if max_folds is not None and contador - (repeticao_inicial - 1) * folds_externos >= max_folds:
            break
        busca = ajustar_gaussiano(X[tr], y[tr], semente=semente + 1000*rep + fold)
        modelo = busca.best_estimator_
        prev = modelo.predict(X[te])
        prob = modelo.predict_proba(X[te])
        m = calcular_metricas(y[te], prev, classes=classes)
        info = {
            'alvo':nome_alvo,
            'repeticao':rep,
            'fold':fold,
            'n_treino':len(tr),
            'n_teste':len(te),
            'regularizacao_relativa':busca.best_params_['gaussiano__regularizacao_relativa'],
            'f1_macro_cv_interna':float(busca.best_score_),
            **{k:v for k,v in m.items() if k != 'por_classe'},
            'metricas_por_classe':m['por_classe'],
        }
        linhas.append(info)
        if guardar_predicoes:
            for idx, yr, yp, p in zip(te, y[te], prev, prob):
                predicoes.append({'alvo':nome_alvo,'repeticao':rep,'fold':fold,
                    'indice_original':int(idx),'real':int(yr),'predito':int(yp),
                    **{f'prob_classe_{int(c)}':float(v) for c,v in zip(classes,p)}})
    return linhas, predicoes
