"""CV aninhada para Bayes com kernel Gaussiano produto e janela de Parzen.

5 folds internos: escolher h pelo F1 macro. Para eficiência, cada fold
interno calcula uma vez as distâncias ao quadrado no espaço Log1p+MinMax,
e avalia todos os h sobre as mesmas distâncias sem usar o teste externo.

A identidade Bayes+MLE a priori leva a:
  P(c | x) ~ [n_c / n] * [1/n_c * sum_{i:y_i=c} K_h(x,x_i)]
          ~ sum_{i:y_i=c} K_h(x,x_i),
onde K_h é produto de gaussianas univariadas. Esta equivalência possibilita
calcular as posteriors em massa sem reestimar a matriz de distância.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from .metricas import calcular_metricas
from .modelos.bayes_parzen import BayesParzen
from .validacao import gerar_folds_externos, gerar_folds_internos

# Valores no espaço [log1p + MinMax]. Grade definida ANTES de consultar dados de teste.
VALORES_H = (0.04, 0.08, 0.12, 0.2, 0.35, 0.55)


@dataclass
class ResultadoAjusteParzen:
    best_estimator_: Pipeline
    best_params_: dict
    best_score_: float
    cv_resultados_: list


def pipeline_parzen(largura_janela: float = 0.2) -> Pipeline:
    return Pipeline([
        ('log1p', FunctionTransformer(np.log1p, validate=True)),
        ('minmax', MinMaxScaler()),
        ('parzen', BayesParzen(largura_janela=largura_janela)),
    ])


def _predicoes_para_grade(distancia2, y_codigos, classes, larguras):
    """Retorna {h: predição} para distâncias já calculadas (somente treino interno).

    Centra o log kernel por linha antes de exponenciar. A constante
    gaussiana do produto é idêntica entre classes, logo cancela na
    normalização das posteriores. O vetor de massas resulta do produto
    matricial entre pesos de Parzen e indicador das classes do TREINO.
    """
    mascara_classe = (y_codigos[:, None] == np.arange(len(classes))[None, :]).astype(float)
    respostas = {}
    for h in larguras:
        energia = -0.5 * distancia2 / (h * h)
        energia -= energia.max(axis=1, keepdims=True)
        # A divisão por n_c da densidade cancela com a priori n_c/n.
        pesos = np.exp(energia)
        massa = pesos @ mascara_classe
        respostas[h] = classes[np.argmax(massa, axis=1)]
    return respostas


def ajustar_parzen(X_treino, y_treino, semente: int, valores_h=VALORES_H, repeticao_inicial: int = 1):
    X_treino = np.asarray(X_treino, dtype=np.float64)
    y_treino = np.asarray(y_treino)
    if X_treino.ndim != 2 or y_treino.ndim != 1 or len(y_treino) != len(X_treino):
        raise ValueError('X_treino e y_treino incompatíveis')
    if not np.isfinite(X_treino).all() or (X_treino < 0).any():
        raise ValueError('Log1p exige valores finitos, não negativos')
    if np.unique(y_treino, return_counts=True)[1].min() < 5:
        raise ValueError('Exige ao menos 5 exemplos por classe na validação interna')
    valores_h = tuple(dict.fromkeys(valores_h))
    if not valores_h:
        raise ValueError('Grade h vazia')
    for h in valores_h:
        if isinstance(h, (bool, np.bool_)) or not isinstance(h, (float, int, np.floating, np.integer)) or not np.isfinite(h) or h <= 0:
            raise ValueError('Todos os valores de h devem ser números positivos finitos')
    valores_h = tuple(float(h) for h in valores_h)
    classes = np.unique(y_treino)
    pontos = {h: [] for h in valores_h}

    for treino_int, valid_int in gerar_folds_internos(y_treino, semente=semente):
        Xlog_treino = np.log1p(X_treino[treino_int])
        esc = MinMaxScaler().fit(Xlog_treino)
        Xt = esc.transform(Xlog_treino)
        Xv = esc.transform(np.log1p(X_treino[valid_int]))
        d2 = cdist(Xv, Xt, metric='sqeuclidean')
        codigos = np.searchsorted(classes, y_treino[treino_int])
        for h, predito in _predicoes_para_grade(d2, codigos, classes, valores_h).items():
            score = f1_score(y_treino[valid_int], predito,
                             labels=classes, average='macro', zero_division=0)
            pontos[h].append(float(score))
    resultados = [
        {'largura_janela':h, 'f1_macro_medio':float(np.mean(pontos[h])),
         'f1_macro_por_fold':pontos[h]}
        for h in valores_h
    ]
    # Empates: selecionar menor largura entre as que apresentam maior F1.
    melhor = max(sorted(resultados, key=lambda a:a['largura_janela']), key=lambda a:a['f1_macro_medio'])
    estimador = pipeline_parzen(largura_janela=melhor['largura_janela'])
    estimador.fit(X_treino, y_treino)
    return ResultadoAjusteParzen(
        best_estimator_=estimador,
        best_params_={'parzen__largura_janela':melhor['largura_janela']},
        best_score_=melhor['f1_macro_medio'], cv_resultados_=resultados,
    )


def avaliar_parzen(X, y, nome_alvo: str, repeticoes: int = 1,
                  folds_externos: int = 10, semente: int = 2026,
                  max_folds: int | None = None, guardar_predicoes: bool = False,
                  valores_h=VALORES_H, repeticao_inicial: int = 1):
    """Gera métricas por fold e, opcionalmente, previsões externas."""
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
        ajuste = ajustar_parzen(X[tr], y[tr], semente=semente + 1000*rep + fold,
                                valores_h=valores_h)
        modelo = ajuste.best_estimator_
        prev = modelo.predict(X[te])
        prob = modelo.predict_proba(X[te])
        m = calcular_metricas(y[te], prev, classes=classes)
        linhas.append({
            'alvo':nome_alvo, 'repeticao':rep, 'fold':fold,
            'n_treino':len(tr), 'n_teste':len(te),
            'largura_janela':ajuste.best_params_['parzen__largura_janela'],
            'f1_macro_cv_interna':float(ajuste.best_score_),
            **{k:v for k,v in m.items() if k != 'por_classe'},
            'metricas_por_classe':m['por_classe'],
        })
        if guardar_predicoes:
            for idx, yr, yp, p in zip(te, y[te], prev, prob):
                predicoes.append({
                    'alvo':nome_alvo, 'repeticao':rep, 'fold':fold,
                    'indice_original':int(idx), 'real':int(yr), 'predito':int(yp),
                    **{f'prob_classe_{int(c)}':float(v) for c,v in zip(classes,p)}
                })
    return linhas, predicoes
