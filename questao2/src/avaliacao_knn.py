"""Avaliação aninhada do Bayes k-NN, com busca eficiente, sem vazamento.

A validação interna 5-fold avalia (distância, k) pelo F1 macro. Para cada
fold interno e distância, calcula os vizinhos uma única vez para k_max e
reaproveita os índices para todos os valores de k. Isso é equivalente a
ajustar separadamente os estimadores, pois k não altera o treinamento do k-NN.

O Log1p e o MinMax são ajustados SEMPRE em cada treino interno. O modelo
vencedor é reajustado do zero em todo o treino externo. O teste externo
nunca é utilizado para escolher k, distância ou parâmetros do MinMax.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from .modelos.bayes_knn import BayesKVizinhos
from .validacao import gerar_folds_externos, gerar_folds_internos
from .metricas import calcular_metricas

VALORES_K = (1, 3, 5, 7, 11, 15, 21)
DISTANCIAS = ('euclidiana', 'city_block', 'chebyshev')


@dataclass
class ResultadoAjusteKNN:
    best_estimator_: Pipeline
    best_params_: dict
    best_score_: float
    cv_resultados_: list


def pipeline_knn(n_vizinhos: int = 5, distancia: str = 'euclidiana') -> Pipeline:
    """Pipeline sem parâmetros ajustados previamente."""
    return Pipeline([
        ('log1p', FunctionTransformer(np.log1p, validate=True)),
        ('minmax', MinMaxScaler()),
        ('knn', BayesKVizinhos(n_vizinhos=n_vizinhos, distancia=distancia)),
    ])


def ajustar_knn(X_treino, y_treino, semente: int, valores_k=VALORES_K, distancias=DISTANCIAS):
    """Seleção 5-fold estratificada: pares (k, distância) sem usar teste externo.

    O F1 macro dos folds internos é usado na busca, de modo idêntico ao
    classificador gaussiano. Empates de score: ordem fixa em distancias e k.
    """
    X_treino = np.asarray(X_treino, dtype=np.float64)
    y_treino = np.asarray(y_treino)
    if X_treino.ndim != 2 or y_treino.ndim != 1 or len(y_treino) != len(X_treino):
        raise ValueError('Dimensões incompatíveis entre X_treino e y_treino')
    if not np.isfinite(X_treino).all() or (X_treino < 0).any():
        raise ValueError('Log1p exige atributos finitos e não negativos')
    if np.unique(y_treino, return_counts=True)[1].min() < 5:
        raise ValueError('São necessários pelo menos 5 exemplos por classe na CV interna')
    valores_k = tuple(dict.fromkeys(valores_k))
    distancias = tuple(dict.fromkeys(distancias))
    if not valores_k or not distancias:
        raise ValueError('A grade de k e distâncias não pode estar vazia')
    if any(not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1 for k in valores_k):
        raise ValueError('Todos os valores de k devem ser inteiros positivos')
    if any(d not in DISTANCIAS for d in distancias):
        raise ValueError(f'Distâncias permitidas: {DISTANCIAS}')
    min_amostra = min(len(tr) for tr, _ in gerar_folds_internos(y_treino, semente=semente))
    if max(valores_k) > min_amostra:
        raise ValueError('Um valor de k excede o tamanho mínimo de treino interno')

    nclasses = np.unique(y_treino)
    pontos = {(d, k): [] for d in distancias for k in valores_k}
    for tr, va in gerar_folds_internos(y_treino, semente=semente):
        # fit apenas no treino INTERNO: na validação podem aparecer valores > 1.
        escalador = MinMaxScaler().fit(np.log1p(X_treino[tr]))
        Xt = escalador.transform(np.log1p(X_treino[tr]))
        Xv = escalador.transform(np.log1p(X_treino[va]))
        for distancia in distancias:
            modelo = BayesKVizinhos(n_vizinhos=max(valores_k), distancia=distancia).fit(Xt, y_treino[tr])
            idx = modelo.indices_vizinhos(Xv, k=max(valores_k))
            codigos = modelo.y_codificado_[idx]
            for k in valores_k:
                decisoes = BayesKVizinhos._vencedores_com_desempate(codigos[:, :k], len(modelo.classes_))
                prev = modelo.classes_[decisoes]
                score = f1_score(y_treino[va], prev, labels=nclasses, average='macro', zero_division=0)
                pontos[(distancia, k)].append(float(score))

    # Empates resolvidos deterministicamente: distâncias na ordem dada,
    # depois menor k, em vez de escolher um par arbitrariamente.
    resultados = []
    for distancia in distancias:
        for k in sorted(valores_k):
            escores = pontos[(distancia, k)]
            resultados.append({'distancia':distancia, 'n_vizinhos':int(k),
                               'f1_macro_medio':float(np.mean(escores)),
                               'f1_macro_por_fold':escores})
    melhor = max(resultados, key=lambda r: r['f1_macro_medio'])
    estimador = pipeline_knn(n_vizinhos=melhor['n_vizinhos'], distancia=melhor['distancia'])
    estimador.fit(X_treino, y_treino)
    return ResultadoAjusteKNN(
        best_estimator_=estimador,
        best_params_={'knn__n_vizinhos':melhor['n_vizinhos'], 'knn__distancia':melhor['distancia']},
        best_score_=melhor['f1_macro_medio'], cv_resultados_=resultados,
    )


def avaliar_knn(X, y, nome_alvo: str, repeticoes: int = 1,
                folds_externos: int = 10, semente: int = 2026,
                max_folds: int | None = None, guardar_predicoes: bool = False,
                valores_k=VALORES_K, distancias=DISTANCIAS, repeticao_inicial: int = 1):
    """Treina e avalia Bayes k-NN no teste externo, retornando métricas por fold."""
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
        busca = ajustar_knn(X[tr], y[tr], semente=semente + 1000 * rep + fold,
                            valores_k=valores_k, distancias=distancias)
        modelo = busca.best_estimator_
        prev = modelo.predict(X[te])
        prob = modelo.predict_proba(X[te])
        m = calcular_metricas(y[te], prev, classes=classes)
        info = {
            'alvo':nome_alvo, 'repeticao':rep, 'fold':fold,
            'n_treino':len(tr), 'n_teste':len(te),
            'n_vizinhos':busca.best_params_['knn__n_vizinhos'],
            'distancia':busca.best_params_['knn__distancia'],
            'f1_macro_cv_interna':float(busca.best_score_),
            **{k:v for k,v in m.items() if k != 'por_classe'},
            'metricas_por_classe':m['por_classe'],
        }
        linhas.append(info)
        if guardar_predicoes:
            for idx, yr, yp, p in zip(te, y[te], prev, prob):
                predicoes.append({'alvo':nome_alvo, 'repeticao':rep, 'fold':fold,
                                  'indice_original':int(idx), 'real':int(yr), 'predito':int(yp),
                                  **{f'prob_classe_{int(c)}':float(v) for c,v in zip(classes,p)}})
    return linhas, predicoes
