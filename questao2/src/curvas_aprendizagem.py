"""Curvas de aprendizagem da Questão 2 — F1 macro em treino e teste.

Para cada repetição e cada proporção 5, 10, ..., 95, sorteia uma divisão
estratificada COMPLEMENTAR do conjunto de dados (sem sobreposição).
Os cinco modelos usam exatamente essa divisão. Os quatro modelos-base têm
hiperparâmetros escolhidos via 5-fold estratificado EXCLUSIVAMENTE no treino;
o voto usa as quatro predições e prioris aprendidas no mesmo treino.

Não reusa os resultados de 30x10, cujos folds não correspondem às divisões
requeridas para estas curvas. O experimento usa todos os 57 atributos originais;
Log1p/MinMax são ajustados em Pipeline nos treinos interno/externo.
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split

from .avaliacao_gaussiano import ajustar_gaussiano
from .avaliacao_knn import ajustar_knn
from .avaliacao_parzen import ajustar_parzen
from .avaliacao_logistica import ajustar_logistica
from .metricas import calcular_metricas
from .modelos.voto_majoritario import VotoMajoritario

PERCENTUAIS = tuple(range(5, 100, 5))
MODELOS_BASE = ('gaussiano', 'knn', 'parzen', 'logistica')
TODOS_MODELOS = (*MODELOS_BASE, 'voto')
AJUSTADORES = {
    'gaussiano': ajustar_gaussiano,
    'knn': ajustar_knn,
    'parzen': ajustar_parzen,
    'logistica': ajustar_logistica,
}


def gerar_divisao(y, percentual: int, repeticao: int = 1, semente: int = 2026):
    """Divisão estratificada reprodutível; índices referem-se à base original.

    Mesma semente por repetição em todos os percentuais, para manter o mesmo
    plano de aleatorização. Os subconjuntos NÃO são necessariamente aninhados.
    """
    if percentual not in PERCENTUAIS:
        raise ValueError('Percentual de treino deve ser 5, 10, ..., 95')
    if repeticao < 1 or not isinstance(repeticao, int):
        raise ValueError('Repetição deve ser um inteiro positivo')
    y = np.asarray(y)
    if y.ndim != 1 or len(np.unique(y)) < 2:
        raise ValueError('y deve ser um vetor com pelo menos duas classes')
    proporcao = percentual / 100
    # Valor inteiro de treino com arredondamento controlado pelo sklearn.
    tr, te = train_test_split(
        np.arange(len(y)), train_size=proporcao, stratify=y,
        random_state=semente + repeticao - 1,
        shuffle=True,
    )
    todas_classes = np.unique(y)
    for part in (tr, te):
        if len(np.unique(y[part])) != len(todas_classes):
            raise ValueError('Divisão sem todas as classes; não é admissível')
    # Exigir ao menos cinco por classe no treino para CV 5-fold.
    if np.min(np.unique(y[tr], return_counts=True)[1]) < 5:
        raise ValueError('Divisão possui classe com <5 itens no treino, impedindo CV interna')
    if len(tr) + len(te) != len(y) or np.intersect1d(tr, te).size:
        raise AssertionError('Divisão inválida: perda ou duplicação de observações')
    return tr, te


def avaliar_ponto(X, y, alvo: str, percentual: int, repeticao: int = 1,
                  semente: int = 2026,
                  ajustadores: dict[str, Callable] | None = None):
    """Executa 5 modelos sobre o mesmo split e devolve 5 linhas auditáveis.

    Para testes, ``ajustadores`` permite fornecer implementações pequenas; a
    execução normal sempre usa as quatro rotinas validadas da Questão 2.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError('Dimensões X/y incompatíveis')
    if not np.isfinite(X).all() or np.any(X < 0):
        raise ValueError('Atributos devem ser finitos e não negativos (Log1p)')
    if alvo not in ('y_original', 'y_double_kmeans'):
        raise ValueError('Alvo desconhecido')
    if ajustadores is None:
        ajustadores = AJUSTADORES
    if set(ajustadores) != set(MODELOS_BASE):
        raise ValueError('Exige exatamente os quatro classificadores-base')

    tr, te = gerar_divisao(y, percentual, repeticao=repeticao, semente=semente)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    classes = np.unique(y)
    # A CV 5-fold interna usa o MESMO conjunto de treino e uma semente
    # independente por alvo/repetição/ponto; jamais usa yte ou Xte para ajuste.
    seed_cv = semente + 10000*repeticao + 101*percentual
    linhas, previsoes_treino, previsoes_teste = [], [], []
    for nome in MODELOS_BASE:
        ajuste = ajustadores[nome](Xtr, ytr, semente=seed_cv)
        modelo = ajuste.best_estimator_
        p_tr, p_te = np.asarray(modelo.predict(Xtr)), np.asarray(modelo.predict(Xte))
        previsoes_treino.append(p_tr)
        previsoes_teste.append(p_te)
        linhas.append(_linha_metricas(
            nome, alvo, percentual, repeticao, semente, tr, te,
            ytr, yte, p_tr, p_te, classes,
            ajuste.best_params_, ajuste.best_score_,
        ))
    voto = VotoMajoritario().fit(ytr)
    voto_tr, _, emp_tr = voto.prever_detalhes(np.stack(previsoes_treino))
    voto_te, _, emp_te = voto.prever_detalhes(np.stack(previsoes_teste))
    linhas.append(_linha_metricas(
        'voto', alvo, percentual, repeticao, semente, tr, te,
        ytr, yte, voto_tr, voto_te, classes,
        {'desempate': 'maior_priori_no_treino'}, None,
        empates_treino=int(emp_tr.sum()), empates_teste=int(emp_te.sum()),
    ))
    return linhas


def _linha_metricas(nome, alvo, percentual, repeticao, semente, tr, te,
                    ytr, yte, predito_tr, predito_te, classes,
                    hiperparametros, pontuacao_cv, empates_treino=0, empates_teste=0):
    metr_tr = calcular_metricas(ytr, predito_tr, classes=classes)
    metr_te = calcular_metricas(yte, predito_te, classes=classes)
    return {
        'alvo': alvo, 'modelo': nome,
        'repeticao': repeticao, 'percentual_treino': percentual,
        'percentual_teste': 100-percentual,
        'semente_particao': semente+repeticao-1,
        'n_treino': len(tr), 'n_teste': len(te),
        'f1_macro_treino': metr_tr['f_measure_macro'],
        'f1_macro_teste': metr_te['f_measure_macro'],
        'precisao_macro_treino': metr_tr['precisao_macro'],
        'precisao_macro_teste': metr_te['precisao_macro'],
        'cobertura_macro_treino': metr_tr['cobertura_macro'],
        'cobertura_macro_teste': metr_te['cobertura_macro'],
        'taxa_erro_treino': metr_tr['taxa_erro'],
        'taxa_erro_teste': metr_te['taxa_erro'],
        'f1_cv_interna': '' if pontuacao_cv is None else pontuacao_cv,
        'hiperparametros': json.dumps(hiperparametros, ensure_ascii=False, sort_keys=True),
        'empates_treino': empates_treino,
        'empates_teste': empates_teste,
    }
