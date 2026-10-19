"""Validação de integridade dos CSVs de validação cruzada externa.

Os arquivos precisam corresponder à divisão estratificada de seed=2026, aos
rótulos originais e à lista exata de índices de teste em cada fold.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .metricas import calcular_metricas
from .validacao import gerar_folds_externos

METRICAS = ('taxa_erro', 'precisao_macro', 'cobertura_macro', 'f_measure_macro')


def ler_csv(caminho):
    with Path(caminho).open(newline='', encoding='utf-8-sig') as fp:
        leitor = csv.DictReader(fp)
        if not leitor.fieldnames:
            raise ValueError(f'CSV sem cabeçalho: {caminho}')
        return list(leitor)


def auditar_resultados(folds_csv, predicoes_csv, y, alvo, repeticoes=30,
                       semente=2026, repeticao_inicial=1, repeticao_final=None):
    """Audita 10 folds por repetição, cobertura de índices e métricas.

    Os CSVs podem conter uma repetição isolada (shard) ou todas as 30.
    Retorna as linhas validadas para análise/consolidação.
    """
    y = np.asarray(y)
    if repeticao_final is None:
        repeticao_final = repeticoes
    if not (1 <= repeticao_inicial <= repeticao_final <= repeticoes):
        raise ValueError('Intervalo inválido de repetições')
    resumos = ler_csv(folds_csv)
    predicoes = ler_csv(predicoes_csv)
    esperado = {}
    for rep, fold, tr, te in gerar_folds_externos(y, repeticoes=repeticoes, semente=semente):
        if repeticao_inicial <= rep <= repeticao_final:
            esperado[(rep, fold)] = set(map(int, te))
    if len(resumos) != len(esperado):
        raise ValueError(f'Resumos: {len(resumos)} folds, esperados {len(esperado)}')
    keys_resumo = {}
    classes = np.unique(y)
    for linha in resumos:
        if linha.get('alvo') != alvo:
            raise ValueError('Alvo incorreto no resumo')
        key = (int(linha['repeticao']), int(linha['fold']))
        if key not in esperado or key in keys_resumo:
            raise ValueError(f'Fold inesperado ou duplicado: {key}')
        if int(linha['n_teste']) != len(esperado[key]) or int(linha['n_treino']) != len(y)-len(esperado[key]):
            raise ValueError(f'Tamanho incorreto do fold {key}')
        if any(not np.isfinite(float(linha[m])) or not 0 <= float(linha[m]) <= 1 for m in METRICAS):
            raise ValueError(f'Métrica inválida no fold {key}')
        keys_resumo[key] = linha
    vistos = {}
    for l in predicoes:
        if l.get('alvo') != alvo:
            raise ValueError('Alvo incorreto nas predições')
        key = (int(l['repeticao']), int(l['fold']))
        if key not in esperado:
            raise ValueError(f'Fold inesperado na predição: {key}')
        ix = int(l['indice_original'])
        if ix not in esperado[key] or (key, ix) in vistos:
            raise ValueError(f'Índice fora do teste ou repetido: {key}, {ix}')
        if int(l['real']) != int(y[ix]) or int(l['predito']) not in classes:
            raise ValueError(f'Classe inconsistente: {key}, {ix}')
        vistos[(key, ix)] = int(l['predito'])
    if len(vistos) != len(y)*(repeticao_final-repeticao_inicial+1):
        raise ValueError('Cobertura incorreta de predições')
    for key, inds in esperado.items():
        if any((key, ix) not in vistos for ix in inds):
            raise ValueError(f'Predições faltantes: {key}')
        indices = sorted(inds)
        m = calcular_metricas(y[indices], [vistos[(key, ix)] for ix in indices], classes=classes)
        for metrica in METRICAS:
            if not np.isclose(m[metrica], float(keys_resumo[key][metrica]), atol=1e-10, rtol=1e-9):
                raise ValueError(f'Métrica {metrica} não confere com predições no fold {key}')
    return resumos
