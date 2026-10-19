"""Avaliação externa pareada do voto majoritário por predições salvas.

Reutiliza os CSVs de predições OUT-OF-FOLD produzidos pelos quatro
classificadores. Não reexecuta treino nem avalia modelos sobre seu treino.
Audita rigorosamente repetições, folds, índices e y_real. A priori de
cada fold é calculada só nos índices de treino, nunca no teste.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .dados import carregar_dados
from .metricas import calcular_metricas
from .modelos.voto_majoritario import VotoMajoritario
from .validacao import gerar_folds_externos

MODELOS_VOTANTES = ('gaussiano', 'knn', 'parzen', 'logistica')
PASTA_RESULTADOS = Path(__file__).resolve().parents[1] / 'resultados'
CHAVES = ('alvo', 'repeticao', 'fold', 'indice_original')


def _ler_predicoes(caminho: Path, nome_alvo: str, y: np.ndarray, repeticoes: int):
    """Lê arquivo e verifica duplicatas e cobertura básica, sem tolerar folds trocados."""
    if not caminho.is_file():
        raise FileNotFoundError(f'Não existe {caminho}; execute o classificador correspondente ' 
                                f'com --todos-folds --repeticoes {repeticoes}')
    with caminho.open(encoding='utf-8-sig', newline='') as fp:
        leitor = csv.DictReader(fp)
        if leitor.fieldnames is None or not set((*CHAVES, 'real', 'predito')).issubset(leitor.fieldnames):
            raise ValueError(f'Colunas obrigatórias ausentes em {caminho}')
        linhas = list(leitor)
    esperado = len(y) * repeticoes
    if len(linhas) != esperado:
        raise ValueError(f'{caminho.name}: {len(linhas)} predições; esperadas {esperado}. ' 
                         'O arquivo deve conter todos os folds de todas as repetições')
    encontrados = {}
    classes = set(map(int, np.unique(y)))
    for linha in linhas:
        if linha['alvo'] != nome_alvo:
            raise ValueError(f'Alvo inconsistente em {caminho.name}')
        try:
            rep, fold, ix = (int(linha[c]) for c in ('repeticao','fold','indice_original'))
            yr, yp = int(linha['real']), int(linha['predito'])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Valores não inteiros em {caminho.name}') from exc
        chave = (rep, fold, ix)
        if rep < 1 or rep > repeticoes or fold < 1 or fold > 10 or ix < 0 or ix >= len(y):
            raise ValueError(f'Identificadores fora do intervalo em {caminho.name}: {chave}')
        if chave in encontrados:
            raise ValueError(f'Predição duplicada em {caminho.name}: {chave}')
        if yr != int(y[ix]) or yp not in classes:
            raise ValueError(f'Classe real ou predita incompatível em {caminho.name}: {chave}')
        encontrados[chave] = yp
    return encontrados


def avaliar_voto_de_csvs(nome_alvo: str, repeticoes: int = 1,
                         max_folds: int | None = None, semente: int = 2026,
                         diretorio_resultados: Path = PASTA_RESULTADOS,
                         guardar_predicoes: bool = True):
    """Devolve (resumos por fold, predições por observação).

    Para cada repetição exige todos os folds nos arquivos, inclusive se
    max_folds for usado para inspecionar apenas o início.
    """
    if repeticoes < 1 or max_folds is not None and max_folds < 1:
        raise ValueError('repeticoes e max_folds devem ser >=1')
    _, respostas, _ = carregar_dados()
    if nome_alvo not in respostas:
        raise ValueError(f'Alvo desconhecido: {nome_alvo}')
    y = respostas[nome_alvo]
    classes = np.unique(y)
    diretorio_resultados = Path(diretorio_resultados)
    por_modelo = {}
    for nome in MODELOS_VOTANTES:
        arquivo = diretorio_resultados/nome/f'{nome_alvo}_{repeticoes}x10_predicoes.csv'
        por_modelo[nome] = _ler_predicoes(arquivo, nome_alvo, y, repeticoes)
    resumos, predicoes = [], []
    n_folds = 0
    for rep, fold, tr, te in gerar_folds_externos(y, repeticoes=repeticoes, semente=semente):
        if max_folds is not None and n_folds >= max_folds:
            break
        chaves = [(rep, fold, int(indice)) for indice in te]
        votos = []
        for nome in MODELOS_VOTANTES:
            try:
                votos.append([por_modelo[nome][chave] for chave in chaves])
            except KeyError as exc:
                raise ValueError(f'{nome}: falta previsão ou os folds divergem da semente {semente}: {exc}') from exc
        modelo = VotoMajoritario().fit(y[tr])
        pred, contagens, empates = modelo.prever_detalhes(np.asarray(votos))
        met = calcular_metricas(y[te], pred, classes=classes)
        resumos.append({
            'alvo': nome_alvo, 'repeticao': rep, 'fold': fold,
            'n_treino':len(tr), 'n_teste':len(te),
            'empates_resolvidos_priori':int(empates.sum()),
            **{f'priori_classe_{int(c)}':float(p) for c,p in zip(modelo.classes_, modelo.prioris_)},
            **{k:v for k,v in met.items() if k != 'por_classe'},
            'metricas_por_classe':met['por_classe']
        })
        if guardar_predicoes:
            for j, idx in enumerate(te):
                predicoes.append({
                    'alvo':nome_alvo,'repeticao':rep,'fold':fold,
                    'indice_original':int(idx),'real':int(y[idx]),'predito':int(pred[j]),
                    **{f'voto_{nome}':int(votos[i][j]) for i,nome in enumerate(MODELOS_VOTANTES)},
                    **{f'votos_classe_{int(c)}':int(contagens[j,k]) for k,c in enumerate(classes)},
                    'empate':bool(empates[j]),
                })
        n_folds += 1
    # Os arrays só podem diferir no mesmo fold se coincidir de novo por acaso.
    # A auditoria completa de pertencimento é feita para todas as repetições
    # (inclusive quando max_folds limita o número de folds resumidos).
    esperado = set()
    for rep, fold, _, te in gerar_folds_externos(y, repeticoes=repeticoes, semente=semente):
        esperado.update((rep, fold, int(i)) for i in te)
    for nome, valores in por_modelo.items():
        if set(valores) != esperado:
            extras = list(set(valores) - esperado)[:3]
            faltas = list(esperado - set(valores))[:3]
            raise ValueError(f'{nome}: folds não coincidem com partição CV; extras={extras}, faltas={faltas}')
    return resumos, predicoes
