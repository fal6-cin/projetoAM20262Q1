"""Carregamento e validação da Spambase com as duas versões de rótulos.

O arquivo está na escala ORIGINAL dos atributos. A transformação usada na Questão 1
apenas construiu y_double_kmeans. Na Questão 2, todo ajuste de transformação
para modelos supervisionados deve ocorrer DENTRO dos folds de treino.
"""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

PASTA_Q2 = Path(__file__).resolve().parents[1]
CAMINHO_CSV = PASTA_Q2 / 'dados' / 'spambase_q2.csv'
CAMINHO_PROVENIENCIA = PASTA_Q2 / 'dados' / 'proveniencia_q1.json'
ALVOS = ('y_original', 'y_double_kmeans')


def carregar_dados(caminho: str | Path = CAMINHO_CSV):
    """Retorna X (float64), dict de y (int64) e nomes dos 57 atributos."""
    with Path(caminho).open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('CSV sem cabeçalho')
        campos = list(reader.fieldnames)
        if len(campos) != 59 or campos[-2:] != list(ALVOS):
            raise ValueError(f'Cabeçalho inesperado: {len(campos)} colunas, últimos {campos[-2:]}')
        nomes = campos[:-2]
        linhas = list(reader)
    X = np.asarray([[float(linha[n]) for n in nomes] for linha in linhas], dtype=np.float64)
    y = {alvo: np.asarray([int(linha[alvo]) for linha in linhas], dtype=np.int64) for alvo in ALVOS}
    return X, y, nomes


def validar_base(caminho: str | Path = CAMINHO_CSV, verificar_hash: bool = True):
    caminho = Path(caminho)
    X, y, nomes = carregar_dados(caminho)
    if X.shape != (4601, 57):
        raise ValueError(f'Dimensões incorretas: {X.shape}')
    if not np.isfinite(X).all() or (X < 0).any():
        raise ValueError('Atributos com valores ausentes, infinitos ou negativos')
    contagens = {alvo: dict(sorted(Counter(map(int, rotulos)).items())) for alvo, rotulos in y.items()}
    esperado = {'y_original': {0: 2788, 1: 1813}, 'y_double_kmeans': {0: 2619, 1: 265, 2: 1717}}
    if contagens != esperado:
        raise ValueError(f'Contagens incompatíveis: {contagens}')
    if verificar_hash and caminho.resolve() == CAMINHO_CSV.resolve():
        referencia = json.loads(CAMINHO_PROVENIENCIA.read_text(encoding='utf-8'))
        atual = hashlib.sha256(caminho.read_bytes()).hexdigest()
        if atual != referencia['sha256_csv']:
            raise ValueError('Arquivo CSV alterado: SHA-256 não coincide com a cópia original')
    return {'observacoes':int(X.shape[0]),'atributos':len(nomes), 'contagens':contagens}
