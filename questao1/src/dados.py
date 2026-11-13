"""Carregamento e validação do dataset Spambase bruto para a Questão 1."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PASTA_Q1 = Path(__file__).resolve().parents[1]
RAIZ_REPO = PASTA_Q1.parent
CAMINHO_DADOS = RAIZ_REPO / "data" / "raw" / "spambase.data"
CAMINHO_NOMES = RAIZ_REPO / "data" / "raw" / "spambase.names"
PASTA_RESULTADOS = PASTA_Q1 / "resultados"


def carregar_nomes_atributos(names_path: str | Path = CAMINHO_NOMES) -> list[str]:
    """Lê os 57 nomes de atributos numéricos de `spambase.names`."""
    nomes: list[str] = []
    with Path(names_path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if ":" in line and "continuous" in line.lower():
                nomes.append(line.split(":", 1)[0].strip())
    if len(nomes) != 57:
        raise ValueError(f"Esperava 57 atributos em {names_path}, encontrei {len(nomes)}.")
    return nomes


def carregar_spambase(
    data_path: str | Path = CAMINHO_DADOS,
    names_path: str | Path = CAMINHO_NOMES,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Carrega o Spambase bruto (sem usar y_original no pré-processamento).

    Retorna `(X_df, X_original, y_original, feature_names)`.
    """
    feature_names = carregar_nomes_atributos(names_path)
    raw = pd.read_csv(data_path, header=None)

    n_features = raw.shape[1] - 1
    if n_features != 57:
        raise ValueError(f"Esperava 57 colunas preditoras em {data_path}, encontrei {n_features}.")

    X_df = raw.iloc[:, :-1].copy()
    X_df.columns = feature_names
    X_original = X_df.to_numpy(dtype=float)
    y_original = raw.iloc[:, -1].astype(int).to_numpy()
    return X_df, X_original, y_original, feature_names
