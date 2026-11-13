"""Estratégias de pré-processamento da Questão 1 (estritamente não supervisionadas)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

TransformFn = Callable[[np.ndarray, list[str], int], tuple[np.ndarray, list[str]]]

TERMOS_DOMINIO_HP = (
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_650",
    "word_freq_415",
    "word_freq_857",
)


@dataclass(frozen=True)
class EstrategiaConfig:
    """Metadados e função de transformação de uma estratégia da Questão 1."""

    nome: str
    titulo: str
    descricao_scaling: str
    csv_saida: str
    diagnostico_comparativo: str
    transform: TransformFn


def _std_puro(
    X_original: np.ndarray, feature_names: list[str], base_seed: int
) -> tuple[np.ndarray, list[str]]:
    return StandardScaler().fit_transform(X_original), list(feature_names)


def _binarizacao(
    X_original: np.ndarray, feature_names: list[str], base_seed: int
) -> tuple[np.ndarray, list[str]]:
    X_bin = (X_original > 0).astype(float)
    X_bin[:, -3:] = np.log1p(X_original[:, -3:])
    return StandardScaler().fit_transform(X_bin), list(feature_names)


def _clip_log_standard(
    X_original: np.ndarray, feature_names: list[str], base_seed: int
) -> tuple[np.ndarray, list[str]]:
    p99 = np.percentile(X_original, 99, axis=0)
    X_log = np.log1p(np.clip(X_original, 0, p99))
    return StandardScaler().fit_transform(X_log), list(feature_names)


def _filtragem_dominio(
    X_original: np.ndarray, feature_names: list[str], base_seed: int
) -> tuple[np.ndarray, list[str]]:
    keep_mask = [f not in TERMOS_DOMINIO_HP for f in feature_names]
    active_names = [f for f in feature_names if f not in TERMOS_DOMINIO_HP]
    X_filtered = X_original[:, keep_mask]
    return StandardScaler().fit_transform(X_filtered), active_names


def _log_minmax(
    X_original: np.ndarray, feature_names: list[str], base_seed: int
) -> tuple[np.ndarray, list[str]]:
    if not np.all(X_original >= 0):
        raise ValueError("Todos os atributos devem ser não negativos para log1p.")
    X_log = np.log1p(X_original)
    return MinMaxScaler().fit_transform(X_log), list(feature_names)


def _quantile_uniform(
    X_original: np.ndarray, feature_names: list[str], base_seed: int
) -> tuple[np.ndarray, list[str]]:
    scaler = QuantileTransformer(output_distribution="uniform", random_state=base_seed)
    return scaler.fit_transform(X_original), list(feature_names)


def _tratamento_diferenciado_runs(
    X_original: np.ndarray, feature_names: list[str], base_seed: int
) -> tuple[np.ndarray, list[str]]:
    X_freq_scaled = StandardScaler().fit_transform(X_original[:, :54])
    X_runs_scaled = StandardScaler().fit_transform(np.log1p(X_original[:, 54:]))
    return np.hstack([X_freq_scaled, X_runs_scaled]), list(feature_names)


ESTRATEGIAS: dict[str, EstrategiaConfig] = {
    "revisada": EstrategiaConfig(
        nome="revisada",
        titulo="Original / Revisada (StandardScaler puro)",
        descricao_scaling="StandardScaler (z-score)",
        csv_saida="spambase_com_clusters.csv",
        diagnostico_comparativo="Isolamento de 34 outliers da HP Labs; Silhueta inflada artificialmente.",
        transform=_std_puro,
    ),
    "binarizacao": EstrategiaConfig(
        nome="binarizacao",
        titulo="Binarização (x > 0) + log1p(maiúsculas) + StandardScaler",
        descricao_scaling="Binarizacao (x>0) + log1p(runs) + StandardScaler",
        csv_saida="spambase_com_clusters_binarizado.csv",
        diagnostico_comparativo="Sem microclusters; partição robusta e balanceada alinhada a spam/não-spam.",
        transform=_binarizacao,
    ),
    "clip_log_standard": EstrategiaConfig(
        nome="clip_log_standard",
        titulo="Clip 99% + log(1+x) + StandardScaler",
        descricao_scaling="Clip 99% + log1p + StandardScaler",
        csv_saida="spambase_com_clusters_clip_log_standard.csv",
        diagnostico_comparativo="Sem microclusters; controle de caudas pesadas preservando escala z-score.",
        transform=_clip_log_standard,
    ),
    "filtragem_dominio": EstrategiaConfig(
        nome="filtragem_dominio",
        titulo="Filtragem de Termos de Domínio (49 atributos) + StandardScaler",
        descricao_scaling="Filtragem de Termos de Domínio (hp, hpl, george, lab, labs, 650, 415, 857) + StandardScaler(49 variáveis)",
        csv_saida="spambase_com_clusters_filtragem_dominio.csv",
        diagnostico_comparativo="Eliminação do microcluster de 34 após remoção dos 8 termos específicos.",
        transform=_filtragem_dominio,
    ),
    "log_minmax": EstrategiaConfig(
        nome="log_minmax",
        titulo="log(1+x) + MinMaxScaler",
        descricao_scaling="log1p + MinMaxScaler",
        csv_saida="spambase_com_clusters_log_minmax.csv",
        diagnostico_comparativo="Entrada oficial da Q2; concordância parcial (ARI=+0,3911) em (3,3).",
        transform=_log_minmax,
    ),
    "quantile_uniform": EstrategiaConfig(
        nome="quantile_uniform",
        titulo="QuantileTransformer (uniforme)",
        descricao_scaling="QuantileTransformer (output_distribution='uniform')",
        csv_saida="spambase_com_clusters_quantile_uniform.csv",
        diagnostico_comparativo="Sem microclusters; mapeamento não-paramétrico uniforme [0, 1].",
        transform=_quantile_uniform,
    ),
    "tratamento_diferenciado_runs": EstrategiaConfig(
        nome="tratamento_diferenciado_runs",
        titulo="Tratamento Diferenciado de Maiúsculas por Bloco",
        descricao_scaling="Tratamento Diferenciado: StandardScaler(freq 0..53) + StandardScaler(log1p(runs 54..56))",
        csv_saida="spambase_com_clusters_runs_diferenciado.csv",
        diagnostico_comparativo="Controle de escala das contagens de maiúsculas preservando peso equilibrado de blocos.",
        transform=_tratamento_diferenciado_runs,
    ),
}


def listar_estrategias() -> list[str]:
    """Retorna os nomes válidos das 7 estratégias da Questão 1."""
    return list(ESTRATEGIAS.keys())


def aplicar_preprocessamento(
    nome: str,
    X_original: np.ndarray,
    feature_names: list[str],
    base_seed: int = 42,
) -> tuple[np.ndarray, list[str], EstrategiaConfig]:
    """Aplica o pré-processamento especificado sem acesso à coluna de classe."""
    if nome not in ESTRATEGIAS:
        disponiveis = ", ".join(ESTRATEGIAS)
        raise ValueError(f"Estratégia desconhecida '{nome}'. Disponíveis: {disponiveis}")
    config = ESTRATEGIAS[nome]
    X_proc, active_names = config.transform(X_original, feature_names, base_seed)
    return X_proc, active_names, config
