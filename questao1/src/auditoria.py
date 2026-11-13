"""Auditoria de microclusters e manifesto de reprodutibilidade da Questão 1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

from .dados import CAMINHO_DADOS, PASTA_Q1
from .double_kmeans import DoubleKMeansResult


def executar_auditoria_baseline(
    X: np.ndarray,
    X_df: pd.DataFrame,
    X_original: np.ndarray,
    y_original: np.ndarray,
    feature_names: list[str],
    best_runs: dict[tuple[int, int], DoubleKMeansResult],
    K_star: int,
    H_star: int,
    audit_dir: Path,
    salvar: bool = True,
    exibir: bool = False,
) -> dict[str, pd.DataFrame]:
    """Gera tabelas de tamanho, Silhueta por objeto/cluster e perfil do menor grupo."""
    labels_kstar = best_runs[(K_star, H_star)].object_labels
    size_rows, sil_rows, sample_rows = [], [], []
    sil_by_pair: dict[tuple[int, int], np.ndarray] = {}

    for (K, H), result in sorted(best_runs.items()):
        labels = result.object_labels
        samples = silhouette_samples(X, labels, metric="euclidean")
        sil_by_pair[(K, H)] = samples
        for k in range(K):
            values = samples[labels == k]
            size_rows.append(
                dict(K=K, H=H, cluster=k, n=len(values), proporcao=len(values) / len(X))
            )
            sil_rows.append(
                dict(
                    K=K,
                    H=H,
                    cluster=k,
                    n=len(values),
                    media=values.mean(),
                    mediana=np.median(values),
                    minimo=values.min(),
                    q05=np.quantile(values, 0.05),
                    q25=np.quantile(values, 0.25),
                    q75=np.quantile(values, 0.75),
                    q95=np.quantile(values, 0.95),
                    maximo=values.max(),
                    proporcao_negativa=np.mean(values < 0),
                )
            )
        sample_rows.append(
            pd.DataFrame(
                dict(
                    K=K,
                    H=H,
                    objeto_id=np.arange(len(X)),
                    linha_spambase=np.arange(1, len(X) + 1),
                    cluster=labels,
                    silhouette=samples,
                )
            )
        )

    sizes_df = pd.DataFrame(size_rows)
    sil_summary_df = pd.DataFrame(sil_rows)
    samples_df = pd.concat(sample_rows, ignore_index=True)

    counts = pd.Series(labels_kstar).value_counts()
    small_label = int(counts.idxmin())
    mask = labels_kstar == small_label
    winner_sil = sil_by_pair[(K_star, H_star)]
    z_original = StandardScaler().fit_transform(X_original)

    small_objects = X_df.loc[mask].copy()
    small_objects.insert(0, "objeto_id", np.flatnonzero(mask))
    small_objects.insert(1, "linha_spambase", np.flatnonzero(mask) + 1)
    small_objects["y_original"] = y_original[mask]
    small_objects["cluster"] = labels_kstar[mask]
    small_objects["silhouette"] = winner_sil[mask]
    small_objects["max_abs_z"] = np.abs(z_original[mask]).max(axis=1)
    small_objects["n_atributos_abs_z_maior_3"] = (np.abs(z_original[mask]) > 3).sum(axis=1)
    small_objects["distancia_centro_global_z"] = np.linalg.norm(z_original[mask], axis=1)
    small_objects["tem_duplicata_atributos_na_base"] = X_df.duplicated(keep=False).to_numpy()[mask]

    std = X_original.std(axis=0)
    delta = X_original[mask].mean(axis=0) - X_original[~mask].mean(axis=0)
    ranking = pd.DataFrame(
        dict(
            variavel=feature_names,
            media_pequeno=X_original[mask].mean(axis=0),
            media_resto=X_original[~mask].mean(axis=0),
            mediana_pequeno=np.median(X_original[mask], axis=0),
            mediana_resto=np.median(X_original[~mask], axis=0),
            prop_nao_zero_pequeno=(X_original[mask] != 0).mean(axis=0),
            prop_nao_zero_resto=(X_original[~mask] != 0).mean(axis=0),
            diferenca_media=delta,
            desvio_global=std,
            diferenca_padronizada=np.divide(delta, std, out=np.zeros_like(delta), where=std > 0),
        )
    )
    ranking["abs_diferenca_padronizada"] = ranking.diferenca_padronizada.abs()
    ranking = ranking.sort_values("abs_diferenca_padronizada", ascending=False).reset_index(drop=True)

    means = np.vstack([X_original[labels_kstar == k].mean(axis=0) for k in range(K_star)])
    weights = np.bincount(labels_kstar, minlength=K_star) / len(X)
    between = (weights[:, None] * (means - X_original.mean(axis=0)) ** 2).sum(axis=0)
    eta2 = np.divide(between, std**2, out=np.zeros_like(between), where=std > 0)
    ranking_todos = pd.DataFrame({"variavel": feature_names, "eta_quadrado": eta2}).sort_values(
        "eta_quadrado", ascending=False
    )

    if salvar:
        audit_dir.mkdir(parents=True, exist_ok=True)
        sizes_df.to_csv(audit_dir / "tamanhos_todos_KH.csv", index=False)
        sil_summary_df.to_csv(audit_dir / "silhouette_por_cluster.csv", index=False)
        samples_df.to_csv(audit_dir / "silhouette_por_objeto_todos_KH.csv", index=False)
        small_objects.to_csv(audit_dir / "objetos_cluster_pequeno.csv", index=False)
        small_objects[feature_names].describe().T.to_csv(
            audit_dir / "descritivas_cluster_pequeno.csv"
        )
        pd.crosstab(labels_kstar, y_original).to_csv(audit_dir / "classes_por_cluster.csv")
        ranking.to_csv(audit_dir / "ranking_variaveis_pequeno_vs_resto.csv", index=False)
        ranking_todos.to_csv(audit_dir / "ranking_todos_clusters.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        top = ranking.head(15).iloc[::-1]
        ax.barh(top.variavel, top.diferenca_padronizada)
        ax.set_xlabel("Média do cluster pequeno − média do resto / desvio padrão global")
        fig.tight_layout()
        fig.savefig(audit_dir / "ranking_variaveis.png", dpi=160)
        if exibir:
            plt.show()
        else:
            plt.close(fig)

    return {
        "tamanhos": sizes_df,
        "silhouette_por_cluster": sil_summary_df,
        "objetos_pequenos": small_objects,
        "ranking_variaveis": ranking,
        "ranking_todos_clusters": ranking_todos,
    }


def gerar_manifesto(
    output_dir: Path,
    n_restarts: int = 100,
    max_iter: int = 200,
    base_seed: int = 42,
    reutiliza_zip: bool = True,
) -> dict:
    """Registra versões de bibliotecas e hashes SHA-256 dos dados e do algoritmo."""
    manifest = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "n_restarts": n_restarts,
        "max_iter": max_iter,
        "base_seed": base_seed,
        "standard_reutiliza_zip": reutiliza_zip,
        "sha256_dados": hashlib.sha256(CAMINHO_DADOS.read_bytes()).hexdigest(),
        "sha256_algoritmo": hashlib.sha256(
            (PASTA_Q1 / "src" / "double_kmeans.py").read_bytes()
        ).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifesto_execucao.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
