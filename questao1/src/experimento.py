"""Execução, reprodução, validação e geração de artefatos da Questão 1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .dados import PASTA_RESULTADOS, carregar_spambase
from .double_kmeans import DoubleKMeansResult, double_kmeans
from .preprocessamentos import ESTRATEGIAS, EstrategiaConfig, aplicar_preprocessamento

GRADE_KH: tuple[tuple[int, int], ...] = tuple(
    (K, H) for K in (2, 3, 4) for H in range(1, K + 1)
)


@dataclass
class AvaliacaoParticao:
    """Resultados consolidados para o par vencedor (K*, H*) de uma estratégia."""

    K_star: int
    H_star: int
    best_solution: DoubleKMeansResult
    results_df: pd.DataFrame
    silhouette: float
    ari: float
    G_df: pd.DataFrame
    contingency: pd.DataFrame
    var_df: pd.DataFrame
    metadata_df: pd.DataFrame
    q2_df: pd.DataFrame
    comp_df: pd.DataFrame


def run_restarts(
    X: np.ndarray,
    K: int,
    H: int,
    n_restarts: int = 100,
    base_seed: int = 42,
    max_iter: int = 200,
    max_attempts_factor: int = 50,
) -> dict:
    """Executa exatamente `n_restarts` rodadas válidas e retorna a de menor W."""
    valid_runs: list[DoubleKMeansResult] = []
    invalid_runs = 0
    attempt = 0
    max_attempts = n_restarts * max_attempts_factor

    while len(valid_runs) < n_restarts and attempt < max_attempts:
        seed = base_seed + 10000 * K + 1000 * H + attempt
        attempt += 1
        res = double_kmeans(X=X, K=K, H=H, seed=seed, max_iter=max_iter)
        if res is None:
            invalid_runs += 1
            continue
        valid_runs.append(res)

    if len(valid_runs) < n_restarts:
        raise RuntimeError(
            f"Só foi possível obter {len(valid_runs)} execuções válidas para (K={K}, H={H})."
        )

    best = min(valid_runs, key=lambda r: r.objective)
    return {
        "best": best,
        "valid_runs": len(valid_runs),
        "invalid_runs": invalid_runs,
        "attempts": attempt,
    }


def reproduzir_grade(
    X: np.ndarray,
    results_df: pd.DataFrame,
    max_iter: int = 200,
) -> dict[tuple[int, int], DoubleKMeansResult]:
    """Reproduz as 9 melhores partições a partir de `resultados_KH.csv` conferindo W."""
    best_runs: dict[tuple[int, int], DoubleKMeansResult] = {}
    for row in results_df.itertuples():
        K, H, seed = int(row.K), int(row.H), int(row.best_seed)
        best = double_kmeans(X=X, K=K, H=H, seed=seed, max_iter=max_iter)
        if best is None:
            raise AssertionError(f"Seed {seed} gerou partição inválida em ({K}, {H}).")
        if not np.isclose(best.objective, row.best_objective_W, atol=1e-3):
            raise AssertionError(
                f"W não reproduzido em ({K}, {H}): obtido {best.objective}, esperado {row.best_objective_W}."
            )
        best_runs[(K, H)] = best
    return best_runs


def executar_grade(
    X: np.ndarray,
    n_restarts: int = 100,
    base_seed: int = 42,
    max_iter: int = 200,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[int, int], DoubleKMeansResult]]:
    """Executa a grade completa dos 9 pares (K, H) com `n_restarts` válidos por par."""
    grid_rows = []
    best_runs: dict[tuple[int, int], DoubleKMeansResult] = {}
    for K, H in GRADE_KH:
        if verbose:
            print(f"Executando par K={K}, H={H} ({n_restarts} restarts válidos)...")
        exp = run_restarts(
            X=X,
            K=K,
            H=H,
            n_restarts=n_restarts,
            base_seed=base_seed,
            max_iter=max_iter,
        )
        best = exp["best"]
        sil = float(silhouette_score(X, best.object_labels, metric="euclidean"))
        best_runs[(K, H)] = best
        grid_rows.append(
            {
                "K": K,
                "H": H,
                "best_objective_W": best.objective,
                "silhouette": sil,
                "best_seed": best.seed,
                "n_iterations_best": best.n_iter,
                "valid_runs": exp["valid_runs"],
                "invalid_attempts": exp["invalid_runs"],
                "total_attempts": exp["attempts"],
            }
        )
    results_df = pd.DataFrame(grid_rows).sort_values(["K", "H"]).reset_index(drop=True)
    return results_df, best_runs


def verificar_sanidade(
    best_runs: dict[tuple[int, int], DoubleKMeansResult],
    monotonic_tol: float = 1e-7,
) -> pd.DataFrame:
    """Verifica cardinalidade dos clusters encontrados e monotonicidade de W."""
    checks = []
    for (K, H), result in sorted(best_runs.items()):
        n_obj = int(np.unique(result.object_labels).size)
        n_var = int(np.unique(result.variable_labels).size)
        diffs = np.diff(result.objective_history)
        monotonic = bool(np.all(diffs <= monotonic_tol))
        checks.append(
            {
                "K": K,
                "H": H,
                "object_clusters_found": n_obj,
                "variable_clusters_found": n_var,
                "objective_non_increasing": monotonic,
            }
        )
    checks_df = pd.DataFrame(checks)
    assert len(checks_df) == len(GRADE_KH), "Grade incompleta: esperava 9 pares (K, H)."
    assert (checks_df["object_clusters_found"] == checks_df["K"]).all()
    assert (checks_df["variable_clusters_found"] == checks_df["H"]).all()
    assert checks_df["objective_non_increasing"].all()
    return checks_df


def avaliar_melhor_particao(
    X_df: pd.DataFrame,
    y_original: np.ndarray,
    active_feature_names: list[str],
    results_df: pd.DataFrame,
    best_runs: dict[tuple[int, int], DoubleKMeansResult],
    config: EstrategiaConfig,
    n_restarts: int = 100,
) -> AvaliacaoParticao:
    """Seleciona (K*, H*) pela maior Silhueta e constrói todas as tabelas de saída."""
    best_row = results_df.loc[results_df["silhouette"].idxmax()]
    K_star = int(best_row["K"])
    H_star = int(best_row["H"])
    sil_max = float(best_row["silhouette"])
    best_solution = best_runs[(K_star, H_star)]
    labels_kstar = best_solution.object_labels.copy()

    ari = float(adjusted_rand_score(y_original, labels_kstar))
    G_df = pd.DataFrame(
        best_solution.G,
        index=[f"obj_cluster_{k}" for k in range(K_star)],
        columns=[f"var_cluster_{h}" for h in range(H_star)],
    )
    contingency = pd.crosstab(
        pd.Series(labels_kstar, name="cluster_double_kmeans"),
        pd.Series(y_original, name="classe_original"),
    )
    var_df = (
        pd.DataFrame(
            {
                "feature": active_feature_names,
                "variable_cluster": best_solution.variable_labels,
            }
        )
        .sort_values(["variable_cluster", "feature"])
        .reset_index(drop=True)
    )

    q2_df = X_df.copy()
    q2_df["y_original"] = y_original
    q2_df["y_double_kmeans"] = labels_kstar

    metadata_df = pd.DataFrame(
        [
            {
                "K_star": K_star,
                "H_star": H_star,
                "silhouette": sil_max,
                "objective_W": float(best_solution.objective),
                "best_seed": int(best_solution.seed),
                "adjusted_rand_index": ari,
                "n_restarts_per_KH": n_restarts,
                "scaling_used_for_clustering": config.descricao_scaling,
            }
        ]
    )

    comp_df = pd.DataFrame(
        [
            {
                "Experimento": "Original (StandardScaler puro)",
                "Melhor (K*, H*)": "(2, 2)",
                "Tamanhos dos Clusters": "34 / 4567 (0,74% / 99,26%)",
                "Silhueta Euclidiana": 0.6596,
                "ARI (vs. Spam/Non-spam)": -0.00495,
                "Diagnóstico": "Isolamento de 34 outliers da HP Labs; Silhueta inflada artificialmente.",
            },
            {
                "Experimento": config.titulo,
                "Melhor (K*, H*)": f"({K_star}, {H_star})",
                "Tamanhos dos Clusters": " / ".join(str(s) for s in np.bincount(labels_kstar)),
                "Silhueta Euclidiana": round(sil_max, 4),
                "ARI (vs. Spam/Non-spam)": round(ari, 4),
                "Diagnóstico": config.diagnostico_comparativo,
            },
        ]
    )

    return AvaliacaoParticao(
        K_star=K_star,
        H_star=H_star,
        best_solution=best_solution,
        results_df=results_df,
        silhouette=sil_max,
        ari=ari,
        G_df=G_df,
        contingency=contingency,
        var_df=var_df,
        metadata_df=metadata_df,
        q2_df=q2_df,
        comp_df=comp_df,
    )


def _finalizar_figura(fig: plt.Figure, save_path: Optional[Path], exibir: bool) -> None:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if exibir:
        plt.show()
    else:
        plt.close(fig)


def plotar_graficos(
    avaliacao: AvaliacaoParticao,
    config: EstrategiaConfig,
    output_dir: Optional[Path] = None,
    exibir: bool = False,
) -> None:
    """Gera os três gráficos obrigatórios (Silhueta, Contingência e Convergência)."""
    df = avaliacao.results_df.copy()
    df["KH"] = [f"({int(k)},{int(h)})" for k, h in zip(df["K"], df["H"])]

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(df["KH"], df["silhouette"], color="#2b5c8f", edgecolor="black")
    ax1.set_xlabel("(K, H)", fontsize=12)
    ax1.set_ylabel("Silhueta Euclidiana", fontsize=12)
    ax1.set_title(f"Silhueta por par (K, H) — {config.titulo}", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)
    _finalizar_figura(fig1, (output_dir / "silhouette_KH.png") if output_dir else None, exibir)

    cont = avaliacao.contingency
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.imshow(cont.values, aspect="auto")
    ax2.set_xticks(np.arange(cont.shape[1]))
    ax2.set_xticklabels(cont.columns)
    ax2.set_yticks(np.arange(cont.shape[0]))
    ax2.set_yticklabels(cont.index)
    ax2.set_xlabel("Classe original")
    ax2.set_ylabel("Cluster Double K-means")
    ax2.set_title(f"Clusters × classes — {config.nome}")
    for i in range(cont.shape[0]):
        for j in range(cont.shape[1]):
            ax2.text(j, i, str(cont.iloc[i, j]), ha="center", va="center")
    _finalizar_figura(fig2, (output_dir / "matriz_contingencia.png") if output_dir else None, exibir)

    history = np.asarray(avaliacao.best_solution.objective_history)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(np.arange(len(history)), history, marker="o", markersize=3)
    ax3.set_xlabel("Iteração")
    ax3.set_ylabel("Função objetivo W")
    ax3.set_title(f"Convergência — {config.nome} (K*={avaliacao.K_star}, H*={avaliacao.H_star})")
    ax3.grid(alpha=0.25)
    _finalizar_figura(fig3, (output_dir / "objetivo_iteracoes.png") if output_dir else None, exibir)


def salvar_artefatos_experimento(
    output_dir: Path,
    avaliacao: AvaliacaoParticao,
    config: EstrategiaConfig,
    exibir_graficos: bool = False,
) -> None:
    """Grava os arquivos CSV, TXT e PNG padronizados de uma estratégia."""
    output_dir.mkdir(parents=True, exist_ok=True)
    avaliacao.results_df.to_csv(output_dir / "resultados_KH.csv", index=False)
    avaliacao.G_df.to_csv(output_dir / "matriz_prototipos_G.csv")
    avaliacao.contingency.to_csv(output_dir / "matriz_contingencia.csv")
    avaliacao.var_df.to_csv(output_dir / "particao_variaveis.csv", index=False)
    avaliacao.q2_df.to_csv(output_dir / config.csv_saida, index=False)
    avaliacao.metadata_df.to_csv(output_dir / "metadata_para_questao2.csv", index=False)
    if config.nome != "revisada":
        avaliacao.comp_df.to_csv(output_dir / "comparativo_com_original.csv", index=False)

    resumo = "\n".join(
        [
            f"K* = {avaliacao.K_star}",
            f"H* = {avaliacao.H_star}",
            f"Silhouette = {avaliacao.silhouette:.10f}",
            f"Best objective W = {avaliacao.best_solution.objective:.10f}",
            f"Best seed = {avaliacao.best_solution.seed}",
            f"Adjusted Rand Index = {avaliacao.ari:.10f}",
        ]
    )
    (output_dir / "resumo_resultado_final.txt").write_text(resumo + "\n", encoding="utf-8")
    plotar_graficos(avaliacao, config, output_dir=output_dir, exibir=exibir_graficos)


def executar_estrategia(
    nome: str,
    reutilizar_grid_salvo: bool = True,
    n_restarts: int = 100,
    base_seed: int = 42,
    max_iter: int = 200,
    salvar: bool = False,
    output_dir: Optional[Path] = None,
    exibir_graficos: bool = False,
    verbose: bool = True,
) -> dict:
    """Executa ou reproduz o protocolo completo da Questão 1 para uma estratégia."""
    X_df, X_original, y_original, feature_names = carregar_spambase()
    X_proc, active_names, config = aplicar_preprocessamento(
        nome=nome, X_original=X_original, feature_names=feature_names, base_seed=base_seed
    )

    pasta_saida = Path(output_dir) if output_dir is not None else (PASTA_RESULTADOS / config.nome)
    grid_csv = pasta_saida / "resultados_KH.csv"

    if reutilizar_grid_salvo and grid_csv.exists():
        results_df = pd.read_csv(grid_csv)
        best_runs = reproduzir_grade(X_proc, results_df, max_iter=max_iter)
    else:
        results_df, best_runs = executar_grade(
            X_proc, n_restarts=n_restarts, base_seed=base_seed, max_iter=max_iter, verbose=verbose
        )

    checks_df = verificar_sanidade(best_runs)
    avaliacao = avaliar_melhor_particao(
        X_df=X_df,
        y_original=y_original,
        active_feature_names=active_names,
        results_df=results_df,
        best_runs=best_runs,
        config=config,
        n_restarts=n_restarts,
    )

    if salvar:
        salvar_artefatos_experimento(
            output_dir=pasta_saida,
            avaliacao=avaliacao,
            config=config,
            exibir_graficos=exibir_graficos,
        )
    elif exibir_graficos:
        plotar_graficos(avaliacao, config, output_dir=None, exibir=True)

    return {
        "config": config,
        "X_df": X_df,
        "X_original": X_original,
        "X": X_proc,
        "y_original": y_original,
        "feature_names": feature_names,
        "active_feature_names": active_names,
        "results_df": results_df,
        "best_runs": best_runs,
        "checks_df": checks_df,
        "avaliacao": avaliacao,
        "output_dir": pasta_saida,
    }


def carregar_resumo_comparativo(resultados_dir: Path = PASTA_RESULTADOS) -> pd.DataFrame:
    """Consolida métricas vencedoras e tamanhos de clusters de todas as estratégias."""
    linhas = []
    for nome, config in ESTRATEGIAS.items():
        pasta = Path(resultados_dir) / nome
        meta_path = pasta / "metadata_para_questao2.csv"
        csv_path = pasta / config.csv_saida
        if not meta_path.exists():
            continue
        meta = pd.read_csv(meta_path).iloc[0]
        tamanhos = ""
        if csv_path.exists():
            rotulos = pd.read_csv(csv_path, usecols=["y_double_kmeans"])["y_double_kmeans"].to_numpy()
            tamanhos = " / ".join(str(int(c)) for c in np.bincount(rotulos))
        linhas.append(
            {
                "estrategia": nome,
                "titulo": config.titulo,
                "K_star": int(meta["K_star"]),
                "H_star": int(meta["H_star"]),
                "silhouette": float(meta["silhouette"]),
                "objective_W": float(meta["objective_W"]),
                "best_seed": int(meta["best_seed"]),
                "ARI": float(meta["adjusted_rand_index"]),
                "tamanhos_clusters": tamanhos,
                "arquivo_q2": config.csv_saida,
            }
        )
    return pd.DataFrame(linhas)
