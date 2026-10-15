
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DoubleKMeansResult:
    object_labels: np.ndarray
    variable_labels: np.ndarray
    G: np.ndarray
    objective: float
    objective_history: list[float]
    n_iter: int
    seed: int


def _initial_partition(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Gera uma partição aleatória hard garantindo inicialmente
    pelo menos um elemento em cada grupo.
    """
    labels = np.arange(n) % k
    rng.shuffle(labels)
    return labels


def _compute_prototypes(
    X: np.ndarray,
    object_labels: np.ndarray,
    variable_labels: np.ndarray,
    K: int,
    H: int,
) -> Optional[np.ndarray]:
    """
    g_kh = média dos valores x_ij pertencentes ao bloco (k,h).
    """
    G = np.empty((K, H), dtype=float)

    for k in range(K):
        rows = object_labels == k
        if not np.any(rows):
            return None

        for h in range(H):
            cols = variable_labels == h
            if not np.any(cols):
                return None

            G[k, h] = X[np.ix_(rows, cols)].mean()

    return G


def _objective(
    X: np.ndarray,
    object_labels: np.ndarray,
    variable_labels: np.ndarray,
    G: np.ndarray,
) -> float:
    """
    W(G,U,V) = sum_{k,h,i,j} u_ik v_jh (x_ij - g_kh)^2
    """
    fitted = G[object_labels[:, None], variable_labels[None, :]]
    return float(np.square(X - fitted).sum())


def double_kmeans(
    X: np.ndarray,
    K: int,
    H: int,
    seed: int,
    max_iter: int = 200,
    monotonic_tol: float = 1e-7,
) -> Optional[DoubleKMeansResult]:
    """
    Uma execução do Double K-means.

    Retorna None se durante a otimização surgir cluster vazio.
    Nesse caso, a inicialização deve ser descartada e uma nova seed testada.

    A implementação segue a alternância:
      1) atualizar protótipos G com U,V fixos;
      2) atualizar partição de objetos U com G,V fixos;
      3) atualizar partição de variáveis V com G,U fixos;
      4) repetir até não haver transferências.
    """
    X = np.asarray(X, dtype=float)
    N, P = X.shape

    if not (1 <= K <= N):
        raise ValueError("K deve estar entre 1 e o número de objetos.")
    if not (1 <= H <= P):
        raise ValueError("H deve estar entre 1 e o número de variáveis.")

    rng = np.random.default_rng(seed)

    object_labels = _initial_partition(N, K, rng)
    variable_labels = _initial_partition(P, H, rng)

    G = _compute_prototypes(X, object_labels, variable_labels, K, H)
    if G is None:
        return None

    objective_history = [_objective(X, object_labels, variable_labels, G)]
    X2 = X * X

    for iteration in range(1, max_iter + 1):
        old_object_labels = object_labels.copy()
        old_variable_labels = variable_labels.copy()

        # ---------------------------------------------------------
        # Passo 2 — atualizar objetos, com G e V fixos
        # D_ik = sum_h sum_j v_jh (x_ij - g_kh)^2
        # ---------------------------------------------------------
        object_costs = np.zeros((N, K), dtype=float)

        for h in range(H):
            cols = variable_labels == h
            nh = int(cols.sum())

            row_sum = X[:, cols].sum(axis=1)
            row_sum_sq = X2[:, cols].sum(axis=1)

            for k in range(K):
                g = G[k, h]
                object_costs[:, k] += (
                    row_sum_sq - 2.0 * g * row_sum + nh * g * g
                )

        new_object_labels = object_costs.argmin(axis=1)

        # Uma partição hard em K grupos deve manter K grupos não vazios.
        if np.unique(new_object_labels).size < K:
            return None

        object_labels = new_object_labels

        # ---------------------------------------------------------
        # Passo 3 — atualizar variáveis, com G e U fixos
        # D_jh = sum_k sum_i u_ik (x_ij - g_kh)^2
        # ---------------------------------------------------------
        variable_costs = np.zeros((P, H), dtype=float)

        for k in range(K):
            rows = object_labels == k
            nk = int(rows.sum())

            col_sum = X[rows, :].sum(axis=0)
            col_sum_sq = X2[rows, :].sum(axis=0)

            for h in range(H):
                g = G[k, h]
                variable_costs[:, h] += (
                    col_sum_sq - 2.0 * g * col_sum + nk * g * g
                )

        new_variable_labels = variable_costs.argmin(axis=1)

        if np.unique(new_variable_labels).size < H:
            return None

        variable_labels = new_variable_labels

        # ---------------------------------------------------------
        # Novo Passo 1 — atualizar G com as novas partições
        # ---------------------------------------------------------
        G = _compute_prototypes(
            X, object_labels, variable_labels, K, H
        )
        if G is None:
            return None

        W = _objective(X, object_labels, variable_labels, G)
        objective_history.append(W)

        # Verificação de sanidade: W não deve aumentar de modo relevante.
        if W > objective_history[-2] + monotonic_tol:
            raise RuntimeError(
                "A função objetivo aumentou. "
                "Revise a implementação ou a tolerância numérica."
            )

        # Critério descrito nos slides: parar quando não houver transferências.
        if (
            np.array_equal(object_labels, old_object_labels)
            and np.array_equal(variable_labels, old_variable_labels)
        ):
            return DoubleKMeansResult(
                object_labels=object_labels,
                variable_labels=variable_labels,
                G=G,
                objective=W,
                objective_history=objective_history,
                n_iter=iteration,
                seed=seed,
            )

    # Caso alcance max_iter, ainda devolvemos a última solução válida.
    return DoubleKMeansResult(
        object_labels=object_labels,
        variable_labels=variable_labels,
        G=G,
        objective=objective_history[-1],
        objective_history=objective_history,
        n_iter=max_iter,
        seed=seed,
    )
