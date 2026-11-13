"""CLI unificada para executar, reproduzir ou consolidar os experimentos da Questão 1."""
from __future__ import annotations

import argparse
from questao1.src.experimento import carregar_resumo_comparativo, executar_estrategia
from questao1.src.preprocessamentos import listar_estrategias


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa ou valida experimentos da Questão 1.")
    parser.add_argument(
        "--estrategia",
        default="todas",
        choices=["todas", *listar_estrategias()],
        help="Estratégia a executar/validar (padrão: todas).",
    )
    parser.add_argument(
        "--reexecutar",
        action="store_true",
        help="Refaz os 100 restarts por (K,H) em vez de reproduzir pelas seeds salvas.",
    )
    parser.add_argument(
        "--salvar",
        action="store_true",
        help="Grava CSVs e gráficos em questao1/resultados/<estrategia>/.",
    )
    parser.add_argument(
        "--resumo",
        action="store_true",
        help="Apenas exibe o comparativo consolidado das estratégias já salvas.",
    )
    args = parser.parse_args()

    if args.resumo:
        print(carregar_resumo_comparativo().to_string(index=False))
        return

    alvos = listar_estrategias() if args.estrategia == "todas" else [args.estrategia]
    for nome in alvos:
        res = executar_estrategia(
            nome=nome,
            reutilizar_grid_salvo=not args.reexecutar,
            salvar=args.salvar,
            exibir_graficos=False,
        )
        av = res["avaliacao"]
        print(
            f"[{nome}] (K*,H*)=({av.K_star},{av.H_star}) | "
            f"Sil={av.silhouette:.6f} | ARI={av.ari:.6f} | W={av.best_solution.objective:.4f}"
        )


if __name__ == "__main__":
    main()
