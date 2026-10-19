"""Na raiz do repositório: python -m questao2.scripts.rodar_logistica

Experimento completo: python -m questao2.scripts.rodar_logistica --todos-folds --repeticoes 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..src.avaliacao_logistica import avaliar_logistica, VALORES_C, PESOS_CLASSES
from ..src.dados import carregar_dados, validar_base
from .rodar_gaussiano import salvar_csv


def main(argv=None):
    p = argparse.ArgumentParser(description='Regressão logística L2 + CV 5-fold interna')
    p.add_argument('--alvo', choices=('ambos', 'y_original', 'y_double_kmeans'), default='ambos')
    p.add_argument('--repeticoes', type=int, default=1)
    p.add_argument('--max-folds', type=int, default=1)
    p.add_argument('--todos-folds', action='store_true')
    p.add_argument('--saida', default=None)
    args = p.parse_args(argv)
    if args.repeticoes < 1 or args.max_folds < 1:
        p.error('--repeticoes e --max-folds devem ser >= 1')
    print('Base validada:', validar_base(), flush=True)
    print('Grade C:', VALORES_C, 'Pesos:', PESOS_CLASSES, flush=True)
    X, ys, _ = carregar_dados()
    nomes = list(ys) if args.alvo == 'ambos' else [args.alvo]
    saida = Path(args.saida) if args.saida else Path(__file__).resolve().parents[1] / 'resultados' / 'logistica'
    for nome in nomes:
        linhas, predicoes = avaliar_logistica(
            X, ys[nome], nome, repeticoes=args.repeticoes,
            max_folds=None if args.todos_folds else args.max_folds,
            guardar_predicoes=True)
        tag = f'{args.repeticoes}x10' if args.todos_folds else f'parcial_{len(linhas)}folds'
        salvar_csv(saida / f'{nome}_{tag}_folds.csv', linhas)
        salvar_csv(saida / f'{nome}_{tag}_predicoes.csv', predicoes)
        print(f'{nome}: {len(linhas)} folds. '
              f'F1 macro={sum(z["f_measure_macro"] for z in linhas)/len(linhas):.4f}; '
              f'erro={sum(z["taxa_erro"] for z in linhas)/len(linhas):.4f}', flush=True)
        print('Hiperparâmetros por fold:',
              [(z['C'], z['class_weight']) for z in linhas], flush=True)
        print('CSV:', saida, flush=True)
    if not args.todos_folds:
        print('ATENÇÃO: apenas avaliação parcial, NÃO equivale a 30 × 10 folds.')
    elif args.repeticoes != 30:
        print('ATENÇÃO: experimento final exige --todos-folds --repeticoes 30.')


if __name__ == '__main__':
    main()
