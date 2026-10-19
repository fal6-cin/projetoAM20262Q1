"""Combinar por voto predições externas já salvas dos quatro classificadores.

Da raiz do repo:
  python -m questao2.scripts.rodar_voto --todos-folds --repeticoes 1
  python -m questao2.scripts.rodar_voto --todos-folds --repeticoes 30

Para 30x10, primeiro produzir os arquivos *_30x10_predicoes.csv dos quatro.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..src.avaliacao_voto import avaliar_voto_de_csvs, PASTA_RESULTADOS
from ..src.dados import validar_base
from .rodar_gaussiano import salvar_csv


def main(argv=None):
    p = argparse.ArgumentParser(description='Quinto classificador: voto majoritário de 4 modelos')
    p.add_argument('--alvo', choices=('ambos','y_original','y_double_kmeans'), default='ambos')
    p.add_argument('--repeticoes', type=int, default=1)
    p.add_argument('--max-folds', type=int, default=1)
    p.add_argument('--todos-folds', action='store_true')
    p.add_argument('--entrada', default=None, help='Pasta resultados/ com subpastas gaussiano, knn, parzen, logistica')
    p.add_argument('--saida', default=None, help='Pasta para salvar métricas e predições do voto')
    args = p.parse_args(argv)
    if args.repeticoes < 1 or args.max_folds < 1:
        p.error('--repeticoes e --max-folds devem ser >= 1')
    print('Base:', validar_base(), flush=True)
    entrada = Path(args.entrada) if args.entrada else PASTA_RESULTADOS
    saida = Path(args.saida) if args.saida else PASTA_RESULTADOS / 'voto'
    nomes = ('y_original', 'y_double_kmeans') if args.alvo == 'ambos' else (args.alvo,)
    for alvo in nomes:
        folds, prev = avaliar_voto_de_csvs(
            alvo, repeticoes=args.repeticoes,
            max_folds=None if args.todos_folds else args.max_folds,
            diretorio_resultados=entrada, guardar_predicoes=True)
        tag = f'{args.repeticoes}x10' if args.todos_folds else f'parcial_{len(folds)}folds'
        salvar_csv(saida / f'{alvo}_{tag}_folds.csv', folds)
        salvar_csv(saida / f'{alvo}_{tag}_predicoes.csv', prev)
        print(f'{alvo}: {len(folds)} folds, {len(prev)} predições; '
              f'F1 macro={sum(f["f_measure_macro"] for f in folds)/len(folds):.4f}; '
              f'erro={sum(f["taxa_erro"] for f in folds)/len(folds):.4f}; '
              f'empates={sum(f["empates_resolvidos_priori"] for f in folds)}', flush=True)
        print('Arquivos:', saida, flush=True)
    if not args.todos_folds or args.repeticoes != 30:
        print('ATENÇÃO: diagnóstico preliminar; a Questão 2 pede 30 × 10 folds.')


if __name__ == '__main__':
    main()
