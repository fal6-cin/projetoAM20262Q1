"""Executar na raiz: python -m questao2.scripts.rodar_knn --max-folds 1

Experimento completo: python -m questao2.scripts.rodar_knn --todos-folds --repeticoes 30
"""
from __future__ import annotations
import argparse
from pathlib import Path

from ..src.avaliacao_knn import avaliar_knn, VALORES_K, DISTANCIAS
from ..src.dados import carregar_dados, validar_base
from .rodar_gaussiano import salvar_csv


def main(argv=None):
    p = argparse.ArgumentParser(description='Avaliação aninhada do Bayes k-vizinhos')
    p.add_argument('--alvo', choices=('ambos','y_original','y_double_kmeans'), default='ambos')
    p.add_argument('--repeticoes', type=int, default=1, help='30 para estudo completo')
    p.add_argument('--max-folds', type=int, default=1, help='Limite para teste rápido; requer --todos-folds para avaliar todos')
    p.add_argument('--todos-folds', action='store_true')
    p.add_argument('--saida', default=None)
    args = p.parse_args(argv)
    if args.repeticoes < 1 or args.max_folds < 1:
        p.error('--repeticoes e --max-folds devem ser >= 1')
    print('Base validada:', validar_base())
    print('Grade k =', VALORES_K, '| distâncias =', DISTANCIAS)
    X, ys, _ = carregar_dados()
    nomes = list(ys) if args.alvo == 'ambos' else [args.alvo]
    saida = Path(args.saida) if args.saida else Path(__file__).resolve().parents[1] / 'resultados' / 'knn'
    for nome in nomes:
        linhas, preditos = avaliar_knn(X, ys[nome], nome, repeticoes=args.repeticoes,
                          max_folds=None if args.todos_folds else args.max_folds,
                          guardar_predicoes=True)
        tag = f'{args.repeticoes}x10' if args.todos_folds else f'parcial_{len(linhas)}folds'
        salvar_csv(saida / f'{nome}_{tag}_folds.csv', linhas)
        salvar_csv(saida / f'{nome}_{tag}_predicoes.csv', preditos)
        print(f'{nome}: {len(linhas)} folds. F1 macro={sum(z["f_measure_macro"] for z in linhas)/len(linhas):.4f}; '
              f'erro={sum(z["taxa_erro"] for z in linhas)/len(linhas):.4f}')
        print('Contagem de pares k/distância:', {f'{z["n_vizinhos"]}/{z["distancia"]}':sum(t['n_vizinhos']==z['n_vizinhos'] and t['distancia']==z['distancia'] for t in linhas) for z in linhas if z})
        print('Resultados:', saida)
    if not args.todos_folds:
        print('ATENÇÃO: execução PARCIAL, apenas depuração; NÃO é 30×10.')
    elif args.repeticoes != 30:
        print('ATENÇÃO: são necessárias 30 repetições para o estudo final.')


if __name__ == '__main__':
    main()
