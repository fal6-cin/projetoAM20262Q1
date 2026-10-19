"""Executar: python -m questao2.scripts.rodar_gaussiano --max-folds 1

Para executar o protocolo COMPLETO (custoso):
    python -m questao2.scripts.rodar_gaussiano --todos-folds --repeticoes 30
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from ..src.avaliacao_gaussiano import avaliar_gaussiano
from ..src.dados import carregar_dados, validar_base


def salvar_csv(caminho, linhas):
    if not linhas:
        raise ValueError('Nenhum resultado para salvar')
    caminho.parent.mkdir(parents=True, exist_ok=True)
    cols = list(linhas[0].keys())
    with caminho.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in linhas:
            w.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v,dict) else v for k,v in row.items()})


def main(argv=None):
    p=argparse.ArgumentParser(description='Avaliação aninhada Bayes gaussiano completo')
    p.add_argument('--alvo', choices=('ambos','y_original','y_double_kmeans'), default='ambos')
    p.add_argument('--repeticoes', type=int, default=1, help='30 para estudo completo')
    p.add_argument('--max-folds', type=int, default=1, help='Número de folds de smoke-test; requer --todos-folds para executar todos')
    p.add_argument('--todos-folds', action='store_true', help='Ignora --max-folds; 10 folds/repetição')
    p.add_argument('--saida', default=None, help='Diretório para arquivos CSV de saída')
    args=p.parse_args(argv)
    if args.repeticoes < 1: p.error('--repeticoes deve ser >=1')
    if args.max_folds < 1: p.error('--max-folds deve ser >=1')
    info = validar_base()
    print('Dados conferidos:', info)
    X, ys, _ = carregar_dados()
    nomes = list(ys) if args.alvo == 'ambos' else [args.alvo]
    caminho_saida = Path(args.saida) if args.saida else Path(__file__).resolve().parents[1]/'resultados'/'gaussiano'
    parcial = not args.todos_folds
    for nome in nomes:
        linhas, predicoes = avaliar_gaussiano(X,ys[nome],nome,repeticoes=args.repeticoes,
            max_folds=None if args.todos_folds else args.max_folds, guardar_predicoes=True)
        tag = 'PARCIAL' if parcial else f'{args.repeticoes}x10'
        nome_saida = f'parcial_{len(linhas)}folds' if parcial else f'{args.repeticoes}x10'
        salvar_csv(caminho_saida / f'{nome}_{nome_saida}_folds.csv', linhas)
        salvar_csv(caminho_saida / f'{nome}_{nome_saida}_predicoes.csv',predicoes)
        print(f'{nome}: {len(linhas)} folds {tag}. F1 macro = {sum(r["f_measure_macro"] for r in linhas)/len(linhas):.4f}; '
              f'taxa erro = {sum(r["taxa_erro"] for r in linhas)/len(linhas):.4f}')
        print('Arquivo:', caminho_saida)
    if parcial:
        print('ATENÇÃO: execução parcial para depuração; NÃO constitui 30 × 10-folds.')
    elif args.repeticoes !=30:
        print('ATENÇÃO: executar --repeticoes 30 --todos-folds para o protocolo 30×10.')


if __name__ == '__main__': main()
