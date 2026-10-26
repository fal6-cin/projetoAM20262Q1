"""Experimento suplementar de ablacao: sem, MinMax, Log1p+MinMax.

Execucao desde a raiz do repositorio:
  python -m questao2.scripts.comparar_preprocessamento --repeticoes 1
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from questao2.src.comparacao_preprocessamento import executar, consolidar, VARIANTES, MODELOS_BASE

PASTA_Q2 = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alvo',choices=['todos','y_original','y_double_kmeans'],default='todos')
    p.add_argument('--variante',choices=['todas',*VARIANTES],default='todas')
    p.add_argument('--modelos', nargs='+',choices=MODELOS_BASE,default=list(MODELOS_BASE))
    p.add_argument('--repeticoes', type=int, default=1)
    p.add_argument('--max-folds', type=int, default=None,
                   help='Rodada diagnostica: nao representa 10-fold completo')
    p.add_argument('--apenas-consolidar', action='store_true')
    args=p.parse_args()
    pasta=PASTA_Q2/'resultados'/'comparacao_preprocessamento'
    alvos = ('y_original','y_double_kmeans') if args.alvo=='todos' else (args.alvo,)
    variantes = VARIANTES if args.variante=='todas' else (args.variante,)
    if args.apenas_consolidar:
        status=consolidar(pasta,alvos=alvos,variantes=variantes,repeticoes=args.repeticoes)
    else:
        status=executar(pasta,alvo=args.alvo,variante=args.variante,modelos=tuple(args.modelos),
                       repeticoes=args.repeticoes,max_folds=args.max_folds)
    print(json.dumps(status,indent=2,ensure_ascii=False))
    if not status['completo']:
        print('Experimento INCOMPLETO: resultados de diagnostico nao sao conclusivos.')

if __name__ == '__main__':
    main()
