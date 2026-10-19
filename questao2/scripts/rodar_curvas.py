"""Curvas: python -m questao2.scripts.rodar_curvas --repeticoes 1

Cada curva tem 19 pontos (5 a 95%) x 5 classificadores x 2 alvos.
Retoma ponto a ponto; gráficos finais só surgem após todos os pontos.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from ..src.dados import carregar_dados, validar_base
from ..src.curvas_aprendizagem import PERCENTUAIS, TODOS_MODELOS, avaliar_ponto
from ..src.graficos_curvas import consolidar_curvas

PASTA = Path(__file__).resolve().parents[1]/'resultados'/'curvas_aprendizagem'


def salvar_ponto(arquivo: Path, linhas):
    if len(linhas)!=len(TODOS_MODELOS) or {x['modelo'] for x in linhas}!=set(TODOS_MODELOS):
        raise ValueError('Ponto incompleto: exigidos os cinco modelos')
    arquivo.parent.mkdir(parents=True,exist_ok=True)
    tmp=arquivo.with_name(arquivo.name+'.temporario')
    try:
        with tmp.open('w', newline='', encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(linhas[0]))
            w.writeheader();w.writerows(linhas)
        os.replace(tmp,arquivo)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None):
    p=argparse.ArgumentParser(description='Curvas de aprendizagem 5% a 95%, com CV interna 5-fold')
    p.add_argument('--repeticoes',type=int,default=1,help='Sorteios estratificados independentes; a letra d não exige 30')
    p.add_argument('--inicio',type=int,default=1,help='Primeira repetição a realizar')
    p.add_argument('--fim',type=int,default=None,help='Última repetição; padrão = --repeticoes')
    p.add_argument('--alvo',choices=['ambos','y_original','y_double_kmeans'],default='ambos')
    p.add_argument('--pct-inicio',type=int,default=5,help='Primeiro percentual (múltiplo de 5)')
    p.add_argument('--pct-fim',type=int,default=95,help='Último percentual (múltiplo de 5)')
    p.add_argument('--saida',type=Path,default=PASTA)
    p.add_argument('--semente',type=int,default=2026)
    p.add_argument('--apenas-consolidar',action='store_true',help='Não treina; audita pontos e gera gráficos se completo')
    args=p.parse_args(argv)
    fim = args.fim if args.fim is not None else args.repeticoes
    if not 1 <= args.inicio <= fim <= args.repeticoes: p.error('Inicio/fim/repeticoes inválidos')
    if (args.pct_inicio not in PERCENTUAIS or args.pct_fim not in PERCENTUAIS
            or args.pct_inicio>args.pct_fim): p.error('Percentuais devem ser 5,10,...,95')
    if args.semente < 0: p.error('Semente deve ser >=0')
    alvos = ('y_original','y_double_kmeans') if args.alvo=='ambos' else (args.alvo,)
    if not args.apenas_consolidar:
        print('Base:',validar_base(),flush=True)
        X, respostas, _=carregar_dados()
        for alvo in alvos:
            for rep in range(args.inicio, fim+1):
                for pct in PERCENTUAIS:
                    if not args.pct_inicio <=pct<=args.pct_fim: continue
                    arquivo=args.saida/'execucoes'/alvo/f'rep_{rep:02d}_pct_{pct:02d}.csv'
                    if arquivo.exists():
                        # Validação do arquivo existente feita na consolidação.
                        from pandas import read_csv
                        t=read_csv(arquivo)
                        if (len(t)!=5 or set(t.modelo)!=set(TODOS_MODELOS) or
                            set(t.alvo)!={alvo} or set(t.repeticao)!={rep} or
                            set(t.percentual_treino)!={pct} or set(t.semente_particao)!={args.semente+rep-1}):
                            raise ValueError(f'Ponto existente inválido: {arquivo}')
                        print(f'{alvo} rep={rep} treino={pct}%: já salvo',flush=True)
                        continue
                    print(f'{alvo} rep={rep} treino={pct}%: ajustando 4 modelos e voto...',flush=True)
                    linhas=avaliar_ponto(X,respostas[alvo],alvo,pct,repeticao=rep,semente=args.semente)
                    salvar_ponto(arquivo,linhas)
                    print(f'  F1 teste:',{r['modelo']:round(r['f1_macro_teste'],4) for r in linhas},flush=True)
    try:
        resumo, faltantes=consolidar_curvas(args.saida,alvos=alvos,repeticoes=args.repeticoes,
                                            exigir_completo=False)
    except ValueError as exc:
        print('Sem dados suficientes para consolidar:',exc,flush=True)
        return
    if faltantes:
        print(f'PARCIAL: faltam {len(faltantes)} pontos; NÃO há curvas finais. Retome o comando.',flush=True)
    else:
        print(f'CURVAS COMPLETAS para alvos {alvos} e {args.repeticoes} repetição(ões). '
              f'Arquivos: {args.saida / "resumo_curvas.csv"} e {args.saida / "graficos"}',flush=True)

if __name__=='__main__':main()
