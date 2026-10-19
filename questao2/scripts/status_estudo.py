"""Exibe estado da execução 30x10; nenhum treinamento ocorre aqui."""
from __future__ import annotations
import argparse
from pathlib import Path
from ..src.estatistica import MODELOS
from ..src.dados import carregar_dados
from ..src.auditoria_experimentos import auditar_resultados
from .executar_estudo import caminhos_shard, RAIZ


def main(argv=None):
    p=argparse.ArgumentParser(description='Status do experimento por modelo/alvo')
    p.add_argument('--saida',type=Path,default=RAIZ/'resultados')
    p.add_argument('--repeticoes',type=int,default=30)
    p.add_argument('--auditar',action='store_true',help='Ler e validar todos os shards encontrados')
    args=p.parse_args(argv)
    _,respostas,_=carregar_dados()
    print(f'Progresso da avaliação ({args.repeticoes} repetições por modelo e alvo)')
    for alvo,y in respostas.items():
        for modelo in MODELOS:
            if modelo=='voto':
                full=args.saida/'voto'/f'{alvo}_{args.repeticoes}x10_folds.csv'
                status='completo' if full.exists() else 'pendente'
                print(f'{alvo:19} {modelo:10} {status}')
                continue
            reps=[]
            for r in range(1,args.repeticoes+1):
                f,pr=caminhos_shard(args.saida,modelo,alvo,r)
                if f.exists()!=pr.exists(): raise ValueError(f'Par de arquivos incompleto: {f}')
                if f.exists():
                    if args.auditar:
                        auditar_resultados(f,pr,y,alvo,repeticoes=args.repeticoes,
                                           repeticao_inicial=r,repeticao_final=r)
                    reps.append(r)
            finais=(args.saida/modelo/f'{alvo}_{args.repeticoes}x10_folds.csv').exists()
            print(f'{alvo:19} {modelo:10} {len(reps):2d}/{args.repeticoes} repetições '
                  f'| consolidado={"sim" if finais else "não"} | faltantes={[r for r in range(1,args.repeticoes+1) if r not in reps][:10]}')

if __name__=='__main__': main()
