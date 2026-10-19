"""Execução incremental 30x10, retomável, com validação dos CSVs por repetição.

Exemplo: python -m questao2.scripts.executar_estudo --modelos todos --repeticoes 30
A base é lida em cada processo, mas cada repetição completa é salva isoladamente.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil

from ..src.dados import carregar_dados, validar_base
from ..src.avaliacao_gaussiano import avaliar_gaussiano
from ..src.avaliacao_knn import avaliar_knn
from ..src.avaliacao_parzen import avaliar_parzen
from ..src.avaliacao_logistica import avaliar_logistica
from ..src.avaliacao_voto import avaliar_voto_de_csvs
from ..src.auditoria_experimentos import auditar_resultados, ler_csv
from .rodar_gaussiano import salvar_csv

AVALIADORES = {'gaussiano': avaliar_gaussiano, 'knn': avaliar_knn,
               'parzen': avaliar_parzen, 'logistica': avaliar_logistica}
RAIZ = Path(__file__).resolve().parents[1]


def _gravar_atomicamente(caminho, linhas):
    caminho=Path(caminho)
    temp=caminho.with_name(caminho.name+'.temporario')
    try:
        salvar_csv(temp,linhas)
        os.replace(temp,caminho)
    finally:
        temp.unlink(missing_ok=True)


def _concatenar_csv(arquivos, destino):
    destino=Path(destino)
    destino.parent.mkdir(parents=True,exist_ok=True)
    tmp=destino.with_name(destino.name+'.temporario')
    try:
        with tmp.open('w',newline='',encoding='utf-8') as out:
            writer=None
            for arquivo in arquivos:
                with Path(arquivo).open(newline='',encoding='utf-8') as src:
                    reader=csv.DictReader(src)
                    if writer is None:
                        writer=csv.DictWriter(out,fieldnames=reader.fieldnames)
                        writer.writeheader()
                    elif reader.fieldnames != writer.fieldnames:
                        raise ValueError('Colunas diferentes entre repetições')
                    writer.writerows(reader)
        os.replace(tmp,destino)
    finally:
        tmp.unlink(missing_ok=True)


def caminhos_shard(saida,modelo,alvo,rep):
    pasta=Path(saida)/'execucoes'/modelo/alvo
    return pasta/f'rep_{rep:02d}_folds.csv',pasta/f'rep_{rep:02d}_predicoes.csv'


def consolidar(saida, modelo, alvo, y, repeticoes=30, semente=2026):
    arquivos_f,arquivos_p=[],[]
    for rep in range(1,repeticoes+1):
        f,p=caminhos_shard(saida,modelo,alvo,rep)
        if not f.exists() or not p.exists():
            return False
        auditar_resultados(f,p,y,alvo,repeticoes=repeticoes, semente=semente,
                           repeticao_inicial=rep,repeticao_final=rep)
        arquivos_f.append(f); arquivos_p.append(p)
    pasta=Path(saida)/modelo
    tag=f'{repeticoes}x10'
    final_f=pasta/f'{alvo}_{tag}_folds.csv'
    final_p=pasta/f'{alvo}_{tag}_predicoes.csv'
    _concatenar_csv(arquivos_f,final_f)
    _concatenar_csv(arquivos_p,final_p)
    auditar_resultados(final_f,final_p,y,alvo,repeticoes=repeticoes, semente=semente)
    return True


def executar_alvo_modelo(X,y,alvo,modelo,saida,repeticoes,inicio,fim,semente,
                          usar_diagnostico=True,apenas_consolidar=False):
    for rep in range(inicio,fim+1):
        f,p=caminhos_shard(saida,modelo,alvo,rep)
        if f.exists() and p.exists():
            auditar_resultados(f,p,y,alvo,repeticoes=repeticoes,semente=semente,
                               repeticao_inicial=rep,repeticao_final=rep)
            print(f'{alvo}/{modelo} repetição {rep}: já concluída e auditada',flush=True)
            continue
        if f.exists() != p.exists():
            raise ValueError(f'Repetição incompleta em disco: {f} / {p}; confira antes de retomar')
        if apenas_consolidar:
            print(f'Faltando {alvo}/{modelo} repetição {rep}: nenhuma execução realizada',flush=True)
            continue
        if rep==1 and usar_diagnostico and semente==2026:
            original_f=Path(saida)/modelo/f'{alvo}_1x10_folds.csv'
            original_p=Path(saida)/modelo/f'{alvo}_1x10_predicoes.csv'
            if original_f.exists() and original_p.exists():
                auditar_resultados(original_f,original_p,y,alvo,repeticoes=1, semente=semente)
                f.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(original_f,f); shutil.copyfile(original_p,p)
                print(f'{alvo}/{modelo} repetição 1: diagnóstico auditado reaproveitado',flush=True)
                continue
        print(f'{alvo}/{modelo} repetição {rep}/{repeticoes}: iniciando 10 folds',flush=True)
        linhas,pred=AVALIADORES[modelo](X,y,alvo,repeticoes=rep,repeticao_inicial=rep,
                                         semente=semente, guardar_predicoes=True)
        if len(linhas)!=10 or len(pred)!=len(y):
            raise AssertionError('O avaliador não produziu os 10 folds esperados')
        _gravar_atomicamente(f,linhas)
        _gravar_atomicamente(p,pred)
        auditar_resultados(f,p,y,alvo,repeticoes=repeticoes,semente=semente,
                           repeticao_inicial=rep,repeticao_final=rep)
        print(f'{alvo}/{modelo} repetição {rep}: salva e auditada; '
              f'F1={sum(x["f_measure_macro"] for x in linhas)/10:.4f}',flush=True)
    pronto=consolidar(saida,modelo,alvo,y,repeticoes=repeticoes,semente=semente)
    if pronto:
        print(f'{alvo}/{modelo}: {repeticoes}x10 consolidado e auditado',flush=True)
    else:
        print(f'{alvo}/{modelo}: ainda sem todas as {repeticoes} repetições; '
              'NÃO existe arquivo final 30x10',flush=True)
    return pronto


def main(argv=None):
    p=argparse.ArgumentParser(description='Rodar 30x10 aninhado com salvamento por repetição e retomada')
    p.add_argument('--modelos',nargs='+',choices=(*AVALIADORES,'todos'),default=['todos'])
    p.add_argument('--alvo',choices=('ambos','y_original','y_double_kmeans'),default='ambos')
    p.add_argument('--repeticoes',type=int,default=30)
    p.add_argument('--inicio',type=int,default=1,help='primeira repetição a processar')
    p.add_argument('--fim',type=int,default=None,help='última repetição; por padrão, todas')
    p.add_argument('--saida',type=Path,default=RAIZ/'resultados')
    p.add_argument('--apenas-consolidar',action='store_true')
    p.add_argument('--nao-reutilizar-1x10',action='store_true')
    p.add_argument('--sem-voto',action='store_true')
    p.add_argument('--semente',type=int,default=2026)
    args=p.parse_args(argv)
    fim=args.fim if args.fim is not None else args.repeticoes
    if not 1 <= args.inicio <= fim <= args.repeticoes:
        p.error('É necessário 1 <= inicio <= fim <= repeticoes')
    modelos=tuple(AVALIADORES) if 'todos' in args.modelos else tuple(dict.fromkeys(args.modelos))
    alvos=('y_original','y_double_kmeans') if args.alvo=='ambos' else (args.alvo,)
    print('Base:',validar_base(),flush=True)
    X, respostas, _ = carregar_dados()
    for alvo in alvos:
        for modelo in modelos:
            executar_alvo_modelo(X,respostas[alvo],alvo,modelo,args.saida,args.repeticoes,
                                 args.inicio,fim,args.semente,
                                 usar_diagnostico=not args.nao_reutilizar_1x10,
                                 apenas_consolidar=args.apenas_consolidar)
        if not args.sem_voto and all((args.saida/m/f'{alvo}_{args.repeticoes}x10_predicoes.csv').exists()
                                     for m in AVALIADORES):
            print(f'{alvo}: calculando voto dos quatro classificadores',flush=True)
            res,pred=avaliar_voto_de_csvs(alvo,repeticoes=args.repeticoes,
                                          diretorio_resultados=args.saida,semente=args.semente)
            destino=args.saida/'voto'; tag=f'{args.repeticoes}x10'
            _gravar_atomicamente(destino/f'{alvo}_{tag}_folds.csv',res)
            _gravar_atomicamente(destino/f'{alvo}_{tag}_predicoes.csv',pred)
            auditar_resultados(destino/f'{alvo}_{tag}_folds.csv',
                               destino/f'{alvo}_{tag}_predicoes.csv',
                               respostas[alvo],alvo,args.repeticoes,semente=args.semente)
            print(f'{alvo}: voto gerado e auditado',flush=True)
        elif not args.sem_voto:
            print(f'{alvo}: voto pendente; faltam resultados completos de classificadores-base',flush=True)
    print('Fim da execução. O relatório final requer cinco modelos x dois alvos x 30 repetições.',flush=True)

if __name__=='__main__': main()
