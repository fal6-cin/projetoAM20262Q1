"""Resumos e comparação pareada sobre CSVs externos já concluídos.

Nunca aplica Friedman/Nemenyi nem chama IC "final" a dados 1x10.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..src.dados import carregar_dados, validar_base
from ..src.auditoria_experimentos import auditar_resultados, ler_csv, METRICAS
from ..src.estatistica import MODELOS, medias_por_repeticao, sumarizar_metricas, friedman_nemenyi

RAIZ=Path(__file__).resolve().parents[1]


def escrever_csv(path, linhas):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not linhas: raise ValueError('Nenhuma linha gerada')
    with path.open('w',newline='',encoding='utf-8') as fp:
        escritor=csv.DictWriter(fp,fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def executar(entrada, saida, repeticoes=30, diagnostico=False, alvos=None, semente=2026):
    if not diagnostico and repeticoes!=30:
        raise ValueError('Análise final exige 30 repetições. Use --diagnostico para quantidades menores.')
    if alvos is None: alvos=('y_original','y_double_kmeans')
    _,respostas,_=carregar_dados()
    saida=Path(saida); entrada=Path(entrada)
    gerados={}
    for alvo in alvos:
        y=respostas[alvo]
        por_modelo={}
        for modelo in MODELOS:
            fp=entrada/modelo/f'{alvo}_{repeticoes}x10_folds.csv'
            pp=entrada/modelo/f'{alvo}_{repeticoes}x10_predicoes.csv'
            por_modelo[modelo]=auditar_resultados(fp,pp,y,alvo,repeticoes=repeticoes,semente=semente)
            print(f'Auditado {alvo}/{modelo}: {len(por_modelo[modelo])} folds',flush=True)
        medias=medias_por_repeticao(por_modelo,repeticoes=repeticoes)
        resumos=sumarizar_metricas(medias)
        for row in resumos: row['alvo']=alvo
        for row in medias: row['alvo']=alvo
        caminho=saida/alvo
        escrever_csv(caminho/'medias_por_repeticao.csv',medias)
        escrever_csv(caminho/'resumo_metricas.csv',resumos)
        gerados[alvo]=['medias_por_repeticao.csv','resumo_metricas.csv']
        if not diagnostico:
            fried,pairs,ranks=friedman_nemenyi(medias)
            for x in fried: x['alvo']=alvo
            for x in pairs: x['alvo']=alvo
            for x in ranks: x['alvo']=alvo
            escrever_csv(caminho/'friedman.csv',fried)
            escrever_csv(caminho/'nemenyi_pares.csv',pairs)
            escrever_csv(caminho/'ranking_medio.csv',ranks)
            gerados[alvo]+=['friedman.csv','nemenyi_pares.csv','ranking_medio.csv']
        titulo=f'Relatório {alvo} — {repeticoes}×10-folds'
        aviso=('**ESTUDO 30×10 COMPLETO**, com ICs descritivos aproximados e testes estatísticos exploratórios.'
               if not diagnostico else '**DIAGNÓSTICO PARCIAL. NÃO É O PROTOCOLO FINAL. SEM TESTES DE SIGNIFICÂNCIA.**')
        lines=[f'# {titulo}', '', aviso, '',
               'As métricas são médias macro de precisão, cobertura e F-measure; taxa de erro = 1−acurácia.',
               'A unidade resumida é a repetição (média dos 10 folds), não 300 folds tratados como amostras independentes.',
               '', '| Classificador | Taxa de erro | Precisão macro | Cobertura macro | F1 macro |',
               '|---|---:|---:|---:|---:|']
        indexed={(x['modelo'],x['metrica']):x for x in resumos}
        for modelo in MODELOS:
            valores=[indexed[(modelo,m)]['media'] for m in METRICAS]
            lines.append(f'| {modelo} | '+' | '.join(f'{v:.4f}' for v in valores)+' |')
        lines+=['', '**Limitações estatísticas:** as repetições do CV compartilham os mesmos 4.601 exemplos; portanto,',
                'os ICs t calculados sobre as médias de repetição são aproximações descritivas, não intervalos',
                'com garantia rigorosa de cobertura para novas amostras independentes. Friedman e Nemenyi',
                'utilizam 30 repetições como blocos pareados, mas os blocos não são plenamente independentes.',
                'O voto é derivado dos outros modelos, e essa dependência também deve constar da interpretação.',
                'Não interpretar a classificação dos rótulos Double K-means como validação externa independente dos clusters.',
                '']
        if not diagnostico:
            lines+=['## Friedman e Nemenyi (exploratórios)', '']
            for entry in fried:
                lines.append(f'- **{entry["metrica"]}**: Friedman χ²={entry["estatistica_friedman"]:.4f}, '
                             f'p={entry["p_valor_friedman"]:.5g}; '
                             f'diferença crítica de ranks Nemenyi={entry["diferenca_critica_nemenyi"]:.4f}.')
            lines+=['','O arquivo `nemenyi_pares.csv` contém os p-valores por par e cada métrica.']
        else:
            lines+=['', 'Não foram calculados Friedman/Nemenyi: número insuficiente de repetições.',
                    'IC95 não disponível com uma repetição.']
        (caminho/'relatorio_avaliacao.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        gerados[alvo].append('relatorio_avaliacao.md')
    (saida/'status_analise.json').write_text(json.dumps({
        'protocolo':'30x10' if (repeticoes==30 and not diagnostico) else 'diagnostico',
        'repeticoes':repeticoes,'alvos':list(alvos),'resultados':gerados,
        'advertencia':'Blocos de CV repetido compartilham observações; ICs/testes são exploratórios.'},
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return gerados


def main(argv=None):
    p=argparse.ArgumentParser(description='Auditar e analisar cinco modelos sobre 30×10 folds')
    p.add_argument('--entrada',type=Path,default=RAIZ/'resultados')
    p.add_argument('--saida',type=Path,default=RAIZ/'resultados'/'analise_estatistica')
    p.add_argument('--repeticoes',type=int,default=30)
    p.add_argument('--diagnostico',action='store_true',help='Descritivo, sem Friedman/Nemenyi; para 1×10')
    p.add_argument('--alvo',choices=('ambos','y_original','y_double_kmeans'),default='ambos')
    p.add_argument('--semente',type=int,default=2026)
    args=p.parse_args(argv)
    if args.repeticoes<1: p.error('repeticoes deve ser >= 1')
    nomes=None if args.alvo=='ambos' else (args.alvo,)
    print('Base:',validar_base(),flush=True)
    executar(args.entrada,args.saida,args.repeticoes,args.diagnostico,nomes,args.semente)
    print('Análise e auditoria concluídas em',args.saida,flush=True)

if __name__=='__main__': main()
