"""Resumos por repetição, ICs descritivos, Friedman e pós-teste Nemenyi.

A unidade de análise é a média dos dez folds de cada repetição: NÃO tratamos
300 folds correlacionados como 300 amostras independentes. As 30 repetições
reutilizam os mesmos exemplos, logo ICs/testes são aproximados/exploratórios.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, studentized_range, t

from .auditoria_experimentos import METRICAS

MODELOS = ('gaussiano', 'knn', 'parzen', 'logistica', 'voto')


def medias_por_repeticao(por_modelo, repeticoes=30):
    """por_modelo: {nome: lista de linhas CSV}; valida emparelhamento."""
    if set(por_modelo) != set(MODELOS):
        raise ValueError('São necessários os cinco modelos')
    saida = []
    pares_esperados = {(r, f) for r in range(1, repeticoes+1) for f in range(1,11)}
    for modelo in MODELOS:
        linhas = por_modelo[modelo]
        dados = {}
        for l in linhas:
            par = (int(l['repeticao']),int(l['fold']))
            if par in dados or par not in pares_esperados:
                raise ValueError(f'{modelo}: fold repetido/inválido: {par}')
            dados[par] = l
        if set(dados) != pares_esperados:
            raise ValueError(f'{modelo}: faltam folds externos')
        for rep in range(1,repeticoes+1):
            grupo = [dados[(rep,f)] for f in range(1,11)]
            for metrica in METRICAS:
                valores = np.asarray([float(l[metrica]) for l in grupo])
                if not np.isfinite(valores).all():
                    raise ValueError('Métricas não finitas')
                saida.append(dict(modelo=modelo, repeticao=rep, metrica=metrica,
                                  media_folds=float(np.mean(valores))))
    return saida


def sumarizar_metricas(medias, confianca=0.95):
    if not 0 < confianca < 1:
        raise ValueError('Confiança fora de (0,1)')
    saida=[]
    for modelo in MODELOS:
        for metrica in METRICAS:
            a=np.array([x['media_folds'] for x in medias if x['modelo']==modelo and x['metrica']==metrica],float)
            n=len(a)
            if not n: raise ValueError('Métrica ausente')
            media=float(a.mean())
            sem=float(a.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan
            margem=float(t.ppf((1+confianca)/2,n-1)*sem) if n>1 else np.nan
            saida.append(dict(modelo=modelo,metrica=metrica,n_repeticoes=n,
                media=media,desvio_repeticoes=float(a.std(ddof=1)) if n>1 else np.nan,
                ic95_inferior=max(0.,media-margem) if n>1 else np.nan,
                ic95_superior=min(1.,media+margem) if n>1 else np.nan,
                metodo_ic='t de Student sobre 30 medias por repeticao; aproximacao descritiva' if n>1 else 'indisponivel: menos de 2 repeticoes'))
    return saida


def friedman_nemenyi(medias, confianca=0.95):
    """Produz resultados por métrica com blocos=30 repetições; empates têm rank médio.

    Friedman: scipy.stats.friedmanchisquare, classificação em blocos pareados.
    Nemenyi: distribuição studentized range com q=sqrt(2)*dif_rank/SE.
    """
    if not 0 < confianca < 1: raise ValueError('Confiança fora do intervalo')
    alpha=1-confianca
    tabelas_f, tabelas_p, tabelas_r = [], [], []
    for metrica in METRICAS:
        por_chave={(r['modelo'],int(r['repeticao'])):float(r['media_folds']) for r in medias if r['metrica']==metrica}
        reps=sorted({k[1] for k in por_chave})
        if len(reps)<2: raise ValueError('Friedman/Nemenyi exige pelo menos duas repetições')
        x=np.asarray([[por_chave[(modelo, rep)] for modelo in MODELOS] for rep in reps],float)
        if not np.isfinite(x).all(): raise ValueError('Dados incompletos ou não finitos')
        rank=np.vstack([rankdata(v if metrica=='taxa_erro' else -v,method='average') for v in x])
        medias_rank=rank.mean(axis=0)
        if np.allclose(x-x[:, :1],0):
            estat,pvalor=0.,1.
        else:
            teste=friedmanchisquare(*[x[:,j] for j in range(len(MODELOS))])
            estat,pvalor=float(teste.statistic),float(teste.pvalue)
        n,k=len(reps),len(MODELOS)
        se=np.sqrt(k*(k+1)/(6*n))
        cd=float(studentized_range.ppf(1-alpha,k,np.inf)/np.sqrt(2)*se)
        tabelas_f.append(dict(metrica=metrica,blocos_repeticoes=n,classificadores=k,
            estatistica_friedman=estat,p_valor_friedman=pvalor,alpha=alpha,
            significativo=pvalor<alpha,diferenca_critica_nemenyi=cd,
            interpretacao='exploratoria: mesmas observacoes reutilizadas nas repeticoes'))
        for i,modelo in enumerate(MODELOS):
            tabelas_r.append(dict(metrica=metrica,modelo=modelo,rank_medio=float(medias_rank[i])))
        for i,j in combinations(range(k),2):
            dif=abs(medias_rank[i]-medias_rank[j])
            p_par=float(studentized_range.sf(np.sqrt(2)*dif/se,k,np.inf))
            tabelas_p.append(dict(metrica=metrica,modelo_a=MODELOS[i],modelo_b=MODELOS[j],
                rank_medio_a=float(medias_rank[i]),rank_medio_b=float(medias_rank[j]),
                diferenca_absoluta_rank=float(dif),p_valor_nemenyi=p_par,
                significativo=(dif>cd and pvalor<alpha),
                observacao='pos-teste reportado mesmo se Friedman nao significativo'))
    return tabelas_f,tabelas_p,tabelas_r
