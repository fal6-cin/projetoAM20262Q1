"""Métricas da atividade para uso posterior pelos cinco classificadores.

Precisão/cobertura/F1 macro não ocultam artificialmente o desempenho de classes
minoritárias. Guardar também valores por classe e support para cada fold.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calcular_metricas(y_real, y_previsto, classes=None):
    y_real=np.asarray(y_real)
    y_previsto=np.asarray(y_previsto)
    if classes is None:
        classes=np.unique(np.concatenate([y_real,y_previsto]))
    p,r,f,s=precision_recall_fscore_support(y_real,y_previsto,labels=classes,zero_division=0)
    return {
        'taxa_erro':float(1-accuracy_score(y_real,y_previsto)),
        'precisao_macro':float(np.mean(p)),
        'cobertura_macro':float(np.mean(r)),
        'f_measure_macro':float(np.mean(f)),
        'por_classe': {str(int(c)):{'precisao':float(pp),'cobertura':float(rr),'f_measure':float(ff),'suporte':int(ss)} for c,pp,rr,ff,ss in zip(classes,p,r,f,s)}
    }
