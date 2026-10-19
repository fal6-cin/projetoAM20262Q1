import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from questao2.scripts.rodar_curvas import salvar_ponto
from questao2.src.curvas_aprendizagem import (
    PERCENTUAIS, TODOS_MODELOS, avaliar_ponto, gerar_divisao,
)
from questao2.src.graficos_curvas import consolidar_curvas


class TestCurvasAprendizagem(unittest.TestCase):
    def test_19_pontos_5_a_95(self):
        self.assertEqual(PERCENTUAIS, tuple(range(5,100,5)))
        self.assertEqual(len(PERCENTUAIS),19)

    def test_divisao_reprodutivel_estratificada_sem_sobreposicao(self):
        y=np.array([0]*2788+[1]*1813)
        for pct in PERCENTUAIS:
            tr,te=gerar_divisao(y,pct,1,2026)
            tr2,te2=gerar_divisao(y,pct,1,2026)
            np.testing.assert_array_equal(tr,tr2)
            np.testing.assert_array_equal(te,te2)
            self.assertEqual(len(tr)+len(te),len(y))
            self.assertEqual(len(set(tr)&set(te)),0)
            self.assertEqual(len(np.unique(y[tr])),2)
            self.assertEqual(len(np.unique(y[te])),2)
            self.assertLess(abs(np.mean(y[tr])-np.mean(y)),0.02)

    def test_percentual_invalido(self):
        with self.assertRaises(ValueError):
            gerar_divisao([0]*200+[1]*200, 7)
        with self.assertRaises(ValueError):
            gerar_divisao([0]*200+[1]*200, 100)

    def test_um_ponto_todos_modelos_mesmos_indices_e_cv_apenas_treino(self):
        n=2000
        X=np.column_stack((np.arange(n,dtype=float),np.arange(n,dtype=float)%10+1))
        y=np.array([0]*1200+[1]*500+[2]*300)
        captures=[]
        def fabrica(nome):
            def ajustar(Xt,yt,semente):
                captures.append((nome,Xt.copy(),yt.copy(),semente))
                return SimpleNamespace(best_estimator_=DummyClassifier(strategy='most_frequent').fit(Xt,yt),
                                       best_score_=0.5,best_params_={'teste':nome})
            return ajustar
        funcs={nome:fabrica(nome) for nome in TODOS_MODELOS if nome!='voto'}
        linhas=avaliar_ponto(X,y,'y_double_kmeans',5,repeticao=1,ajustadores=funcs)
        self.assertEqual([v['modelo'] for v in linhas],list(TODOS_MODELOS))
        self.assertEqual(len(captures),4)
        tr,te=gerar_divisao(y,5)
        for _,xx,yy,_ in captures:
            np.testing.assert_array_equal(xx,X[tr])
            np.testing.assert_array_equal(yy,y[tr])
        self.assertEqual(linhas[-1]['n_treino'],len(tr))
        self.assertEqual(linhas[-1]['n_teste'],len(te))
        for linha in linhas:
            self.assertTrue(0 <= linha['f1_macro_treino'] <=1)
            self.assertTrue(0 <= linha['f1_macro_teste'] <=1)
            self.assertEqual(linha['percentual_teste'],95)
        self.assertEqual(linhas[-1]['empates_teste'],0)

    def test_salvamento_atomico_e_retomada_consistencia(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rep_01_pct_05.csv'
            linhas=[]
            for nome in TODOS_MODELOS:
                linhas.append(dict(modelo=nome,alvo='y_original',repeticao=1,percentual_treino=5,
                                   n_treino=230,n_teste=4371,f1_macro_treino=0.9,f1_macro_teste=0.8))
            salvar_ponto(p,linhas)
            self.assertTrue(p.exists())
            self.assertEqual(len(pd.read_csv(p)),5)
            self.assertFalse(p.with_suffix('.csv.temporario').exists())
            with self.assertRaises(ValueError):
                salvar_ponto(p,linhas[:-1])

    def test_consolidacao_rejeita_incompleto_e_metadados_falsos(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'execucoes'/'y_original'/'rep_01_pct_05.csv'
            linhas=[]
            for nome in TODOS_MODELOS:
                linhas.append(dict(modelo=nome,alvo='y_original',repeticao=1,percentual_treino=5,
                                   n_treino=230,n_teste=4371,f1_macro_treino=0.9,f1_macro_teste=0.8))
            salvar_ponto(p,linhas)
            with self.assertRaisesRegex(ValueError,'INCOMPLETAS'):
                consolidar_curvas(td,alvos=('y_original',),repeticoes=1,exigir_completo=True)
            resumo,faltantes=consolidar_curvas(td,alvos=('y_original',),repeticoes=1,exigir_completo=False)
            self.assertEqual(len(faltantes),18)
            self.assertEqual(len(resumo),5)
            t=pd.read_csv(p); t.loc[0,'percentual_treino']=10;t.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Metadados'):
                consolidar_curvas(td,alvos=('y_original',),repeticoes=1,exigir_completo=False)

if __name__=='__main__':unittest.main()
