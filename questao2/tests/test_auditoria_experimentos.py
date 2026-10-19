import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from questao2.src.auditoria_experimentos import auditar_resultados
from questao2.src.metricas import calcular_metricas
from questao2.src.validacao import gerar_folds_externos


def gravar(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class TestAuditoria(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.f=Path(self.tmp.name)/'folds.csv'
        self.p=Path(self.tmp.name)/'predicoes.csv'
        self.y=np.array([0]*35+[1]*35)
        self.rows=[];self.preds=[]
        for rep,fold,tr,te in gerar_folds_externos(self.y,repeticoes=2):
            yp=self.y[te].copy()
            m=calcular_metricas(self.y[te],yp)
            self.rows.append(dict(alvo='teste',repeticao=rep,fold=fold,n_treino=len(tr),n_teste=len(te),
                **{k:v for k,v in m.items() if k!='por_classe'}))
            for ix,v in zip(te,yp):
                self.preds.append(dict(alvo='teste',repeticao=rep,fold=fold,indice_original=int(ix),
                                       real=int(self.y[ix]),predito=int(v)))
        gravar(self.f,self.rows);gravar(self.p,self.preds)

    def tearDown(self): self.tmp.cleanup()
    def test_cobertura_perfeita(self):
        self.assertEqual(len(auditar_resultados(self.f,self.p,self.y,'teste',repeticoes=2)),20)
    def test_rejeita_predicao_fora_do_fold(self):
        self.preds[0]['indice_original']=999
        gravar(self.p,self.preds)
        with self.assertRaises(ValueError): auditar_resultados(self.f,self.p,self.y,'teste',repeticoes=2)
    def test_rejeita_metrica_inventada(self):
        self.rows[0]['f_measure_macro']=0.1
        gravar(self.f,self.rows)
        with self.assertRaises(ValueError): auditar_resultados(self.f,self.p,self.y,'teste',repeticoes=2)
    def test_valida_repeticao_isolada(self):
        gravar(self.f,self.rows[10:]);gravar(self.p,[x for x in self.preds if x['repeticao']==2])
        self.assertEqual(len(auditar_resultados(self.f,self.p,self.y,'teste',repeticoes=2,
                                                repeticao_inicial=2,repeticao_final=2)),10)

if __name__=='__main__':unittest.main()
