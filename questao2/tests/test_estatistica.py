import unittest
import numpy as np
from questao2.src.estatistica import (
    MODELOS, medias_por_repeticao, sumarizar_metricas, friedman_nemenyi)
from questao2.src.auditoria_experimentos import METRICAS


class TestEstatistica(unittest.TestCase):
    def setUp(self):
        self.res={}
        for i,modelo in enumerate(MODELOS):
            self.res[modelo]=[]
            for r in range(1,31):
                for f in range(1,11):
                    score=0.5+0.05*i+0.0002*((r+f)%7)
                    self.res[modelo].append(dict(repeticao=r,fold=f,
                        taxa_erro=1-score,precisao_macro=score,
                        cobertura_macro=score,f_measure_macro=score))

    def test_aggregation_completo(self):
        m=medias_por_repeticao(self.res)
        self.assertEqual(len(m),5*4*30)
        c=sumarizar_metricas(m)
        self.assertEqual(len(c),20)
        self.assertTrue(all(np.isfinite(x['ic95_inferior']) for x in c))

    def test_friedman_nemenyi(self):
        m=medias_por_repeticao(self.res)
        fried,pares,ranks=friedman_nemenyi(m)
        self.assertEqual(len(fried),4)
        self.assertEqual(len(pares),4*10)
        self.assertEqual(len(ranks),4*5)
        self.assertTrue(all(x['significativo'] for x in fried))
        self.assertTrue(all(0<=p['p_valor_nemenyi']<=1 for p in pares))
        melhores=[x for x in ranks if x['metrica']=='f_measure_macro']
        self.assertEqual(min(melhores,key=lambda x:x['rank_medio'])['modelo'],'voto')

    def test_dado_duplicado_rejeitado(self):
        self.res['knn'][0]['fold']=2
        with self.assertRaises(ValueError): medias_por_repeticao(self.res)

    def test_nemenyi_identico(self):
        for modelo in MODELOS:
            for l in self.res[modelo]:
                for k in METRICAS: l[k]=0.5
        _,pares,ranks=friedman_nemenyi(medias_por_repeticao(self.res))
        self.assertTrue(all(p['p_valor_nemenyi']>0.99 for p in pares))
        self.assertTrue(all(x['rank_medio']==3 for x in ranks))

    def test_sem_ic_com_um_bloco(self):
        r=[dict(modelo=m,metrica=k,repeticao=1,media_folds=0.55) for m in MODELOS for k in METRICAS]
        self.assertTrue(np.isnan(sumarizar_metricas(r)[0]['ic95_inferior']))
        with self.assertRaises(ValueError): friedman_nemenyi(r)

if __name__=='__main__':unittest.main()
