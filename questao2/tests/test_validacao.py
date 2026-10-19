import unittest
import numpy as np
from questao2.src.dados import carregar_dados
from questao2.src.validacao import gerar_folds_externos, gerar_folds_internos, conferir_folds
from questao2.src.metricas import calcular_metricas

class TestValidacao(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.ys,_=carregar_dados()
    def test_repeticoes_e_cobertura(self):
        for y in self.ys.values():
            d=conferir_folds(y)
            self.assertEqual(d['total_divisoes'],300)
    def test_sem_sobreposicao_e_5_folds_internos(self):
        y=self.ys['y_double_kmeans']
        _,_,tr,te=next(gerar_folds_externos(y))
        self.assertEqual(len(np.intersect1d(tr,te)),0)
        cv=list(gerar_folds_internos(y[tr]))
        self.assertEqual(len(cv),5)
        for it,iv in cv: self.assertEqual(len(np.intersect1d(it,iv)),0)
    def test_metricas_identicas(self):
        v=calcular_metricas([0,0,1,1],[0,0,1,1])
        self.assertAlmostEqual(v['taxa_erro'],0)
        self.assertAlmostEqual(v['f_measure_macro'],1)
