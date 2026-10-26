import unittest
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from questao2.src.comparacao_preprocessamento import (transformar, pipeline, VARIANTES,
                                                       consolidar)
from questao2.src.modelos.bayes_knn import BayesKVizinhos
from tempfile import TemporaryDirectory
from pathlib import Path

class TestComparacaoPreprocessamento(unittest.TestCase):
    def test_sem_mantem_escala_original(self):
        treino=np.array([[1.,100.],[4.,10.]])
        teste=np.array([[5.,999.]])
        a,b=transformar(treino,teste,'sem')
        np.testing.assert_array_equal(a,treino)
        np.testing.assert_array_equal(b,teste)

    def test_minmax_apenas_treino(self):
        treino=np.array([[0.,10.],[2.,20.]])
        teste=np.array([[4.,40.]])
        _,b=transformar(treino,teste,'minmax')
        np.testing.assert_allclose(b,[[2.,3.]])

    def test_logminmax_apenas_treino(self):
        treino=np.array([[0.],[3.]])
        teste=np.array([[15.]])
        _,b=transformar(treino,teste,'log1p_minmax')
        np.testing.assert_allclose(b,[[2.]])

    def test_pipelines(self):
        for variante in VARIANTES:
            p=pipeline(BayesKVizinhos(),variante)
            self.assertEqual('modelo',p.steps[-1][0])
            self.assertEqual('minmax' in p.named_steps, variante!='sem')
            self.assertEqual('log1p' in p.named_steps, variante=='log1p_minmax')

    def test_consolidacao_incompleta_nao_publica_resultado(self):
        with TemporaryDirectory() as d:
            resposta=consolidar(Path(d),alvos=('y_original',),variantes=('sem',),repeticoes=1)
            self.assertFalse(resposta['completo'])
            self.assertFalse((Path(d)/'comparacao_resumo.csv').exists())

if __name__=='__main__':
    unittest.main()
