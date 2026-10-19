import unittest
import numpy as np
from questao2.src.avaliacao_gaussiano import avaliar_gaussiano
from questao2.src.avaliacao_knn import avaliar_knn
from questao2.src.avaliacao_parzen import avaliar_parzen
from questao2.src.avaliacao_logistica import avaliar_logistica

class TestRepeticaoIntermediaria(unittest.TestCase):
    def test_inicio_segunda_repeticao_mesmos_folds(self):
        rng=np.random.default_rng(100)
        X=rng.uniform(0,2,size=(80,3))
        y=np.tile(np.array([0,1]),40)
        opcoes=[
            (avaliar_gaussiano,{}),
            (avaliar_knn,{'valores_k':(1,3),'distancias':('euclidiana',)}),
            (avaliar_parzen,{'valores_h':(0.1,0.4)}),
            (avaliar_logistica,{'valores_C':(1.,),'pesos_classes':(None,)}),
        ]
        for func, kwargs in opcoes:
            with self.subTest(modelo=func.__name__):
                linhas,pred=func(X,y,'teste',repeticoes=2,repeticao_inicial=2,
                                max_folds=1,guardar_predicoes=True,**kwargs)
                self.assertEqual(len(linhas),1)
                self.assertEqual((linhas[0]['repeticao'],linhas[0]['fold']),(2,1))
                self.assertEqual(len(pred),8)
                self.assertTrue(all(z['repeticao']==2 for z in pred))

if __name__=='__main__':unittest.main()
