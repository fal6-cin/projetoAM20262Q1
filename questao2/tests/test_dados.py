import unittest
from questao2.src.dados import carregar_dados, validar_base

class TestDados(unittest.TestCase):
    def test_shape_labels(self):
        X,y,nomes=carregar_dados()
        self.assertEqual(X.shape,(4601,57))
        self.assertEqual(len(nomes),57)
        self.assertEqual(set(y),{'y_original','y_double_kmeans'})
    def test_integridade(self):
        info=validar_base()
        self.assertEqual(info['contagens']['y_double_kmeans'],{0:2619,1:265,2:1717})
