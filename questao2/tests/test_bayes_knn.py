"""Testes matemáticos, distâncias, desempate, grades e ausência de vazamento."""
import unittest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from questao2.src.modelos.bayes_knn import BayesKVizinhos
from questao2.src.avaliacao_knn import ajustar_knn, pipeline_knn, avaliar_knn


class TestBayesKVizinhos(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0., 0.], [0., 1.], [2., 0.], [8., 8.], [8., 9.], [9., 8.]])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_posteriors_sao_fracoes_e_prioris_empiricas(self):
        m = BayesKVizinhos(n_vizinhos=3).fit(self.X, self.y)
        np.testing.assert_allclose(m.prioris_, [0.5, 0.5])
        pts = np.array([[0., 0.], [9., 9.], [5., 5.]])
        prob = m.predict_proba(pts)
        np.testing.assert_allclose(prob.sum(axis=1), np.ones(len(pts)))
        np.testing.assert_allclose(prob[0], [1., 0.])
        np.testing.assert_allclose(prob[1], [0., 1.])
        np.testing.assert_allclose(prob * 3, np.rint(prob * 3))

    def test_tres_distancias_correspondem_scipy(self):
        for nome, met in [('euclidiana', 'euclidean'), ('city_block','cityblock'), ('chebyshev','chebyshev')]:
            with self.subTest(metrica=nome):
                m = BayesKVizinhos(n_vizinhos=3, distancia=nome, tamanho_lote=1).fit(self.X, self.y)
                pontos = np.array([[1.23, 0.4], [8.56, 6.7]])
                indices = m.indices_vizinhos(pontos)
                esperado = np.argsort(cdist(pontos, self.X, metric=met), axis=1, kind='stable')[:, :3]
                np.testing.assert_array_equal(indices, esperado)

    def test_equivalencia_knn_sklearn_quando_nao_ha_empate(self):
        p = np.array([[0., 0.], [1., 0.], [9., 9.], [8.3, 8.2]])
        for distancia, nome in [('euclidiana','euclidean'),('city_block','manhattan'),('chebyshev','chebyshev')]:
            with self.subTest(metrica=distancia):
                m = BayesKVizinhos(n_vizinhos=3, distancia=distancia).fit(self.X,self.y)
                ref = KNeighborsClassifier(n_neighbors=3, metric=nome).fit(self.X,self.y)
                np.testing.assert_array_equal(m.predict(p), ref.predict(p))
                np.testing.assert_allclose(m.predict_proba(p), ref.predict_proba(p))

    def test_multiclasse_empate_remove_mais_distante(self):
        # Vizinhos ordenados 0, 1, 2: um voto por classe -> ganha 0
        x = np.array([[0., 0.], [1., 0.], [2., 0.]])
        y = np.array([2, 1, 0])
        m = BayesKVizinhos(n_vizinhos=3).fit(x, y)
        p = m.predict_proba([[0.01, 0]])
        np.testing.assert_allclose(p, np.full((1,3),1/3))
        np.testing.assert_array_equal(m.predict([[0.01, 0]]), [2])

    def test_distancias_empatadas_sao_deterministicas(self):
        x = np.array([[0.,1.], [1.,0.], [-1.,0.], [0.,-1.]])
        m = BayesKVizinhos(2).fit(x,[0,1,1,0])
        np.testing.assert_array_equal(m.indices_vizinhos([[0,0]]), [[0,1]])

    def test_parametros_invalidos_e_dimensionalidade(self):
        for k in [0, 7, 1.5, True]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                BayesKVizinhos(n_vizinhos=k).fit(self.X,self.y)
        with self.assertRaises(ValueError):
            BayesKVizinhos(distancia='minkowski').fit(self.X,self.y)
        with self.assertRaises(ValueError):
            BayesKVizinhos(n_vizinhos=3).fit(self.X, self.y).predict([[1.,2.,3.]])

    def test_escalador_so_vê_treino(self):
        X=np.array([[0,0],[1,1],[2,2],[8,8],[9,9],[100,100]],float)
        y=np.array([0,0,0,1,1,1])
        p=pipeline_knn(n_vizinhos=1).fit(X[:5],y[:5])
        ultimo=p.named_steps['minmax'].transform(np.log1p(X[-1:]))
        self.assertTrue((ultimo>1).all())

    def test_cv_interna_funciona_e_nao_le_testes_externos(self):
        rng=np.random.default_rng(5)
        X=np.abs(rng.normal(size=(75,4)))
        y=np.repeat([0,1,2],25)
        r=ajustar_knn(X,y,semente=123,valores_k=(1,3),distancias=('euclidiana','city_block'))
        self.assertEqual(len(r.cv_resultados_),4)
        self.assertEqual(len(r.cv_resultados_[0]['f1_macro_por_fold']),5)
        self.assertIsInstance(r.best_estimator_, Pipeline)
        self.assertIn(r.best_params_['knn__n_vizinhos'], (1,3))
        self.assertTrue(np.isfinite(r.best_score_))
        self.assertEqual(len(r.best_estimator_.predict(X[:7])),7)

    def test_cv_otimizacao_corresponde_ao_pipeline_individual(self):
        rng=np.random.default_rng(44)
        X=np.vstack([rng.random((25,3)), rng.random((25,3))+1.5])
        y=np.array([0]*25+[1]*25)
        r=ajustar_knn(X,y,2028,valores_k=(1,3),distancias=('euclidiana',))
        scores={i['n_vizinhos']:i['f1_macro_medio'] for i in r.cv_resultados_}
        from sklearn.model_selection import StratifiedKFold
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=2028)
        for k in (1,3):
            refs=[]
            for tr,va in cv.split(X,y):
                modelo=pipeline_knn(n_vizinhos=k)
                modelo.fit(X[tr],y[tr]); yp=modelo.predict(X[va])
                refs.append(f1_score(y[va],yp,average='macro'))
            self.assertAlmostEqual(scores[k],np.mean(refs),places=12)

    def test_avaliacao_externa_basica_e_contagem_predicoes(self):
        rng=np.random.default_rng(9)
        X=np.vstack([rng.random((30,3)), rng.random((30,3))+1])
        y=np.array([0]*30+[1]*30)
        linhas, preditos=avaliar_knn(X,y,'teste',max_folds=1,guardar_predicoes=True,
                   valores_k=(1,3),distancias=('euclidiana',))
        self.assertEqual(len(linhas),1)
        self.assertEqual(len(preditos),6)
        self.assertEqual(linhas[0]['repeticao'],1)
        self.assertEqual(linhas[0]['fold'],1)
        self.assertAlmostEqual(preditos[0]['prob_classe_0']+preditos[0]['prob_classe_1'],1)


if __name__=='__main__': unittest.main()
