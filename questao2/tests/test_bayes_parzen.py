"""Parzen: fórmulas, estabilidade, seleção h e isolamento entre folds."""
import unittest

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from questao2.src.modelos.bayes_parzen import BayesParzen
from questao2.src.avaliacao_parzen import (
    VALORES_H, _predicoes_para_grade, ajustar_parzen, avaliar_parzen,
    pipeline_parzen,
)


class TestBayesParzen(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0., 0.], [0., 1.], [1., 0.],
                           [3., 4.], [4., 3.], [4., 4.]])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_posteriori_referencia_kernel_produto_bayes(self):
        h = 0.7
        modelo = BayesParzen(largura_janela=h, tamanho_lote=1).fit(self.X, self.y)
        consultas = np.array([[0.1, 0.3], [3.8, 3.6], [1.7, 1.8]])
        # Referência explícita: p(x|classe) por média de produtos de N(0,h²).
        pesos = np.zeros((len(consultas), 2), dtype=float)
        for r, ponto in enumerate(consultas):
            for c in [0, 1]:
                densidade = np.mean([multivariate_normal.pdf(ponto, mean=x, cov=np.eye(2)*h*h)
                                     for x in self.X[self.y == c]])
                pesos[r, c] = densidade * np.mean(self.y == c)
        np.testing.assert_allclose(modelo.predict_proba(consultas), pesos / pesos.sum(axis=1, keepdims=True), atol=1e-13)
        np.testing.assert_allclose(np.exp(modelo.predict_log_joint(consultas)), pesos, atol=1e-13)
        np.testing.assert_allclose(np.exp(modelo.predict_log_proba(consultas)), modelo.predict_proba(consultas))
        np.testing.assert_allclose(modelo.prioris_, [0.5, 0.5])

    def test_priori_desbalanceada_conta_corretamente(self):
        # Amostras idênticas: as densidades de classe coincidem, posterior = priori.
        X = np.zeros((5, 2))
        y = np.array([0, 0, 0, 0, 1])
        p = BayesParzen(largura_janela=1).fit(X, y).predict_proba([[0., 0.]])
        np.testing.assert_allclose(p, [[0.8, 0.2]], atol=1e-14)

    def test_alta_dimensao_largura_pequena_sem_nan(self):
        rng = np.random.default_rng(10)
        Xt = np.vstack([rng.random((10, 57)), rng.random((10, 57)) + 1])
        y = np.array([0]*10+[1]*10)
        consulta = np.full((3,57), 10.0)
        m = BayesParzen(largura_janela=0.001, tamanho_lote=2).fit(Xt, y)
        posterior = m.predict_proba(consulta)
        self.assertTrue(np.isfinite(posterior).all())
        np.testing.assert_allclose(posterior.sum(axis=1), 1., atol=1e-12)

    def test_parametros_invalidos_e_dimensoes(self):
        for h in [0, -1, float('nan'), float('inf'), True, 'xxx']:
            with self.subTest(h=h), self.assertRaises(ValueError):
                BayesParzen(largura_janela=h).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            BayesParzen(tamanho_lote=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            BayesParzen().fit(self.X, self.y).predict([[1, 2, 3]])

    def test_predicoes_grade_mesmas_do_modelo_por_h(self):
        consultas = np.array([[1.5, 2.1], [3.2, 3.6], [0., 0.]])
        from scipy.spatial.distance import cdist
        classes = np.unique(self.y)
        d2 = cdist(consultas, self.X, 'sqeuclidean')
        resultados = _predicoes_para_grade(d2, self.y, classes, (0.04, 0.3, 0.9))
        for h, pred in resultados.items():
            modelo = BayesParzen(largura_janela=h).fit(self.X, self.y)
            np.testing.assert_array_equal(pred, modelo.predict(consultas))

    def test_cv_interna_mesma_pontuacao_de_ajustes_individuais(self):
        rng = np.random.default_rng(71)
        X = np.vstack([rng.random((25,3)), rng.random((25,3))+0.5])
        y = np.array([0]*25+[1]*25)
        larguras = (0.1, 0.35)
        ajustado = ajustar_parzen(X, y, semente=193, valores_h=larguras)
        resultados = {r['largura_janela']:r for r in ajustado.cv_resultados_}
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=193).split(X, y))
        for h in larguras:
            ref=[]
            for tr, va in folds:
                m = pipeline_parzen(largura_janela=h).fit(X[tr], y[tr])
                ref.append(f1_score(y[va], m.predict(X[va]), average='macro'))
            np.testing.assert_allclose(resultados[h]['f1_macro_por_fold'], ref, atol=1e-12)
        self.assertIn(ajustado.best_params_['parzen__largura_janela'], larguras)
        self.assertEqual(len(ajustado.best_estimator_.predict(X[:4])), 4)

    def test_escalonamento_so_treino_e_cada_fold_externo(self):
        X = np.array([[0,0],[1,1],[2,2],[8,8],[9,9],[100,100]],float)
        y = np.array([0,0,0,1,1,1])
        pipe = pipeline_parzen(0.2).fit(X[:5], y[:5])
        xtrans = pipe.named_steps['minmax'].transform(np.log1p(X[-1:]))
        self.assertTrue(np.all(xtrans>1))
        rng = np.random.default_rng(5)
        dados = np.vstack([rng.random((30, 3)), rng.random((30, 3))+1])
        classes = np.array([0]*30 + [1]*30)
        linhas, predito = avaliar_parzen(dados, classes, 'mini', max_folds=1,
                             guardar_predicoes=True, valores_h=(0.1,0.3))
        self.assertEqual(len(linhas), 1)
        self.assertEqual(len(predito), 6)
        self.assertAlmostEqual(sum(predito[0][f'prob_classe_{c}'] for c in [0, 1]),1)
        self.assertEqual(len(linhas[0]['metricas_por_classe']), 2)


if __name__ == '__main__':
    unittest.main()
