"""Testes do quarto classificador: regressão logística (binária e softmax)."""
import unittest
import numpy as np
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from questao2.src.modelos.regressao_logistica import RegressaoLogistica
from questao2.src.avaliacao_logistica import (
    VALORES_C, PESOS_CLASSES, pipeline_logistica, ajustar_logistica,
    avaliar_logistica,
)


class TestRegressaoLogistica(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(91)
        cls.X2 = np.vstack([rng.random((24, 4)) * 0.4,
                            rng.random((24, 4)) * 0.4 + 1.0])
        cls.y2 = np.array([0] * 24 + [1] * 24)
        cls.X3 = np.vstack([rng.random((22, 4)) * 0.35,
                            rng.random((22, 4)) * 0.35 + 1,
                            rng.random((22, 4)) * 0.35 + 2])
        cls.y3 = np.repeat([0, 1, 2], 22)

    def test_binaria_probabilidades_coincidem_com_referencia(self):
        modelo = RegressaoLogistica(C=0.6).fit(self.X2, self.y2)
        ref = LogisticRegression(C=0.6, solver='lbfgs', max_iter=2000, tol=1e-4,
                                 random_state=2026).fit(self.X2, self.y2)
        consultas = self.X2[[0, 3, 26, 30]]
        np.testing.assert_allclose(modelo.predict_proba(consultas), ref.predict_proba(consultas), atol=1e-12)
        decisao = modelo.decision_function(consultas)
        np.testing.assert_allclose(modelo.predict_proba(consultas)[:, 1], expit(decisao), atol=1e-12)
        np.testing.assert_array_equal(modelo.predict(consultas), ref.predict(consultas))

    def test_multiclasse_softmax_probabilidades(self):
        m = RegressaoLogistica(C=3., class_weight='balanced').fit(self.X3, self.y3)
        d = m.decision_function(self.X3[:7])
        self.assertEqual(d.shape, (7, 3))
        p = m.predict_proba(self.X3[:7])
        self.assertEqual(p.shape, (7, 3))
        np.testing.assert_allclose(p, softmax(d, axis=1), atol=1e-12)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
        self.assertTrue(np.all((p > 0) & (p < 1)))
        np.testing.assert_array_equal(m.classes_, [0, 1, 2])

    def test_pipeline_multiclasse_correto(self):
        m = pipeline_logistica(C=0.3).fit(self.X3, self.y3)
        self.assertEqual(m.named_steps['logistica'].classes_.size, 3)
        p = m.predict_proba(self.X3[:5])
        np.testing.assert_allclose(p.sum(axis=1), 1.)

    def test_validacao_hiperparametros_e_cv_interna(self):
        # Checagem independente de cada pontuação da GridSearchCV.
        Cs = (0.03, 0.3)
        pesos = (None, 'balanced')
        busca = ajustar_logistica(self.X3, self.y3, semente=99,
                                 valores_C=Cs, pesos_classes=pesos)
        self.assertIn(busca.best_params_['logistica__C'], Cs)
        self.assertIn(busca.best_params_['logistica__class_weight'], pesos)
        for C in Cs:
            for peso in pesos:
                scores = []
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
                for tr, va in cv.split(self.X3, self.y3):
                    m = pipeline_logistica(C, peso).fit(self.X3[tr], self.y3[tr])
                    scores.append(f1_score(self.y3[va], m.predict(self.X3[va]),
                                           labels=[0, 1, 2], average='macro'))
                sel = next(j for j, params in enumerate(busca.cv_results_['params'])
                           if params['logistica__C'] == C and params['logistica__class_weight'] == peso)
                self.assertAlmostEqual(float(busca.cv_results_['mean_test_score'][sel]), np.mean(scores))

    def test_escalonador_usa_apenas_treino(self):
        X = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [1000, 1000]], float)
        y = np.array([0, 0, 0, 1, 1, 1])
        pipe = pipeline_logistica().fit(X[:5], y[:5])
        transformado = pipe.named_steps['minmax'].transform(np.log1p(X[-1:]))
        self.assertTrue((transformado > 1).all())

    def test_um_fold_externo_e_probabilidades_individuais(self):
        linhas, predicoes = avaliar_logistica(self.X3, self.y3, 'artificial',
                        max_folds=1, guardar_predicoes=True,
                        valores_C=(0.3,), pesos_classes=(None,))
        self.assertEqual(len(linhas), 1)
        self.assertEqual(len(predicoes), 7)  # 66 registros, ~7 por fold
        self.assertEqual(linhas[0]['C'], 0.3)
        self.assertEqual(linhas[0]['class_weight'], 'None')
        self.assertEqual(len(linhas[0]['metricas_por_classe']), 3)
        for linha in predicoes:
            self.assertAlmostEqual(sum(linha[f'prob_classe_{i}'] for i in [0, 1, 2]), 1.)
            self.assertIn(linha['predito'], [0, 1, 2])

    def test_validacoes_de_parametros_e_dimensoes(self):
        for C in [0, -1, float('nan'), float('inf'), True, 'ruim']:
            with self.subTest(C=C), self.assertRaises(ValueError):
                RegressaoLogistica(C=C).fit(self.X2, self.y2)
        for peso in ['automatico', 2, {'A':3}]:
            with self.subTest(peso=peso), self.assertRaises(ValueError):
                RegressaoLogistica(class_weight=peso).fit(self.X2, self.y2)
        with self.assertRaises(ValueError):
            RegressaoLogistica().fit(self.X2, np.zeros(len(self.X2)))
        with self.assertRaises(ValueError):
            RegressaoLogistica().fit(self.X2, self.y2[:-1])
        with self.assertRaises(ValueError):
            RegressaoLogistica().fit(self.X2, self.y2).predict_proba(np.ones((3, 7)))
        with self.assertRaises(ValueError):
            ajustar_logistica(self.X2, self.y2, semente=2, valores_C=())
        with self.assertRaises(ValueError):
            ajustar_logistica(-self.X2, self.y2, semente=2)


if __name__ == '__main__':
    unittest.main()
