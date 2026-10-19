"""Testes matemáticos do Bayes multivariado e da ausência de vazamento."""
import unittest
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.pipeline import Pipeline

from questao2.src.modelos.bayes_gaussiano import BayesGaussianoMultivariado
from questao2.src.avaliacao_gaussiano import pipeline_gaussiano, ajustar_gaussiano


class TestBayesGaussiano(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(123)
        X0=rng.multivariate_normal([-1,0], [[2,.75],[.75,1]], size=75)
        X1=rng.multivariate_normal([2,1], [[1.5,-.4],[-.4,1]], size=50)
        self.X=np.vstack((X0,X1))
        self.y=np.array([0]*75+[1]*50)

    def test_parametros_sao_mv_e_prioris_empiricas(self):
        m=BayesGaussianoMultivariado(0).fit(self.X,self.y)
        np.testing.assert_allclose(m.priors_,[75/125,50/125])
        np.testing.assert_allclose(m.media_[0],self.X[:75].mean(axis=0))
        np.testing.assert_allclose(m.covariancia_ml_[0],np.cov(self.X[:75],rowvar=False,ddof=0))
        self.assertGreater(abs(m.covariancia_ml_[0,0,1]),0.1)  # NÃO diagonal

    def test_posterior_corresponde_ao_scipy_multivariado(self):
        m=BayesGaussianoMultivariado(0).fit(self.X,self.y)
        pontos=np.array([[0.,0.],[-1.,1.],[2.,2.]])
        bruto=np.column_stack([multivariate_normal.logpdf(pontos,mean=m.media_[j],cov=m.covariancia_ml_[j]) + np.log(m.priors_[j]) for j in range(2)])
        bruto=np.exp(bruto - np.max(bruto,axis=1,keepdims=True))
        esperado=bruto/bruto.sum(axis=1,keepdims=True)
        np.testing.assert_allclose(m.predict_proba(pontos),esperado,rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(m.predict_proba(pontos).sum(axis=1),1)

    def test_singular_necessita_de_regularizacao(self):
        x=np.array([[0,1,3],[0,2,4],[0,4,5],[0,5,7],
                    [1,1,4],[1,2,5],[1,3,6],[1,4,7]],float)
        y=np.array([0]*4+[1]*4)
        with self.assertRaisesRegex(ValueError,'Covariância não definida positiva'):
            BayesGaussianoMultivariado(0).fit(x,y)
        m=BayesGaussianoMultivariado(1e-3).fit(x,y)
        self.assertTrue(np.isfinite(m.predict_log_proba(x)).all())
        np.testing.assert_allclose(m.predict_proba(x).sum(axis=1),np.ones(len(x)))

    def test_lambda_nao_negativa(self):
        with self.assertRaises(ValueError):
            BayesGaussianoMultivariado(-1).fit(self.X,self.y)

    def test_escalonamento_ajustado_apenas_no_treino(self):
        X=np.array([[0,0],[1,1],[2,2],[10,10],[15,15],[30,30]],float)
        y=np.array([0,0,0,1,1,1])
        pipeline=pipeline_gaussiano(1e-2)
        pipeline.fit(X[:5],y[:5])
        teste=pipeline.named_steps['minmax'].transform(np.log1p(X[5:]))
        self.assertTrue((teste>1).all())  # transform NÃO ajustado com amostra 30

    def test_5folds_busca_no_treino_e_modelo_fittado(self):
        X=np.abs(self.X)
        busca=ajustar_gaussiano(X,self.y,2026,grade=(1e-4,1e-2))
        self.assertEqual(len(busca.cv_results_['params']),2)
        self.assertTrue(np.isfinite(busca.best_score_))
        self.assertIsInstance(busca.best_estimator_,Pipeline)
        self.assertEqual(busca.predict(X[:4]).shape,(4,))


if __name__=='__main__': unittest.main()
