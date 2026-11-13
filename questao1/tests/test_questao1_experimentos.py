"""Testes unitários e de reprodutibilidade para os módulos unificados da Questão 1."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from questao1.src.dados import PASTA_RESULTADOS, carregar_spambase
from questao1.src.experimento import (
    carregar_resumo_comparativo,
    executar_estrategia,
)
from questao1.src.preprocessamentos import (
    aplicar_preprocessamento,
    listar_estrategias,
)


class TestQuestao1Experimentos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_df, cls.X_original, cls.y_original, cls.feature_names = carregar_spambase()

    def test_carregar_spambase_integridade(self):
        self.assertEqual(self.X_original.shape, (4601, 57))
        self.assertEqual(len(self.feature_names), 57)
        self.assertTrue(np.isfinite(self.X_original).all())
        self.assertTrue((self.X_original >= 0).all())
        counts = dict(zip(*np.unique(self.y_original, return_counts=True)))
        self.assertEqual(counts, {0: 2788, 1: 1813})

    def test_preprocessamentos_dimensoes_e_dominios(self):
        estrategias = listar_estrategias()
        self.assertEqual(len(estrategias), 7)
        for nome in estrategias:
            with self.subTest(estrategia=nome):
                X_proc, active_names, cfg = aplicar_preprocessamento(
                    nome, self.X_original, self.feature_names
                )
                expected_cols = 49 if nome == "filtragem_dominio" else 57
                self.assertEqual(X_proc.shape, (4601, expected_cols))
                self.assertEqual(len(active_names), expected_cols)
                self.assertTrue(np.isfinite(X_proc).all())
                if nome in {"log_minmax", "quantile_uniform"}:
                    self.assertGreaterEqual(float(X_proc.min()), 0.0)
                    self.assertLessEqual(float(X_proc.max()), 1.0)
                self.assertTrue(cfg.csv_saida.endswith(".csv"))

    def test_reproducao_exata_das_sete_estrategias(self):
        for nome in listar_estrategias():
            with self.subTest(estrategia=nome):
                res = executar_estrategia(
                    nome=nome,
                    reutilizar_grid_salvo=True,
                    salvar=False,
                    exibir_graficos=False,
                    verbose=False,
                )
                av = res["avaliacao"]
                meta_salvo = pd.read_csv(
                    PASTA_RESULTADOS / nome / "metadata_para_questao2.csv"
                ).iloc[0]
                self.assertEqual(av.K_star, int(meta_salvo["K_star"]))
                self.assertEqual(av.H_star, int(meta_salvo["H_star"]))
                self.assertAlmostEqual(
                    av.best_solution.objective, float(meta_salvo["objective_W"]), places=4
                )
                self.assertAlmostEqual(
                    av.silhouette, float(meta_salvo["silhouette"]), places=6
                )
                self.assertAlmostEqual(
                    av.ari, float(meta_salvo["adjusted_rand_index"]), places=6
                )
                # Garante que os 57 atributos exportados continuam na escala original
                np.testing.assert_allclose(
                    av.q2_df[self.feature_names].to_numpy(), self.X_original
                )

    def test_resumo_comparativo_consolidado(self):
        resumo = carregar_resumo_comparativo()
        self.assertEqual(len(resumo), 7)
        self.assertIn("log_minmax", set(resumo["estrategia"]))
        self.assertIn("revisada", set(resumo["estrategia"]))


if __name__ == "__main__":
    unittest.main()
