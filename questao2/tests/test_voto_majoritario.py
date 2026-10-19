"""Testes da regra matemática, dos empates e da auditoria de folds/CSV."""
from __future__ import annotations
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from questao2.src.modelos.voto_majoritario import VotoMajoritario, decidir_votos
from questao2.src.avaliacao_voto import avaliar_voto_de_csvs, _ler_predicoes, PASTA_RESULTADOS
from questao2.src.dados import carregar_dados
from questao2.src.validacao import gerar_folds_externos


class TestVoto(unittest.TestCase):
    def test_maioria_sem_empate(self):
        m = VotoMajoritario().fit([0, 0, 0, 1, 1])
        pred, votos, emp = m.prever_detalhes(np.array([
            [1,0,1], [1,0,0], [1,0,1], [0,1,1]]))
        np.testing.assert_array_equal(pred, [1,0,1])
        np.testing.assert_array_equal(votos, [[1,3], [3,1], [1,3]])
        self.assertFalse(emp.any())

    def test_empate_dois_dois_prioris_treino(self):
        m=VotoMajoritario().fit([1, 1, 1, 0])
        votos=np.array([[1],[0],[0],[1]])
        self.assertEqual(m.predict(votos).tolist(), [1])
        self.assertEqual(m.prever_detalhes(votos)[2].tolist(), [True])
        # Invertendo só as prioris do treino, muda desempate.
        self.assertEqual(VotoMajoritario().fit([0,0,0,1]).predict(votos).tolist(), [0])

    def test_empate_entre_tres_classes(self):
        m=VotoMajoritario().fit([0,0,1,1,1,2,2,2,2])
        pred, contagens, emp=m.prever_detalhes(np.array([[0], [1], [2], [2]]))
        self.assertEqual(pred.tolist(), [2])
        self.assertFalse(emp[0])  # 2 tem dois votos, não é empate
        pred,_,emp=m.prever_detalhes(np.array([[0],[1],[1],[0]]))
        self.assertEqual(pred.tolist(), [1])
        self.assertTrue(emp[0])

    def test_prioris_iguais_menor_rotulo(self):
        m=VotoMajoritario().fit([2,0,1,2,0,1])
        # 2x de 0 e 2x de 2 => empataram voto e priori
        self.assertEqual(m.predict(np.array([[2],[0],[2],[0]])).tolist(), [0])

    def test_fracoes_somam_um_nao_sao_posteriores(self):
        m=VotoMajoritario().fit([0,0,1])
        frac=m.fracoes_votos(np.array([[0,1],[1,1],[0,1],[1,0]]))
        np.testing.assert_allclose(frac, [[0.5,0.5],[0.25,0.75]])
        np.testing.assert_allclose(frac.sum(axis=1), [1.,1.])

    def test_inputs_invalidos(self):
        m=VotoMajoritario()
        with self.assertRaises(RuntimeError): m.predict(np.array([[0],[0],[0],[0]]))
        m.fit([0,1,1])
        with self.assertRaises(ValueError): m.predict(np.array([[0],[0],[0]]))
        with self.assertRaises(ValueError): m.predict(np.array([[0],[0],[0],[2]]))
        with self.assertRaises(ValueError): decidir_votos(np.array([[0],[0],[0],[1]]), [0,1], [0.5,0.8])

    def test_integracao_um_fold_referencia(self):
        for nome_alvo in ('y_original','y_double_kmeans'):
            folds, linhas = avaliar_voto_de_csvs(nome_alvo, repeticoes=1, max_folds=1)
            self.assertEqual(len(folds),1)
            self.assertEqual(len(linhas), folds[0]['n_teste'])
            self.assertEqual(set(linhas[0]) & {'voto_gaussiano','voto_knn','voto_parzen','voto_logistica'},
                             {'voto_gaussiano','voto_knn','voto_parzen','voto_logistica'})
            _, ys, _ = carregar_dados()
            y=ys[nome_alvo]
            _,_,tr,te=next(gerar_folds_externos(y,repeticoes=1))
            self.assertEqual([p['indice_original'] for p in linhas],te.tolist())
            self.assertEqual(sum(x['empate'] for x in linhas), folds[0]['empates_resolvidos_priori'])
            # Prioris salvas são frequências do treinamento, não do teste.
            for c in np.unique(y):
                self.assertAlmostEqual(folds[0][f'priori_classe_{int(c)}'],np.mean(y[tr]==c))
            for row in linhas:
                votes=[row['voto_'+s] for s in ('gaussiano','knn','parzen','logistica')]
                cont=[votes.count(c) for c in np.unique(y)]
                self.assertEqual(sum(cont),4)
                self.assertEqual(row['predito'], np.unique(y)[np.argmax([cnt*10+folds[0][f'priori_classe_{int(c)}'] for cnt,c in zip(cont,np.unique(y))])])

    def test_csv_duplicado_rejeitado(self):
        _,ys,_=carregar_dados()
        original=PASTA_RESULTADOS/'gaussiano/y_original_1x10_predicoes.csv'
        with tempfile.TemporaryDirectory() as d:
            temp=Path(d)/'corrompido.csv'
            with original.open(encoding='utf-8',newline='') as f:
                rows=list(csv.DictReader(f))
            rows[1]['indice_original']=rows[0]['indice_original']
            with temp.open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0].keys())
                writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'duplicada'):
                _ler_predicoes(temp,'y_original',ys['y_original'],1)

    def test_csv_semente_diferente_rejeitada(self):
        with self.assertRaisesRegex(ValueError,'folds divergem'):
            avaliar_voto_de_csvs('y_original',repeticoes=1,max_folds=1,semente=2030)

    def test_incompleto_30_rejeita_com_mensagem(self):
        # Simular um diretório SEM arquivos 30x10, independentemente do estado
        # real do repositório (inclusive após finalizar todos os experimentos).
        with tempfile.TemporaryDirectory() as diretorio_vazio:
            with self.assertRaisesRegex(FileNotFoundError, 'Não existe'):
                avaliar_voto_de_csvs('y_original', repeticoes=30, max_folds=1,
                                     diretorio_resultados=Path(diretorio_vazio))


if __name__=='__main__':
    unittest.main()
