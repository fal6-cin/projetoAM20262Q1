import tempfile
import unittest
from pathlib import Path
from questao2.scripts.executar_estudo import _concatenar_csv, _gravar_atomicamente
from questao2.src.auditoria_experimentos import ler_csv

class TestRetomada(unittest.TestCase):
    def test_arquivo_atomicamente_e_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.csv';b=Path(tmp)/'b.csv';merged=Path(tmp)/'merged.csv'
            _gravar_atomicamente(a,[{'repeticao':1,'fold':1,'metricas_por_classe':{'0':{'precisao':1}}}])
            _gravar_atomicamente(b,[{'repeticao':2,'fold':1,'metricas_por_classe':{'0':{'precisao':1}}}])
            self.assertEqual(len(list(Path(tmp).glob('*.temporario'))),0)
            _concatenar_csv([a,b],merged)
            self.assertEqual(len(ler_csv(merged)),2)

if __name__=='__main__':unittest.main()
