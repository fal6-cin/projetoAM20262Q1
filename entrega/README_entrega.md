# Checklist de Entrega — Projeto AM 2026-2 (18/11/2026)

> **Regra obrigatória do enunciado ([`Projeto-AM-2026-2.pdf`](../Enunciado%20do%20Projeto/Projeto-AM-2026-2.pdf)):**  
> **NÃO compactar em `.zip`!** Enviar **todos os arquivos individualmente** no Google Classroom.  
> **Apresentação:** quarta-feira, **18/11/2026**, limite rigoroso de **10 minutos** (incluindo discussão), com presença e participação obrigatórias de toda a equipe (Douglas Araújo Silva, Felipe Araújo de Lima e Letícia De Albuquerque S. Leitão).

## 1. Relatório do Projeto

- [`relatorio_final_projeto_am_2026_2.md`](relatorio_final_projeto_am_2026_2.md) — Relatório final consolidado em Markdown (Questão 1 + Questão 2, discussão, ameaças à validade, reprodutibilidade, referências e anexos).
- [`relatorio_final_projeto_am_2026_2.tex`](relatorio_final_projeto_am_2026_2.tex) — Código-fonte LaTeX do relatório final consolidado.
- [`relatorio_final_projeto_am_2026_2.pdf`](relatorio_final_projeto_am_2026_2.pdf) — Versão compilada em PDF do relatório final consolidado.

## 2. Código-Fonte Principal

### Base e Ambiente
- [`../requirements.txt`](../requirements.txt)
- [`../data/raw/spambase.data`](../data/raw/spambase.data)
- [`../data/raw/spambase.names`](../data/raw/spambase.names)

### Questão 1 (Double K-means)
- [`../questao1/questao1_experimentos.ipynb`](../questao1/questao1_experimentos.ipynb)
- [`../questao1/src/double_kmeans.py`](../questao1/src/double_kmeans.py)
- [`../questao1/src/dados.py`](../questao1/src/dados.py)
- [`../questao1/src/preprocessamentos.py`](../questao1/src/preprocessamentos.py)
- [`../questao1/src/experimento.py`](../questao1/src/experimento.py)
- [`../questao1/src/auditoria.py`](../questao1/src/auditoria.py)
- [`../questao1/scripts/rodar_experimentos.py`](../questao1/scripts/rodar_experimentos.py)

### Questão 2 (Classificadores e Análise Estatística)
- [`../questao2/dados/spambase_q2.csv`](../questao2/dados/spambase_q2.csv)
- [`../questao2/dados/proveniencia_q1.json`](../questao2/dados/proveniencia_q1.json)
- [`../questao2/questao2_experimentos.ipynb`](../questao2/questao2_experimentos.ipynb)
- Módulos em [`../questao2/src/`](../questao2/src/) e [`../questao2/src/modelos/`](../questao2/src/modelos/)
- Scripts executáveis em [`../questao2/scripts/`](../questao2/scripts/)

## 3. Slides da Apresentação

- Slides a preparar e anexar individualmente no Google Classroom junto ao código-fonte e ao relatório em **18/11/2026**.
