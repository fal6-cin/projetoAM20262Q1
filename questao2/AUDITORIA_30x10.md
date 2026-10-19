# Auditoria da Questão 2 — experimento 30 × 10-folds

**Status: completo e revalidado.** O protocolo externo contém 30 repetições de 10 folds por classificador e versão da variável resposta, com 4.601 predições por repetição.

## Integridade dos arquivos

- **Alvos:** `y_original` (2.788 classe 0; 1.813 classe 1) e `y_double_kmeans` (2.619 classe 0; 265 classe 1; 1.717 classe 2).
- **Base:** 4.601 e-mails e 57 atributos; SHA-256 do arquivo `dados/spambase_q2.csv` coincide com `dados/proveniencia_q1.json`.
- **Completude:** 10 combinações alvo/modelo × 300 folds = **3.000 avaliações externas**; 10 × 30 × 4.601 = **1.380.300 predições** auditadas.
- **Auditoria:** reexecução de `python -m questao2.scripts.analisar_resultados --repeticoes 30` confirmou índices de teste, repetições, classes reais e métricas recalculadas. As médias e os IC95 reproduziram exatamente os valores depositados; diferenças nos p-valores de Nemenyi limitaram-se a arredondamentos numéricos próximos da precisão de máquina.
- **Testes:** Os 56 testes unitários passaram.

## Médias e intervalos descritivos

Métricas macro conforme `resumo_metricas.csv`. Os IC95 são obtidos por aproximação t de Student sobre as 30 médias por repetição; **não representam amostras estatisticamente independentes**.

### Classes originais (spam/não spam)

| Classificador | Erro | Precisão macro | Cobertura macro | F1 macro [IC95 descritivo] |
|---|---:|---:|---:|---:|
| Bayes gaussiano | 15.16% | 0.8525 | 0.8683 | **0.8471** [0.8466; 0.8476] |
| Bayes k-vizinhos | 6.74% | 0.9321 | 0.9266 | **0.9290** [0.9282; 0.9298] |
| Bayes Parzen | 5.75% | 0.9422 | 0.9372 | **0.9394** [0.9389; 0.9400] |
| Regressão logística | 6.16% | 0.9357 | 0.9359 | **0.9355** [0.9351; 0.9360] |
| Voto majoritário | 5.71% | 0.9439 | 0.9364 | **0.9397** [0.9392; 0.9402] |

### Clusters Double K-means (K=3,H=3)

| Classificador | Erro | Precisão macro | Cobertura macro | F1 macro [IC95 descritivo] |
|---|---:|---:|---:|---:|
| Bayes gaussiano | 16.99% | 0.7624 | 0.8774 | **0.7976** [0.7968; 0.7984] |
| Bayes k-vizinhos | 6.55% | 0.9392 | 0.8903 | **0.9114** [0.9099; 0.9129] |
| Bayes Parzen | 5.66% | 0.9521 | 0.9000 | **0.9224** [0.9217; 0.9232] |
| Regressão logística | 0.95% | 0.9840 | 0.9819 | **0.9827** [0.9821; 0.9833] |
| Voto majoritário | 4.64% | 0.9612 | 0.9141 | **0.9347** [0.9338; 0.9357] |

## Friedman e Nemenyi — interpretação cuidadosa

- Para `y_original`, o maior F1 macro foi do **voto majoritário** (0,9397), praticamente empatado com **Parzen** (0,9394). Nemenyi para F1: **p≈0,988** entre voto e Parzen; a diferença numérica não foi considerada significativa nesse procedimento.
- Para `y_double_kmeans`, o maior F1 macro foi da **regressão logística** (0,9827), com taxa de erro de 0,95%. Nemenyi para F1: **p≈0,102** entre logística e voto, apesar da diferença numérica; para logística versus Parzen, **p≈0,00000954**.
- O teste de Friedman retornou `p < 10⁻¹⁹` nas quatro métricas para ambos os alvos. Isso indica diferenças de ranking segundo o procedimento aplicado; **não valida independência dos 30 blocos**, uma vez que reutilizam os mesmos 4.601 exemplos.
- A própria classificação `y_double_kmeans` deriva do agrupamento da base inteira. Classificá-la bem não prova que os clusters representam spam/não spam nem garante generalização para novas bases.