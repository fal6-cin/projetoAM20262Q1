# Questão 1 — Spambase / Double K-means (AM 2026-2)

> **Relatório técnico consolidado:** [docs/relatorio_questao1_double_kmeans.md](docs/relatorio_questao1_double_kmeans.md).

## Estrutura

```text
questao1/
├── README.md
├── src/
│   └── double_kmeans.py                  # Implementação do algoritmo Double K-means
├── docs/
│   └── relatorio_questao1_double_kmeans.md
├── notebooks/
│   ├── revisada.ipynb                    # Baseline (StandardScaler) + auditoria e sensibilidade
│   ├── binarizacao.ipynb                 # Frequências binarizadas + log1p(maiúsculas) + StandardScaler
│   ├── clip_log_standard.ipynb           # Clip 99% + log1p + StandardScaler
│   ├── filtragem_dominio.ipynb           # Remoção de 8 atributos de domínio + StandardScaler
│   ├── log_minmax.ipynb                  # log1p + MinMaxScaler (origem dos rótulos oficiais da Q2)
│   ├── quantile_uniform.ipynb            # QuantileTransformer uniforme
│   └── tratamento_diferenciado_runs.ipynb
└── resultados/
    ├── originais/                        # Saídas originais preservadas
    ├── revisada/                         # Auditoria e sensibilidade da linha de base
    ├── binarizacao/
    ├── clip_log_standard/
    ├── filtragem_dominio/
    ├── log_minmax/                       # K*=3, H*=3, seed=33044 -> spambase_com_clusters_log_minmax.csv
    ├── quantile_uniform/
    └── tratamento_diferenciado_runs/
```

## Origem dos Dados e Ligação com a Questão 2

- **Dados brutos:** lidos de `../data/raw/spambase.data` e `../data/raw/spambase.names` (4.601 instâncias, 57 atributos numéricos).
- **Entrada oficial da Questão 2:** o arquivo `questao2/dados/spambase_q2.csv` corresponde a `resultados/log_minmax/spambase_com_clusters_log_minmax.csv`, mantendo os **57 atributos na escala original** junto com `y_original` e `y_double_kmeans` ($K^*=3, H^*=3$).
