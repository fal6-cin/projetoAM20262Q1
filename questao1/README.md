# Questão 1 — Spambase / Double K-means (AM 2026-2)

> **Notebook unificado:** [questao1_experimentos.ipynb](questao1_experimentos.ipynb) (centraliza todas as 7 estratégias de pré-processamento).  
> **Relatório técnico consolidado:** [docs/relatorio_questao1_double_kmeans.md](docs/relatorio_questao1_double_kmeans.md).

## Estrutura

```text
questao1/
├── README.md
├── __init__.py
├── questao1_experimentos.ipynb           # Notebook unificado (espelho de questao2/questao2_experimentos.ipynb)
├── docs/
│   └── relatorio_questao1_double_kmeans.md
├── src/
│   ├── double_kmeans.py                  # Implementação do algoritmo Double K-means
│   ├── dados.py                          # Leitura do Spambase bruto em ../data/raw/
│   ├── preprocessamentos.py              # Registro das 7 estratégias de pré-processamento
│   ├── experimento.py                    # Grade (K,H), reprodução, sanidade, gráficos e exportação
│   └── auditoria.py                      # Auditoria detalhada de microclusters (baseline revisada)
├── scripts/
│   └── rodar_experimentos.py             # CLI para executar, reproduzir ou resumir estratégias
├── tests/
│   └── test_questao1_experimentos.py     # Testes automatizados de reprodutibilidade das 7 estratégias
├── notebooks/                            # Notebooks individuais originais preservados (histórico)
│   ├── revisada.ipynb
│   ├── binarizacao.ipynb
│   ├── clip_log_standard.ipynb
│   ├── filtragem_dominio.ipynb
│   ├── log_minmax.ipynb
│   ├── quantile_uniform.ipynb
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

## Como Executar e Testar

A partir da **raiz do repositório**:

```powershell
# Reproduzir e validar todas as 7 estratégias pela CLI
python -m questao1.scripts.rodar_experimentos --estrategia todas

# Apenas exibir o comparativo consolidado
python -m questao1.scripts.rodar_experimentos --resumo

# Rodar os testes automatizados da Questão 1
pytest questao1/tests
```

## Origem dos Dados e Ligação com a Questão 2

- **Dados brutos:** lidos de `../data/raw/spambase.data` e `../data/raw/spambase.names` (4.601 instâncias, 57 atributos numéricos).
- **Dependências:** instaladas pelo [../requirements.txt](../requirements.txt) na raiz do repositório (`pip install -r requirements.txt`).
- **Entrada oficial da Questão 2:** o arquivo `questao2/dados/spambase_q2.csv` corresponde a `resultados/log_minmax/spambase_com_clusters_log_minmax.csv`, mantendo os **57 atributos na escala original** junto com `y_original` e `y_double_kmeans` ($K^*=3, H^*=3$).
