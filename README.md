# Projeto AM 2026-2 — Spambase (Double K-means e Classificação)

Repositório do projeto da disciplina **Aprendizagem de Máquina (CIn/UFPE — 2026.2)** sobre a base **Spambase** (4.601 instâncias, 57 atributos numéricos contínuos e 2 classes a priori: `non-spam` / `spam`).

Enunciado completo: [Projeto-AM-2026-2.pdf](Enunciado%20do%20Projeto/Projeto-AM-2026-2.pdf).

---

## Estrutura do Repositório

```text
.
├── data/
│   └── raw/                             # Fonte única da base original (spambase.data, spambase.names)
├── Enunciado do Projeto/
│   └── Projeto-AM-2026-2.pdf            # Especificação oficial do projeto
├── questao1/                            # Co-clustering não supervisionado (ver questao1/README.md)
│   ├── questao1_experimentos.ipynb      # Notebook unificado para as 7 estratégias
│   ├── src/                             # Double K-means, dados, pré-processamentos, experimento e auditoria
│   ├── scripts/                         # CLI unificada (rodar_experimentos.py)
│   ├── tests/                           # Testes automatizados de reprodutibilidade da Q1
│   ├── docs/                            # Relatório técnico da Questão 1
│   ├── notebooks/                       # Notebooks individuais preservados
│   └── resultados/                      # Resultados organizados por experimento (originais, revisada, log_minmax, ...)
├── questao2/                            # Pipeline modular da Questão 2 (ver questao2/README.md)
│   ├── dados/                           # spambase_q2.csv (escala original + y_double_kmeans) e proveniencia_q1.json
│   ├── src/                             # Carregamento, CV 30×10 / 5-fold, métricas, Friedman/Nemenyi e 5 modelos
│   ├── scripts/                         # Scripts CLI de execução, comparação e análise
│   ├── tests/                           # Suíte de testes automatizados (67 casos)
│   ├── resultados/                      # Métricas, predições, análise estatística e curvas de aprendizagem
│   └── docs/                            # Relatórios e auditorias da Questão 2
├── entrega/
│   └── README_entrega.md                # Checklist dos arquivos individuais para o Google Classroom
└── requirements.txt                     # Dependências unificadas do projeto (Q1, Q2, notebooks e testes)
```

---

## Fluxo do Projeto

1. **Questão 1 (Co-clustering não supervisionado com Double K-means):**
   - Grade $(K, H)$ com $K \in \{2, 3, 4\}$ e $H \in \{1, \ldots, K\}$ (9 pares, 100 reinicializações válidas por par).
   - Seleção por menor função objetivo $W$ por par, seguida de Silhueta média para escolha de $(K^*, H^*)$ e avaliação externa via ARI.
   - Os rótulos oficiais exportados para a Questão 2 (`questao2/dados/spambase_q2.csv`) vêm de `questao1/resultados/log_minmax/` ($K^*=3, H^*=3$, mantendo os 57 atributos na escala original).
2. **Questão 2 (Classificação Supervisionada e Avaliação Estatística):**
   - Comparação de 5 classificadores (**Bayesiano Gaussiano**, **k-NN**, **Janela de Parzen**, **Regressão Logística** e **Voto Majoritário**) em duas versões do alvo: `y_original` (2 classes) e `y_double_kmeans` (3 classes).
   - Validação cruzada estratificada **30 × 10-folds**, ajuste de hiperparâmetros via **5-fold** restrito aos 9 folds de treino, intervalos de confiança, testes **Friedman / Nemenyi** e curvas de aprendizagem (5% a 95%).
   - Detalhes completos em [questao2/README.md](questao2/README.md) e [questao2/docs/RELATORIO_FINAL_QUESTAO2.md](questao2/docs/RELATORIO_FINAL_QUESTAO2.md).

---

## Configuração do Ambiente e Testes

Com Python 3.12+ na raiz do repositório:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para executar todas as suítes de testes (Questão 1 e Questão 2):

```powershell
pytest questao1/tests questao2/tests
```

---

## Entrega (18/11/2026)

* **Google Classroom:** enviar **arquivos individuais** (código-fonte, executável se houver, slides e relatório) — **não compactar em `.zip`**.
* **Apresentação:** limite rigoroso de **10 minutos** (incluindo discussão), com presença e participação obrigatórias de toda a equipe.
* Consulte o checklist em [entrega/README_entrega.md](entrega/README_entrega.md).
