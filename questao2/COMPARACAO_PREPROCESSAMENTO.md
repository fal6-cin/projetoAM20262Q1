# Estudo complementar — sensibilidade ao pré-processamento (Questão 2)

## Pergunta e finalidade

A escolha de `Log1p + MinMaxScaler`, motivada inicialmente pelo Double K-means na Questão 1, também se mostra adequada para a classificação supervisionada na Questão 2?

O objetivo é realizar uma **análise exploratória de sensibilidade**, sem substituir os resultados oficiais obtidos com validação cruzada de **30 × 10 folds** nem escolher retrospectivamente uma configuração com base nos testes externos oficiais.

## Estratégias comparadas

| Nome no programa | Atributos utilizados |
|---|---|
| `sem` | 57 atributos originais, sem Log1p e sem normalização |
| `minmax` | 57 atributos originais com MinMaxScaler |
| `log1p_minmax` | Log1p seguido de MinMaxScaler (estratégia da avaliação oficial da Questão 2) |

O CSV e os dois vetores de resposta permanecem inalterados. Os experimentos são executados separadamente para `y_original` (spam/não spam) e `y_double_kmeans` (três clusters).

## Protocolo comparativo

- Foi utilizada **uma repetição completa de dez folds externos estratificados por alvo**, com semente `2026`; os mesmos índices de teste foram compartilhados entre as três estratégias e os cinco classificadores dentro de cada alvo.
- Em cada treinamento externo, os quatro classificadores-base selecionaram seus hiperparâmetros por **cinco folds internos estratificados**, maximizando F1 macro.
- O MinMaxScaler, quando presente, foi ajustado **somente com os dados de treinamento** em cada fold interno e novamente no treino externo. O conjunto de teste não participou do ajuste.
- Foram utilizados Bayes gaussiano multivariado, Bayes k-vizinhos, Bayes Parzen, regressão logística e voto majoritário. O voto usa as predições dos quatro modelos nos mesmos exemplos de teste e as probabilidades a priori calculadas exclusivamente no treino externo.
- Foram registrados F1 macro, taxa de erro, precisão macro e cobertura macro por fold. O experimento salva resultados por classificador e fold para permitir retomada.
- A largura de janela do Parzen na estratégia `sem` foi pesquisada na grade adaptada à escala bruta `h ∈ {0,05; 0,2; 1; 5; 20; 100; 500}`; a grade oficial de Parzen em `log1p_minmax` permanece `h ∈ {0,04; 0,08; 0,12; 0,20; 0,35; 0,55}`.

O estudo consolidado compreende **300 avaliações externas**: 2 alvos × 3 estratégias × 5 classificadores × 10 folds. Essas avaliações **não equivalem a 300 repetições independentes**.

## Ajustes posteriores à primeira rodada

A primeira análise revelou dois problemas que poderiam prejudicar a interpretação de alternativas ao pré-processamento oficial. Foram então reavaliados os seguintes cenários:

1. **Regressão logística sem pré-processamento:** aumento do limite de iterações para `max_iter=20000`, com `tol=1e-6`, mantendo os mesmos folds, hiperparâmetros candidatos e critérios de avaliação.
2. **Bayes Parzen com apenas MinMaxScaler:** ampliação da grade para `h ∈ {0,005; 0,01; 0,02; 0,03; 0,04; 0,08; 0,12; 0,20; 0,35; 0,55}`.

O **voto majoritário foi recalculado** em cada uma das duas estratégias afetadas. Os demais cenários, inclusive `log1p_minmax`, tiveram seus registros válidos preservados. Foram gerados **80 registros revisados** (2 alvos × 10 folds × 4 combinações de estratégia/modelo, incluindo o voto).

Os arquivos históricos permanecem em `resultados/comparacao_preprocessamento/execucoes/` e os cenários reexecutados estão em `resultados/comparacao_preprocessamento/execucoes_ajustes_v2/`. Os CSVs consolidados incorporam os resultados revisados.

## Resultados consolidados — F1 macro de teste

Os valores abaixo são médias dos dez folds externos, arredondadas a quatro casas decimais.

### Classes originais (`y_original`)

| Classificador | Sem pré-processamento | Apenas MinMax | Log1p + MinMax |
|---|---:|---:|---:|
| Bayes gaussiano multivariado | 0,8279 | 0,8287 | **0,8467** |
| Bayes k-vizinhos | 0,8445 | 0,9054 | **0,9287** |
| Bayes Parzen | 0,8268 | 0,9022 | **0,9416** |
| Regressão logística | 0,9247 | 0,9230 | **0,9352** |
| Voto majoritário | 0,8937 | 0,9189 | **0,9404** |

### Três clusters (`y_double_kmeans`)

| Classificador | Sem pré-processamento | Apenas MinMax | Log1p + MinMax |
|---|---:|---:|---:|
| Bayes gaussiano multivariado | 0,6822 | 0,5550 | **0,7967** |
| Bayes k-vizinhos | 0,8229 | 0,8953 | **0,9155** |
| Bayes Parzen | 0,7818 | 0,8958 | **0,9254** |
| Regressão logística | 0,8977 | 0,8928 | **0,9813** |
| Voto majoritário | 0,8458 | 0,9062 | **0,9349** |

**Resultado principal:** Log1p + MinMax apresentou o maior F1 macro médio nos cinco classificadores e nos dois alvos desta comparação exploratória. Isso fornece evidência favorável à estratégia adotada, mas **não comprova superioridade estatística definitiva**.

## Resultados dos ajustes

### Regressão logística sem pré-processamento

| Alvo | F1 anterior | F1 revisado | Avisos de não convergência antes | Depois |
|---|---:|---:|---:|---:|
| Classes originais | 0,9260 | 0,9247 | 311 | 171 |
| Três clusters | 0,9045 | 0,8977 | 510 | 308 |

Os números de avisos são contagens acumuladas da validação interna e dos ajustes finais, **não** quantidade de folds. Apesar da redução, a regressão logística com dados brutos **continuou apresentando avisos**; por isso, não é possível afirmar que seu desempenho corresponda a ajustes plenamente convergidos.

### Parzen com MinMaxScaler

| Alvo | F1 anterior | F1 revisado | Larguras selecionadas nos dez folds |
|---|---:|---:|---|
| Classes originais | 0,9037 | 0,9022 | `h=0,03` em 4 folds; `h=0,04` em 6 |
| Três clusters | 0,8958 | 0,8958 | `h=0,04` em 10 folds |

Nenhuma escolha atingiu a **nova borda inferior** da grade (`h=0,005`). Assim, a limitação anterior da busca foi investigada e mitigada, embora não se possa afirmar que a grade seja universalmente ótima.

## Diagnóstico inicial de desenvolvimento — registro histórico

Antes da comparação completa, foi executado **um único fold externo** para a regressão logística com as classes originais, usando o limite de **2.000 iterações**. Esse diagnóstico não corresponde à comparação consolidada posterior:

| Estratégia | F1 macro de teste em um fold | Avisos de não convergência na CV/ajuste |
|---|---:|---:|
| Sem pré-processamento | 0,9254 | 31 |
| Apenas MinMax | 0,9363 | 0 |
| Log1p + MinMax | 0,9293 | 0 |

Esses valores são preservados exclusivamente para rastreabilidade, **não substituem** os resultados médios dos dez folds apresentados acima.

## Execução e reprodução

Execute na **raiz do repositório**:

```bash
python -m pip install -r questao2/requirements.txt
python -m questao2.scripts.comparar_preprocessamento --repeticoes 1
```

Para conferir os arquivos já gerados e consolidar sem retreinar:

```bash
python -m questao2.scripts.comparar_preprocessamento --repeticoes 1 --apenas-consolidar
```

Para um diagnóstico limitado, sem representatividade da rodada completa:

```bash
python -m questao2.scripts.comparar_preprocessamento --modelos logistica --alvo y_original --max-folds 1
```

Os resultados ficam em `resultados/comparacao_preprocessamento/`, incluindo `comparacao_folds.csv`, `comparacao_resumo.csv` e os JSONs de execução. O código preserva registros anteriores e utiliza diretório separado para os cenários revisados.

## Interpretação, limitações e preservação da versão oficial

A vantagem numérica de Log1p + MinMax é **exploratória**, pois foi avaliada em uma única repetição de dez folds. A regressão logística em dados brutos não alcançou convergência em todos os ajustes, e não foi conduzida uma comparação confirmatória independente. Além disso, `y_double_kmeans` foi construído usando o conjunto completo na Questão 1; classificá-lo mede a capacidade de reproduzir a partição, não sua validade externa.

**Os avaliadores originais, `config.json`, os resultados oficiais de 30 × 10 folds e as curvas de aprendizagem não são modificados pela análise complementar.** O relatório principal inclui apenas uma síntese contextualizada dessa investigação.

A interpretação detalhada e os resultados completos constam em [Análise da comparação de pré-processamento](ANALISE_COMPARACAO_PREPROCESSAMENTO_QUESTAO2.md).
