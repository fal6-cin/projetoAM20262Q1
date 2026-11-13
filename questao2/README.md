# Questão 2 — Spambase / Aprendizagem de Máquina 2026.2

> **Relatório final consolidado:** [docs/RELATORIO_FINAL_QUESTAO2.md](docs/RELATORIO_FINAL_QUESTAO2.md). Reúne as métricas e os intervalos de confiança descritivos do 30 × 10, Friedman/Nemenyi, curvas de aprendizagem e interpretação.

## Estado deste pacote
**Etapas 1, 2a, 2b, 2c, 2d e 2e funcionais:** base oficial da Questão 1, metadados, carregamento, verificações, 30×10-folds estratificados, 5-folds internos e funções de métricas. Os classificadores bayesianos gaussiano MULTIVARIADO, k-vizinhos, Parzen (kernel gaussiano multivariado produto) e regressão logística L2 estão implementados e integrados à CV aninhada; o voto majoritário dos quatro também está implementado, com auditoria dos arquivos de predições externas.

**Etapa 3 concluída (30×10):** os cinco classificadores, para `y_original` e `y_double_kmeans`, têm 30 repetições de 10 folds externos completas, auditadas e consolidadas. Já foram calculadas as quatro métricas, intervalos de confiança descritivos aproximados, Friedman e Nemenyi. Consulte `resultados/analise_estatistica/` e `docs/AUDITORIA_30x10.md`.

**Etapa 4 concluída:** as curvas de aprendizagem foram executadas de 5% a 95% de treinamento, em passos de 5%, para os cinco classificadores e as duas variáveis resposta. CSVs, 14 gráficos e discussão estão em `resultados/curvas_aprendizagem/` e `docs/RELATORIO_CURVAS_APRENDIZAGEM.md`. O relatório final está em `docs/RELATORIO_FINAL_QUESTAO2.md`. Resta a preparação da apresentação da disciplina.

**Ressalva estatística:** os testes de Friedman/Nemenyi e os ICs da Etapa 3 usam médias por repetição sobre os mesmos 4.601 e-mails; por isso, suas inferências devem ser interpretadas como exploratórias.

## Origem dos dados

`dados/spambase_q2.csv` é uma cópia byte a byte do arquivo:

`questao1_log_minmax/resultados_questao1_log_minmax/spambase_com_clusters_log_minmax.csv`

- 4.601 objetos e 57 atributos numéricos, mantidos em suas escalas originais.
- `y_original`: 0=2.788, 1=1.813.
- `y_double_kmeans`: 0=2.619, 1=265, 2=1.717.
- Rótulos do Double K-means: **Log1p + MinMaxScaler**, K*=3, H*=3, seed 33044, 100 execuções válidas por configuração na Questão 1.
- Para todos os detalhes de procedência e SHA-256, consulte `dados/proveniencia_q1.json`.

**Pré-processamento:** na Questão 1, Log1p + MinMaxScaler foi aplicado para obter os **rótulos dos clusters**. O CSV exportado para a Questão 2 mantém os **57 atributos nas escalas originais**. Na implementação efetivamente utilizada na Questão 2, os **quatro classificadores-base** aplicam Log1p + MinMaxScaler aos preditores por meio de um pipeline ajustado **somente sobre o treinamento** de cada fold externo e interno; o conjunto de teste é apenas transformado com os parâmetros aprendidos no treinamento. O voto majoritário combina as previsões dos quatro modelos e não requer um escalonamento próprio.

## Estrutura

```text
questao2/
├── README.md
├── docs/
│   ├── RELATORIO_FINAL_QUESTAO2.md
│   ├── RELATORIO_CURVAS_APRENDIZAGEM.md
│   ├── AUDITORIA_30x10.md
│   ├── ESTADO_ETAPA3.md
│   ├── COMPARACAO_PREPROCESSAMENTO.md
│   └── ANALISE_COMPARACAO_PREPROCESSAMENTO_QUESTAO2.md
├── config.json
├── dados/
│   ├── spambase_q2.csv
│   └── proveniencia_q1.json
├── src/
│   ├── dados.py
│   ├── validacao.py
│   ├── metricas.py
│   ├── avaliacao_gaussiano.py
│   ├── avaliacao_knn.py
│   ├── avaliacao_parzen.py
│   ├── avaliacao_logistica.py
│   ├── avaliacao_voto.py
│   ├── comparacao_preprocessamento.py
│   └── modelos/
│       ├── bayes_gaussiano.py
│       ├── bayes_knn.py
│       ├── bayes_parzen.py
│       ├── regressao_logistica.py
│       └── voto_majoritario.py
├── scripts/
│   ├── validar_base.py
│   ├── verificar_folds.py
│   ├── rodar_gaussiano.py
│   ├── rodar_knn.py
│   ├── rodar_parzen.py
│   ├── rodar_logistica.py
│   ├── rodar_voto.py
│   ├── executar_estudo.py
│   ├── status_estudo.py
│   ├── analisar_resultados.py
│   ├── comparar_preprocessamento.py
│   └── rodar_curvas.py
├── tests/                 # testes dos classificadores, da CV e dos resultados
├── resultados/
│   ├── gaussiano/
│   ├── knn/
│   ├── parzen/
│   ├── logistica/
│   ├── voto/
│   ├── execucoes/
│   ├── analise_estatistica/
│   ├── comparacao_preprocessamento/
│   └── curvas_aprendizagem/
└── questao2_experimentos.ipynb
```

## Como instalar e testar

1. Descompacte a pasta `questao2/` **na raiz do mesmo repositório** que contém `questao1_log_minmax/`.
2. Abra um terminal na **raiz** do repositório (o diretório que contém a pasta `questao2/`).
3. Opcionalmente crie um ambiente virtual e instale os requisitos:

```bash
python -m pip install -r requirements.txt
```

4. Execute as verificações:

```bash
python -m questao2.scripts.validar_base
python -m questao2.scripts.verificar_folds
python -m unittest discover -s questao2/tests -v
```

5. Abra `questao2/questao2_experimentos.ipynb` no Jupyter e execute as células de diagnóstico.

## Protocolo exigido pela Questão 2

Para **cada uma** das variáveis resposta (`y_original`, `y_double_kmeans`):

1. Avaliar cinco classificadores (Bayes normal multivariado completo, Bayes k-vizinhos, Bayes Parzen, regressão logística, voto majoritário dos quatro).
2. Usar **validação cruzada estratificada 30 × 10-fold** (30 repetições; em cada uma, 10 folds), com **5-folds estratificados internos** para ajuste de hiperparâmetros quando necessário.
3. Usar os **mesmos folds** para os cinco modelos ao comparar um mesmo conjunto de rótulos.
4. Estimar **taxa de erro, precisão, cobertura e F-measure**; conservar valores por classe e médias macro, depois construir estimativas pontuais/ICs com procedimento estatístico explicitado.
5. Comparar classificadores com Friedman e Nemenyi; interpretar adequadamente a dependência dos folds repetidos (uma possibilidade é usar média por repetição como bloco, registrando a limitação).
6. Fazer curvas de aprendizagem de **5% a 95%, passo 5%**, com métricas de treino e teste e amostragem estratificada.

## Cuidados estatísticos

- A classe minoritária da partição Double K-means tem **265 objetos**.
- A matriz de covariância completa do grupo minoritário apresenta singularidade devido a atributos constantes/colineares. A implementação utiliza o classificador **gaussiano multivariado com covariância completa** (não `GaussianNB`) e regularização explícita da estimativa de covariância. A intensidade da regularização é escolhida por CV interna, apenas com os dados de treinamento, e sua justificativa está detalhada na seção 2a.
- O `y_double_kmeans` foi criado por agrupamento sobre a base toda. A validação supervisionada nessa segunda versão mede a **reprodução dessa partição**, não a generalização independente dos clusters para novos e-mails.
- A regra de desempate do voto majoritário considera a **maior probabilidade a priori no treino**, conforme os slides do professor.
- Os scripts de verificação de folds não treinam classificadores e não ajustam escalonamento global.

## Situação das entregas

- Etapa 2: **concluída** — quatro classificadores-base, voto majoritário e testes automatizados.
- Etapa 3: **concluída e auditada** — 30 × 10 avaliações externas por classificador e variável resposta, métricas, ICs descritivos, Friedman/Nemenyi exploratórios.
- Etapa 4: **concluída** — curvas de aprendizagem de 5% a 95%, F1 macro de treino e teste, CSVs, 14 gráficos e discussão.
- Relatório final: **concluído** — `docs/RELATORIO_FINAL_QUESTAO2.md`.
- Estudo complementar de pré-processamento: **concluído como análise exploratória**, incluindo as duas reavaliações localizadas e as limitações documentadas.
- Testes automatizados: **67 casos executados e aprovados** nesta versão (os 62 da auditoria antiga são um marco histórico).
- Apresentação da disciplina: **pendente**.


## Classificador bayesiano gaussiano implementado (etapa 2a)

### Fórmula e escolhas matemáticas

A classe `BayesGaussianoMultivariado` estima, no treino:

- `P(classe=c)=n_c/n` (priori de máxima verossimilhança);
- `mu_c = mean(X_c)`;
- `Sigma_c_ML = (X_c-mu_c)^T(X_c-mu_c)/n_c`, matriz de covariância COMPLETA;
- `Sigma_c_usada = Sigma_c_ML + lambda * s_global * I`, com `s_global`
  igual à média das variâncias ML de todos os atributos de **treinamento**.

Sem regularização (`lambda=0`), a expressão é a estimativa de máxima
verossimilhança integral, mas pode ser singular. **Com lambda>0, a matriz
usada na regra bayesiana é uma adaptação regularizada, não a covariância de
máxima verossimilhança pura.** Documente a adaptação no relatório e justifique
pela singularidade observada. As densidades são avaliadas com Cholesky e
log-verossimilhança; nada é treinado com `y_original` ao avaliar
`y_double_kmeans` (e vice-versa).

### Protocolo sem vazamento

O pipeline `log1p → MinMaxScaler → BayesGaussianoMultivariado` é ajustado
DO ZERO em cada treino de validação cruzada interna e depois reajustado no
treino externo. `GridSearchCV` busca `lambda` em
`[1e-6, 1e-4, 1e-2, 1e-1]` por F1 macro, com 5 folds estratificados.
O fold externo de teste é usado **apenas** na predição/avaliação final.
É uma escolha explícita aplicar também Log1p + MinMax aos preditores da
Questão 2, sem reaproveitar scaler ajustado na Questão 1.

### Como rodar (a partir da raiz do repositório)

Execução curta de conferência (um fold para cada versão de rótulo):

```bash
python -m questao2.scripts.rodar_gaussiano
```

Um ciclo completo de 10 folds em cada alvo (ainda não são 30 repetições):

```bash
python -m questao2.scripts.rodar_gaussiano --todos-folds --repeticoes 1
```

Protocolo completo exigido (computacionalmente mais custoso):

```bash
python -m questao2.scripts.rodar_gaussiano --todos-folds --repeticoes 30
```

Os resultados por fold e as probabilidades individuais são salvos em
`resultados/gaussiano/`. **Saídas do modo parcial NÃO representam métricas
conclusivas**. A pontuação da CV interna jamais substitui a métrica de teste
externa. Esta seção descreve o primeiro classificador; a seção seguinte apresenta o segundo.

### Resultados diagnósticos e finais incluídos no pacote

O diretório `resultados/gaussiano/` preserva os diagnósticos iniciais de **1 × 10 folds**, além dos resultados definitivos de **30 × 10 folds** para os dois alvos.

| Alvo | F1 macro diagnóstico (1 × 10) | Erro diagnóstico | F1 macro final (30 × 10) | Erro final |
|---|---:|---:|---:|---:|
| `y_original` | 0,8467 | 0,1521 | **0,8471** | **0,1516** |
| `y_double_kmeans` | 0,7967 | 0,1695 | **0,7976** | **0,1699** |

Os arquivos `<alvo>_30x10_folds.csv` contêm **300 avaliações externas por alvo** e fundamentam o relatório final, os intervalos de confiança descritivos e as comparações Friedman/Nemenyi. Os resultados de 1 × 10 folds ficam preservados apenas para rastreabilidade. A regularização foi selecionada por CV interna de cinco folds, sem uso do teste externo no ajuste. Consulte `resultados/gaussiano/README.md`.


## Classificador Bayesiano k-vizinhos (etapa 2b)

**Modelo implementado:** `src/modelos/bayes_knn.py`, classe `BayesKVizinhos`. A estimativa de posterior é `P(classe=c | x) = k_c/k`, onde `k_c` é a contagem de exemplos de treino da classe c entre os k vizinhos mais próximos. As distâncias estão de acordo com o enunciado e os slides `Tecnicas-Nao-Parametricas.pdf` (páginas 23 a 29):

- `euclidiana`: norma L2;
- `city_block`: norma L1, também chamada Manhattan;
- `chebyshev`: norma L-infinito, máximo das diferenças absolutas.

A regra é **bayesiana por proporção empírica de vizinhos**, não Naive Bayes, não densidade normal e não votação ponderada. Para empates na decisão, removem-se os vizinhos mais distantes um a um até existir um vencedor único, conforme estratégia discutida na aula de aprendizado baseado em instâncias. `predict_proba` sempre fornece as frações calculadas com os k vizinhos originais; portanto, em caso de empate de probabilidades, `predict` aplica explicitamente a regra de desempate. Em empates exatos de distância, os exemplos mais antigos (menor índice de treino) vêm primeiro, para garantir reprodutibilidade.

**Seleção de hiperparâmetros:** `src/avaliacao_knn.py` pesquisa `k` em `(1,3,5,7,11,15,21)` e as três distâncias por F1 macro médio em **5 folds internos estratificados**. O `log1p` é aplicado aos atributos não negativos e `MinMaxScaler` é ajustado exclusivamente no conjunto de treinamento de cada fold interno. Para acelerar a busca sem alterar seus resultados, os 21 candidatos reutilizam a busca de k=21 vizinhos dentro de cada distância/fold, e cada candidato recebe seu próprio F1. Em empate de F1 médio, a seleção usa ordem determinística (Euclidiana, City-Block, Chebyshev; depois menor k). O par escolhido é reajustado do zero em todo o treino externo. Os folds externos são exatamente os mesmos do gaussiano para cada conjunto de rótulos, usando a mesma semente.

**Limitações:** proporções de vizinhos podem conter probabilidades zero; isso é esperado. A implementação calcula distâncias por lotes para reduzir uso de memória, mas as 30×10 avaliações com ajuste interno continuam custosas. O k-NN pode ser afetado pela dimensionalidade e pela escolha da escala.

### Comandos (a partir da raiz do repositório)

Conferência rápida, **um fold externo por alvo**:

```bash
python -m questao2.scripts.rodar_knn
```

Uma repetição inteira, **dez folds por alvo**:

```bash
python -m questao2.scripts.rodar_knn --todos-folds --repeticoes 1
```

Protocolo completo da Questão 2, **30×10 folds por alvo**:

```bash
python -m questao2.scripts.rodar_knn --todos-folds --repeticoes 30
```

Os CSVs de `resultados/knn/` guardam os hiperparâmetros escolhidos por fold, F1 da CV interna, métricas externas e probabilidades/previsões individuais. Uma execução com `--max-folds` produz arquivos `parcial_*` e **não serve como avaliação final**.

**Referências didáticas:** `Tecnicas-Nao-Parametricas.pdf` (método k-vizinhos, distâncias e ajuste de k), `Instance-Based-Learning-1.pdf` (normalização e empates), `Projeto-AM-2026-2.pdf` (Questão 2, distâncias e CV aninhada).


## Classificador Bayes por janela de Parzen (etapa 2c)

**Implementado em `src/modelos/bayes_parzen.py`:** estimador de densidade de
Parzen com **kernel gaussiano multivariado produto**. A densidade condicional
para cada classe c é a média de produtos de 57 densidades normais univariadas,
com mesma largura de janela escalar h em cada dimensão:

```text
p_hat(x | c) = (1/n_c) sum_{i:y_i=c} prod_{j=1}^{57}
                           exp(-(x_j - x_ij)^2/(2 h^2)) / (sqrt(2 pi) h)
P_hat(c) = n_c/n
P_hat(c | x) = p_hat(x | c) P_hat(c) / sum_r p_hat(x | r) P_hat(r)
```

Isto **não é GaussianNB**, nem um classificador de média/covariância normal:
utiliza todas as amostras de treino como centros da mistura de kernels. A
opção h compartilhado é uma restrição metodológica explícita da família
mais geral de larguras por variável que aparece nos slides.

**Escolha de h por cinco folds internos:** `src/avaliacao_parzen.py` avalia a
grade `h = (0.04, 0.08, 0.12, 0.20, 0.35, 0.55)`, fixada antes de observar
o teste, por **F1 macro**. Em empate de F1 interno, escolhe o menor h.
O pipeline `log1p → MinMaxScaler → BayesParzen` é ajustado DO ZERO em cada
treino interno e reajustado no treino externo. Não há uso dos dados do teste
externo para escolher largura, ajuste de escala, estimativa de densidade ou
probabilidades a priori. Para reduzir o custo, as distâncias do treino e
validação internos são reaproveitadas entre os seis h. Os escores de densidade
são computados com **log-sum-exp**, evitando o underflow de produtos de 57
kernels muito pequenos; as probabilidades permanecem normalizadas.

Para rodar, na **raiz do repositório**:

```bash
# Apenas um fold externo por versão; rápido, para teste
python -m questao2.scripts.rodar_parzen

# Primeira repetição completa: 10 folds externos por versão
python -m questao2.scripts.rodar_parzen --todos-folds --repeticoes 1

# Protocolo completo exigido (custo computacional significativo)
python -m questao2.scripts.rodar_parzen --todos-folds --repeticoes 30
```

As saídas incluem métricas por fold, largura selecionada, F1 da CV interna, resultados por classe e probabilidades/predições de cada observação do teste. Estão no diretório `resultados/parzen/`. **Os CSVs `*_1x10_*` são diagnósticos históricos**; os CSVs `*_30x10_*` contêm os resultados definitivos, utilizados nas análises estatísticas e no relatório final. As curvas de aprendizagem, já concluídas, estão em `resultados/curvas_aprendizagem/`.

Diagnóstico inicial (média de 10 folds externos em cada variável resposta):

| Alvo | F1 macro | Taxa de erro | Precisão macro | Cobertura macro | h selecionado |
|---|---:|---:|---:|---:|---|
| y_original | 0,9416 | 0,0554 | 0,9441 | 0,9397 | 0,12 nos 10 folds |
| y_double_kmeans | 0,9254 | 0,0543 | 0,9530 | 0,9034 | 0,20 em 9 folds; 0,12 em 1 |

A avaliação final de 30 × 10, os intervalos de confiança descritivos e os testes estatísticos já foram executados. O enunciado solicita envio dos arquivos individuais ao Google Classroom; os ZIPs servem apenas para facilitar a transferência entre versões do repositório.


## Regressão logística — quarto classificador (etapa 2d)

O arquivo `src/modelos/regressao_logistica.py` implementa o classificador
`RegressaoLogistica` como uma interface verificável para o otimizador LBFGS do
scikit-learn, com **regressão binária** para `y_original` e **regressão
multinomial softmax** para `y_double_kmeans`. A função de perda é a
log-verossimilhança negativa (entropia cruzada), com **penalidade L2**.
O otimizador fornece `predict_proba` para os dois casos. Não é uma
implementação do zero do método de otimização; a implementação e os testes
conferem a formulação e integram a CV exigida pela disciplina.

Na CV interna estratificada de 5 folds (`src/avaliacao_logistica.py`),
selecionamos por **F1 macro** as combinações de:

- `C ∈ {0.03, 0.3, 3.0, 30.0, 300.0}` — inverso da força de regularização L2;
- `class_weight ∈ {None, 'balanced'}` — pesos uniformes ou inversamente proporcionais à frequência de classes.

O `log1p` e o `MinMaxScaler` são ajustados dentro dos conjuntos de treinamento
de cada fold interno e novamente nos 9 folds de treinamento externo após a
seleção. O conjunto de teste externo não é usado para escolher os parâmetros.
A busca e os fold externos usam F1 macro; os CSVs guardam, por fold, os valores
escolhidos, as quatro métricas pedidas, resultados por classe, predições e
probabilidades. A ponderação de classes é treinada somente dentro do `fit`
e não cria nem exclui exemplos.

Na raiz do repositório:

```bash
# Diagnóstico com um fold externo de cada alvo
python -m questao2.scripts.rodar_logistica

# Dez folds (uma única repetição; diagnóstico)
python -m questao2.scripts.rodar_logistica --todos-folds --repeticoes 1

# Estudo completo solicitado (computacionalmente mais custoso)
python -m questao2.scripts.rodar_logistica --todos-folds --repeticoes 30
```

**Situação:** o diagnóstico `1×10` não substitui o protocolo `30×10`, que já foi concluído. Os intervalos de confiança descritivos, Friedman/Nemenyi e as curvas de aprendizagem também estão concluídos. O voto majoritário está implementado na etapa 2e. Consulte `resultados/logistica/README.md` para detalhes.

**Entrega final da disciplina:** conforme `Projeto-AM-2026-2.pdf`, os arquivos
devem ser submetidos *individualmente* no Google Classroom, **não em ZIP**.
O ZIP produzido em nossas etapas serve apenas para atualizar o repositório.


## Voto majoritário — quinto classificador (etapa 2e)

**Implementado e testado:** `src/modelos/voto_majoritario.py`, com a integração
`src/avaliacao_voto.py` e o comando `scripts/rodar_voto.py`. O método faz
a **votação não ponderada** entre as decisões dos quatro classificadores,
na ordem: Bayes gaussiano multivariado, Bayes k-vizinhos, Bayes Parzen e
regressão logística. A classe com maior número de votos vence. Em caso de
empate (por exemplo, 2 × 2), a decisão é favorável à classe empatada que
tiver a **maior probabilidade a priori no TREINO EXTERNO** do fold. Se
as prioris das empatadas forem idênticas, escolhe-se o menor rótulo
como desempate determinístico secundário.

A integração combina os **quatro CSVs de predições externas**, sem refazer
o treinamento ou consultar predições sobre o treinamento. Os quatro
modelos-base já passaram, cada um separadamente, pela busca interna de
hiperparâmetros quando necessário. O voto não introduz novos hiperparâmetros.
`avaliacao_voto.py` audita a **igualdade exata de alvo, repetição, fold,
índice da amostra e rótulo real**, rejeitando arquivos truncados, duplicados,
com semente/folds diferentes ou com classes inválidas.

A priori é calculada por contagem de `y_treino` **para cada fold**, portanto
não existe consulta ao rótulo do fold de teste no desempate. O CSV de
saída registra os quatro votos individuais, as contagens por classe, os
empates, as prioris de treinamento por fold e as quatro métricas solicitadas.
As **frações de votos** são proporções de decisões discretas e não devem
ser interpretadas como probabilidades a posteriori calibradas.

### Como executar

Na raiz do repositório, após gerar os CSVs correspondentes dos quatro
modelos-base (a primeira repetição 1×10 já vem neste pacote):

```bash
# 1 fold de diagnóstico por alvo
python -m questao2.scripts.rodar_voto

# Uma repetição integral de 10 folds — diagnóstico incluído no ZIP
python -m questao2.scripts.rodar_voto --todos-folds --repeticoes 1

# Depois de executar os quatro modelos-base em 30×10:
python -m questao2.scripts.rodar_gaussiano --todos-folds --repeticoes 30
python -m questao2.scripts.rodar_knn --todos-folds --repeticoes 30
python -m questao2.scripts.rodar_parzen --todos-folds --repeticoes 30
python -m questao2.scripts.rodar_logistica --todos-folds --repeticoes 30
python -m questao2.scripts.rodar_voto --todos-folds --repeticoes 30
```

A **execução final do voto não treina os modelos**; é necessária a presença
dos quatro arquivos `<alvo>_30x10_predicoes.csv` em cada subpasta de
`resultados/`. Caso faltem, o programa emite erro explicando o que
precisa ser executado. As métricas `1×10` são diagnósticas; o conjunto `30×10` está presente e auditado, incluindo ICs descritivos e Friedman/Nemenyi exploratórios. As curvas de aprendizagem também foram concluídas e estão na pasta `resultados/curvas_aprendizagem/`.

### Diagnóstico de uma repetição 1×10 (média por fold)

| Alvo | F1 macro | Erro | Precisão macro | Cobertura macro | Empates totais |
|---|---:|---:|---:|---:|---:|
| `y_original` | 0,9404 | 0,0565 | 0,9439 | 0,9375 | 179 |
| `y_double_kmeans` | 0,9349 | 0,0452 | 0,9579 | 0,9160 | 171 |

O voto majoritário **não necessariamente supera** o melhor classificador
individual. A comparação 30×10 foi executada; interprete seus testes de significância como exploratórios devido à dependência entre repetições. Os arquivos de diagnóstico desta etapa estão em
`resultados/voto/`.

---

## Etapa 3A — Histórico da execução retomável (já concluída)

Este tópico documenta o procedimento usado para gerar o estudo. **A execução completa 30×10 já está presente neste repositório**, e suas tabelas finais estão em `resultados/analise_estatistica/`. A execução abrangeu,
para os quatro modelos-base, **2 alvos × 30 repetições × 10 folds externos**, cada
um deles incluindo 5 folds internos para seleção de hiperparâmetros. A execução
é custosa, sobretudo para Parzen.

### Executar sem perder todo o progresso ao interromper

Abra um terminal na **raiz do repositório** (pasta que contém `questao2/`), instale
`requirements.txt` e rode:

```bash
# Confirma as implementações e a base
python -m unittest discover -s questao2/tests -v
python -m questao2.scripts.validar_base

# Verifica o estado dos arquivos 30×10 (não treina modelos)
python -m questao2.scripts.status_estudo

# Todos os quatro classificadores-base, os dois alvos, 30 repetições
# Pode levar muito tempo: deixar o computador funcionando até terminar.
python -m questao2.scripts.executar_estudo --repeticoes 30

# Se interromper, execute o MESMO comando novamente:
# repetições já salvas são auditadas e ignoradas.
python -m questao2.scripts.executar_estudo --repeticoes 30

# Só após completar todos os modelos: análise estatística
python -m questao2.scripts.analisar_resultados --repeticoes 30
```

É possível rodar de modo parcelado, por exemplo:

```bash
python -m questao2.scripts.executar_estudo --modelos gaussiano knn --alvo y_original --inicio 1 --fim 5
python -m questao2.scripts.executar_estudo --modelos gaussiano knn --alvo y_original --inicio 6 --fim 30
python -m questao2.scripts.executar_estudo --modelos parzen logistica --alvo y_original
python -m questao2.scripts.executar_estudo --modelos todos --alvo y_double_kmeans
```

Cada repetição é escrita em `resultados/execucoes/<modelo>/<alvo>/rep_XX_{folds,predicoes}.csv`.
Cada arquivo é auditado para verificar **fold, índice de teste, classe real e métricas
recalculadas**. Arquivos válidos são preservados em retomadas; arquivos faltantes
são produzidos sem recomputar os anteriores. O gerador usa semente de CV 2026;
os modelos continuam a escolher hiperparâmetros em cinco folds internos.

Se os resultados preexistentes `*_1x10_*.csv` passarem na auditoria, o sistema
reaproveita esses diagnósticos como primeira repetição. Para desativar esse
reaproveitamento use `--nao-reutilizar-1x10`. Para **somente consolidar**
repetições previamente executadas, use `--apenas-consolidar`.

O arquivo final `resultados/<modelo>/<alvo>_30x10_{folds,predicoes}.csv` é
criado **somente quando as 30 repetições daquele modelo/alvo estão disponíveis
e auditadas**. Depois que os quatro modelos-base de um alvo estiverem completos,
o orquestrador produz o **voto majoritário** usando as predições pareadas e as
prioris do treinamento de cada fold. É proibido combinar resultados de seeds,
índices ou folds incompatíveis.

### Estatísticas dos 30 blocos pareados

`python -m questao2.scripts.analisar_resultados --repeticoes 30` exige os cinco
modelos completos nos dois alvos. Confere todos os CSVs e produz, para cada alvo:

- `medias_por_repeticao.csv`: média dos dez folds de cada repetição/modelo/métrica;
- `resumo_metricas.csv`: média das 30 repetições, desvio-padrão e IC aproximado de 95%;
- `friedman.csv`: Friedman com 30 repetições como blocos pareados;
- `nemenyi_pares.csv`: pós-teste Nemenyi (10 pares para cada métrica);
- `ranking_medio.csv`: postos médios dos classificadores para cada métrica;
- `relatorio_avaliacao.md`: tabela e observações estatísticas.

**Cautela:** as 30 repetições reutilizam os mesmos 4.601 e-mails, e os folds dentro
de cada repetição se sobrepõem nos dados de treinamento. Por isso, o IC t de Student
sobre as médias das repetições e as inferências Friedman/Nemenyi são
**aproximações exploratórias**, não garantias formais de generalização para
novas bases independentes. Não consideramos 300 folds como 300 amostras
independentes. O quinto classificador (voto) depende das predições dos outros
quatro; sua dependência também deve ser observada na interpretação dos testes.
A atividade exige ICs e Friedman/Nemenyi; esta implementação declara claramente
as hipóteses/limitações das estimativas.

### Diagnósticos (não são o experimento final)

Os CSVs anteriores `*_1x10_*.csv` foram auditados e resumidos separadamente em
`resultados/analise_estatistica/`. Eles **não** geram testes Friedman/Nemenyi ou
ICs. Você pode reproduzir a análise descritiva:

```bash
python -m questao2.scripts.analisar_resultados --repeticoes 1 --diagnostico
```

### O que continua pendente

**Somente a apresentação final da disciplina.** As Etapas 3 (avaliação 30 × 10, métricas, ICs descritivos e Friedman/Nemenyi exploratórios) e 4 (curvas de aprendizagem de 5% a 95%) foram concluídas e documentadas no relatório final.


## Etapa 4 — Curvas de aprendizagem de F1 (letra d)

O programa `python -m questao2.scripts.rodar_curvas` executa separadamente,
para `y_original` e `y_double_kmeans`, **19 divisões complementares** treino/teste
estratificadas: 5%/95%, 10%/90%, ... 95%/5%. Todos os cinco modelos usam
**a mesma divisão** para um percentual e uma repetição.

- Quatro modelos-base: hiperparâmetros selecionados com 5-fold estratificado **apenas no conjunto de treino da divisão**. O pipeline Log1p+MinMax é ajustado internamente e reajustado no treino, sem vazamento para o teste.
- O voto reúne as predições desses quatro modelos sobre treino e teste, usando para desempatar **somente** as priori estimadas no treinamento.
- Calcula **F1 macro nos conjuntos de treinamento e teste** e salva, adicionalmente, precisão, cobertura e taxa de erro. Valores de treino são in-sample (particularmente otimistas para modelos por vizinhos/kernel).
- Usa como entrada os 57 atributos do CSV na escala original; nenhuma característica ou resposta é recalculada. Os clusters foram gerados sobre a base inteira durante a Questão 1; por isso sua tarefa supervisionada avalia reprodução dessa partição.
- Os percentuais são de **todo o conjunto de 4.601 observações**. Divisões entre percentuais têm estratificação e semente controladas, mas **não precisam ser subconjuntos aninhados**.
- **A Questão 2(d) não exige expressamente 30 repetições** das curvas. O padrão é 1 divisão por proporção (19 por alvo); `--repeticoes 5`, por exemplo, adiciona replicações estratificadas opcionais. Não confundir com os 30×10 folds obrigatórios da letra (a), já completos.

A partir da raiz do repositório:

```bash
# Reproduzir/retomar as 19 proporções × duas versões × 5 modelos
python -m questao2.scripts.rodar_curvas --repeticoes 1

# Só auditar/consolidar os CSVs existentes e gerar os gráficos quando completos
python -m questao2.scripts.rodar_curvas --repeticoes 1 --apenas-consolidar

# Opcional, não exigido na letra (d): 5 sorteios por proporção
python -m questao2.scripts.rodar_curvas --repeticoes 5

# Rodada curta de desenvolvimento (NÃO é curva final)
python -m questao2.scripts.rodar_curvas --alvo y_original --pct-inicio 5 --pct-fim 10
```

Os arquivos individuais são salvos **atomicamente por ponto** em
`resultados/curvas_aprendizagem/execucoes/{alvo}/rep_XX_pct_YY.csv`, o que permite
retomar sem perder resultados. O consolidado e os 14 PNGs (comparação de cinco
modelos em treino/teste e curvas pareadas por modelo) são gerados **somente
quando todas as 19 proporções estão disponíveis**. Para resultados de
mais de um sorteio, cada ponto apresenta média e desvio padrão dos F1.

O documento `docs/RELATORIO_CURVAS_APRENDIZAGEM.md` descreve o protocolo e os
resultados executados. Os gráficos não substituem os testes Friedman/Nemenyi
calculados para a etapa anterior.

## Análise complementar de pré-processamento

Foi realizado um estudo exploratório de sensibilidade à preparação dos 57 atributos da Spambase, comparando **dados sem pré-processamento**, **apenas MinMaxScaler** e **Log1p seguido de MinMaxScaler**. Foram utilizados os mesmos dez folds externos estratificados para cada variável resposta e as mesmas divisões entre os cinco classificadores; os hiperparâmetros dos quatro modelos-base foram escolhidos exclusivamente por CV interna de cinco folds.

A comparação totalizou **300 avaliações externas** (2 alvos × 3 estratégias × 5 classificadores × 10 folds). A estratégia Log1p + MinMax apresentou o maior F1 macro médio nas dez combinações de classificador e alvo avaliadas, mas o estudo é **exploratório** e não constitui prova de superioridade estatística definitiva.

Dois cenários foram reavaliados após a primeira rodada:

- **Regressão logística sem pré-processamento:** `max_iter=20000` e `tol=1e-6`; os avisos de não convergência diminuíram (311→171 para as classes originais; 510→308 para os clusters), mas persistiram.
- **Parzen com apenas MinMaxScaler:** a grade de `h` foi ampliada para incluir valores abaixo de 0,04; em nenhum fold a largura selecionada ficou no novo limite inferior de 0,005.

O voto majoritário foi recalculado nos cenários afetados. Os resultados oficiais **30 × 10**, os arquivos de avaliação principal e as curvas de aprendizagem permaneceram inalterados.

Para reproduzir ou consolidar a análise a partir da raiz do repositório:

```bash
python -m questao2.scripts.comparar_preprocessamento --repeticoes 1
python -m questao2.scripts.comparar_preprocessamento --repeticoes 1 --apenas-consolidar
```

Documentos: [protocolo e diagnóstico histórico](docs/COMPARACAO_PREPROCESSAMENTO.md) e [análise completa revisada](docs/ANALISE_COMPARACAO_PREPROCESSAMENTO_QUESTAO2.md). CSVs e registros por fold: `resultados/comparacao_preprocessamento/`.
