# Questão 2 — Estudo complementar de sensibilidade ao pré-processamento

## Objetivo

Investigar, sem substituir a avaliação oficial de 30 × 10 folds, se a combinação **Log1p + MinMaxScaler** adotada na Questão 2 se mostra adequada aos classificadores supervisionados, comparando-a com dados sem pré-processamento e com a aplicação isolada de MinMaxScaler.

## Protocolo auditado

- **Fontes:** `questao2/resultados/comparacao_preprocessamento/comparacao_folds.csv` e `comparacao_resumo.csv`, consolidados após a reavaliação dos cenários de sensibilidade.
- **Variáveis resposta:** `y_original` (spam/não spam) e `y_double_kmeans` (três clusters).
- **Classificadores:** Bayes gaussiano multivariado, Bayes k-vizinhos, Bayes Parzen, regressão logística e voto majoritário.
- **Validação:** uma repetição completa de validação cruzada estratificada de 10 folds por alvo, com cinco folds internos para seleção de hiperparâmetros dos quatro classificadores-base.
- **Comparabilidade:** as três estratégias compartilham os mesmos índices de teste em cada fold e variável resposta.
- **Prevenção de vazamento:** as transformações são ajustadas exclusivamente nos dados de treinamento, inclusive durante a validação cruzada interna.

O experimento compreende **300 avaliações externas**, resultantes de:

2 variáveis resposta × 3 estratégias × 5 classificadores × 10 folds.

Essas avaliações não representam 300 repetições independentes, mas 30 combinações de classificador, alvo e estratégia, cada uma avaliada em dez folds.

### Reavaliação complementar

Após a primeira comparação, foram identificados dois cenários que exigiam investigação adicional:

1. Regressão logística sem pré-processamento, devido aos avisos de não convergência.
2. Bayes Parzen com apenas MinMaxScaler, devido à seleção recorrente da menor largura de janela disponível.

Foram realizados os seguintes ajustes:

- Na regressão logística sem pré-processamento, utilizaram-se `max_iter=20000` e `tol=1e-6`.
- No Parzen com MinMaxScaler, a grade de larguras de janela foi ampliada para incluir valores inferiores a `0,04`.
- As previsões do voto majoritário foram recalculadas nas duas estratégias afetadas.
- Os demais resultados válidos foram reaproveitados.

A reavaliação produziu **80 registros revisados**, considerando os dois alvos, dez folds e quatro combinações de modelo e estratégia, incluindo os votos majoritários.

Os registros originais foram preservados em `execucoes/`, enquanto os revisados foram armazenados em `execucoes_ajustes_v2/`.

Os resultados da estratégia `log1p_minmax` permanecem idênticos aos diagnósticos oficiais de 1 × 10 folds anteriormente verificados.

Todos os valores apresentados a seguir representam médias dos dez folds externos, arredondadas a quatro casas decimais.

## Resultados — F1 macro

### Classes originais (`y_original`)

| Classificador | Sem pré-processamento | Apenas MinMax | Log1p + MinMax | Ganho sobre dados brutos |
|---|---:|---:|---:|---:|
| Gaussiano multivariado | 0,8279 | 0,8287 | **0,8467** | +0,0188 |
| Bayes k-vizinhos | 0,8445 | 0,9054 | **0,9287** | +0,0842 |
| Bayes Parzen | 0,8268 | 0,9022 | **0,9416** | +0,1148 |
| Regressão logística | 0,9247 | 0,9230 | **0,9352** | +0,0105 |
| Voto majoritário | 0,8937 | 0,9189 | **0,9404** | +0,0467 |

### Clusters do Double K-means (`y_double_kmeans`)

| Classificador | Sem pré-processamento | Apenas MinMax | Log1p + MinMax | Ganho sobre dados brutos |
|---|---:|---:|---:|---:|
| Gaussiano multivariado | 0,6822 | 0,5550 | **0,7967** | +0,1145 |
| Bayes k-vizinhos | 0,8229 | 0,8953 | **0,9155** | +0,0927 |
| Bayes Parzen | 0,7818 | 0,8958 | **0,9254** | +0,1436 |
| Regressão logística | 0,8977 | 0,8928 | **0,9813** | +0,0836 |
| Voto majoritário | 0,8458 | 0,9063 | **0,9349** | +0,0891 |

Os ganhos representam diferenças descritivas entre médias de F1 macro. Não devem ser interpretados isoladamente como estimativas causais ou evidências de significância estatística.

Os resultados da regressão logística sem pré-processamento exigem cautela devido à persistência de avisos de não convergência.

## Efeito dos ajustes complementares

### Regressão logística sem pré-processamento

| Variável resposta | F1 anterior | F1 revisado | Diferença |
|---|---:|---:|---:|
| Classes originais | 0,9260 | 0,9247 | −0,0014 |
| Double K-means | 0,9045 | 0,8977 | −0,0067 |

A reavaliação também reduziu os avisos de não convergência:

| Variável resposta | Avisos anteriores | Avisos revisados |
|---|---:|---:|
| Classes originais | 311 | 171 |
| Double K-means | 510 | 308 |

Os avisos contabilizam ocorrências durante a seleção interna de hiperparâmetros e os ajustes finais, não a quantidade de folds externos.

Embora tenha ocorrido redução, a alteração dos parâmetros de otimização não eliminou os problemas de convergência.

Consequentemente, não é possível afirmar que o desempenho da regressão logística com dados brutos represente uma solução plenamente convergida.

### Bayes Parzen com MinMaxScaler

A grade de larguras de janela foi ampliada para:

`h = (0,005; 0,01; 0,02; 0,03; 0,04; 0,08; 0,12; 0,20; 0,35; 0,55)`

Na primeira comparação, `h = 0,04` havia sido selecionado em todos os folds dos dois alvos, coincidindo com o limite inferior da grade original.

Após a ampliação:

- **Classes originais:** `h = 0,03` em quatro folds e `h = 0,04` em seis folds.
- **Double K-means:** `h = 0,04` nos dez folds.

Nenhum valor selecionado coincidiu com o novo limite inferior de `0,005`.

#### Impacto no desempenho

| Variável resposta | F1 anterior | F1 revisado |
|---|---:|---:|
| Classes originais | 0,9037 | 0,9022 |
| Double K-means | 0,8958 | 0,8958 |

A ampliação da grade não produziu melhoria no F1 macro médio de teste.

Entretanto, permitiu verificar que a seleção anterior não estava necessariamente limitada pela ausência de valores menores.

Assim, a limitação específica da grade original foi investigada e tratada.

### Voto majoritário

O voto majoritário foi recalculado utilizando as previsões atualizadas dos classificadores-base nos mesmos exemplos de teste.

Essa atualização afetou as estratégias sem pré-processamento e apenas MinMaxScaler.

Os resultados revisados do voto já estão incorporados às tabelas principais.

## Achados e interpretação

1. **Log1p + MinMaxScaler apresentou o maior F1 macro médio nas dez combinações de classificador e variável resposta avaliadas.** A conclusão permaneceu a mesma após os ajustes complementares.

2. A vantagem foi especialmente evidente nos classificadores baseados em distâncias, como Bayes k-vizinhos e Bayes Parzen. Nas classes originais, o F1 macro do Parzen aumentou de 0,8268 com dados brutos para 0,9416 com a transformação completa. Nos clusters, aumentou de 0,7818 para 0,9254.

3. A transformação completa também superou a aplicação isolada de MinMaxScaler em todos os classificadores desta repetição. Esse comportamento é compatível com a hipótese de que a compressão não linear realizada por Log1p pode contribuir além da normalização das amplitudes. Entretanto, a comparação não isola um efeito causal nem comprova superioridade universal.

4. A normalização isolada não beneficiou todos os modelos. No gaussiano multivariado aplicado aos clusters, o F1 macro diminuiu de 0,6822 nos dados brutos para 0,5550 com MinMaxScaler. Com Log1p + MinMaxScaler, alcançou 0,7967.

5. Os rótulos dos clusters foram previamente obtidos a partir da base completa na Questão 1. Portanto, o desempenho dos classificadores em `y_double_kmeans` representa a capacidade de reproduzir essa partição, e não uma demonstração independente da qualidade dos agrupamentos ou da capacidade de generalização do processo de agrupamento.

## Limitações e ajustes realizados

### Uma única repetição de dez folds

A comparação utilizou uma repetição de validação cruzada estratificada de dez folds para cada variável resposta.

Embora esse protocolo permita uma avaliação pareada das estratégias, não constitui evidência confirmatória de superioridade estatística.

Os resultados devem ser interpretados como exploratórios e não substituem os experimentos oficiais de 30 × 10 folds.

### Convergência parcialmente melhorada

A regressão logística sem pré-processamento foi reexecutada com maior limite de iterações e tolerância mais rigorosa.

Apesar da redução dos avisos, a não convergência persistiu nos dois alvos.

Portanto, o desempenho dessa configuração deve ser interpretado com cautela.

A investigação de estratégias adicionais de otimização poderá ser realizada futuramente, caso seja necessário obter uma comparação livre desses avisos.

### Grade de Parzen ampliada

A limitação identificada na grade inicial de Parzen com MinMaxScaler foi investigada mediante a inclusão de larguras de janela menores.

Os valores selecionados após a ampliação não coincidiram com o novo limite inferior.

Essa observação reduz a preocupação de que os resultados estivessem restritos pela menor largura originalmente disponível, embora não demonstre que a nova grade seja globalmente ótima.

### Ausência de avaliação confirmatória independente

Uma investigação adicional poderia repetir a comparação com outras divisões previamente definidas e condições de otimização mais abrangentes.

O experimento complementar não deve ser utilizado para selecionar retrospectivamente configurações com base nos conjuntos de teste da avaliação oficial.

A escolha de Log1p + MinMaxScaler permanece justificada por suas propriedades metodológicas e pelos resultados exploratórios observados, sem pressupor superioridade universal.

## Conclusão

Como análise complementar de sensibilidade, foram comparados cinco classificadores em duas variáveis resposta sob três condições: dados sem pré-processamento, normalização MinMaxScaler isolada e transformação Log1p seguida de MinMaxScaler.

A avaliação utilizou uma repetição de validação cruzada estratificada de dez folds, com divisões externas idênticas entre as estratégias e seleção de hiperparâmetros por validação cruzada interna.

**A combinação Log1p + MinMaxScaler apresentou o maior F1 macro médio nas dez combinações de classificador e variável resposta avaliadas.**

Após a primeira comparação, foram realizados dois ajustes complementares: ampliação da grade de larguras de janela do classificador Parzen com MinMaxScaler e reavaliação da regressão logística sem pré-processamento.

A ampliação da grade de Parzen permitiu selecionar larguras inferiores às disponíveis anteriormente, sem produzir melhoria no F1 macro médio de teste. A regressão logística apresentou redução dos avisos de não convergência, embora esses avisos não tenham sido completamente eliminados.

Os ajustes não modificaram a principal observação do estudo: a transformação Log1p + MinMaxScaler permaneceu com o melhor desempenho médio em todas as combinações avaliadas.

Esses resultados fornecem **evidência empírica exploratória favorável à estratégia de pré-processamento adotada na Questão 2**, especialmente para os classificadores sensíveis às distâncias entre observações.

Entretanto, a comparação não demonstra superioridade estatística definitiva, considerando a realização de apenas uma repetição de dez folds e a persistência de problemas de convergência na regressão logística com dados brutos.

Dessa forma, mantém-se Log1p + MinMaxScaler como a estratégia de pré-processamento da Questão 2. Os experimentos oficiais de 30 × 10 folds permanecem como avaliação principal, enquanto a presente comparação é documentada como um estudo complementar de sensibilidade, com suas limitações explicitadas.
