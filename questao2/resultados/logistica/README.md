# Regressão logística — resultados diagnósticos e finais

**Situação: avaliação final 30 × 10 concluída e consolidada para as duas variáveis resposta.** Os arquivos de 1 × 10 folds permanecem apenas para fins de diagnóstico e rastreabilidade.

## Procedimento

Classificador binário para `y_original` e multinomial softmax para `y_double_kmeans`, ajustado por LBFGS com regularização L2. Os hiperparâmetros `C ∈ {0,03; 0,3; 3; 30; 300}` e `class_weight ∈ {None, balanced}` são selecionados por F1 macro em cinco folds internos estratificados. `log1p → MinMaxScaler` é ajustado dentro de cada treino. `class_weight` pondera a função de perda e não altera as classes ou as amostras.

Os cinco classificadores foram comparados usando as **mesmas partições externas estratificadas** para um mesmo alvo. Cada repetição contém 10 folds externos; os hiperparâmetros dos quatro modelos-base foram escolhidos em validação cruzada interna de cinco folds, somente com os dados de treinamento.

## Resultados finais — 30 repetições × 10 folds

Os números abaixo são médias das 30 médias por repetição (cada uma calculada sobre seus 10 folds de teste). Precisão, cobertura e F-measure usam a **média macro**.

| Variável resposta | Folds | Predições externas | F1 macro | Taxa de erro | Precisão macro | Cobertura macro |
|---|---:|---:|---:|---:|---:|---:|
| `y_original` | 300 | 138.030 | 0,9355 | 0,0616 | 0,9357 | 0,9359 |
| `y_double_kmeans` | 300 | 138.030 | 0,9827 | 0,0095 | 0,9840 | 0,9819 |

**Arquivos finais, para cada alvo** (`y_original` e `y_double_kmeans`):

- `{alvo}_30x10_folds.csv` — métricas externas e informações por fold.
- `{alvo}_30x10_predicoes.csv` — predições externas auditáveis (30 × 4.601 = 138.030 registros por alvo).

As estimativas pontuais, os **intervalos de confiança aproximados de 95%** e os testes **Friedman/Nemenyi** foram consolidados por alvo em [`../analise_estatistica/y_original/`](../analise_estatistica/y_original/) e [`../analise_estatistica/y_double_kmeans/`](../analise_estatistica/y_double_kmeans/). Consulte também o [relatório final](../../RELATORIO_FINAL_QUESTAO2.md), que integra as curvas de aprendizagem de 5% a 95%.

**Ressalva estatística:** as repetições reutilizam as mesmas 4.601 observações; os intervalos t e os testes Friedman/Nemenyi baseados em médias de repetições são análises exploratórias, não inferência com 30 amostras independentes. A avaliação de `y_double_kmeans` mede a reprodução da partição da Questão 1, e não validação independente dos clusters.

## Resultados diagnósticos — apenas 1 repetição × 10 folds

Esta seção preserva os números iniciais. **Não são os resultados finais.**

| Variável resposta | Folds | Predições externas | F1 macro | Taxa de erro | Precisão macro | Cobertura macro |
|---|---:|---:|---:|---:|---:|---:|
| `y_original` | 10 | 4.601 | 0,9352 | 0,0619 | 0,9354 | 0,9353 |
| `y_double_kmeans` | 10 | 4.601 | 0,9813 | 0,0100 | 0,9821 | 0,9809 |

No diagnóstico 1×10, o máximo de iterações LBFGS observado foi 61 para `y_original` e 43 para `y_double_kmeans`.

Os arquivos `{alvo}_1x10_folds.csv` e `{alvo}_1x10_predicoes.csv` correspondem somente à execução diagnóstica; cada objeto foi testado uma vez nessa repetição. Esses arquivos não são a base dos intervalos de confiança nem dos testes comparativos finais.
