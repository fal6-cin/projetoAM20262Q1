# Bayes por janela de Parzen — resultados diagnósticos e finais

**Situação: avaliação final 30 × 10 concluída e consolidada para as duas variáveis resposta.** Os arquivos de 1 × 10 folds permanecem apenas para fins de diagnóstico e rastreabilidade.

## Procedimento

Estimativa da densidade condicional por kernel gaussiano multivariado produto, com largura comum `h`, selecionada por F1 macro na validação cruzada interna estratificada de cinco folds. `log1p → MinMaxScaler` é ajustado somente com exemplos dos treinos correspondentes; os cálculos utilizam log-sum-exp para estabilidade numérica.

Os cinco classificadores foram comparados usando as **mesmas partições externas estratificadas** para um mesmo alvo. Cada repetição contém 10 folds externos; os hiperparâmetros dos quatro modelos-base foram escolhidos em validação cruzada interna de cinco folds, somente com os dados de treinamento.

## Resultados finais — 30 repetições × 10 folds

Os números abaixo são médias das 30 médias por repetição (cada uma calculada sobre seus 10 folds de teste). Precisão, cobertura e F-measure usam a **média macro**.

| Variável resposta | Folds | Predições externas | F1 macro | Taxa de erro | Precisão macro | Cobertura macro |
|---|---:|---:|---:|---:|---:|---:|
| `y_original` | 300 | 138.030 | 0,9394 | 0,0575 | 0,9422 | 0,9372 |
| `y_double_kmeans` | 300 | 138.030 | 0,9224 | 0,0566 | 0,9521 | 0,9000 |

**Arquivos finais, para cada alvo** (`y_original` e `y_double_kmeans`):

- `{alvo}_30x10_folds.csv` — métricas externas e informações por fold.
- `{alvo}_30x10_predicoes.csv` — predições externas auditáveis (30 × 4.601 = 138.030 registros por alvo).

As estimativas pontuais, os **intervalos de confiança aproximados de 95%** e os testes **Friedman/Nemenyi** foram consolidados por alvo em [`../analise_estatistica/y_original/`](../analise_estatistica/y_original/) e [`../analise_estatistica/y_double_kmeans/`](../analise_estatistica/y_double_kmeans/). Consulte também o [relatório final](../../RELATORIO_FINAL_QUESTAO2.md), que integra as curvas de aprendizagem de 5% a 95%.

**Ressalva estatística:** as repetições reutilizam as mesmas 4.601 observações; os intervalos t e os testes Friedman/Nemenyi baseados em médias de repetições são análises exploratórias, não inferência com 30 amostras independentes. A avaliação de `y_double_kmeans` mede a reprodução da partição da Questão 1, e não validação independente dos clusters.

## Resultados diagnósticos — apenas 1 repetição × 10 folds

Esta seção preserva os números iniciais. **Não são os resultados finais.**

| Variável resposta | Folds | Predições externas | F1 macro | Taxa de erro | Precisão macro | Cobertura macro |
|---|---:|---:|---:|---:|---:|---:|
| `y_original` | 10 | 4.601 | 0,9416 | 0,0554 | 0,9441 | 0,9397 |
| `y_double_kmeans` | 10 | 4.601 | 0,9254 | 0,0543 | 0,9530 | 0,9034 |

Na execução inicial de dez folds, `h=0,12` foi escolhido em todos os folds de `y_original`; em `y_double_kmeans`, `h=0,20` foi escolhido em nove folds e `h=0,12` em um.

Os arquivos `{alvo}_1x10_folds.csv` e `{alvo}_1x10_predicoes.csv` correspondem somente à execução diagnóstica; cada objeto foi testado uma vez nessa repetição. Esses arquivos não são a base dos intervalos de confiança nem dos testes comparativos finais.
