# Voto majoritário — resultados diagnósticos e finais

**Situação: avaliação final 30 × 10 concluída e consolidada para as duas variáveis resposta.** Os arquivos de 1 × 10 folds permanecem apenas para fins de diagnóstico e rastreabilidade.

## Procedimento

Combina as predições externas, fora do treino, dos quatro classificadores-base nos **mesmos folds**, após auditoria dos índices e rótulos. Em empate, escolhe a classe empatada com **maior frequência a priori no treino externo**; não usa rótulos de teste. Os componentes foram ajustados com CV interna de cinco folds; o voto não exige seleção adicional de hiperparâmetros.

Os cinco classificadores foram comparados usando as **mesmas partições externas estratificadas** para um mesmo alvo. Cada repetição contém 10 folds externos; os hiperparâmetros dos quatro modelos-base foram escolhidos em validação cruzada interna de cinco folds, somente com os dados de treinamento.

## Resultados finais — 30 repetições × 10 folds

Os números abaixo são médias das 30 médias por repetição (cada uma calculada sobre seus 10 folds de teste). Precisão, cobertura e F-measure usam a **média macro**.

| Variável resposta | Folds | Predições externas | F1 macro | Taxa de erro | Precisão macro | Cobertura macro |
|---|---:|---:|---:|---:|---:|---:|
| `y_original` | 300 | 138.030 | 0,9397 | 0,0571 | 0,9439 | 0,9364 |
| `y_double_kmeans` | 300 | 138.030 | 0,9347 | 0,0464 | 0,9612 | 0,9141 |

**Arquivos finais, para cada alvo** (`y_original` e `y_double_kmeans`):

- `{alvo}_30x10_folds.csv` — métricas externas e informações por fold.
- `{alvo}_30x10_predicoes.csv` — predições externas auditáveis (30 × 4.601 = 138.030 registros por alvo).

As estimativas pontuais, os **intervalos de confiança aproximados de 95%** e os testes **Friedman/Nemenyi** foram consolidados por alvo em [`../analise_estatistica/y_original/`](../analise_estatistica/y_original/) e [`../analise_estatistica/y_double_kmeans/`](../analise_estatistica/y_double_kmeans/). Consulte também o [relatório final](../../RELATORIO_FINAL_QUESTAO2.md), que integra as curvas de aprendizagem de 5% a 95%.

**Ressalva estatística:** as repetições reutilizam as mesmas 4.601 observações; os intervalos t e os testes Friedman/Nemenyi baseados em médias de repetições são análises exploratórias, não inferência com 30 amostras independentes. A avaliação de `y_double_kmeans` mede a reprodução da partição da Questão 1, e não validação independente dos clusters.

## Resultados diagnósticos — apenas 1 repetição × 10 folds

Esta seção preserva os números iniciais. **Não são os resultados finais.**

| Variável resposta | Folds | Predições externas | F1 macro | Taxa de erro | Precisão macro | Cobertura macro |
|---|---:|---:|---:|---:|---:|---:|
| `y_original` | 10 | 4.601 | 0,9404 | 0,0565 | 0,9439 | 0,9375 |
| `y_double_kmeans` | 10 | 4.601 | 0,9349 | 0,0452 | 0,9579 | 0,9160 |

No diagnóstico 1×10 ocorreram 179 empates para `y_original` e 171 para `y_double_kmeans`. Essas contagens correspondem apenas à repetição diagnóstica e **não** ao total dos 30×10.

Os arquivos `{alvo}_1x10_folds.csv` e `{alvo}_1x10_predicoes.csv` correspondem somente à execução diagnóstica; cada objeto foi testado uma vez nessa repetição. Esses arquivos não são a base dos intervalos de confiança nem dos testes comparativos finais.
