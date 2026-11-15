# Roteiro de apresentação — Spambase: decisões que mudaram os resultados

**Objetivo:** explicar as decisões técnicas adotadas e justificá-las com evidências do projeto, não reproduzir integralmente o relatório.

**Tempo sugerido:** 7 min 40 s de apresentação e 2 min 20 s para perguntas, totalizando os 10 minutos do enunciado. O tempo deve ser ensaiado pela equipe e todos os integrantes devem participar.

| Slide | Tempo | Mensagem principal |
|---|---:|---|
| 1. Capa | 0:15 | Apresentaremos decisões, evidências e consequências das duas questões. |
| 2. Três perguntas | 0:30 | O desafio combina representação, avaliação controlada e interpretação. |
| 3. Diagnóstico da Silhueta | 0:45 | O z-score deu Silhueta 0,6596, mas ARI quase zero: microcluster de 34 e-mails. |
| 4. Representação escolhida | 0:55 | Comparamos sete estratégias. log1p + MinMax preserva atributos e gerou (K,H)=(3,3). O ARI informou a escolha *entre representações*, não o algoritmo em si. |
| 5. O que os clusters mostram | 0:40 | Descrever os três grupos e a concordância parcial com as classes conhecidas (ARI 0,3911). |
| 6. Validação aninhada | 0:55 | Justificar 30 × 10 folds externos, ajuste 5-folds interno, pipeline e métricas macro. Diferenciar reprodução dos clusters de generalização externa. |
| 7. Escolhas de modelos | 0:40 | Mostrar que as cinco abordagens testam hipóteses diferentes; cada uma teve seleção de parâmetros apropriada. |
| 8. Resultados nos dois alvos | 1:00 | Voto e Parzen quase empatam no original; Logística lidera pontualmente ao reproduzir clusters. Não comparar F1 entre tarefas como se fossem equivalentes. |
| 9. Comparações estatísticas | 0:50 | Friedman detecta diferenças globais, mas Nemenyi não diferencia Voto/Parzen nem Logística/Voto no F1; ICs/testes são exploratórios. |
| 10. Curvas de aprendizagem | 0:45 | Destacar como o desempenho varia com a quantidade de dados e a limitação de testes menores nos extremos. |
| 11. Conclusões | 0:25 | Reforçar representação, rigor do protocolo e dependência do alvo. |

## Perguntas que vale preparar

- **Por que não escolhemos a maior Silhueta global entre diferentes pré-processamentos?** Porque Silhuetas em espaços transformados diferentes não são diretamente comparáveis. Em cada representação, a escolha de (K,H) seguiu a Silhueta; entre representações, consideramos propriedades dos atributos e o ARI a posteriori. Isso significa que a escolha de representação não foi inteiramente não supervisionada.
- **Por que o agrupamento tem três classes e qual é a qualidade dele?** O par (3,3) venceu na grade da representação log1p + MinMax, com ARI 0,3911 frente à classe original: concordância parcial, não equivalência semântica.
- **Por que usar F1 macro?** Para dar peso equivalente às classes, incluindo o cluster minoritário de 265 e-mails.
- **A Logística é estatisticamente superior ao Voto no alvo de clusters?** Não foi detectada diferença no Nemenyi exploratório de F1: p=0,1024, embora a média pontual de F1 da Logística seja maior.
- **O que significa avaliar rótulos gerados no conjunto completo?** Que se mede reprodução supervisionada de uma partição transdutiva fixa, não desempenho de descoberta de clusters inéditos.
- **Onde estão os gráficos/matrizes exigidos pelo enunciado?** Estão no relatório final e na apresentação técnica completa já produzida. Na versão curta, o foco são as decisões e os resultados principais.

**Fonte dos números:** `entrega/relatorio_final_projeto_am_2026_2.md` e saídas experimentais do projeto anexado. O arquivo `.tex` acompanha os gráficos originais e permanece editável.
