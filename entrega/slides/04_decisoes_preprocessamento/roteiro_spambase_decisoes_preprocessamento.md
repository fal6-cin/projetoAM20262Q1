# Roteiro de apresentação — Spambase: decisões que mudaram os resultados

**Objetivo:** explicar as decisões técnicas adotadas e justificá-las com evidências do projeto, incluindo os problemas do Double K-means na base Spambase e a necessidade de pré-processamento.

**Tempo sugerido:** ~8 min de apresentação e ~2 min para perguntas, totalizando os 10 minutos do enunciado. O tempo deve ser ensaiado pela equipe e todos os integrantes devem participar.

| Slide | Tempo | Mensagem principal |
|---|---:|---|
| 1. Capa | 0:15 | Apresentaremos decisões, evidências e consequências das duas questões. |
| 2. Agenda | 0:20 | Divisão em duas partes: Q1 (Double K-means e pré-processamento) e Q2 (classificação e análise estatística). |
| 3. Três perguntas | 0:25 | O desafio combina representação, avaliação controlada e interpretação. |
| 4. Problemas do Double K-means na base | 0:40 | Escalas muito diferentes, esparsidade (>80% zeros) e caudas longas levam a perda quadrática $W$ (com média única $g_{kh}$ por bloco) a isolar outliers. |
| 5. Diagnóstico da Silhueta | 0:40 | O z-score deu Silhueta 0,6596, mas ARI quase zero: microcluster de 34 e-mails definido por termos técnicos raros. |
| 6. Necessidade de pré-processamento | 0:35 | Z-score puro ou log só nas maiúsculas mantêm o microcluster de 34 e-mails. É preciso comprimir caudas (`log1p`) e equalizar escala (`MinMax`) nas 57 variáveis. |
| 7. Comparação das 7 estratégias | 0:40 | Tabela comparativa das 7 representações (6.300 execuções válidas): $(K^*,H^*)$, Silhueta, ARI e tamanhos dos grupos encontrados. |
| 8. Representação escolhida | 0:40 | `log1p + MinMax` preserva os 57 atributos e gerou (K,H)=(3,3). O ARI informou a escolha *entre representações*, não o algoritmo em si. |
| 9. O que os clusters mostram | 0:35 | Descrever os três grupos e a concordância parcial com as classes conhecidas (ARI 0,3911). |
| 10. Validação aninhada | 0:40 | Justificar 30 × 10 folds externos, ajuste 5-folds interno, pipeline e métricas macro. Diferenciar reprodução dos clusters de generalização externa. |
| 11. Escolhas de modelos | 0:35 | Mostrar que as cinco abordagens testam hipóteses diferentes; cada uma teve seleção de parâmetros apropriada. |
| 12. Resultados nos dois alvos | 0:45 | Voto e Parzen quase empatam no original; Logística lidera pontualmente ao reproduzir clusters. Não comparar F1 entre tarefas como se fossem equivalentes. |
| 13. Comparações estatísticas | 0:35 | Friedman detecta diferenças globais, mas Nemenyi não diferencia Voto/Parzen nem Logística/Voto no F1; ICs/testes são exploratórios. |
| 14. Curvas de aprendizagem | 0:35 | Destacar como o desempenho varia com a quantidade de dados e a limitação de testes menores nos extremos. |
| 15. Conclusões | 0:25 | Reforçar representação, rigor do protocolo e dependência do alvo. |
| 16. Obrigado! | 0:10 | Abertura para perguntas, link do repositório, contatos da equipe e 4 referências principais. |

## Perguntas que vale preparar

- **Por que o Double K-means tem dificuldade com a base Spambase sem pré-processamento específico?** Porque o método resume cada bloco de linhas e colunas por uma única média $g_{kh}$ sob perda quadrática. Como a base mistura contagens de maiúsculas (até 15.841) com frequências percentuais muito esparsas (>80% zeros) e picos raros (vocabulário da HP Labs), os desvios extremos dominam $W$ e forçam a criação de microclusters de outliers.
- **Por que não escolhemos a maior Silhueta global entre diferentes pré-processamentos?** Porque Silhuetas em espaços transformados diferentes não são diretamente comparáveis. Em cada representação, a escolha de (K,H) seguiu a Silhueta; entre representações, consideramos propriedades dos atributos e o ARI a posteriori. Isso significa que a escolha de representação não foi inteiramente não supervisionada.
- **Por que o agrupamento tem três classes e qual é a qualidade dele?** O par (3,3) venceu na grade da representação `log1p + MinMax`, com ARI 0,3911 frente à classe original: concordância parcial, não equivalência semântica.
- **Por que usar F1 macro?** Para dar peso equivalente às classes, incluindo o cluster minoritário de 265 e-mails.
- **A Logística é estatisticamente superior ao Voto no alvo de clusters?** Não foi detectada diferença no Nemenyi exploratório de F1: p=0,1024, embora a média pontual de F1 da Logística seja maior.
- **O que significa avaliar rótulos gerados no conjunto completo?** Que se mede reprodução supervisionada de uma partição transdutiva fixa, não desempenho de descoberta de clusters inéditos.

**Fonte dos números:** `entrega/relatorio_final_projeto_am_2026_2.md` e saídas experimentais do projeto anexado. O arquivo `.tex` acompanha os gráficos originais e permanece editável.
