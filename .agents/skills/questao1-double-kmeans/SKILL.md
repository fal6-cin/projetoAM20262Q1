---
name: questao1-double-kmeans
description: >-
  Executa, audita e valida o protocolo experimental da Questão 1 do Projeto AM (Double K-means no Spambase).
  Use esta skill quando for necessário executar ou revisar o agrupamento co-clustering do Spambase,
  avaliar pares (K, H), calcular Silhueta e ARI, inspecionar a matriz de protótipos G e a matriz de contingência,
  ou preparar os datasets de entrada para a Questão 2.
---

# Protocolo Experimental: Questão 1 (Double K-means no Spambase)

Este guia padroniza a execução, avaliação e auditoria da Questão 1 da disciplina de Aprendizagem de Máquina (CIn/UFPE), baseada no co-clustering do dataset Spambase via Double K-means.

## 1. Estrutura do Problema e Dados
* **Dataset:** Spambase (UCI ML Repository), 4.601 observações e 57 atributos numéricos contínuos (54 frequências percentuais e 3 contagens de letras maiúsculas).
* **Rótulos a priori:** Coluna binária final (0 = non-spam, 1 = spam). **Não participa** do agrupamento nem do cálculo de protótipos.

## 2. Grade Experimental Obrigatória
* **Clusters de objetos:** $K \in \{2, 3, 4\}$.
* **Clusters de atributos:** $H \in \{1, \ldots, K\}$ para cada $K$.
* **Total de pares avaliados:** 9 pares:
  $$(2,1), (2,2), (3,1), (3,2), (3,3), (4,1), (4,2), (4,3), (4,4)$$
* **Reinicializações:** Exatamente **100 execuções válidas** por par $(K,H)$ a partir de sementes aleatórias reprodutíveis. Se ocorrer cluster vazio durante uma iteração, descartar a tentativa e rodar uma nova semente.

## 3. Protocolo Estrito de Seleção e Avaliação
A ordem das operações é fixa e não pode ser invertida:
1. **Otimização:** Em cada par $(K,H)$, selecionar a execução que obteve a **menor função objetivo** $W(G, U, V)$ dentre as 100 tentativas válidas:
   $$W(G, U, V) = \sum_{k=1}^K \sum_{h=1}^H \sum_{i=1}^N \sum_{j=1}^P u_{ik} v_{jh} (x_{ij} - g_{kh})^2$$
2. **Cálculo da Silhueta:** Calcular o escore médio de Silhueta Euclidiana da partição de objetos **somente sobre a melhor execução** selecionada no passo 1.
3. **Seleção de $(K^*, H^*)$:**
   $$K^* = \arg\max_{(K,H)} \text{Sil}((K, H))$$
   (Guardar também o $H^*$ correspondente da solução vencedora).
4. **Avaliação Externa com ARI:**
   Calcular o Adjusted Rand Index (ARI) comparando os rótulos preditos $y_{double\_kmeans}$ ($K^*$ grupos) com os rótulos reais $y_{original}$ (spam/non-spam).

## 4. Artefatos de Entrega Obrigatórios
Para a partição vencedora $(K^*, H^*)$, gerar:
* **Matriz de protótipos dos blocos ($G$):** Tabela de dimensão $K^* \times H^*$, onde $g_{kh}$ é a média do bloco.
* **Matriz de contingência / confusão:** Tabela cruzada entre a partição de objetos encontrada e a classe a priori.
* **Curva de convergência:** Gráfico da função objetivo $W$ versus número de iterações (verificando monotonicidade não crescente).
* **Gráfico de Silhuetas:** Gráfico de barras $\text{Sil} \times (K, H)$ para os 9 pares.

## 5. Diretrizes Metodológicas de Pré-processamento
* **StandardScaler (z-score clássico):** Mantém a conformidade com o roteiro inicial do projeto, mas requer documentação crítica no relatório:
  * A assimetria das frequências percentuais e a presença de outliers geram z-scores de até $+50$, induzindo o Double K-means a isolar microclusters atípicos (ex.: 34 objetos da HP Labs) com Silhueta inflada e ARI próximo a zero.
* **Análise de Sensibilidade Recomendada:** Comparar com pré-processamentos robustos como $\log(1+x) + \text{StandardScaler}$, binarização ($x > 0$) ou `MinMaxScaler`, demonstrando controle teórico sobre a sensibilidade do K-means a caudas pesadas.

## 6. Transição para a Questão 2 (Prevenção de Data Leakage)
* O arquivo compartilhado para a Questão 2 (`spambase_com_clusters.csv`) deve conter:
  1. Os 57 atributos numéricos **na escala original** (sem padronização prévia).
  2. A coluna `y_original` (2 classes a priori).
  3. A coluna `y_double_kmeans` ($K^*$ classes encontradas).
* Qualquer transformação (z-score, log, etc.) necessária aos classificadores supervisionados da Questão 2 deve ser ajustada **estritamente dentro dos dados de treinamento de cada fold da validação cruzada 30x10**.
