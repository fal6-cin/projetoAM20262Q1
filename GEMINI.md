# Diretrizes do Projeto de Aprendizagem de Máquina (AM 2026-2)

## 1. Integridade da Questão 1 (Double K-means)
* **Não supervisionamento estrito:** A coluna de classe original (`y_original`) não deve participar de nenhuma etapa de pré-processamento, cálculo de médias, atualização de partições ou critério de parada do Double K-means.
* **Invariante de Seleção:** Para cada par $(K, H)$, selecione sempre a execução de menor função objetivo $W$ antes de computar a Silhueta. Nunca selecione a execução diretamente pela Silhueta entre os restarts.
* **Reprodutibilidade:** Toda execução estocástica deve utilizar sementes determinísticas (`seed = base_seed + 10000*K + 1000*H + restart_idx`) registradas em metadados.

## 2. Transição Segura para a Questão 2 (Sem Data Leakage)
* **Escala Original nos Datasets Compartilhados:** Arquivos de dados gerados na Questão 1 para consumo na Questão 2 (como `spambase_com_clusters.csv`) devem manter os 57 atributos originais sem pré-processamento global.
* **Isolamento de Fold:** Na Questão 2, qualquer escalonamento (StandardScaler, normalização, etc.) e ajuste de hiperparâmetros por validação cruzada aninhada (nested CV) deve ser feito exclusivamente no conjunto de treino de cada partição, aplicando apenas `transform()` no teste.
* **Proibição de Vazamento de Rótulos:** Ao avaliar os classificadores predizendo `y_double_kmeans`, nunca utilize `y_original` como preditor, e vice-versa.
