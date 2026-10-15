# Questão 1 revisada

Abra questao1_double_kmeans_revisada.ipynb nesta pasta e execute em ordem. Bibliotecas: numpy, pandas, matplotlib e scikit-learn. A primeira célula instala dependências se necessário.

O notebook e os resultados originais permanecem em ../original_zip/projeto1_am. resultados_originais_zip é uma cópia das saídas fornecidas; os novos resultados ficam em resultados_questao1_revisada.

REUTILIZAR_ZIP=True reconstrói as nove melhores partições usando as seeds salvas, conferindo W, Silhouette, dados e rótulos vencedores. False repete o grid de 900 execuções válidas. Sensibilidade acrescenta 1800 execuções válidas e não troca os rótulos oficiais.

Auditoria exporta tamanhos, Silhouette individual e por cluster, detalhes dos objetos do grupo pequeno, ranking e figuras. Sensibilidade exporta grids, partições completas NPZ, rótulos diagnósticos, contingências, convergência, comparação de vencedores e estabilidade via ARI. Os IDs seguem a ordem do spambase.data, sem reordenar linhas.
