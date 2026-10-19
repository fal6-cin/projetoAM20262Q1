# Modelos da Questão 2

- `bayes_gaussiano.py`: **IMPLEMENTADO** — normal multivariada com covariância completa ML e regularização diagonal para singularidade; seleção por 5-folds internos.
- `bayes_knn.py`: **IMPLEMENTADO** — probabilidades pelas frações dos vizinhos; distâncias Euclidiana, City-Block e Chebyshev, escolha por 5-folds internos.
- `bayes_parzen.py`: **IMPLEMENTADO** — kernel gaussiano multivariado produto, log-sum-exp e seleção da janela h por 5-folds internos.
- `regressao_logistica.py`: **IMPLEMENTADO** — logística binária/multiclasse com L2, seleção de C e ponderação de classe por 5-folds internos.
- `voto_majoritario.py`: **IMPLEMENTADO** — quatro votos não ponderados; empates pela maior priori do **treino externo**. Sem ajuste de hiperparâmetros do voto. Integração em `../avaliacao_voto.py` sobre predições **externas** alinhadas dos quatro modelos.

Não usar `GaussianNB` em lugar do gaussiano multivariado exigido.
