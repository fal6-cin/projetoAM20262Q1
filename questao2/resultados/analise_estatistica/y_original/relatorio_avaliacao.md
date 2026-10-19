# Relatório y_original — 30×10-folds

**ESTUDO 30×10 COMPLETO**, com ICs descritivos aproximados e testes estatísticos exploratórios.

As métricas são médias macro de precisão, cobertura e F-measure; taxa de erro = 1−acurácia.
A unidade resumida é a repetição (média dos 10 folds), não 300 folds tratados como amostras independentes.

| Classificador | Taxa de erro | Precisão macro | Cobertura macro | F1 macro |
|---|---:|---:|---:|---:|
| gaussiano | 0.1516 | 0.8525 | 0.8683 | 0.8471 |
| knn | 0.0674 | 0.9321 | 0.9266 | 0.9290 |
| parzen | 0.0575 | 0.9422 | 0.9372 | 0.9394 |
| logistica | 0.0616 | 0.9357 | 0.9359 | 0.9355 |
| voto | 0.0571 | 0.9439 | 0.9364 | 0.9397 |

**Limitações estatísticas:** as repetições do CV compartilham os mesmos 4.601 exemplos; portanto,
os ICs t calculados sobre as médias de repetição são aproximações descritivas, não intervalos
com garantia rigorosa de cobertura para novas amostras independentes. Friedman e Nemenyi
utilizam 30 repetições como blocos pareados, mas os blocos não são plenamente independentes.
O voto é derivado dos outros modelos, e essa dependência também deve constar da interpretação.
Não interpretar a classificação dos rótulos Double K-means como validação externa independente dos clusters.

## Friedman e Nemenyi (exploratórios)

- **taxa_erro**: Friedman χ²=114.5175, p=7.9104e-24; diferença crítica de ranks Nemenyi=1.1136.
- **precisao_macro**: Friedman χ²=115.8933, p=4.0229e-24; diferença crítica de ranks Nemenyi=1.1136.
- **cobertura_macro**: Friedman χ²=98.2400, p=2.3306e-20; diferença crítica de ranks Nemenyi=1.1136.
- **f_measure_macro**: Friedman χ²=114.2400, p=9.0662e-24; diferença crítica de ranks Nemenyi=1.1136.

O arquivo `nemenyi_pares.csv` contém os p-valores por par e cada métrica.
