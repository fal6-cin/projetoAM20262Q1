# Relatório y_double_kmeans — 30×10-folds

**ESTUDO 30×10 COMPLETO**, com ICs descritivos aproximados e testes estatísticos exploratórios.

As métricas são médias macro de precisão, cobertura e F-measure; taxa de erro = 1−acurácia.
A unidade resumida é a repetição (média dos 10 folds), não 300 folds tratados como amostras independentes.

| Classificador | Taxa de erro | Precisão macro | Cobertura macro | F1 macro |
|---|---:|---:|---:|---:|
| gaussiano | 0.1699 | 0.7624 | 0.8774 | 0.7976 |
| knn | 0.0655 | 0.9392 | 0.8903 | 0.9114 |
| parzen | 0.0566 | 0.9521 | 0.9000 | 0.9224 |
| logistica | 0.0095 | 0.9840 | 0.9819 | 0.9827 |
| voto | 0.0464 | 0.9612 | 0.9141 | 0.9347 |

**Limitações estatísticas:** as repetições do CV compartilham os mesmos 4.601 exemplos; portanto,
os ICs t calculados sobre as médias de repetição são aproximações descritivas, não intervalos
com garantia rigorosa de cobertura para novas amostras independentes. Friedman e Nemenyi
utilizam 30 repetições como blocos pareados, mas os blocos não são plenamente independentes.
O voto é derivado dos outros modelos, e essa dependência também deve constar da interpretação.
Não interpretar a classificação dos rótulos Double K-means como validação externa independente dos clusters.

## Friedman e Nemenyi (exploratórios)

- **taxa_erro**: Friedman χ²=120.0000, p=5.3415e-25; diferença crítica de ranks Nemenyi=1.1136.
- **precisao_macro**: Friedman χ²=120.0000, p=5.3415e-25; diferença crítica de ranks Nemenyi=1.1136.
- **cobertura_macro**: Friedman χ²=119.2267, p=7.8132e-25; diferença crítica de ranks Nemenyi=1.1136.
- **f_measure_macro**: Friedman χ²=120.0000, p=5.3415e-25; diferença crítica de ranks Nemenyi=1.1136.

O arquivo `nemenyi_pares.csv` contém os p-valores por par e cada métrica.
