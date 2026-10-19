# Estado da Questão 2 — Etapa 3 concluída

- Base e rótulos preservados: `Log1p + MinMaxScaler`, K*=3, H*=3, 4.601 observações, 57 atributos.
- Os cinco modelos, nos dois alvos, concluíram **30 repetições × 10 folds externos**.
- **3.000 folds externos** e **1.380.300 predições** revalidados em auditoria independente da sessão de execução.
- Estão presentes as 4 métricas, resumos por repetição, IC95 descritivos aproximados, Friedman e Nemenyi exploratórios.
- O auditor do projeto recalculou métricas e verificou a correspondência de predições e índices de teste: aprovado.
- Os **56 testes unitários passam**.
- As inferências estatísticas exigem cautela, pois as 30 repetições de CV reutilizam os mesmos e-mails.
- **A Etapa 4 foi implementada:** rotina para curvas F1 macro de treino/teste, de 5% a 95%, com saída de gráficos.

Detalhes e resultados: `AUDITORIA_30x10.md` e `resultados/analise_estatistica/`.

## Conferência

Na raiz do repositório:

```bash
python -m unittest discover -s questao2/tests -v
python -m questao2.scripts.status_estudo
python -m questao2.scripts.analisar_resultados --repeticoes 30
```

O último comando apenas revalida as predições já armazenadas e regenera a análise estatística; **não retreina** os classificadores.
