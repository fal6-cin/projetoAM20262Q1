# Estado da Questão 2 — Etapas 3 e 4 concluídas

## Experimentos concluídos

- **Origem dos rótulos:** Questão 1 com Log1p + MinMaxScaler, K*=3, H*=3. O CSV de entrada da Questão 2 conserva os 57 preditores em escala original.
- **Etapa 3:** cinco classificadores, duas variáveis resposta, 30 repetições × 10 folds estratificados externos por classificador/alvo; **3.000 avaliações externas** e **1.380.300 predições**.
- **Auditoria dos resultados:** folds, índices das amostras, rótulos verdadeiros, métricas por fold e alinhamento das predições foram verificados pela rotina de auditoria do projeto.
- **Estatísticas:** quatro métricas, resumos por repetição, intervalos de confiança descritivos aproximados de 95%, Friedman e Nemenyi exploratórios, disponíveis em `resultados/analise_estatistica/`.
- **Etapa 4:** 19 proporções (5% a 95%) × 2 alvos × 5 modelos = **190 resultados de modelos**, provenientes de **38 divisões estratificadas**; estão presentes os CSVs, os **14 gráficos** e o relatório de curvas de aprendizagem.
- **Testes automatizados:** **62 testes aprovados**.

As inferências estatísticas devem ser lidas com cautela: as 30 repetições reutilizam as mesmas 4.601 observações; os rótulos dos clusters foram gerados usando a base completa e sua classificação mede a reprodução dessa partição.

## Documentos e resultados finais

- `RELATORIO_FINAL_QUESTAO2.md`: documento consolidado dos métodos, das métricas e das conclusões.
- `AUDITORIA_30x10.md`: conferência do estudo e resultados estatísticos.
- `RELATORIO_CURVAS_APRENDIZAGEM.md`: execução, resultados e interpretação das curvas.
- `resultados/analise_estatistica/` e `resultados/curvas_aprendizagem/`: tabelas e gráficos.

## Conferência sem refazer os treinamentos

Na raiz do repositório:

```bash
python -m unittest discover -s questao2/tests -v
python -m questao2.scripts.status_estudo
python -m questao2.scripts.validar_base
python -m questao2.scripts.rodar_curvas --repeticoes 1 --apenas-consolidar
```

Opcionalmente, `python -m questao2.scripts.analisar_resultados --repeticoes 30` revalida as predições gravadas e regenera a análise estatística, **sem retreinar** os classificadores.
