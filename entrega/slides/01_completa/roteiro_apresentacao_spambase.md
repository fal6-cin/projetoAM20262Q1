# Roteiro de apresentação — Projeto de Aprendizagem de Máquina (IN1102)

**Projeto:** Spambase — Double K-means e classificação supervisionada  
**Equipe:** Douglas Araújo Silva, Felipe Araújo de Lima e Letícia De Albuquerque S. Leitão  
**Data:** 18/11/2026  
**Duração exigida pelo enunciado:** 10 minutos por equipe, **incluindo discussão**.

## Como usar as versões

- **Apresentação completa — 37 slides:** 23 slides principais e 14 slides de apoio técnico. Abrange o enunciado e permite consultar matrizes, ICs de 95%, tabelas completas de Nemenyi e curvas individuais durante perguntas.
- **Apresentação principal — 23 slides:** concentra o percurso do problema às conclusões. Ainda é bastante densa; é importante ensaiar e avançar rapidamente por slides metodológicos, em vez de ler tabelas célula por célula.

## Roteiro sugerido — aproximadamente 7min30 de fala + 2min30 de discussão

| Tempo | Slides | Quem pode apresentar | Mensagem principal |
|---|---|---|---|
| 0:00–0:25 | 1–2 | Douglas | Base Spambase, 57 atributos e as duas questões. |
| 0:25–1:15 | 3–4 | Douglas | Double K-means, 9 pares, 100 reinícios válidos e seleção por menor W e maior Silhueta. |
| 1:15–1:45 | 5–6 | Douglas | Paradoxo da Silhueta; escolha da representação log_minmax com ressalva de uso exploratório do ARI. |
| 1:45–2:55 | 7–10 | Felipe | Mostrar Silhueta x (K,H), matriz G, contingência/ARI e evolução de W. |
| 2:55–3:20 | 11 | Felipe | Dois alvos e separação do pré-processamento nos folds. |
| 3:20–4:10 | 12–15 | Felipe | Cinco classificadores, 30 x 10-folds, CV interna de 5 folds, ajuste e métricas. |
| 4:10–5:05 | 16–17 | Letícia | Voto/Parzen lideram no alvo original; Logística lidera na reprodução dos clusters. |
| 5:05–6:00 | 18–19 | Letícia | Friedman rejeita nas oito comparações; Nemenyi diferencia pares selecionados com CD = 1,1136. |
| 6:00–6:55 | 20–21 | Letícia | Comparar treino e teste em 19 percentuais de 5% a 95%; destacar lacunas/variabilidade. |
| 6:55–7:30 | 22–23 | Toda a equipe | Resumo, limitações e abertura para perguntas. |
| 7:30–10:00 | Apoio se perguntarem | Toda a equipe | Responder com tabelas completas, figuras e detalhes do protocolo. |

**Observação:** os tempos são uma sugestão de ensaio, não uma medição da duração real. Para a apresentação rigorosa de 10 minutos, o grupo deve cronometrar a passagem real e cortar falas se necessário. Se o tempo estiver curto, os slides 6, 14 e 15 são os primeiros candidatos a explicação de uma frase, pois as informações completas permanecem no material entregue.

## Pontos de atenção nas respostas ao professor

1. **Seleção correta em Q1:** o melhor reinício de cada par é selecionado pelo menor W; depois o par $(K^*,H^*)$ é definido pela maior Silhueta na representação escolhida. O ARI foi calculado a posteriori e também informado na escolha *entre* representações; esse último passo não foi cego aos rótulos.
2. **Interpretação do ARI:** ARI = 0,391111 denota concordância parcial, não reprodução perfeita das duas classes originais.
3. **Gaussiano multivariado:** a covariância de máxima verossimilhança foi regularizada para uso numérico; não se deve afirmar covariância ML pura na decisão.
4. **Validação:** há 300 folds externos por combinação alvo/classificador, tuning estratificado de 5 folds nos nove folds de treino externo, e pré-processamento sem consulta ao fold de teste.
5. **ICs/Friedman/Nemenyi:** resultados descritivos e exploratórios; 30 repetições compartilham as mesmas 4.601 observações.
6. **Clusters como alvo supervisionado:** a tarefa mede fidelidade à partição descoberta sobre a base completa, e não prova generalização para e-mails inéditos.
7. **Curvas de aprendizagem:** foram usadas amostras estratificadas separadas por proporção; 95% de treino deixa somente 231 instâncias de teste e aumenta a variabilidade dos resultados.

## Onde encontrar as evidências

- Slides 7–10: Silhueta, matriz G, contingência e W.
- Slides 16–19: quatro métricas, Friedman e Nemenyi (resumo).
- Slides 20–21: treino/teste para os cinco classificadores em ambos os alvos.
- Slides 24–26: grade da Q1 e diagnóstico da linha de base.
- Slides 27–28: estimativas pontuais com ICs de 95%.
- Slides 29–32 e 36–37: ranks e comparações de Nemenyi em todas as métricas.
- Slides 33–34: exemplos individuais de treino/teste.
