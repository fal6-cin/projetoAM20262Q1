Pessoal, atualizei o notebook da Questão 1 com as auditorias para entender melhor o resultado.

A solução original continua sendo K=2 e H=2, com Silhouette de 0,6596, mas ela separou 34 objetos dos outros 4567. Os 34 são da classe non-spam e têm frequências bem altas em palavras/códigos como 857, 415, direct, telnet e technology. O ARI ficou perto de zero, então essa divisão não corresponde à separação spam/non-spam.

O grupo pequeno não aparece só na solução vencedora: oito dos nove pares têm um cluster com menos de 1% da base. Por isso, precisamos explicar o que o algoritmo encontrou antes de usar os rótulos na Questão 2.

Também executei a comparação dos três pré-processamentos, mantendo 100 execuções válidas por par. Nos dados originais, o vencedor foi (2,1), com grupos de 238 e 4363 objetos. Com log1p+StandardScaler, foi (2,2), com 176 e 4425. Ou seja, o pré-processamento muda a partição, mas as três versões continuam desbalanceadas e com ARI perto de zero. A Silhouette maior dos dados originais não basta para escolher essa versão, porque ela foi calculada em outra escala.

Próximos passos:

1. Olhar as tabelas de tamanho e Silhouette por cluster, os 34 objetos e o ranking das variáveis.
2. Conferir a comparação entre dados originais, StandardScaler e log1p+StandardScaler. A ideia é entender quanto a solução depende do pré-processamento, sem escolher só o resultado com aparência melhor.
3. Juntar esses diagnósticos com a resposta do professor sobre a padronização e registrar a decisão no relatório.
4. Depois disso, confirmar qual arquivo de rótulos vamos usar na Questão 2. Por enquanto, o CSV principal mantém os rótulos originais do experimento; as outras versões são diagnósticos.
5. Na Questão 2, ajustar qualquer padronização só nos dados de treino de cada divisão da validação cruzada.
