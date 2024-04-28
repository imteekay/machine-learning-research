# ML in Healthcare

## Outline to curso

- Capacidade preditiva: acurácia das decisões
- Intro to R and Python
- Machine Learning models
- Sobreajustes
- Pré-processamento de dados
- Regressão e classificação
- Medição de performance
- Importância preditiva das variáveis
- Regressões penalizadas
- Mínimos quadrados parciais
- Support Vector Machines (SVM)
- Redes neurais
- MARS
- Árvores de decisão, random forests e gradient boosted trees
- Deep Learning

## Inversão da regra de regressão

```
   Yi     = Bo + B1 * X1i + B2 * X2i
Glicemia           IMC       Dieta
```

Como o IMC e Dieta influenciam em gerar problemas de Glicemia

## Basic Resources

- [Applied Predictive Modeling](https://www.goodreads.com/en/book/show/17299542)
- [An Introduction to Statistical Learning: with Applications in Python](https://www.goodreads.com/book/show/178815107-an-introduction-to-statistical-learning)

## Modelos de machine learning; pré-processamento de variáveis preditoras

- **Aprendizado supervisionado**: quando os dados incluídos para treinar o algoritmo incluem a solução desejada, o rótulo (label)
  - e.g. Dado variáveis de X1 e X2, quanto maior as duas variáveis, maior a probabilidade de câncer ("ter câncer" ou "não ter câncer" é o rótulo/label nesse caso)
- **Aprendizado não supervisionado**: não existe label, o algoritmo aprende sem uma resposta certa
  - e.g. algoritmos: clustering, redução de dimensão
  - e.g. dados sobre pessoas com doenças cardio vascular e utilizamos clustering para agrupar diferentes tipos de pessoas com essa doença, podendo achar um grupo X (e.g. grupo que tem como causa a genética), um grupo Y (e.g. grupo que tem como causa a obesidade e sedentarismo), Z (e.g. grupo de pessoas idosas)
- **Aprendizado semi-supervisionado**: presença de alguns dados com rótulo
  - e.g. identificação de fotos. o algoritmo precisaria apenas de apenas um rótulo
- **Aprendizado por reforço**: interação com ambiente dinâmico com feedback (positivo ou premiaçoes e negativo ou punições)
  - e.g. jogo Go
- Modelos preditivos
  - Objetivo: desenvolver algoritmos que fazem boas predições em saúde
  - Exemplos:
    - Classificação: quando a variável a ser predita é qualitativa
    - Regressão: quando a variável a ser predita é quantitativa
  - Performance preditiva ruim:
    - Pré-processamento inadequado
    - Validação inadequada de algoritmos: não tomar uma boa decisão de qual o melhor algoritmo
    - Extrapolação inadequada: treinar o algoritmo com determinados dados mas testa em uma outra população
    - Sobreajuste
- Predição x Interpretação/Inferência
  - Predição: performance preditiva
    - e.g. dados as características de uma pessoa, essa pessoa terá câncer de pulmão daqui a X anos?
  - Interpretação/Inferência: relação entre variáveis
    - e.g. fumar tem relação com câncer de pulmão
- Antes de começar a modelar com Machine Learning
  - Qual o problema a ser resolvido? É um problema de inferência ou de predição?
- Pré-processamento de dados
  - Tratar problemas de outliers, correlações aleatórias e erros de medida: chave para boa performance preditiva
  - Fatores importantes em pré-processamento de dados
    - seleção das variáveis
      - não selecionar todas as variáveis
      - selecionar variáveis que sejam plausíveis. e.g. ter hipertensão correlacionado com a variável de "ter 5 camisetas amarelas". Essa variável pode ser discartada
    - vazamento de dados / data leakage
      - dados "escondidos" faz com que o modelo aprenda padrões desinteressantes
    - padronização de dados
      - alta escala podem afetar qualidade de predição
    - redução de dimensão
    - colinearidade
    - valores missing
    - one-hot encoding

## Sobreajuste

- É o principal problema de ML
- Não é generalizável: funciona muito bem para a amostra atual mas não para dados futuros - influenciado por fatores aleatórios e erros de medidas
- tradeoff entre viés e variância
  - Variação alta (overfit): modela perfeitamente todos os dados, muito complexo, mas não entendeu o padrão dos dados
  - Viés alto (underfit): não tem um bom fit com os dados, bem simplista, não entendeu o padrão dos dados
- Como avaliar se o modelo está com sobreajuste?
  - Testando com novos dados
  - Com dados que o algoritmo nunca viu e a performance do modelo cair, então o modelo tem sobreajuste
  - e.g. modelo treinado com dados de 2023 e testado com dados de 2024
  - e.g. separar dados aleatoriamente
    - Dados para treino: 70-80% para definir o modelo
    - Dados para teste: 20-30% para analisar a performance preditiva
    - Garantir que tanto os dados de teste quanto os dados de treino tenham distribuição parecidas
      - e.g. 30% dos dados tem pessoas com hipertensão tanto para treino quanto para teste
- Utilizar parâmetros e hiperparametros para segurar ou impedir que o modelo fique muito complexo
  - Errar mais no treino e acertar mais no teste (sempre focado nos dados do futuro)

## Algoritmos de Machine Learning

- Como cada algoritmo procura padrões nos dados
- Tomadas de decisão para utilização de algoritmos
  - Inferência ou predição? procurar relações nos dados/variáveis ou predizer o que pode acontecer no futuro?
  - Qual o tipo de problema? supervisionado, não-supervisionado, semi-supervisionado, por reforço?
  - Problema de classificação ou de regressão?
    - Classificação: desfecho a ser predito são categorias
    - Regressão: desfecho é contínuo
  - Selecionar variáveis plausíveis
  - Evitar vazamento de dados
  - Padronização de variáveis contínuas
  - Excluir variáveis correlacionais - para deixar o modelo mais estável
  - Utilização de técnica de redução de dimensão (e.g. PCA)
  - O que fazer com valores missing?
  - Utilização de one-hot encoding para preditores categoricos?
  - Divisão da amostra entre treino e teste (terceira divisão para testar os hiperparâmetros)
  - Quais hiperparâmetros olhar?
  - Utilização de validação cruzadas? 10-folds?
  - Quais algoritmos testar?
- Regressões
  - Comuns em estudo de inferência: relação entre variáveis
  - Mas também pode ser usado em predição
  - Regressão Linear (desfecho contínuo - resultado pode assumir qualquer valor infinito):
    - Usado para predizer valores numéricos: preços de casas, preços de stock option, temperatura
  - Regressão Logística (desfecho categórico - resultado "binário" assume um valor dentro de um conjunto de classes/categories):
    - Usado para classificar e resultar em valores binários: spam, risco de crédito, diagnóstico médico
