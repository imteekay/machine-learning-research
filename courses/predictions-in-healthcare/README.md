# Machine Learning para Predições em Saúde

## Pre-processamento

### Motivos para má performance de algoritmos

- Extrapolação inadequada dos resultados
  - Desenvolver algoritmos para uma população e esperar que funcionam bem para outra população
  - e.g. aplicar algoritmos que funcionam bem para o Brasil e funcione bem na Europa; caractersticas genéticas e socieconômicas muito diferentes
  - e.g. períodos diferentes - doenças sazonais
- Pré-processamento inadequada dos dados
- Sobreajuste
- Validação inadequada da quantidade dos algoritmos

### Pré-processamento inadequada dos dados

- Selecionar variáveis
  - Escolha de variáveis plausíveis e diretamente ligado ao desfecho
  - Não precisa ser uma variável que cause o desfecho mas precisa estar diretamente ligado
    - e.g. predizer se uma pessoa vai a óbito daqui a 1 ano. variável de ter estado na UTI é muito relevante mas não é causadora de ir a óbito. Ela está associada, relacionada.
  - Entrar em contato com os melhores especialistas na área: as variáveis ajudam no desfecho?
- Vazamento de dados
  - Quando os dados de treino apresentam informação escondida que faz com que o modelo aprenda padrões que não são do seu interesse
  - Não é uma variável que está predizendo o desfecho, é o desfecho que está predizendo a variável
    - e.g. incidente de hipertensão no próximo ano: variável de tomar medicação anti-hipertensivo. O paciente já tinha hipertensão, só não estava informado no pontuário
    - e.g. incluir número identificador do paciente como variável preditora; caso pacientes de hospitais de câncer tiverem números semelhantes, o algoritmo irá predizer que paciente no intervalo de identificadores X->Y tem câncer
- Padronização
  - Utilizar score Z: Padronização de variáveis para todas terem média de 0 e desvio padrão de 1
  - Colocar todas as variáveis na mesma escala
- Redução de dimensão
  - Reduz o número de variáveis utilizando a técnica de Análise de Componentes Principais (PCA)
  - Encontrar combinação lineares das variáveis preditoras, assim reduzindo variáveis
- Colinearidade
  - Variáveis colineares trazem informações redundantes (perda de tempo)
  - Aumento da instabilidade dos modelos
  - Se tiver uma correlação muito alta, por exemplo, limite de correlação acima de 0.75, tiramos uma das duas
- Valores missing
  - Pelo fato de ter uma variável faltante, é importante para a predição
  - e.g. pessoas acamadas não é possível medir a altura (entre outras medições), e ter variável missing influencia na predição
  - É importante entender por que valores de uma variável está faltante
- One-hot coding
  - Algoritmos tem dificuldade de entender variáveis que tem mais de uma categoria
  - e.g. Sul, Norte, Nordeste, Sudeste, e Centro -> transformar cada valor em uma variável categórica. Categoria "Sul" é entre 0 (não) e 1 (sim), "Sudeste" entre 0 (não) e 1 (sim) e assim por diante
