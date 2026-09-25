# Exam Prep

## Arrays

1. Qual a complexidade de busca por índice? O(1)
2. Qual a complexidade de busca por elemento? O(N)
3. Qual a complexidade de busca por elemento em array ordenado? O(logN) - busca binária
4. Qual a complexidade de adicionar novo elemento no final da lista em arrays estáticos? O(1)
5. Qual a complexidade de adicionar novo elemento no final da lista em arrays dinâmicos? O(N)
6. Qual a complexidade de adicionar novo elemento que não é o final da lista? O(N) - shift de posição
7. Qual a complexidade de remover o último elemento da lista? O(1)
8. Qual a complexidade de remover o elemento que não é o último da lista? O(N) - shift de posição
9. Qual a complexidade de memória de um array? O(N)

## Listas Ligadas (Linked Lists)

1. Qual a complexidade de busca por elemento? O(N)
2. Qual a complexidade de adicionar elemento no início? O(1)
3. Qual a complexidade de adicionar elemento no meio? O(N)
4. Qual a complexidade de adicionar elemento no final? O(N)
5. Qual a complexidade de remover elemento no início? O(1)
6. Qual a complexidade de remover elemento no meio? O(N)
7. Qual a complexidade de remover elemento no final? O(N)
8. Qual a complexidade de memória de uma lista ligada? O(N)

## Pilhas (Stacks)

1. Qual a complexidade de busca por elemento? O(N)
2. Qual a complexidade de adicionar novo elemento? O(1)
3. Qual a complexidade de remover elemento? O(1)
4. Qual a complexidade de verificar elemento do topo? O(1)
5. Qual a complexidade de memória de uma pilha? O(N)

## Filas (Queues)

1. Qual a complexidade de busca por elemento? O(N)
2. Qual a complexidade de pegar o primeiro elemento (front)? O(1)
3. Qual a complexidade de adicionar novo elemento (enqueue)? O(1)
4. Qual a complexidade de remover elemento (dequeue)? O(1) - se a estrutura for uma lista ligada / O(N) - se a estrutura for um array dinamico que precisa de shift de posição
5. Qual a complexidade de memória de uma fila? O(N)

## Listas de Prioridade (Heaps)

1. Qual a complexidade de busca do elemento mais prioritário (max/min)? O(1)
2. Qual a complexidade de busca por elemento? O(log n) - desce por um dos lados da árvore
3. Qual a complexidade de insert (adicionar)? O(log n) - desce por um dos lados da árvore
4. Qual a complexidade de extract / remover prioritário? O(log n) - a remoção do elemento prioritário é O(1), mas o recolocar outro elemento na parte prioritária necessita de um sift-up passando por um lado da árvore
5. Qual a complexidade de memória da lista de prioridade? O(n)

## Tabelas de Espalhamento (Hash Tables)

1. Qual a complexidade de busca de um elemento? O(1)
2. Qual a complexidade de busca de um elemento (com colisão)? O(n)
3. Qual a complexidade de adicionar elemento? O(1) ou O(N) se ocorrer muitas colisões
4. Qual a complexidade de remover elemento? O(1)
5. Qual a complexidade de memória da tabela de espalhamento? O(n)

## Árvores Binárias e de Busca (BST)

1. Qual a complexidade de busca de um elemento numa árvore balanceada? O(logN)
2. Qual a complexidade de inserção de elemento numa árvore balanceada? O(logN)
3. Qual a complexidade de remoção de elemento numa árvore balanceada? O(logN)
4. Qual a complexidade de busca de um elemento numa árvore desbalanceada? O(N)
5. Qual a complexidade de inserção de elemento numa árvore desbalanceada? O(N)
6. Qual a complexidade de remoção de elemento numa árvore desbalanceada? O(N)
7. Qual a complexidade de memória da árvore binária de busca? O(n)

## Complexidade Algorítmica e Notação Assintótica

- Conceitos:
  - $O(g(n))$ (Big-O): Limite superior (pior caso).  
  - $Ω(g(n))$ (Omega): Limite inferior (melhor caso).  
  - $ϴ(g(n))$ (Theta): Limite justo (comportamento exato).  
- Complexidades:
  - $O(1)$: Constante. Não importa se há 10 ou 1 bilhão de itens, leva o mesmo tempo.
  - $O(\log n)$: Logarítmico. Extremamente eficiente. Se os dados dobrarem, o algoritmo só faz uma operação a mais.
  - $O(n)$: Linear. Se os dados dobrarem, o algoritmo dobra em operações.
  - $O(n^2)$: Quadrático. O pesadelo da escalabilidade. Um loop dentro de outro loop. Se os dados dobrarem, o tempo aumenta em 4 vezes. Se aumentarem 10 vezes, o tempo aumenta 100 vezes.
- **Hierarquia comum:** $O(1) < O(\log n) < O(n) < O(n \log n) < O(n²) < O(2ᴺ) < O(n!)$.

## Bubble Sort

[Ilustração](https://www.youtube.com/watch?v=xli_FI7CuzA)

- Como funciona? Compara 2 elementos, faz swap se o número for maior, com intuito de mover os maiores números para o final da lista
- Qual a complexidade de tempo médio ($θ(n)$)? $θ(n²)$
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)? $O(n²)$
- Qual a complexidade de memória? $O(1)$

## Selection Sort 

- Como funciona?
- Qual a complexidade de tempo médio ($θ(n)$)?
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)?
- Qual a complexidade de memória?

## Insertion Sort

- Como funciona?
- Qual a complexidade de tempo médio ($θ(n)$)?
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)?
- Qual a complexidade de memória?

## Merge Sort

- Como funciona?
- Qual a complexidade de tempo médio ($θ(n)$)?
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)?
- Qual a complexidade de memória?

## Quick Sort

- Como funciona?
- Qual a complexidade de tempo médio ($θ(n)$)?
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)?
- Qual a complexidade de memória?

## Heap Sort

- Como funciona?
- Qual a complexidade de tempo médio ($θ(n)$)?
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)?
- Qual a complexidade de memória?

## Counting Sort

- Como funciona?
- Qual a complexidade de tempo médio ($θ(n)$)?
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)?
- Qual a complexidade de memória?
