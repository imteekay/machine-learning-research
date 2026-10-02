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

[Ilustração](https://www.youtube.com/watch?v=g-PGLbMth_g)

- Como funciona? Procura o menor elemento, faz o swap pra posição atual e escanea o restante dos elementos fazendo a mesma operação
- Qual a complexidade de tempo médio ($θ(n)$)? $θ(n²)$
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)? $O(n²)$
- Qual a complexidade de memória? $O(1)$

## Insertion Sort

[Ilustração](https://www.youtube.com/watch?v=JU767SDMDvA)

- Como funciona? Compara o elemento atual com os da esquerda e faz swap caso seja um elemento menor, até que chegue na posição correta
- Qual a complexidade de tempo médio ($θ(n)$)? $θ(n²)$
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)? $O(n²)$
- Qual a complexidade de memória? $O(1)$

## Merge Sort

[Ilustração](https://www.youtube.com/watch?v=4VqmGXwpLqc)

- Como funciona? Divide na metade recursivamente até que chegue ao nível de um elemento. Ao fazer o merge de duas unidades/grupos, junte em ordem. A divisão é feita recursivamente. A junção é feita iterativamente apontando para cada elemento dos dois grupos.
- Qual a complexidade de tempo médio ($θ(n)$)? $O(n\log n)$
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)? $O(n\log n)$
- Qual a complexidade de memória? $O(n)$

## Quick Sort

[Ilustração](https://www.youtube.com/watch?v=Hoixgm4-P4M)

- Como funciona? Funciona com um elemento pivô. Escolha um elemento maior na esquerda e um elemento menor na direita, faça swap deles, até que os ponteiros se cruzem, isso significa que todos os elemento menores que o pivô ficaram na esquerda e os maiores ficaram na direita. Faça o mesmo procedimento recursivamente para o bloco da esquerda e o bloco da direita.
- Qual a complexidade de tempo médio ($θ(n)$)? $O(n \log n)$
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)? $O(n²)$
- Qual a complexidade de memória? $O(n)$

## Heap Sort

[Ilustração](https://www.youtube.com/watch?v=2DmK_H7IdTo)

- Como funciona? Transforma a lista em uma árvore max heap, remove o elemento prioriário (maior elemento) e move ele para o final da lista ordenada. Recria a árvore max heap e remove o maior elemento novamente. Faça esse procedimento até que todos os elementos forem removidos da heap.
- Qual a complexidade de tempo médio ($θ(n)$)? $O(n \log n)$
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)? $O(n \log n)$
- Qual a complexidade de memória? $O(1)$

## Counting Sort

[Ilustração](https://www.youtube.com/watch?v=OKd534EWcdk)

- Como funciona? Conta a quantidade de cada elemento. Crie um array em order com a soma cumulativa das quantidades. Faça um shift do array para a esquerda e os valores serão o index em que cada elemento começa na array ordenado. Ao atravessar o array desordenado, utilize o valor como index, insira no array na posição correta e incremente o index. Faça isso para todos os elementos.
- Qual a complexidade de tempo médio ($θ(n)$)? $O(n)$
- Qual a complexidade de tempo limite superior (pior caso) ($O(n)$)? $O(n)$
- Qual a complexidade de memória? $O(n)$

## BFS (Largura)

- Como funciona? Explora tudo ao redor (filhos em árvores; vizinhos em grafos) antes de aprofundar. Utiliza filas para priorizar o que há ao redor primeiro e enfilera os próximos nós. Utiliza um HashSet para armazenar quais nós já foram visitados, para quebrar o loop de exploração.
- Complexidade de tempo: $O(V + E)$ (nós explorados (V) + vértices caminhados (E))
- Complexidade de espaço: $O(V)$ (todos os nós na fila)

## DFS (Profundidade)

- Como funciona? Explora um nó de cada vez, aprofundando o caminho ao máximo. Utiliza pilha para priorizar a profundidade. Utiliza um HashSet para armazenar quais nós já foram visitados, para quebrar o loop de exploração.
- Complexidade de tempo: $O(V + E)$ (nós explorados (V) + vértices caminhados (E))
- Complexidade de espaço: $O(V)$ (todos os nós na fila)

## Caminhos Mínimos

### Algoritmo de Dijkstra

- Objetivo: dado um vértice, encontrar o caminho mínimo para todos os outros vértices
- Como funciona: assim como DFS, coloque os vértices vizinhos em uma fila de prioridade. A fila de prioridade precisar ser priorizada/ordenada por menor peso. Para cada vértice visitado, atualize o valor do caminho (soma dos pesos do caminho) caso ache um valor menor do que o atual (inicializa com valor infinito)
- Complexidade de tempo: $O((V + E) \log V)$ (for binary heap) or $O(V²)$ (for unsorted array)
- Complexidade de espaço: $O(V)$

### Bellman-Ford

- Objetivo: encontrar o caminho mínimo
- Como funciona: O algoritmo roda \(V-1\) vezes. Em cada rodada, ele itera por todas as arestas \((u, v)\) e atualiza o caminho mínimo se a distância atual até u + o peso da aresta for menor que o caminho já conhecido até $v$.
- Complexidade de tempo: $O((V + E) \log V)$
- Complexidade de espaço: $O(V)$
