# Exam

[More info](https://www.iq.usp.br/portaliqusp/sites/default/files/anexos/3%20%C3%81reas%20Topicos%20e%20Bibliografia%20da%20prova%20-%20revisado_0.pdf)

- Duração máxima de 3 horas
- A prova será composta por 60 (sessenta) questões de múltipla escolha, com questões divididas igualmente entre as 3 (três) áreas mencionadas acima. 
- O candidato deve responder a um total de 20 (vinte) questões, sendo que no mínimo 14 (quatorze) devem pertencer a uma única área do conhecimento (as outras seis questões podem ser da mesma área que as 14 ou ser uma combinação de questões das outras áreas, a critério do candidato). 
- É vedado ao candidato responder a mais de 20 (vinte) questões.

Serão considerados aprovados para os cursos de Mestrado ou Doutorado os candidatos que obtiveram nota igual ou superior à nota mínima indicada em um dos exames especificados no item 3.3 acima ou obtiverem nota mínima 6,0 na atual prova de seleção do programa.

---

## O que estudar

Aqui está um roteiro detalhado de estudos em tópicos, estruturado para cobrir os **conceitos, implementações e aplicações** de cada assunto listado.

---

## 1. Estrutura de Dados

### 1.1 Listas Ligadas, Pilhas, Filas e Listas de Prioridade

* **Listas Ligadas (Singly, Doubly, Circular):**
* **Conceito:** Alocação dinâmica de memória, ponteiros/referências, nós contendo valor e ponteiro para o próximo.
* **Implementação:** Inserção, remoção e busca (no início, fim e meio); ponteiros `head` e `tail`.
* **Aplicações:** Gerenciamento dinâmico de memória, histórico de navegação, base para pilhas e filas.


* **Pilhas (LIFO - *Last In, First Out*):**
* **Conceito:** Operações restritas ao topo da estrutura (`push`, `pop`, `peek`/`top`).
* **Implementação:** Via array dinâmico ou lista encadeada.
* **Aplicações:** Avaliação e conversão de expressões aritméticas (Infixa $\to$ Pós-fixa), *undo/redo* em editores, pilha de chamadas de funções (*call stack*).


* **Filas (FIFO - *First In, First Out*):**
* **Conceito:** Inserção no fim (`enqueue`) e remoção no início (`dequeue`).
* **Implementação:** Array circular com ponteiros `front`/`rear` ou lista duplamente encadeada.
* **Aplicações:** Buffers de impressão/redes, filas de tarefas (*task queues*), busca em largura (BFS).


* **Listas de Prioridade (Priority Queues & Heaps):**
* **Conceito:** Elementos possuem prioridades associadas; remoção sempre do elemento de maior (ou menor) prioridade.
* **Implementação:** *Max-Heap* e *Min-Heap* (árvores binárias completas representadas em vetores); operações `sift-up` e `sift-down`.
* **Aplicações:** Algoritmo de Dijkstra, codificação de Huffman, escalonamento de processos em SO.



### 1.2 Recursão

* **Conceito:** Resolução de problemas dividindo-os em subproblemas menores do mesmo tipo; caso base (*base case*) vs. passo recursivo.
* **Implementação:** Relações de recorrência, uso interno da pilha de chamadas (*call stack*), estouro de pilha (*stack overflow*) e recursão de cauda (*tail recursion*).
* **Aplicações:** Divisão e conquista (Merge Sort, Quick Sort), travessia em árvores, *backtracking* (Sudoku, N-Rainhas).

### 1.3 Tabelas de Espalhamento (Hash Tables)

* **Conceito:** Mapeamento de chaves para valores em tempo médio $O(1)$ através de uma função hash.
* **Implementação:**
* **Funções Hash:** Módulo, multiplicação, funções de hash universais.
* **Tratamento de Colisões:** Encadeamento (*separate chaining*) vs. Endereçamento Aberto (*linear probing*, *quadratic probing*, *double hashing*).
* **Fator de Carga (*Load Factor*) e Rehash:** Redimensionamento dinâmico da tabela.


* **Aplicações:** Tabelas de símbolos de compiladores, caches (Ex: LRU Cache), conjuntos (*Sets*) e dicionários (*Maps*).

### 1.4 Árvores Binárias e Árvores Binárias de Busca (BST)

* **Árvores Binárias Gerais:**
* **Conceito:** Estrutura hierárquica onde cada nó possui no máximo dois filhos (esquerdo e direito).
* **Implementação e Percursos:** Pré-ordem, Em-ordem, Pós-ordem e Por Nível (Breadth-First).


* **Árvores Binárias de Busca (BST - *Binary Search Tree*):**
* **Conceito:** Propriedade da BST: elementos à esquerda do nó são menores; elementos à direita são maiores.
* **Implementação:** Busca, inserção, remoção (tratando 0, 1 e 2 filhos - sucessor em-ordem) em $O(h)$.
* **Árvores Balanceadas (Visão Geral):** Entendimento de pior caso $O(n)$ e árvores auto-balanceadas (AVL, Rubro-Negra) garantindo $O(\log n)$.


* **Aplicações:** Índices de bancos de dados, sistemas de arquivos, ordenação (*Tree Sort*).

### 1.5 Union-Find (Disjoint-Set)

* **Conceito:** Representação de conjuntos disjuntos com operações para unir conjuntos e determinar se dois elementos pertencem ao mesmo conjunto.
* **Implementação:**
* Operações: `find` (encontrar representante) e `union` (juntar conjuntos).
* Otimizações: **Compressão de caminho** (*Path Compression*) e **União por Rank/Tamanho** (*Union by Rank/Size*). Complexidade amortizada próxima de $O(1)$ ($\alpha(n)$ - função de Ackermann inversa).


* **Aplicações:** Algoritmo de Kruskal (Árvore Geradora Mínima), detecção de ciclos em grafos não direcionados, componentes conectados dinâmicos.

---

## 2. Algoritmos e Linguagens Formais

### 2.1 Complexidade Algorítmica e Notação Assintótica

* **Conceito:** Análise do crescimento do tempo de execução e uso de memória em função do tamanho da entrada $n$.
* **Notações Assintóticas:**
* $O$ (*Big-O*): Limite superior (pior caso).
* $\Omega$ (*Big-Omega*): Limite inferior (melhor caso).
* $\Theta$ (*Big-Theta*): Limite justo (*tight bound*).


* **Classes de Complexidade Comuns:** $O(1)$, $O(\log n)$, $O(n)$, $O(n \log n)$, $O(n^2)$, $O(2^n)$, $O(n!)$.
* **Métodos de Análise:** Análise de laços simples/aninhados, Resolução de Recorrências (Método Mestre, Árvore de Recorrência e Substituição).

### 2.2 Algoritmos de Ordenação e Seleção

* **Algoritmos de Ordenação por Comparação:**
* **Quadráticos:** Insertion Sort, Selection Sort, Bubble Sort.
* **Logarítmicos/Divisão e Conquista:** Merge Sort ($O(n \log n)$ garantido), Quick Sort ($O(n \log n)$ médio, escolha de pivô), Heap Sort ($O(n \log n)$ in-place).


* **Algoritmos Não-Comparativos (Lineares):** Counting Sort, Radix Sort, Bucket Sort (condições de aplicabilidade e complexidade $O(n + k)$).
* **Algoritmos de Seleção:**
* QuickSelect (encontrar o $k$-ésimo menor elemento em tempo médio $O(n)$).


* **Aplicações:** Ordenação de dados em bancos de dados, buscas binárias, computação gráfica.

### 2.3 Algoritmos para Problemas em Grafos

* **Representação de Grafos:** Matriz de Adjacência vs. Lista de Adjacência (espaço e eficiência).
* **Buscas e Aplicações:**
* **Busca em Largura (BFS):** Fila, caminho mínimo em grafos sem peso, verificação de grafos bipartidos.
* **Busca em Profundidade (DFS):** Pilha/Recursão, tempos de descoberta/finalização, classificação de arestas (árvore, retorno, avanço, cruzamento).
* **Componentes Conexos:** Identificação via BFS/DFS em grafos não-direcionados.
* **Componentes Fortemente Conexos (SCC):** Algoritmo de Tarjan ou Kosaraju em grafos direcionados.
* **Ordenação Topológica:** Algoritmo de Kahn (BFS) e ordenação por tempo de finalização da DFS em DAGs (Grafos Acíclicos Direcionados).


* **Árvore Geradora Mínima (MST):**
* **Algoritmo de Kruskal:** Usa ordenação de arestas + Union-Find.
* **Algoritmo de Prim:** Usa Fila de Prioridade (*Min-Heap*).


* **Caminhos Mínimos:**
* **Fonte Única sem pesos negativos:** Algoritmo de Dijkstra ($O((V + E) \log V)$ com heap).
* **Fonte Única com pesos negativos:** Algoritmo de Bellman-Ford (detecção de ciclos negativos).
* **Todos para Todos:** Algoritmo de Floyd-Warshall ($O(V^3)$ por programação dinâmica).



### 2.4 NP-Completude

* **Classes de Problemas:**
* **P:** Problemas solúveis em tempo polinomial por uma Máquina de Turing determinística.
* **NP:** Problemas cujas soluções podem ser verificadas em tempo polinomial por uma Máquina de Turing não-determinística.


* **Conceito de Redução Polinomial:** $A \le_P B$ (se $B$ for fácil, $A$ também é).
* **NP-Duro (NP-Hard) e NP-Completo (NPC):**
* Definição e Teorema de Cook-Levin (SAT é NP-Completo).


* **Problemas Clássicos NP-Completos:** SAT / 3-SAT, Clicagem (*Clique*), Cobertura de Vértices (*Vertex Cover*), Ciclo Hamiltoniano, Caxeiro Viajante (TSP), Soma dos Subconjuntos (*Subset Sum*), Mochila (*Knapsack* 0/1).
* **Estratégias para Lidar com Problemas NPC:** Algoritmos de aproximação, programação dinâmica (soluções pseudo-polinomiais), heurísticas e *backtracking*/Branch & Bound.

### 2.5 Autômatos Finitos e Expressões Regulares

* **Autômatos Finitos Determinísticos (AFD):**
* Definição formal (quíntupla $\Sigma, Q, q_0, F, \delta$), diagrama de estados e tabela de transições.


* **Autômatos Finitos Não-Determinísticos (AFN e AFN-$\epsilon$):**
* Transições múltiplas e transições vazias.
* Algoritmo de conversão AFN $\to$ AFD (Construção do Subconjunto / *Powerset Construction*).


* **Expressões Regulares (ER / Regex):**
* Operadores regulares: União ($\vert{}$), Concatenação, Fecho de Kleene ($*$).
* Equivalência entre Expressões Regulares e Autômatos Finitos (Teorema de Kleene).

* **Linguagens Regulares:**
* Propriedades de fechamento (união, interseção, complemento).
* **Lema do Bombeamento (*Pumping Lemma*):** Utilização para provar que uma determinada linguagem **não** é regular.
* **Aplicações:** Análise léxica de compiladores, validação de entrada, busca de padrões em textos.
