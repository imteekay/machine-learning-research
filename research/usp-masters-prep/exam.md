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

## 1. Estruturas de Dados

### 1.1 Listas Ligadas, Pilhas, Filas e Listas de Prioridade

* **Listas Ligadas (Simples, Duplamente Ligadas e Circulares):** Conceito e implementação baseados em alocação dinâmica de memória, onde nós contêm valores e ponteiros para o próximo (e anterior, se dupla). Destacam-se pelas operações de inserção e remoção mais flexíveis que arrays estáticos. Aplicações incluem gerenciamento dinâmico de memória, histórico de navegação (listas de reprodução) e como base estrutural para construir pilhas e filas.
* **Pilhas (Stacks - LIFO):** Operam no princípio *Last-In, First-Out*. A implementação restringe o acesso ao topo da estrutura através das operações `push`, `pop` e `peek`/`top` (via arrays ou listas encadeadas). Aplicações cobrem desde a avaliação e conversão de expressões matemáticas (Infixa para Pós-fixa), *undo/redo* em editores, até a crucial pilha de execução do sistema (*call stack*).
* **Filas (Queues - FIFO):** Operam no princípio *First-In, First-Out*. A inserção ocorre no fim (`enqueue`) e a remoção no início (`dequeue`). A implementação costuma usar arrays circulares (com ponteiros `front`/`rear`) ou listas duplamente encadeadas, englobando também os Deques (filas de duas pontas). Aplicações reais envolvem escalonamento de processos no SO, buffers de impressão/redes, controle de tarefas assíncronas e a Busca em Largura (BFS).
* **Listas de Prioridade (Priority Queues & Heaps):** Estruturas onde os elementos possuem prioridades e a remoção foca sempre no elemento de maior/menor prioridade. Geralmente implementadas usando *Max-Heaps* e *Min-Heaps* (árvores binárias completas mapeadas em vetores) com operações de manutenção como `sift-up` e `sift-down` em tempo $O(\log n)$. Aplicações essenciais no Algoritmo de Dijkstra, codificação de Huffman e escalonamento de processos por prioridade.

### 1.2 Recursão

* **Conceito e Implementação:** Resolução de problemas pela divisão em subproblemas menores do mesmo tipo. Exige a definição clara de um caso base (condição de parada) e do passo recursivo (relações de recorrência).
* **Cuidados e Otimizações:** Depende do uso interno da pilha de chamadas, o que traz o risco de estouro de memória (*stack overflow*). É vital entender a diferença entre recursão direta/indireta e a otimização de recursão de cauda (*tail recursion*), que permite a compiladores transformarem a recursão em iteração.
* **Aplicações Práticas:** Algoritmos de Divisão e Conquista (Merge Sort, Quick Sort), travessia em estruturas de árvores/grafos e algoritmos de *backtracking* (problema das N-Rainhas, resolução de Sudoku).

### 1.3 Tabelas de Espalhamento (Hash Tables)

* **Conceito e Função Hash:** Mapeamento de chaves para índices de um array através de funções matemáticas (módulo, multiplicação, hash universal), permitindo buscas, inserções e remoções em tempo médio $O(1)$.
* **Tratamento de Colisões e Rehash:** Implementações precisam lidar com colisões via Encadeamento Separado (*separate chaining* usando listas ligadas) ou Endereçamento Aberto (*open addressing* via *linear probing*, *quadratic probing* ou *double hashing*). O redimensionamento dinâmico é controlado pelo fator de carga (*load factor*).
* **Aplicações Práticas:** Construção de bancos de dados, tabelas de símbolos de compiladores, estruturas de dicionários (chave-valor), conjuntos (*Sets*) e sistemas de cache rápido (como LRU Cache).

### 1.4 Árvores Binárias e de Busca (BST)

* **Árvores Binárias Gerais:** Estrutura hierárquica (raiz, folha, altura, profundidade) onde cada nó tem no máximo dois filhos. O domínio das travessias clássicas (Pré-ordem, Em-ordem, Pós-ordem e Por Nível/BFS) usando recursão ou pilhas iterativas é fundamental.
* **Árvores Binárias de Busca (BST):** Adiciona a propriedade em que nós à esquerda são menores e à direita são maiores. Suporta inserção, busca e busca de mínimo/máximo em tempo $O(h)$, mas a remoção é complexa (exige tratar casos de 0, 1 ou 2 filhos, usando o sucessor em-ordem).
* **Balanceamento e Aplicações:** No pior caso (árvore degenerada), uma BST vira uma lista ligada com tempo $O(n)$. O entendimento de árvores auto-balanceadas (como AVL e Rubro-Negra) é essencial para garantir operações em $O(\log n)$. Aplicações incluem índices de buscas, sistemas de arquivos e ordenação (*Tree Sort*).

### 1.5 Union-Find (Conjuntos Disjuntos)

* **Conceito e Operações:** Estrutura para manter o rastro de um conjunto de elementos particionados em subconjuntos disjuntos, operando majoritariamente com os métodos `MakeSet`, `find` (encontra o representante) e `union` (junta dois conjuntos).
* **Otimizações Críticas:** A combinação das técnicas de Compressão de Caminho (*Path Compression*) e União por Tamanho/Posto (*Union by Rank/Size*) garante uma complexidade de tempo amortizada quase constante, descrita pela função inversa de Ackermann ($\alpha(n)$).
* **Aplicações Práticas:** Implementação do Algoritmo de Kruskal para Árvores Geradoras Mínimas, encontrar componentes conexos dinâmicos em grafos e detecção rápida de ciclos em grafos não direcionados.

---

## 2. Algoritmos e Linguagens Formais

### 2.1 Complexidade Algorítmica e Notação Assintótica

* **Conceitos Básicos:** Análise matemática do crescimento do tempo de execução e consumo de memória (espaço) em função do tamanho da entrada $n$, avaliando laços aninhados, recursões e resolução de recorrências (Método Mestre).
* **Notações Assintóticas:** Uso de $O$ (*Big-O*) para o limite superior/pior caso (foco principal de entrevistas e provas), $\Omega$ (*Big-Omega*) para limite inferior/melhor caso, e $\Theta$ (*Big-Theta*) para o limite exato (*tight bound*).
* **Hierarquia de Complexidade:** Compreensão profunda das classes de eficiência, priorizando do mais rápido ao mais lento: $O(1) < O(\log n) < O(n) < O(n \log n) < O(n^2) < O(2^n) < O(n!)$.

### 2.2 Algoritmos de Ordenação e Seleção

* **Ordenação Básica / Quadrática ($O(n^2)$):** Bubble Sort, Selection Sort e Insertion Sort (este último notável por ser muito eficiente para listas pequenas ou quase já ordenadas).
* **Ordenação Eficiente ($O(n \log n)$):** Merge Sort (estável, mas exige memória extra), Quick Sort (*in-place*, rápido na média, mas de pior caso $O(n^2)$ dependendo da escolha do pivô) e Heap Sort (*in-place* e tempo garantido).
* **Ordenação em Tempo Linear e Seleção:** Algoritmos não-comparativos baseados na natureza dos dados, como Counting Sort, Radix Sort e Bucket Sort (complexidade $O(n + k)$). Inclui também o Quickselect para encontrar o $k$-ésimo menor elemento de um array não ordenado em tempo médio linear.

### 2.3 Algoritmos para Problemas em Grafos

* **Representação e Busca:** Escolha entre Matriz de Adjacência (grafos densos) e Lista de Adjacência (grafos esparsos). Domínio da Busca em Largura (BFS, usa fila, acha caminhos curtos em grafos sem peso) e Busca em Profundidade (DFS, usa recursão/pilha, acha ciclos e labirintos).
* **Aplicações Avançadas de Busca:** Identificação de Componentes Conexos, Componentes Fortemente Conexos em grafos direcionados (Algoritmos de Tarjan ou Kosaraju) e Ordenação Topológica em DAGs (via DFS ou Algoritmo de Kahn).
* **Árvore Geradora Mínima (MST):** Algoritmo de Prim (cresce a partir de um vértice usando Fila de Prioridade) e Algoritmo de Kruskal (ordena arestas globais usando Union-Find para evitar ciclos).
* **Caminhos Mínimos:** Algoritmo de Dijkstra (para grafos com pesos não-negativos, $O((V+E)\log V)$), Bellman-Ford (lida com pesos negativos e detecta ciclos negativos) e Floyd-Warshall (encontra caminhos entre todos os pares usando Programação Dinâmica, $O(V^3)$).

### 2.4 NP-Completude

* **Classes de Complexidade:** Distinção exata entre a classe **P** (resolvível em tempo polinomial), **NP** (verificável em tempo polinomial de forma não-determinística), **NP-Completo** (os problemas mais difíceis dentro de NP, baseados no Teorema de Cook-Levin) e **NP-Difícil** (*NP-Hard*).
* **Conceitos e Provas:** Uso da redução polinomial ($A \le_P B$) para provar a dificuldade de um problema transformando-o em outro já conhecido.
* **Problemas Clássicos e Soluções:** Conhecimento de problemas notórios como SAT/3-SAT, Caixeiro Viajante (TSP), Mochila (*Knapsack*), Clique e Cobertura de Vértices (*Vertex Cover*). Na prática, resolvem-se com heurísticas, algoritmos de aproximação, Programação Dinâmica (pseudo-polinomial) ou *Branch & Bound*.

### 2.5 Autômatos Finitos e Expressões Regulares

* **Expressões Regulares (Regex) e Linguagens:** Descrição de linguagens regulares usando os operadores de união ($\vert$), concatenação e Fecho de Kleene ($*$). Compreensão das propriedades de fechamento e do Lema do Bombeamento (*Pumping Lemma*) usado estritamente para provar que linguagens (como $a^n b^n$) **não** são regulares.
* **Autômatos Finitos Determinísticos (AFD):** Modelo formal de 5-tuplas ($\Sigma, Q, q_0, F, \delta$) onde para cada estado e símbolo do alfabeto existe apenas uma transição possível.
* **Autômatos Finitos Não-Determinísticos (AFN e AFN-$\epsilon$):** Permitem ramificações múltiplas de estados para o mesmo símbolo e transições vazias (sem consumir entrada).
* **Teoremas e Conversões (Teorema de Kleene):** Conversão de Regex para AFN através do Algoritmo de Thompson, e de AFN para AFD utilizando a Construção de Subconjuntos (*Powerset Construction*). Fundamentais para a construção de analisadores léxicos em compiladores e *pattern matching*.