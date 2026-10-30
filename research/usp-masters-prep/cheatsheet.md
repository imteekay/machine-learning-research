# Cheatsheet — Revisão de 30 minutos

> Leia de cima para baixo. Cada seção cobre um item da ementa (`exam.md`). No final há as **pegadinhas** mais comuns em múltipla escolha.

---

## 0. Formato da prova

- 17/11/2026, 08:30, 3h, múltipla escolha.
- Responder **exatamente 20** questões (mín. 14 de uma mesma área). **Nunca responda mais de 20.**
- Nota mínima: 6,0 → 12 acertos de 20. Pule as dúvidas e volte depois (~9 min por questão).

---

## 1. Estruturas de Dados

### 1.1 Tabela-mestra de complexidades

| Estrutura | Acesso/Busca | Inserção | Remoção | Observação |
| :---- | :---- | :---- | :---- | :---- |
| Array (índice) | $O(1)$ / busca $O(n)$ | fim $O(1)$ amortizado; meio $O(n)$ | fim $O(1)$; meio $O(n)$ | memória contígua |
| Array ordenado | busca binária $O(\log n)$ | $O(n)$ | $O(n)$ | shift de posições |
| Lista ligada simples | $O(n)$ | início $O(1)$; fim $O(1)$ **com ponteiro de cauda** | início $O(1)$; fim $O(n)$ | sem acesso aleatório |
| Lista duplamente ligada | $O(n)$ | início/fim $O(1)$ | início/fim/nó conhecido $O(1)$ | 2 ponteiros por nó |
| Pilha (LIFO) | topo $O(1)$ | push $O(1)$ | pop $O(1)$ | call stack, undo, parênteses |
| Fila (FIFO) | front $O(1)$ | enqueue $O(1)$ | dequeue $O(1)$ | array circular ou lista; BFS |
| Heap binário | min/max $O(1)$; busca $O(n)$ | $O(\log n)$ (sift-up) | extract $O(\log n)$ (sift-down) | build-heap $O(n)$ |
| Hash table | $O(1)$ médio / $O(n)$ pior | $O(1)$ médio | $O(1)$ médio | fator de carga $\alpha = n/m$ |
| BST | $O(h)$ | $O(h)$ | $O(h)$ | $h = \log n$ balanceada, $h = n$ degenerada |
| Union-Find (rank + compressão) | find $O(\alpha(n))$ | union $O(\alpha(n))$ | — | amortizado, $\alpha(n) \le 4$ |

Memória de todas: $O(n)$.

### 1.2 Heap

- Árvore binária **completa** em array. Max-heap: pai $\ge$ filhos. Min-heap: pai $\le$ filhos.
- Índices (base 0): filhos $2i+1$, $2i+2$; pai $\lfloor (i-1)/2 \rfloor$. Folhas: a partir de $\lfloor n/2 \rfloor$.
- **Insert**: coloca no fim → **sift-up**. **Extract**: troca raiz com último, remove, → **sift-down**.
- **Build-heap** (heapify de baixo para cima): $O(n)$, não $O(n \log n)$.
- Usos: Dijkstra, Prim, Huffman, Heap Sort, escalonamento, k-ésimo maior.

### 1.3 Recursão

- Caso base + caso recursivo. Cada chamada ocupa um quadro na pilha → profundidade $d$ = memória $O(d)$.
- **Tail call**: a chamada recursiva é a **última** ação (nada pendente). `n * f(n-1)` **não** é cauda; `f(n-1, acc*n)` é. Com TCO: memória $O(1)$.
- Fibonacci ingênuo $O(2^n)$; com memoização $O(n)$. Hanói: $2^n - 1$ movimentos.
- Backtracking: N-Rainhas, Sudoku, permutações (explora e desfaz).

### 1.4 Hash Table

- **Encadeamento** (separate chaining): lista em cada bucket; busca esperada $\Theta(1 + \alpha)$; $\alpha$ pode ser $> 1$.
- **Endereçamento aberto** ($\alpha < 1$ obrigatório):
  - Linear: $h(k) + i$ → **agrupamento primário**.
  - Quadrática: $h(k) + c_1 i + c_2 i^2$ → agrupamento secundário.
  - Double hashing: $h_1(k) + i \cdot h_2(k)$ → melhor distribuição ($h_2(k)$ nunca 0, coprimo com $m$).
  - Remoção precisa de marcador **"removido" (tombstone)**.
- Rehash quando $\alpha$ passa do limite (ex.: 0,75) → dobra tamanho, $O(n)$ pontual, $O(1)$ amortizado.
- Método da divisão: $k \bmod m$, com $m$ **primo** longe de potência de 2.

### 1.5 Árvores Binárias / BST

- BST: esquerda < nó < direita. **Em-ordem = sequência ordenada.**
- Travessias: Pré (R-E-D), Em (E-R-D), Pós (E-D-R), Nível (BFS com fila).
- Remoção: 0 filhos → remove; 1 filho → sobe o filho; 2 filhos → troca pelo **sucessor em-ordem** (mínimo da subárvore direita).
- Inserir dados **ordenados** numa BST simples → degenera em lista → $O(n)$.

### 1.6 Union-Find

- `make_set`, `find` (representante/raiz), `union` (liga raízes).
- **Union by rank/size**: raiz da árvore menor vira filha da maior → altura $O(\log n)$.
- **Path compression**: no `find`, todos os nós do caminho apontam direto para a raiz.
- Ambos: $O(\alpha(n))$ amortizado. Usos: **Kruskal**, componentes conexos dinâmicos, ciclo em grafo **não direcionado**.

---

## 2. Algoritmos e Linguagens Formais

### 2.1 Notação assintótica

- $f = O(g)$: $\exists c, n_0$: $f(n) \le c \cdot g(n)$ para $n \ge n_0$ → **limite superior**.
- $f = \Omega(g)$: $f(n) \ge c \cdot g(n)$ → **limite inferior**.
- $f = \Theta(g)$: ambos → **limite justo**.
- $o$ / $\omega$: estritos (para **todo** $c$).
- ⚠️ $O$/$\Omega$/$\Theta$ **não** significam pior/melhor/médio caso. Cada caso tem seu próprio $O$, $\Omega$, $\Theta$.
- Hierarquia: $1 < \log\log n < \log n < \sqrt{n} < n < n\log n < n^2 < n^3 < 2^n < n! < n^n$.
- Base do log não importa. $\log(n!) = \Theta(n \log n)$. $\sum_{i=1}^{n} i = \frac{n(n+1)}{2} = \Theta(n^2)$.

### 2.2 Teorema Mestre — $T(n) = a\,T(n/b) + f(n)$

Compare $f(n)$ com $n^{\log_b a}$:

| Caso | Condição | Resultado |
| :---- | :---- | :---- |
| 1 | $f(n) = O(n^{\log_b a - \epsilon})$ (folhas dominam) | $\Theta(n^{\log_b a})$ |
| 2 | $f(n) = \Theta(n^{\log_b a})$ (empate) | $\Theta(n^{\log_b a} \log n)$ |
| 3 | $f(n) = \Omega(n^{\log_b a + \epsilon})$ + regularidade (raiz domina) | $\Theta(f(n))$ |

Decore:

| Recorrência | Resultado | Exemplo |
| :---- | :---- | :---- |
| $T(n) = T(n/2) + 1$ | $\Theta(\log n)$ | busca binária |
| $T(n) = 2T(n/2) + n$ | $\Theta(n \log n)$ | merge sort |
| $T(n) = 2T(n/2) + 1$ | $\Theta(n)$ | travessia de árvore |
| $T(n) = T(n/2) + n$ | $\Theta(n)$ | quickselect (médio) |
| $T(n) = 4T(n/2) + n$ | $\Theta(n^2)$ | |
| $T(n) = 2T(n/2) + n^2$ | $\Theta(n^2)$ | |
| $T(n) = 7T(n/2) + n^2$ | $\Theta(n^{\log_2 7}) \approx n^{2,81}$ | Strassen |
| $T(n) = T(n-1) + 1$ | $\Theta(n)$ | fatorial (não é Mestre) |
| $T(n) = T(n-1) + n$ | $\Theta(n^2)$ | quicksort pior caso |
| $T(n) = 2T(n-1) + 1$ | $\Theta(2^n)$ | Hanói |

### 2.3 Ordenação

| Algoritmo | Melhor | Médio | Pior | Memória extra | Estável | In-place |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Bubble | $n$ (c/ flag) | $n^2$ | $n^2$ | $1$ | ✅ | ✅ |
| Selection | $n^2$ | $n^2$ | $n^2$ | $1$ | ❌ | ✅ |
| Insertion | $n$ | $n^2$ | $n^2$ | $1$ | ✅ | ✅ |
| Merge | $n\log n$ | $n\log n$ | $n\log n$ | $n$ | ✅ | ❌ |
| Quick | $n\log n$ | $n\log n$ | $n^2$ | $\log n$ (pilha) | ❌ | ✅ |
| Heap | $n\log n$ | $n\log n$ | $n\log n$ | $1$ | ❌ | ✅ |
| Counting | $n+k$ | $n+k$ | $n+k$ | $n+k$ | ✅ | ❌ |
| Radix (LSD) | $d(n+k)$ | $d(n+k)$ | $d(n+k)$ | $n+k$ | ✅ | ❌ |
| Bucket | $n+k$ | $n+k$ | $n^2$ | $n+k$ | ✅* | ❌ |

\* Bucket é estável se a ordenação interna de cada balde for estável (ex.: Insertion).

- **Limite inferior** de ordenação por comparação: $\Omega(n \log n)$ (árvore de decisão com $n!$ folhas).
- Insertion: ótimo para **quase ordenado** / pequeno; nº de trocas = nº de **inversões**.
- Selection: mínimo de trocas ($O(n)$ swaps), mas sempre $\Theta(n^2)$ comparações.
- Quick pior caso: pivô sempre extremo (array já ordenado com pivô no início/fim). Solução: pivô aleatório / mediana de três.
- Merge: estável, bom para listas ligadas e ordenação externa.
- Radix usa ordenação **estável** (Counting) por dígito, do menos significativo para o mais.
- **Quickselect** (k-ésimo menor): $O(n)$ médio, $O(n^2)$ pior. **Mediana das medianas**: $O(n)$ pior caso.

### 2.4 Grafos

**Representação**: Matriz $O(V^2)$ memória, aresta $(u,v)$ em $O(1)$ → densos. Lista $O(V+E)$ → esparsos. Soma dos graus $= 2E$. Máx. arestas (não dir.) $= V(V-1)/2$.

| Algoritmo | Problema | Estrutura | Tempo | Restrição |
| :---- | :---- | :---- | :---- | :---- |
| BFS | caminho mínimo **sem pesos**, níveis | **fila** | $O(V+E)$ | |
| DFS | ciclos, topológica, SCC, labirinto | **pilha**/recursão | $O(V+E)$ | |
| Kahn | ordenação topológica | fila + grau de entrada | $O(V+E)$ | só DAG |
| DFS pós-ordem invertida | ordenação topológica | pilha | $O(V+E)$ | só DAG |
| Kosaraju | SCC | 2 DFS (G e $G^T$) | $O(V+E)$ | dirigido |
| Tarjan | SCC | 1 DFS + pilha + *low-link* | $O(V+E)$ | dirigido |
| Dijkstra | caminho mínimo, fonte única | fila de prioridade | $O((V+E)\log V)$ heap; $O(V^2)$ array | **pesos $\ge 0$** |
| Bellman-Ford | caminho mínimo, fonte única | relaxa todas as arestas $V-1$ vezes | $O(V \cdot E)$ | aceita negativos; **detecta ciclo negativo** |
| Floyd-Warshall | todos os pares | DP, matriz | $\Theta(V^3)$ tempo, $\Theta(V^2)$ memória | aceita negativos, sem ciclo negativo |
| Prim | MST | fila de prioridade | $O(E\log V)$; $O(V^2)$ array | conexo, não dir. |
| Kruskal | MST | ordena arestas + Union-Find | $O(E\log E) = O(E\log V)$ | não dir. |

- **Relaxamento**: se $d[u] + w(u,v) < d[v]$ então $d[v] \leftarrow d[u] + w(u,v)$.
- Bellman-Ford: se na **$V$-ésima** rodada ainda relaxar → **ciclo negativo**.
- Floyd-Warshall: $d[i][j] = \min(d[i][j],\ d[i][k] + d[k][j])$ com $k$ no laço **externo**. Ciclo negativo ⇔ $d[i][i] < 0$.
- Dijkstra falha com pesos negativos porque **finaliza** o vértice quando ele sai da fila.
- DAG: caminho mínimo em $O(V+E)$ (topológica + relaxar), aceita pesos negativos.
- Grafo dirigido tem ciclo ⇔ DFS encontra **aresta de retorno** (back edge).
- **MST**: $V-1$ arestas, sem ciclos. Única se todos os pesos forem distintos. Propriedade do corte: a menor aresta que cruza um corte está em alguma MST. MST ≠ árvore de caminhos mínimos.

### 2.5 P, NP, NP-Completo, NP-Difícil

- **P**: resolvível em tempo polinomial (determinístico).
- **NP**: solução **verificável** em tempo polinomial (certificado) = resolvível por máquina **não-determinística** em tempo polinomial. **P ⊆ NP**. P = NP? **em aberto**.
- **NP-Difícil**: todo problema de NP se reduz a ele. Pode não estar em NP (ex.: Problema da Parada).
- **NP-Completo** = NP ∩ NP-Difícil.
- **Redução** $A \le_p B$: "$B$ é pelo menos tão difícil quanto $A$". Para provar que **$B$ é NP-difícil**, reduza um NP-completo **conhecido $A$ para $B$**.
- Para provar NP-completo: (1) $B \in$ NP; (2) NPC conhecido $\le_p B$.
- **Cook-Levin**: SAT é NP-completo (o primeiro).
- **NPC**: SAT, 3-SAT, Clique, Vertex Cover, Conjunto Independente, Ciclo Hamiltoniano, TSP (decisão), Mochila (decisão), Subset Sum, 3-Coloração, Set Cover.
- **Em P**: 2-SAT, caminho mínimo, MST, ciclo **Euleriano**, 2-coloração (bipartido), emparelhamento, programação linear, primalidade.
- Mochila DP $O(nW)$ é **pseudo-polinomial** (polinomial no **valor** de $W$, exponencial nos bits).
- Na prática: heurísticas, aproximação (Vertex Cover 2-aprox), DP pseudo-polinomial, Branch & Bound.

### 2.6 Autômatos e Expressões Regulares

- **AFD**: $M = (Q, \Sigma, \delta, q_0, F)$, $\delta: Q \times \Sigma \to Q$ (exatamente uma transição por par).
- **AFN**: $\delta: Q \times (\Sigma \cup \{\epsilon\}) \to \mathcal{P}(Q)$. Aceita se **algum** caminho termina em estado final.
- **Kleene**: Regex ≡ AFN-ε ≡ AFN ≡ AFD → todos reconhecem as **linguagens regulares**.
  - Regex → AFN-ε: **Thompson** ($O(m)$ estados).
  - AFN → AFD: **construção de subconjuntos** (até $2^n$ estados; pior caso atingível).
  - AFD → Regex: eliminação de estados.
- Precedência regex: `*` > concatenação > `|`.
- Regulares são **fechadas** sob: união, concatenação, estrela, **complemento**, **interseção**, diferença, reverso.
- **Lema do Bombeamento**: se $L$ é regular, $\exists p$ tal que toda $w \in L$, $|w| \ge p$, se divide em $w = xyz$ com $|xy| \le p$, $|y| \ge 1$ e $xy^iz \in L$ $\forall i \ge 0$. Serve **só para provar que NÃO é regular** (condição necessária, não suficiente).
- **Não regulares**: $a^nb^n$, parênteses balanceados, palíndromos, $ww$, $a^{n^2}$, $a^p$ ($p$ primo).
- **Regulares**: toda linguagem **finita**; $a^*b^*$; $a^nb^m$; nº par de $a$; termina em `01`; binários divisíveis por 3.
- AFD mínimo é **único** (a menos de renomear estados).

---

## 3. Pegadinhas (leia antes de entrar)

1. **$\Omega$ ≠ melhor caso** e **$O$ ≠ pior caso**: são limites. "Insertion Sort é $O(n^2)$" e "é $\Omega(n)$" são ambas verdadeiras.
2. **Build-heap é $O(n)$**, não $O(n \log n)$.
3. **Heap não serve para busca**: buscar elemento arbitrário é $O(n)$.
4. **Array dinâmico**: append é $O(1)$ **amortizado** (dobrando capacidade). Se crescer por constante, vira $O(n)$ amortizado.
5. **Hash é $O(1)$ médio**, $O(n)$ pior caso.
6. **BST não balanceada** com inserção ordenada → $O(n)$.
7. **Quick Sort pior caso $O(n^2)$**; não é estável. **Merge Sort** é estável mas usa $O(n)$ extra. **Heap Sort** é in-place e $O(n \log n)$ garantido, mas **não estável**.
8. Comparação: nada abaixo de $\Omega(n \log n)$. Counting/Radix/Bucket "furam" porque **não comparam**.
9. **Dijkstra + peso negativo = errado** (mesmo sem ciclo negativo). Use Bellman-Ford.
10. **BFS** dá caminho mínimo só em grafo **sem peso** (ou pesos iguais).
11. Ordenação topológica só existe em **DAG**.
12. Redução: para provar $X$ difícil, **reduza o conhecido PARA $X$** ($A_{NPC} \le_p X$), nunca o contrário.
13. **NP não significa "não polinomial"**; P ⊆ NP. Ninguém provou que NPC exige tempo exponencial.
14. **2-SAT ∈ P**, **3-SAT é NPC**. **Euleriano ∈ P**, **Hamiltoniano é NPC**.
15. **AFN não é mais poderoso que AFD** — mesmo poder, AFN só pode ser exponencialmente menor.
16. Lema do Bombeamento **não prova** que uma linguagem é regular.
17. Toda linguagem **finita** é regular. $a^*b^*$ é regular; $a^nb^n$ não.
18. Tail recursion: `return n * fat(n-1)` **não** é de cauda.
19. Kruskal usa **Union-Find**; Prim usa **fila de prioridade**. Dijkstra ≈ Prim, mas a chave é distância **acumulada** (Dijkstra) vs. peso da **aresta** (Prim).
20. Remoção em endereçamento aberto precisa de **tombstone**, senão a busca para cedo e erra.
