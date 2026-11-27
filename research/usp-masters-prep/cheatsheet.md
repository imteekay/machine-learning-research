# Cheatsheet — Revisão de 30 minutos

> Leia de cima para baixo. Cada seção cobre um item da ementa (`exam.md`). No final há as **pegadinhas** mais comuns em múltipla escolha.

---

## 0. Formato da prova

- 01/12/2026, 08:30 (Brasília), 3h, remota (online), múltipla escolha.
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
- Altura: $\lfloor \log_2 n \rfloor$. Base 1: filhos $2i$, $2i+1$; pai $\lfloor i/2 \rfloor$.
- **Top-$k$ maiores**: min-heap de tamanho $k$; se o elemento > topo, troca o topo → $O(n \log k)$.
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
- **Hash universal**: função sorteada de uma família $H$ com $\Pr[h(x) = h(y)] \le 1/m$ para $x \ne y$ → bom caso médio mesmo contra entradas adversárias.
- Hash **não mantém ordem**: para ordem/intervalos use BST balanceada.
- **LRU Cache $O(1)$**: hash table (chave → nó) + lista **duplamente** ligada (ordem de uso).

### 1.5 Árvores Binárias / BST

- BST: esquerda < nó < direita. **Em-ordem = sequência ordenada.**
- Travessias: Pré (R-E-D), Em (E-R-D), Pós (E-D-R), Nível (BFS com fila).
- Remoção: 0 filhos → remove; 1 filho → sobe o filho; 2 filhos → troca pelo **sucessor em-ordem** (mínimo da subárvore direita).
- Inserir dados **ordenados** numa BST simples → degenera em lista → $O(n)$.
- Mínimo = nó mais à esquerda. Máx. de nós com altura $h$: $2^{h+1} - 1$. Árvore **cheia** (0 ou 2 filhos): folhas $L = I + 1$.
- **Pré + em-ordem** determinam a árvore (chaves distintas); **pré + pós** em geral **não**.
- **AVL**: $FB = h(esq) - h(dir) \in \{-1,0,1\}$. LL → rotação simples **direita**; RR → simples **esquerda**; LR → esquerda no filho esq., depois direita no nó; RL → o espelho. Inserir 1,2,3 → rotação à esquerda, raiz 2.
- **Rubro-negra**: (1) raiz preta; (2) folhas NIL pretas; (3) vermelho não tem filho vermelho; (4) todo caminho até as folhas tem o mesmo nº de pretos. Altura $\le 2\log(n+1)$.
- AVL × RB: AVL mais rígida (≈ $1{,}44 \log n$) → **busca** mais rápida; RB faz menos rotações → **inserção/remoção** mais rápidas.

### 1.6 Union-Find

- `make_set`, `find` (representante/raiz), `union` (liga raízes).
- **Union by rank/size**: raiz da árvore menor vira filha da maior → altura $O(\log n)$. O rank só sobe (+1) quando os ranks são iguais.
- **Path compression**: no `find`, todos os nós do caminho apontam direto para a raiz.
- Só rank: $O(\log n)$ por operação. Ambos: $O(\alpha(n))$ amortizado. Usos: **Kruskal**, componentes conexos dinâmicos, ciclo em grafo **não direcionado**.
- **Ciclo**: para cada aresta $(u,v)$, se `find(u) == find(v)` → ciclo; senão `union`.
- **Contar componentes**: começa com $n$; cada `union` bem-sucedida reduz 1 (ou conte os $x$ com `find(x) == x`).

### 1.7 Listas, Pilhas e Filas

- Acesso por índice $O(1)$: endereço = base + índice × tamanho. Inserir no meio de array: $O(n)$ (shift).
- Array dinâmico: crescer **+constante** (ex.: +10) → $O(n)$ amortizado ($O(n^2)$ total); crescimento **geométrico** → $O(1)$ amortizado.
- Lista simples com `head`+`tail`: inserir no fim $O(1)$; remover do fim $O(n)$ (achar o penúltimo). Duplamente ligada: ambos $O(1)$.
- Busca binária não serve em lista ligada (sem acesso aleatório).
- **Ciclo em lista ligada** em $O(1)$ memória: Floyd (tartaruga 1 passo, lebre 2 passos; se encontram → ciclo).
- Pilha LIFO (`push/pop/peek`); fila FIFO (`enqueue/dequeue/front`); **deque** = inserção/remoção $O(1)$ nas duas pontas.
- Fila circular de tamanho $m$: `rear = (rear + 1) mod m`; cheia quando `(rear+1) mod m == front` (sacrifica uma posição) ou com contador.
- **Fila com 2 pilhas**: entrada recebe `push`; no `dequeue`, se a saída estiver vazia, despeja a entrada nela → $O(1)$ amortizado.
- Infixa → pós-fixa: `A + B * C` → `A B C * +`; `(A + B) * C` → `A B + C *`; `A * (B + C)` → `A B C + *`. Avaliar pós-fixa: empilha operandos; operador desempilha 2.
- Parênteses balanceados: pilha (balanceado se termina vazia). BFS usa **fila**; DFS usa **pilha**.

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
- $O$ vs $o$: $n = O(n)$ mas $n \ne o(n)$; $n = o(n^2)$.
- $2^{n+1} = O(2^n)$ (constante 2), mas $2^{2n} = 4^n \ne O(2^n)$. $2^n \ne \Theta(3^n)$.
- Laços: `i = n; while i > 1: i /= 2` → $\Theta(\log n)$; `for i: while j < n: j *= 2` → $\Theta(n \log n)$; `for i: for j in i..n` → $\Theta(n^2)$; laço interno de 10 iterações é constante.
- Insertion Sort: $O(n^2)$ **e** $\Omega(n)$ (melhor caso $\Theta(n)$, pior $\Theta(n^2)$).

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
| $T(n) = T(n/2) + n$ | $\Theta(n)$ | — |
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

- **Limite inferior** de ordenação por comparação: $\Omega(n \log n)$ (árvore de decisão com $n!$ folhas).
- Insertion: ótimo para **quase ordenado** / pequeno; nº de trocas = nº de **inversões**.
- Selection: mínimo de trocas ($O(n)$ swaps), mas sempre $\Theta(n^2)$ comparações.
- Quick pior caso: pivô sempre extremo (array já ordenado com pivô no início/fim). Solução: pivô aleatório / mediana de três.
- Merge: estável, bom para listas ligadas (religa ponteiros, $O(1)$ extra) e ordenação externa.
- Quick: memória $O(\log n)$ médio; $O(n)$ pior (ou $O(\log n)$ recursando primeiro na menor parte). Introsort cai para Heap Sort se a recursão ficar funda.
- Heap Sort: build max-heap $O(n)$; $n-1$ vezes troca raiz com o último, reduz o heap, sift-down.
- Bubble: após a passada $i$, os $i$ maiores estão nas posições finais.
- Estável + pior $O(n \log n)$: **Merge**. Array quase ordenado: **Insertion** ($O(n + \text{inversões})$).
- Partição de **Lomuto** (pivô = último): `i` marca o fim da região $\le$ pivô; no fim troca o pivô para `i+1`.
- Counting: $O(n+k)$ tempo/memória; use quando $k = O(n)$; não compara, usa valores como índices.
### 2.4 Grafos

**Representação**: Matriz $O(V^2)$ memória, aresta $(u,v)$ em $O(1)$ → densos. Lista $O(V+E)$ → esparsos. Soma dos graus $= 2E$. Máx. arestas (não dir.) $= V(V-1)/2$.

| Algoritmo | Problema | Estrutura | Tempo | Restrição |
| :---- | :---- | :---- | :---- | :---- |
| BFS | caminho mínimo **sem pesos**, níveis | **fila** | $O(V+E)$ | |
| DFS | ciclos, topológica, SCC, labirinto | **pilha**/recursão | $O(V+E)$ | |
| DFS pós-ordem invertida | ordenação topológica | pilha | $O(V+E)$ | só DAG |
| Dijkstra | caminho mínimo, fonte única | fila de prioridade | $O((V+E)\log V)$ heap; $O(V^2)$ array | **pesos $\ge 0$** |
| Bellman-Ford | caminho mínimo, fonte única | relaxa todas as arestas $V-1$ vezes | $O(V \cdot E)$ | aceita negativos; **detecta ciclo negativo** |
| Prim | MST | fila de prioridade | $O(E\log V)$; $O(V^2)$ array | conexo, não dir. |
| Kruskal | MST | ordena arestas + Union-Find | $O(E\log E) = O(E\log V)$ | não dir. |
| Floyd-Warshall | caminhos mínimos **todos os pares** | DP, $k$ no laço **externo** | $\Theta(V^3)$ | ciclo negativo ⇔ algum $d[i][i] < 0$ |
| Kahn | ordenação topológica | fila de vértices com grau de entrada 0 | $O(V+E)$ | se sobrarem vértices → ciclo |
| Kosaraju | SCC | 2 DFS | $O(V+E)$ | DFS em $G$ (tempo de término) → transpor → DFS em $G^T$ em ordem decrescente |
| Tarjan | SCC | 1 DFS + pilha + low-link | $O(V+E)$ | `low[v] == idx[v]` → desempilha um SCC |

- **Relaxamento**: se $d[u] + w(u,v) < d[v]$ então $d[v] \leftarrow d[u] + w(u,v)$.
- Bellman-Ford: se na **$V$-ésima** rodada ainda relaxar → **ciclo negativo**.
- Dijkstra falha com pesos negativos porque **finaliza** o vértice quando ele sai da fila.
- Grafo dirigido tem ciclo ⇔ DFS encontra **aresta de retorno** (back edge) para vértice ainda na pilha (cinza).
- Arestas na DFS (dirigido): árvore, retorno, avanço, cruzamento. Não dirigido: só árvore e retorno.
- Checar aresta $(u,v)$: matriz $O(1)$; lista $O(\deg u)$. BFS/DFS: lista $O(V+E)$, matriz $O(V^2)$. Esparso ($E \approx 3V$) → **lista**.
- Dijkstra: heap binário $O((V+E)\log V)$; array $O(V^2)$ (denso); heap de Fibonacci $O(E + V\log V)$.
- Caminho mínimo em **DAG** (pesos negativos ok): ordem topológica + relaxar nessa ordem, $O(V+E)$.
- **Bipartido**: BFS/DFS com 2 cores; aresta entre mesma cor → ciclo ímpar → não bipartido.
- BFS: distância de $s$ a $v$ = **nível** de $v$ na árvore BFS.
- SCC: subconjunto maximal de um grafo **dirigido** com caminho $u \to v$ e $v \to u$ para todo par.
- **MST**: $V-1$ arestas, sem ciclos. Única se todos os pesos forem distintos. Propriedade do corte: a menor aresta que cruza um corte está em alguma MST. MST ≠ árvore de caminhos mínimos.

### 2.5 P, NP, NP-Completo, NP-Difícil

- **P**: resolvível em tempo polinomial (determinístico).
- **NP**: solução **verificável** em tempo polinomial (certificado) = resolvível por máquina **não-determinística** em tempo polinomial. **P ⊆ NP**. P = NP? **em aberto**.
- **NP-Difícil**: todo problema de NP se reduz a ele. Pode não estar em NP (ex.: Problema da Parada).
- **NP-Completo** = NP ∩ NP-Difícil.
- **Redução** $A \le_p B$: "$B$ é pelo menos tão difícil quanto $A$". Para provar que **$B$ é NP-difícil**, reduza um NP-completo **conhecido $A$ para $B$**.
- Para provar NP-completo: (1) $B \in$ NP; (2) NPC conhecido $\le_p B$.
- **NPC**: SAT, 3-SAT, Clique, Vertex Cover, Conjunto Independente, Ciclo Hamiltoniano, TSP (decisão), Mochila (decisão), Subset Sum, 3-Coloração, Set Cover.
- **Em P**: 2-SAT, caminho mínimo, MST, ciclo **Euleriano**, 2-coloração (bipartido), emparelhamento, programação linear, primalidade.
- **Cook-Levin (1971)**: SAT é o primeiro NP-completo. Se **algum** NPC tiver algoritmo polinomial → **P = NP**.
- Se $A \le_p B$ e $B \in$ P, então $A \in$ P. Se P ≠ NP, nenhum NPC está em P.
- Euleriano ∈ P: existe sse conexo e todos os graus pares. Hamiltoniano é NPC. TSP **decisão** ("rota de custo $\le K$?") é NPC; a versão de otimização é NP-difícil.
- Conjunto independente $S$ em $G$ ⇔ $S$ é clique em $\bar G$ ⇔ $V \setminus S$ é vertex cover.
- Vertex Cover 2-aproximação: emparelhamento maximal, inclui os dois extremos de cada aresta.
- 2-SAT ∈ P via grafo de implicação + SCC.
- Mochila DP $O(nW)$ é **pseudo-polinomial** (polinomial no **valor** de $W$, exponencial nos bits).
- Na prática: heurísticas, aproximação (Vertex Cover 2-aprox), DP pseudo-polinomial, Branch & Bound.

### 2.6 Autômatos e Expressões Regulares

- **AFD**: $M = (Q, \Sigma, \delta, q_0, F)$, $\delta: Q \times \Sigma \to Q$ (exatamente uma transição por par).
- **AFN**: $\delta: Q \times (\Sigma \cup \{\epsilon\}) \to \mathcal{P}(Q)$. Aceita se **algum** caminho termina em estado final.
- Precedência regex: `*` > concatenação > `|`.
- Regulares são **fechadas** sob: união, concatenação, estrela, **complemento**, **interseção**, diferença, reverso.
- **Lema do Bombeamento**: se $L$ é regular, $\exists p$ tal que toda $w \in L$, $|w| \ge p$, se divide em $w = xyz$ com $|xy| \le p$, $|y| \ge 1$ e $xy^iz \in L$ $\forall i \ge 0$. Serve **só para provar que NÃO é regular** (condição necessária, não suficiente).
- **Não regulares**: $a^nb^n$, parênteses balanceados, palíndromos, $ww$, $a^{n^2}$, $a^p$ ($p$ primo).
- **Regulares**: toda linguagem **finita**; $a^*b^*$; $a^nb^m$; nº par de $a$; termina em `01`; binários divisíveis por 3.
- AFD mínimo é **único** (a menos de renomear estados; Myhill-Nerode).
- AFN → AFD (construção de subconjuntos): até $2^n$ estados (limite atingível). **ε-fecho** = estados alcançáveis só com transições ε (inclui o próprio).
- Regex → AFN-ε: construção de **Thompson**. **Kleene**: regular ⇔ regex ⇔ autômato finito.
- Complemento: inverte os finais do AFD. Interseção: autômato produto.
- Livres de contexto ($a^nb^n$, parênteses) → **autômato de pilha**. Parênteses balanceados **não** são regulares (exigem contagem ilimitada).
- AFD para binários divisíveis por 3: 3 estados (resto), $\delta(r,b) = (2r+b) \bmod 3$, inicial e final = resto 0.
- Regex: `ab*|c` = `(a(b*))|c`; "contém `11`" = `(0|1)*11(0|1)*`. Analisadores léxicos usam autômatos finitos.

---

## 3. Pegadinhas (leia antes de entrar)

1. **$\Omega$ ≠ melhor caso** e **$O$ ≠ pior caso**: são limites. "Insertion Sort é $O(n^2)$" e "é $\Omega(n)$" são ambas verdadeiras.
2. **Build-heap é $O(n)$**, não $O(n \log n)$.
3. **Heap não serve para busca**: buscar elemento arbitrário é $O(n)$.
4. **Array dinâmico**: append é $O(1)$ **amortizado** (dobrando capacidade). Se crescer por constante, vira $O(n)$ amortizado.
5. **Hash é $O(1)$ médio**, $O(n)$ pior caso.
6. **BST não balanceada** com inserção ordenada → $O(n)$.
7. **Quick Sort pior caso $O(n^2)$**; não é estável. **Merge Sort** é estável mas usa $O(n)$ extra. **Heap Sort** é in-place e $O(n \log n)$ garantido, mas **não estável**.
8. Comparação: nada abaixo de $\Omega(n \log n)$. Counting "fura" porque **não comparam**.
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
