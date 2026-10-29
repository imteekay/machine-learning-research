# Deep Dive — Maestria em todos os tópicos

> Estrutura de cada tópico: **intuição → definição → algoritmo → complexidade (com o porquê) → exemplo passo a passo → aplicações → pegadinhas**.
> Para revisão rápida use `cheatsheet.md`; para treino use `quiz.md`. No final há a seção **"Correções em relação às notas"**.

---

## Sumário

1. Estruturas de Dados
   - 1.1 Arrays, Listas Ligadas, Pilhas, Filas e Listas de Prioridade
   - 1.2 Recursão
   - 1.3 Tabelas de Espalhamento
   - 1.4 Árvores Binárias, BST, AVL, Rubro-Negra
   - 1.5 Union-Find
2. Algoritmos e Linguagens Formais
   - 2.1 Complexidade e Notação Assintótica
   - 2.2 Ordenação e Seleção
   - 2.3 Grafos
   - 2.4 NP-Completude
   - 2.5 Autômatos Finitos e Expressões Regulares

---

## 1. Estruturas de Dados

### 1.1 Arrays, Listas Ligadas, Pilhas, Filas e Listas de Prioridade

#### Arrays estáticos e dinâmicos

**Intuição:** um bloco contíguo de memória. O endereço do elemento $i$ é `base + i * tamanho`, então qualquer posição é acessada em uma única conta.

**Array dinâmico** (ArrayList, `vector`, `list` do Python): mantém `capacidade ≥ tamanho`. Quando cheio, aloca um bloco novo (tipicamente **o dobro**), copia tudo e libera o antigo.

**Análise amortizada do append (dobrando):**
- Começando com capacidade 1 e fazendo $n$ appends, as cópias custam $1 + 2 + 4 + \dots + 2^{\lfloor \log n \rfloor} < 2n$.
- Somando as $n$ escritas: total $< 3n$ → **$O(1)$ amortizado** por append.
- Uma operação isolada pode custar $O(n)$ (a que dispara a realocação).
- Se crescesse por **constante** $c$ (ex.: +10), as cópias seriam $c + 2c + 3c + \dots \approx n^2/(2c)$ → $O(n)$ amortizado. Por isso se cresce **geometricamente**.

| Operação | Custo |
| :---- | :---- |
| Acesso por índice | $O(1)$ |
| Busca por valor | $O(n)$; ordenado: $O(\log n)$ |
| Append | $O(1)$ amortizado, $O(n)$ pior caso |
| Inserir/remover no meio | $O(n)$ (shift) |
| Remover do fim | $O(1)$ |

**Busca binária** (array ordenado):

```
function binarySearch(arr, x) {
  let lo = 0, hi = arr.length - 1;
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (arr[mid] === x) return mid;
    if (arr[mid] < x) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}
```

Cada iteração descarta metade → $T(n) = T(n/2) + O(1) = O(\log n)$. Exige **acesso aleatório** — em lista ligada não compensa.

**Vantagens práticas:** localidade de cache (elementos vizinhos na memória), pouco overhead. **Desvantagem:** inserções no meio e realocações.

#### Listas Ligadas

**Variações:**
- **Simples:** nó = (valor, `next`).
- **Dupla:** nó = (valor, `prev`, `next`) — permite remover um nó conhecido em $O(1)$ e percorrer nos dois sentidos.
- **Circular:** o último aponta para o primeiro (útil em round-robin, buffers).
- **Com sentinela (nó cabeça fictício):** elimina casos especiais de lista vazia.

| Operação | Simples (head) | Simples (head+tail) | Dupla (head+tail) |
| :---- | :---- | :---- | :---- |
| Acesso ao $i$-ésimo | $O(n)$ | $O(n)$ | $O(n)$ |
| Inserir no início | $O(1)$ | $O(1)$ | $O(1)$ |
| Inserir no fim | $O(n)$ | $O(1)$ | $O(1)$ |
| Remover do início | $O(1)$ | $O(1)$ | $O(1)$ |
| Remover do fim | $O(n)$ | $O(n)$ ⚠️ | $O(1)$ |
| Inserir/remover após nó conhecido | $O(1)$ | $O(1)$ | $O(1)$ |
| Remover nó conhecido | $O(n)$ (achar anterior) | $O(n)$ | $O(1)$ |

⚠️ Mesmo com `tail`, remover o último da lista **simples** exige achar o penúltimo.

**Técnicas clássicas:**
- **Inverter lista** em $O(n)$ / $O(1)$ memória: três ponteiros (`prev`, `curr`, `next`).
- **Meio da lista**: ponteiro lento (1 passo) e rápido (2 passos); quando o rápido chega ao fim, o lento está no meio.
- **Ciclo (Floyd, tartaruga e lebre)**: se lento e rápido se encontram, há ciclo. $O(n)$ tempo, $O(1)$ memória.

**Array vs Lista:** array ganha em acesso aleatório e cache; lista ganha em inserções/remoções em posições já conhecidas e não precisa de realocação.

#### Pilhas (LIFO)

```
function push(x)  { top++; data[top] = x; }
function pop()    { return data[top--]; }
function peek()   { return data[top]; }
```

Tudo $O(1)$ (array dinâmico: push $O(1)$ amortizado).

**Aplicações e algoritmos importantes:**

1. **Parênteses balanceados:** empilha aberturas; ao ver fechamento, desempilha e confere o par. Válido se terminar com pilha vazia e nunca desempilhar vazio.
2. **Avaliação de pós-fixa:** operando → empilha; operador → desempilha dois (`b = pop`, `a = pop`), empilha `a op b`.
   - `5 1 2 + 4 * + 3 -` → 5, (1+2)=3, 3·4=12, 5+12=17, 17−3 = **14**.
3. **Infixa → pós-fixa (Shunting Yard, Dijkstra):**
   - Operando → saída.
   - `(` → empilha.
   - `)` → desempilha para a saída até `(`.
   - Operador $o$ → enquanto o topo for operador de precedência **maior ou igual** (para associatividade à esquerda), desempilha para a saída; depois empilha $o$.
   - No fim, desempilha tudo.
   - Exemplo: `A + B * C - D` → `A B C * + D -`.
   - Exemplo: `(A + B) * (C - D)` → `A B + C D - *`.
4. **Call stack:** cada chamada empilha um *frame* (parâmetros, variáveis locais, endereço de retorno).
5. **DFS iterativa, undo/redo, backtracking.**

#### Filas (FIFO) e Deques

**Array circular** (capacidade $m$):

```
function enqueue(x) { data[rear] = x; rear = (rear + 1) % m; size++; }
function dequeue()  { const x = data[front]; front = (front + 1) % m; size--; return x; }
```

Cheia quando `size == m` (ou, sem contador, quando `(rear + 1) % m == front`, desperdiçando uma posição). Evita o shift $O(n)$ de um array comum.

**Com lista ligada:** enqueue no `tail`, dequeue no `head` → ambos $O(1)$.

**Deque:** inserção/remoção $O(1)$ nas duas pontas (lista dupla ou array circular). Útil em janela deslizante (máximo da janela em $O(n)$).

**Fila com duas pilhas:** `in` recebe enqueues; dequeue tira de `out`; se `out` vazia, despeja `in` inteira em `out`. Cada elemento é movido no máximo uma vez → $O(1)$ amortizado.

**Aplicações:** BFS, escalonamento round-robin, buffers (impressora, rede, produtor/consumidor).

#### Listas de Prioridade e Heaps

**TAD Fila de Prioridade:** `insert(x, p)`, `extract-max/min`, `peek`, opcional `decrease-key`.

| Implementação | insert | extract | peek |
| :---- | :---- | :---- | :---- |
| Array não ordenado | $O(1)$ | $O(n)$ | $O(n)$ |
| Array ordenado | $O(n)$ | $O(1)$ | $O(1)$ |
| **Heap binário** | $O(\log n)$ | $O(\log n)$ | $O(1)$ |
| BST balanceada | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ ou $O(1)$ |

**Heap binário:** árvore binária **completa** (todos os níveis cheios exceto talvez o último, preenchido da esquerda para a direita) + **propriedade de heap** (max-heap: pai ≥ filhos). Por ser completa, cabe num array sem ponteiros:

- Base 0: `left(i) = 2i+1`, `right(i) = 2i+2`, `parent(i) = floor((i-1)/2)`.
- Base 1: `left(i) = 2i`, `right(i) = 2i+1`, `parent(i) = floor(i/2)`.
- Altura $\lfloor \log_2 n \rfloor$. Folhas nos índices $\lfloor n/2 \rfloor .. n-1$ (base 0).

```
function siftUp(i) {
  while (i > 0 && heap[parent(i)] < heap[i]) {
    swap(i, parent(i));
    i = parent(i);
  }
}

function siftDown(i, n) {
  while (true) {
    let largest = i;
    const l = 2 * i + 1, r = 2 * i + 2;
    if (l < n && heap[l] > heap[largest]) largest = l;
    if (r < n && heap[r] > heap[largest]) largest = r;
    if (largest === i) return;
    swap(i, largest);
    i = largest;
  }
}

function insert(x) { heap.push(x); siftUp(heap.length - 1); }

function extractMax() {
  const max = heap[0];
  heap[0] = heap.pop();      // último vai para a raiz
  siftDown(0, heap.length);
  return max;
}

function buildHeap() {
  for (let i = Math.floor(heap.length / 2) - 1; i >= 0; i--) siftDown(i, heap.length);
}
```

**Por que build-heap é $O(n)$:** há $\le \lceil n/2^{h+1} \rceil$ nós de altura $h$, e cada sift-down custa $O(h)$:
$\sum_{h=0}^{\log n} \frac{n}{2^{h+1}} \cdot h = n \sum \frac{h}{2^{h+1}} = O(n)$ (a série converge para 1).
Inserir um por um, por outro lado, custa $O(n \log n)$.

**Exemplo (min-heap):** inserir 5, 3, 8, 1, 9 → `[1, 3, 8, 5, 9]`. `extract-min` → `[3, 5, 8, 9]` (ver `quiz.md` Q5/Q6 para o passo a passo).

**Por que não serve para busca:** a ordem só vale entre pai e filho; um valor pode estar em qualquer subárvore → $O(n)$.

**Aplicações:** Dijkstra, Prim, Huffman (junta os dois menos frequentes repetidamente, $O(n \log n)$), Heap Sort, escalonamento por prioridade, mediana em fluxo (dois heaps), top-$k$ ($O(n \log k)$), merge de $k$ listas ordenadas ($O(n \log k)$).

---

### 1.2 Recursão

**Definição:** uma função que resolve o problema chamando a si mesma em instâncias menores, até um **caso base** resolvido diretamente.

**Como a máquina executa:** cada chamada empilha um *frame* na **call stack**. Memória = $O(\text{profundidade máxima})$, não o total de chamadas.

- Fatorial: profundidade $n$ → $O(n)$ memória.
- Busca binária recursiva: profundidade $\log n$ → $O(\log n)$.
- Fibonacci ingênuo: $O(2^n)$ chamadas, mas profundidade só $n$ → $O(n)$ memória.
- Quick Sort: profundidade $O(\log n)$ médio, $O(n)$ pior.

**Árvore de recursão:** desenhe as chamadas; tempo = soma do trabalho em todos os nós.
- `fib(n)` → árvore binária de altura $n$ → $\Theta(\varphi^n)$.
- Merge Sort → $\log n$ níveis, $n$ de trabalho por nível → $n \log n$.

**Tipos:**
- **Direta:** `f` chama `f`.
- **Indireta / mútua:** `par(n) = impar(n-1)`, `impar(n) = par(n-1)`.
- **Linear** (uma chamada por ativação) vs **em árvore** (várias chamadas — Fibonacci, Hanói).
- **De cauda:** a chamada recursiva é a **última** operação; nada fica pendente.

```
// NÃO é cauda: multiplicação pendente
function fat(n) {
  if (n <= 1) return 1;
  return n * fat(n - 1);
}

// É cauda: o resultado é carregado no acumulador
function fatTail(n, acc = 1) {
  if (n <= 1) return acc;
  return fatTail(n - 1, acc * n);
}
```

**TCO (Tail Call Optimization):** o compilador reutiliza o frame atual (vira um `goto` para o início) → memória $O(1)$. Scheme exige; C/C++ com otimização costuma fazer; **Python e Java não fazem**.

**Recursão → iteração:** sempre possível com uma pilha explícita (é o que a máquina faz).

**Memoização:** guardar resultados de subproblemas repetidos. Fibonacci: $O(2^n) \to O(n)$. É a ponte para **programação dinâmica** (top-down).

**Divisão e conquista:** dividir → conquistar recursivamente → combinar. Merge Sort, Quick Sort, busca binária, Strassen, par de pontos mais próximos.

**Backtracking:** recursão que tenta escolhas e desfaz quando falham.

```
function nQueens(row) {
  if (row === n) { count++; return; }
  for (let col = 0; col < n; col++) {
    if (safe(row, col)) {
      place(row, col);
      nQueens(row + 1);
      remove(row, col);     // desfaz (backtrack)
    }
  }
}
```

**Torre de Hanói:** move $n-1$ para o auxiliar, move o maior, move $n-1$ de volta: $T(n) = 2T(n-1) + 1 = 2^n - 1$.

**Pegadinhas:**
- Profundidade ≠ número de chamadas (memória vs tempo).
- `return f(n-1) + 1` **não** é cauda.
- Sem caso base alcançável → stack overflow.

---

### 1.3 Tabelas de Espalhamento (Hash Tables)

**Ideia:** função $h: U \to \{0, \dots, m-1\}$ mapeia a chave para uma posição do array. Se não houver colisão, busca/inserção/remoção são $O(1)$.

**Colisões são inevitáveis** (princípio da casa dos pombos: $|U| \gg m$; paradoxo do aniversário: com ~$\sqrt{m}$ chaves já é provável haver colisão).

#### Funções hash

- **Divisão:** $h(k) = k \bmod m$. Escolher $m$ **primo** longe de potência de 2 (se $m = 2^p$, só os $p$ bits baixos importam).
- **Multiplicação:** $h(k) = \lfloor m \cdot (kA \bmod 1) \rfloor$, $0 < A < 1$ (Knuth sugere $A \approx (\sqrt5 - 1)/2$). $m$ pode ser potência de 2.
- **Universal:** escolher $h$ aleatoriamente de uma família $H$ com $\Pr[h(x) = h(y)] \le 1/m$ para $x \ne y$. Ex.: $h_{a,b}(k) = ((ak + b) \bmod p) \bmod m$. Protege contra entradas adversárias.
- Boa função: determinística, rápida, distribui uniformemente.

#### Encadeamento separado (chaining)

Cada posição guarda uma lista. Sob hash uniforme simples, com $\alpha = n/m$:
- Busca sem sucesso: $\Theta(1 + \alpha)$; com sucesso: $\Theta(1 + \alpha/2)$.
- Se $m = \Theta(n)$ → $\alpha = O(1)$ → **$O(1)$ esperado**.
- Pior caso: todas na mesma lista → $O(n)$. (Java 8 converte buckets grandes em árvores → $O(\log n)$.)
- $\alpha$ pode passar de 1.

#### Endereçamento aberto

Tudo dentro do array; em colisão, **sondar** uma sequência $h(k, 0), h(k, 1), \dots$

| Sondagem | Fórmula | Problema |
| :---- | :---- | :---- |
| Linear | $(h(k) + i) \bmod m$ | **agrupamento primário** (blocos contíguos crescem) |
| Quadrática | $(h(k) + c_1 i + c_2 i^2) \bmod m$ | agrupamento **secundário** (mesma $h(k)$ → mesma sequência); pode não visitar todas as posições |
| Double hashing | $(h_1(k) + i \cdot h_2(k)) \bmod m$ | melhor; $h_2(k) \ne 0$ e coprimo com $m$ |

- Exige $\alpha < 1$. Nº esperado de sondagens em busca sem sucesso $\le \frac{1}{1-\alpha}$ (hash uniforme). Com $\alpha = 0{,}5$ → 2; com $\alpha = 0{,}9$ → 10.
- **Remoção:** marcar como **"removido" (tombstone)**. A busca continua passando por ele; a inserção pode reutilizá-lo.

**Exemplo (linear, $m = 7$):** inserir 10, 17, 24, 3 (todos $\equiv 3$) → posições 3, 4, 5, 6. Buscar 3 custa 4 sondagens: o agrupamento primário em ação.

#### Rehash

Quando $\alpha$ excede um limite (Java: 0,75), cria array ~2× maior e reinsere tudo (as posições mudam porque $m$ mudou). Custo $O(n)$ pontual, $O(1)$ amortizado (mesmo argumento do array dinâmico).

#### Aplicações

Dicionários/mapas, conjuntos, tabela de símbolos de compiladores, índices hash de banco de dados (consulta por igualdade), cache, contagem de frequência, deduplicação, **LRU Cache** (hash + lista duplamente ligada → get/put $O(1)$).

**Hash vs BST balanceada:** hash tem $O(1)$ esperado, mas **não ordena** (sem mínimo, sucessor, consulta por intervalo); BST tem $O(\log n)$ garantido e mantém ordem.

---

### 1.4 Árvores Binárias e de Busca

#### Vocabulário

- **Raiz**, **folha** (sem filhos), **nó interno**.
- **Profundidade** de um nó: nº de arestas da raiz até ele. **Altura** de um nó: nº de arestas no maior caminho até uma folha. Altura da árvore = altura da raiz.
- **Árvore binária cheia (full):** todo nó tem 0 ou 2 filhos → folhas = internos + 1.
- **Completa:** todos os níveis cheios, exceto o último, preenchido da esquerda (heaps).
- **Perfeita:** todos os níveis cheios → $2^{h+1} - 1$ nós.
- Altura mínima com $n$ nós: $\lfloor \log_2 n \rfloor$. Máxima: $n - 1$.

#### Travessias

```
function preOrder(node)  { if (!node) return; visit(node); preOrder(node.left); preOrder(node.right); }
function inOrder(node)   { if (!node) return; inOrder(node.left); visit(node); inOrder(node.right); }
function postOrder(node) { if (!node) return; postOrder(node.left); postOrder(node.right); visit(node); }

function levelOrder(root) {
  const queue = [root];
  while (queue.length) {
    const node = queue.shift();
    visit(node);
    if (node.left) queue.push(node.left);
    if (node.right) queue.push(node.right);
  }
}
```

Todas $O(n)$. Memória: recursivas $O(h)$; por nível $O(w)$ (largura máxima).

**Exemplo** — inserir 50, 30, 70, 20, 40, 60, 80:

```
        50
      /    \
    30      70
   /  \    /  \
  20  40  60  80
```

- Pré: 50 30 20 40 70 60 80
- Em: 20 30 40 50 60 70 80 (ordenado!)
- Pós: 20 40 30 60 80 70 50
- Nível: 50 30 70 20 40 60 80

**Usos:** pré-ordem → copiar/serializar árvore; em-ordem → ordem crescente em BST; pós-ordem → liberar memória, avaliar árvore de expressão (gera pós-fixa); nível → menor profundidade, impressão por camadas.

**Reconstrução:** pré + em (ou pós + em) determinam a árvore unicamente. Pré + pós não (exceto em árvores cheias).

#### BST (Árvore Binária de Busca)

**Invariante:** para todo nó, chaves da subárvore esquerda < chave < chaves da subárvore direita.

```
function search(node, x) {
  if (!node || node.key === x) return node;
  return x < node.key ? search(node.left, x) : search(node.right, x);
}

function insert(node, x) {
  if (!node) return new Node(x);
  if (x < node.key) node.left = insert(node.left, x);
  else node.right = insert(node.right, x);
  return node;
}
```

- **Mínimo:** desce à esquerda. **Máximo:** desce à direita.
- **Sucessor de $x$:** se $x$ tem subárvore direita → mínimo dela; senão, sobe até o primeiro ancestral do qual $x$ está na subárvore esquerda.

**Remoção (3 casos):**
1. **Folha:** remove.
2. **Um filho:** o filho sobe para o lugar do nó.
3. **Dois filhos:** copia o **sucessor em-ordem** (mínimo da subárvore direita, que tem no máximo 1 filho — direito) para o nó e remove o sucessor (cai no caso 1 ou 2). Alternativa simétrica: predecessor (máximo da esquerda).

Exemplo: remover 50 da árvore acima → sucessor 60 vira raiz; 70 fica sem filho esquerdo.

**Complexidade:** tudo $O(h)$.
- Inserção aleatória → altura esperada $O(\log n)$.
- Inserção **ordenada** → árvore degenerada (lista) → $h = n - 1$ → $O(n)$.
- Número de BSTs distintas com $n$ chaves: número de Catalan $C_n = \frac{1}{n+1}\binom{2n}{n}$.

**Tree Sort:** inserir tudo e fazer em-ordem → $O(n \log n)$ se balanceada, $O(n^2)$ no pior caso.

#### AVL

**Invariante:** para todo nó, $FB = h(\text{esq}) - h(\text{dir}) \in \{-1, 0, +1\}$. Altura $\le \approx 1{,}44 \log_2 n$.

Após inserir/remover, sobe atualizando alturas; no primeiro nó com $|FB| = 2$, aplica a rotação:

| Caso | Situação | Correção |
| :---- | :---- | :---- |
| LL | FB(nó) = +2, FB(filho esq) ≥ 0 | rotação simples à **direita** no nó |
| RR | FB(nó) = −2, FB(filho dir) ≤ 0 | rotação simples à **esquerda** no nó |
| LR | FB(nó) = +2, FB(filho esq) = −1 | esquerda no filho esq, depois direita no nó |
| RL | FB(nó) = −2, FB(filho dir) = +1 | direita no filho dir, depois esquerda no nó |

**Rotação à direita em $y$** (com filho esquerdo $x$):

```
      y                x
     / \              / \
    x   C    ==>     A   y
   / \                  / \
  A   B                B   C
```

A ordem em-ordem (A x B y C) é preservada — por isso a rotação mantém a propriedade de BST. Custo $O(1)$.

**Exemplos:**
- Inserir 1, 2, 3 → caso RR em 1 → rotação à esquerda → raiz 2.
- Inserir 3, 1, 2 → caso LR em 3 → esquerda em 1, direita em 3 → raiz 2.
- Inserir 10, 20, 30, 40, 50: após 30 → raiz 20; após 50 → RR em 30 → `20(10, 40(30, 50))`.

Inserção: no máximo **uma** rotação (simples ou dupla) restaura tudo. Remoção: pode precisar de $O(\log n)$ rotações.

#### Rubro-Negra (Red-Black)

**Regras:**
1. Todo nó é vermelho ou preto.
2. Raiz é preta.
3. Folhas (NIL) são pretas.
4. Nó vermelho tem filhos pretos (nunca dois vermelhos seguidos).
5. Todo caminho de um nó até suas folhas tem o mesmo número de nós pretos (**altura negra**).

**Consequência:** o caminho mais longo tem no máximo 2× o mais curto → altura $\le 2\log_2(n+1)$.

Inserção: novo nó entra **vermelho** (não viola regra 5); corrige violações da regra 4 com **recoloração** (tio vermelho) ou **rotações** (tio preto) — no máximo 2 rotações. Remoção: no máximo 3 rotações.

**AVL vs RB:** AVL mais baixa → buscas mais rápidas (bom para leitura intensa). RB menos rotações → inserções/remoções mais rápidas. RB é usada em `std::map`, `TreeMap` do Java, escalonador CFS do Linux.

**Bônus — B-trees:** árvores de busca com muitos filhos por nó, altas taxas de ramificação → poucas leituras de disco. Usadas em índices de bancos de dados e sistemas de arquivos.

---

### 1.5 Union-Find (Conjuntos Disjuntos)

**Problema:** manter uma partição de $\{1..n\}$ sob `union(a, b)` e responder `find(a) == find(b)?`.

**Representação:** floresta; cada conjunto é uma árvore; a raiz é o representante; `parent[raiz] = raiz`.

```
function makeSet(x) { parent[x] = x; rank[x] = 0; }

function find(x) {
  if (parent[x] !== x) parent[x] = find(parent[x]);   // compressão de caminho
  return parent[x];
}

function union(a, b) {
  let ra = find(a), rb = find(b);
  if (ra === rb) return false;                          // já no mesmo conjunto
  if (rank[ra] < rank[rb]) [ra, rb] = [rb, ra];
  parent[rb] = ra;                                      // menor rank sob o maior
  if (rank[ra] === rank[rb]) rank[ra]++;
  return true;
}
```

**Por que cada otimização ajuda:**
- **Sem otimizações:** `union(1,2), union(2,3), …` pode formar uma corrente → `find` $O(n)$.
- **União por rank (ou tamanho):** uma árvore de rank $r$ tem $\ge 2^r$ nós → rank $\le \log n$ → `find` $O(\log n)$ pior caso.
- **Compressão de caminho** sozinha: $O(\log n)$ amortizado.
- **Ambas:** $O(\alpha(n))$ amortizado (Tarjan). $\alpha$ é a inversa da função de Ackermann: $\alpha(n) \le 4$ para $n$ menor que o número de átomos do universo.

**Rank ≠ altura** depois de compressões (é um limite superior da altura).

**Aplicações:**
- **Kruskal** (MST).
- Componentes conexos em grafo que só **ganha** arestas (incremental).
- Detecção de ciclo em grafo **não direcionado**.
- Percolação, segmentação de imagens, equivalência de tipos (unificação), "contas de usuário mescladas".

**Limitação:** não suporta "desunir" (split) eficientemente.

---

## 2. Algoritmos e Linguagens Formais

### 2.1 Complexidade Algorítmica e Notação Assintótica

**Modelo:** contamos operações elementares (comparações, atribuições, acessos) em função do tamanho da entrada $n$. Constantes e termos de ordem menor são ignorados.

#### Definições formais

- $f(n) = O(g(n))$ ⇔ $\exists c > 0, n_0 > 0$ tais que $0 \le f(n) \le c\,g(n)$ $\forall n \ge n_0$. (**limite superior**)
- $f(n) = \Omega(g(n))$ ⇔ $\exists c > 0, n_0$ tais que $f(n) \ge c\,g(n)$ $\forall n \ge n_0$. (**limite inferior**)
- $f(n) = \Theta(g(n))$ ⇔ $f = O(g)$ e $f = \Omega(g)$. (**limite justo**)
- $f(n) = o(g(n))$ ⇔ **para todo** $c > 0$ existe $n_0$… ⇔ $\lim f/g = 0$.
- $f(n) = \omega(g(n))$ ⇔ $\lim f/g = \infty$.

**Teste do limite:** se $\lim_{n\to\infty} f(n)/g(n) = L$:
- $L = 0$ → $f = o(g)$ (logo $O$);
- $0 < L < \infty$ → $f = \Theta(g)$;
- $L = \infty$ → $f = \omega(g)$ (logo $\Omega$).

**Exemplo:** $3n^2 + 10n = \Theta(n^2)$. Prova de $O$: para $n \ge 10$, $10n \le n^2$, então $3n^2 + 10n \le 4n^2$ → $c = 4$, $n_0 = 10$.

#### Notação ≠ caso

Melhor, médio e pior **caso** são funções diferentes; cada uma pode ser descrita com $O$, $\Omega$, $\Theta$.

| Insertion Sort | Melhor caso | Pior caso |
| :---- | :---- | :---- |
| Tempo | $\Theta(n)$ | $\Theta(n^2)$ |

Logo "Insertion Sort é $O(n^2)$" (para toda entrada) e "Insertion Sort é $\Omega(n)$" (para toda entrada) são verdadeiras; "Insertion Sort é $\Theta(n^2)$" só vale para o pior caso.

#### Hierarquia

$1 \prec \log^* n \prec \log\log n \prec \log n \prec \log^k n \prec n^{\epsilon} \prec \sqrt{n} \prec n \prec n\log n \prec n^2 \prec n^3 \prec 2^n \prec 3^n \prec n! \prec n^n$

Fatos úteis:
- Qualquer polilog é $o$ de qualquer polinômio: $\log^{100} n = o(n^{0{,}01})$.
- Qualquer polinômio é $o$ de qualquer exponencial: $n^{100} = o(1{,}01^n)$.
- $\log_a n = \Theta(\log_b n)$; mas $2^n \ne \Theta(3^n)$ e $2^{2n} = 4^n \ne O(2^n)$.
- $\log(n!) = \Theta(n \log n)$; $n! = o(n^n)$; $n! = \omega(2^n)$.

#### Somatórios

- $\sum_{i=1}^n i = \frac{n(n+1)}{2} = \Theta(n^2)$; $\sum i^2 = \Theta(n^3)$.
- Geométrica: $\sum_{i=0}^{k} r^i = \frac{r^{k+1}-1}{r-1}$; se $r > 1$, domina o último termo; se $r < 1$, $\Theta(1)$.
- Harmônica: $\sum_{i=1}^n 1/i = \ln n + O(1)$.

#### Analisando laços

```
for i in 1..n:            // n
  for j in 1..n:          // × n            → Θ(n²)

for i in 1..n:
  for j in 1..i:          // Σ i = n²/2     → Θ(n²)

for i in 1..n:
  j = 1
  while j < n: j *= 2     // log n          → Θ(n log n)

i = n
while i > 0: i /= 2       //                → Θ(log n)

for i in 1..n:
  j = i
  while j < n: j += i     // n/i            → Σ n/i = Θ(n log n) (harmônica)
```

#### Recorrências

**Teorema Mestre:** $T(n) = aT(n/b) + f(n)$, $a \ge 1$, $b > 1$. Seja $n^{\log_b a}$ o "custo das folhas".

1. $f(n) = O(n^{\log_b a - \epsilon})$ → $T(n) = \Theta(n^{\log_b a})$.
2. $f(n) = \Theta(n^{\log_b a})$ → $T(n) = \Theta(n^{\log_b a}\log n)$. (Generalização: $f = \Theta(n^{\log_b a} \log^k n)$ → $\Theta(n^{\log_b a}\log^{k+1} n)$.)
3. $f(n) = \Omega(n^{\log_b a + \epsilon})$ e $a f(n/b) \le c f(n)$ para algum $c < 1$ → $T(n) = \Theta(f(n))$.

**Lacuna:** $T(n) = 2T(n/2) + n/\log n$ não cai em nenhum caso padrão (diferença não é polinomial) — resultado $\Theta(n \log\log n)$.

**Não se aplica** a $T(n) = T(n-1) + \dots$ (subtrai, não divide). Use **expansão/iteração**:
- $T(n) = T(n-1) + c$ → $\Theta(n)$.
- $T(n) = T(n-1) + n$ → $\Theta(n^2)$.
- $T(n) = 2T(n-1) + c$ → $\Theta(2^n)$.
- $T(n) = T(n-1) + T(n-2) + c$ → $\Theta(\varphi^n)$.

**Árvore de recursão** para $T(n) = 2T(n/2) + n$: nível $i$ tem $2^i$ nós de custo $n/2^i$ → $n$ por nível × $\log n$ níveis = $n \log n$.

**Método da substituição:** chuta a forma e prova por indução. Ex.: $T(n) \le cn\log n$ para $T(n) = 2T(n/2) + n$:
$T(n) \le 2c(n/2)\log(n/2) + n = cn\log n - cn + n \le cn \log n$ se $c \ge 1$.

#### Análise amortizada

Custo médio por operação **no pior caso de uma sequência** (não é probabilístico!). Métodos: agregado, contábil, potencial. Exemplos: array dinâmico, fila com duas pilhas, Union-Find, contador binário (incremento $O(1)$ amortizado).

#### Complexidade de espaço

Memória **extra** (auxiliar) além da entrada, inclusive pilha de recursão. Ex.: Merge Sort $O(n)$, Quick Sort $O(\log n)$, Heap Sort $O(1)$.

---

### 2.2 Ordenação e Seleção

#### Propriedades

- **Estável:** chaves iguais mantêm a ordem relativa. Importa quando se ordena por várias chaves (ex.: Radix).
- **In-place:** memória extra $O(1)$ (às vezes aceita-se $O(\log n)$ de pilha).
- **Adaptativo:** mais rápido em entrada quase ordenada (Insertion, Bubble com flag).
- **Inversão:** par $(i, j)$ com $i < j$ e $A[i] > A[j]$. Máximo $n(n-1)/2$.

#### Bubble Sort

Percorre trocando pares adjacentes fora de ordem; após a passada $k$, os $k$ maiores estão no fim.

```
for (let i = 0; i < n - 1; i++) {
  let swapped = false;
  for (let j = 0; j < n - 1 - i; j++)
    if (a[j] > a[j + 1]) { swap(j, j + 1); swapped = true; }
  if (!swapped) break;                 // otimização: melhor caso O(n)
}
```

Estável, in-place. $\Theta(n^2)$ médio/pior; $\Theta(n)$ melhor com flag. Trocas = inversões.

#### Selection Sort

Para cada posição $i$, acha o **mínimo** de $a[i..n-1]$ e troca com $a[i]$ (o menor vai para o **início**).

- Sempre $\Theta(n^2)$ comparações (não adaptativo).
- No máximo $n-1$ trocas → bom quando escrever é caro.
- **Não estável** (a troca pode pular sobre um igual): `[2a, 2b, 1]` → `[1, 2b, 2a]`.

#### Insertion Sort

Mantém prefixo ordenado; insere `a[i]` deslocando os maiores para a direita.

```
for (let i = 1; i < n; i++) {
  const key = a[i];
  let j = i - 1;
  while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
  a[j + 1] = key;
}
```

- Tempo $\Theta(n + I)$, $I$ = nº de inversões → melhor $\Theta(n)$ (ordenado), pior $\Theta(n^2)$ (invertido), médio $\Theta(n^2)$.
- Estável, in-place, adaptativo, **online** (ordena à medida que recebe).
- Usado para subarrays pequenos dentro de Timsort/Introsort.

#### Merge Sort

```
function mergeSort(a) {
  if (a.length <= 1) return a;
  const mid = Math.floor(a.length / 2);
  return merge(mergeSort(a.slice(0, mid)), mergeSort(a.slice(mid)));
}

function merge(l, r) {
  const out = [];
  let i = 0, j = 0;
  while (i < l.length && j < r.length)
    out.push(l[i] <= r[j] ? l[i++] : r[j++]);   // "<=" garante estabilidade
  return out.concat(l.slice(i), r.slice(j));
}
```

- $T(n) = 2T(n/2) + \Theta(n) = \Theta(n \log n)$ em **todos** os casos.
- Memória $\Theta(n)$ em arrays; $O(1)$ extra em listas ligadas.
- Estável. Ideal para **ordenação externa** (dados em disco) e listas ligadas.
- Variante: contar inversões em $O(n \log n)$ durante o merge.

#### Quick Sort

```
function quickSort(a, lo, hi) {
  if (lo >= hi) return;
  const p = partition(a, lo, hi);
  quickSort(a, lo, p - 1);
  quickSort(a, p + 1, hi);
}

// Lomuto: pivô = último
function partition(a, lo, hi) {
  const pivot = a[hi];
  let i = lo - 1;
  for (let j = lo; j < hi; j++)
    if (a[j] <= pivot) { i++; swap(i, j); }
  swap(i + 1, hi);
  return i + 1;
}
```

**Hoare** (a versão das suas notas): dois ponteiros vindo das pontas, trocam elementos fora de lugar até se cruzarem. Faz ~3× menos trocas que Lomuto e lida melhor com repetidos.

**Exemplo Lomuto** `[3, 7, 8, 5, 2, 1, 9, 5, 4]`, pivô 4 → `[3, 2, 1, 4, 7, 8, 9, 5, 5]`, pivô na posição 3 (passo a passo em `quiz.md` Q25).

**Análise:**
- Melhor/médio: partições balanceadas → $T(n) = 2T(n/2) + n = \Theta(n \log n)$. Mesmo partições 9:1 dão $\Theta(n \log n)$.
- Pior: pivô sempre extremo → $T(n) = T(n-1) + n = \Theta(n^2)$. Acontece com array **ordenado/invertido** e pivô fixo na ponta, ou todos iguais com Lomuto.
- Memória: pilha $O(\log n)$ médio, $O(n)$ pior (reduz para $O(\log n)$ recursando primeiro na parte menor).
- Não estável. Na prática é o mais rápido (constantes baixas, cache).
- **Mitigação:** pivô aleatório (esperado $O(n \log n)$ para qualquer entrada), mediana de três, partição em 3 vias para repetidos, **Introsort** (troca para Heap Sort se a profundidade passar de $2\log n$).

#### Heap Sort

```
buildMaxHeap(a);                         // O(n)
for (let end = n - 1; end > 0; end--) {
  swap(0, end);                          // maior vai para o fim
  siftDown(0, end);                      // O(log n), heap encolhe
}
```

$\Theta(n \log n)$ garantido, in-place ($O(1)$), **não estável**, pouco amigo de cache.

#### Limite inferior para ordenação por comparação

Qualquer algoritmo baseado em comparações é uma **árvore de decisão** binária; precisa de $\ge n!$ folhas (uma por permutação). Altura $\ge \log_2 n! = \Theta(n \log n)$. Logo, pior caso $\Omega(n \log n)$. Merge e Heap Sort são **ótimos**.

#### Ordenações lineares (não comparativas)

**Counting Sort** (inteiros em $[0, k]$):

```
count = array(k + 1, 0)
for x in a: count[x]++
for i in 1..k: count[i] += count[i - 1]        // posições finais (cumulativo)
for x in a (de trás para frente):               // de trás → estável
  out[--count[x]] = x
```

$\Theta(n + k)$ tempo e memória. Bom se $k = O(n)$. Estável.

**Radix Sort (LSD):** ordena por dígito, do **menos** significativo ao mais, com ordenação **estável** (Counting). $d$ dígitos, base $k$: $\Theta(d(n + k))$.
- Ex.: `170, 45, 75, 90, 802, 24, 2, 66` → unidades → dezenas → centenas → ordenado.
- $n$ inteiros em $[0, n^c)$: base $n$ → $d = c$ → $\Theta(n)$.

**Bucket Sort:** valores uniformes em $[0, 1)$: distribui em $n$ baldes, ordena cada um (Insertion), concatena. Médio $\Theta(n)$; pior $\Theta(n^2)$ (todos no mesmo balde).

#### Seleção

- Mínimo/máximo: $n - 1$ comparações. Ambos juntos: $\lceil 3n/2 \rceil$ (comparar em pares).
- **Quickselect** ($k$-ésimo menor):

```
function quickSelect(a, lo, hi, k) {
  const p = partition(a, lo, hi);
  if (p === k) return a[p];
  return k < p ? quickSelect(a, lo, p - 1, k) : quickSelect(a, p + 1, hi, k);
}
```

Médio $T(n) = T(n/2) + n = \Theta(n)$; pior $\Theta(n^2)$.

- **Mediana das medianas (BFPRT):** grupos de 5, mediana de cada, mediana delas recursivamente como pivô → garante partição ≥ 30/70 → $T(n) \le T(n/5) + T(7n/10) + O(n) = \Theta(n)$ **pior caso**.
- Alternativas: ordenar e pegar ($O(n \log n)$); heap de tamanho $k$ ($O(n \log k)$).

#### Resumo

| Algoritmo | Melhor | Médio | Pior | Memória | Estável | In-place |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Bubble | $n$ | $n^2$ | $n^2$ | 1 | ✅ | ✅ |
| Selection | $n^2$ | $n^2$ | $n^2$ | 1 | ❌ | ✅ |
| Insertion | $n$ | $n^2$ | $n^2$ | 1 | ✅ | ✅ |
| Merge | $n\log n$ | $n\log n$ | $n\log n$ | $n$ | ✅ | ❌ |
| Quick | $n\log n$ | $n\log n$ | $n^2$ | $\log n$ | ❌ | ✅ |
| Heap | $n\log n$ | $n\log n$ | $n\log n$ | 1 | ❌ | ✅ |
| Counting | $n+k$ | $n+k$ | $n+k$ | $n+k$ | ✅ | ❌ |
| Radix | $d(n+k)$ | $d(n+k)$ | $d(n+k)$ | $n+k$ | ✅ | ❌ |
| Bucket | $n+k$ | $n+k$ | $n^2$ | $n+k$ | ✅ | ❌ |
| Quickselect | $n$ | $n$ | $n^2$ | $\log n$ | — | ✅ |

---

### 2.3 Algoritmos em Grafos

#### Conceitos

- $G = (V, E)$; dirigido ou não; ponderado ou não.
- Não dirigido: $\sum \deg(v) = 2E$; máximo $V(V-1)/2$ arestas. Dirigido: máximo $V(V-1)$.
- **Esparso:** $E = O(V)$. **Denso:** $E = \Theta(V^2)$.
- **Árvore:** conexo e acíclico ⇔ conexo com $V - 1$ arestas.
- **DAG:** dirigido acíclico.
- **Bipartido** ⇔ sem ciclos de comprimento ímpar.

#### Representação

| | Matriz de adjacência | Lista de adjacência |
| :---- | :---- | :---- |
| Memória | $\Theta(V^2)$ | $\Theta(V + E)$ |
| Existe $(u,v)$? | $O(1)$ | $O(\deg u)$ |
| Listar vizinhos de $u$ | $\Theta(V)$ | $\Theta(\deg u)$ |
| BFS/DFS | $\Theta(V^2)$ | $\Theta(V + E)$ |
| Bom para | densos, Floyd-Warshall | esparsos (maioria dos casos) |

#### BFS (Busca em Largura)

```
function bfs(s) {
  dist[s] = 0; visited.add(s);
  const queue = [s];
  while (queue.length) {
    const u = queue.shift();
    for (const v of adj[u]) {
      if (!visited.has(v)) {
        visited.add(v);              // marca ao ENFILEIRAR, não ao tirar
        dist[v] = dist[u] + 1;
        parent[v] = u;
        queue.push(v);
      }
    }
  }
}
```

- $O(V + E)$: cada vértice entra uma vez na fila; cada aresta examinada uma (dirigido) ou duas (não dirigido) vezes.
- Memória $O(V)$.
- `dist[v]` = **menor número de arestas** de $s$ a $v$ → caminho mínimo em grafo **não ponderado**.
- Aplicações: componentes conexos, teste de bipartição (2 cores por nível), menor número de movimentos (puzzles), web crawler, rede social (graus de separação).

#### DFS (Busca em Profundidade)

```
function dfs(u) {
  color[u] = GRAY; d[u] = ++time;          // descoberta
  for (const v of adj[u]) {
    if (color[v] === WHITE) { parent[v] = u; dfs(v); }
    else if (color[v] === GRAY) { /* aresta de retorno → ciclo (dirigido) */ }
  }
  color[u] = BLACK; f[u] = ++time;         // término
}
```

- $O(V + E)$; memória $O(V)$ (pilha).
- **Cores:** branco (não visto), cinza (na pilha), preto (terminado).
- **Teorema dos parênteses:** intervalos $[d, f]$ de dois vértices são disjuntos ou um contém o outro.
- **Classificação de arestas $(u, v)$:**
  - Árvore: $v$ branco.
  - Retorno (back): $v$ cinza (ancestral) → **ciclo**.
  - Avanço (forward): $v$ preto e descendente.
  - Cruzamento (cross): $v$ preto, nem ancestral nem descendente.
  - Grafo **não dirigido**: só árvore e retorno.
- **Ciclo em dirigido** ⇔ existe aresta de retorno. Em não dirigido: aresta para vértice visitado que não é o pai.

#### Ordenação Topológica (só DAG)

Ordem linear em que toda aresta $u \to v$ tem $u$ antes de $v$. Existe ⇔ o grafo é DAG. Pode haver várias.

**Kahn (BFS por grau de entrada):**

```
for v in V: if indeg[v] == 0: queue.push(v)
while queue:
  u = queue.shift(); order.push(u)
  for v in adj[u]:
    if (--indeg[v] == 0) queue.push(v)
if order.length < V: "há ciclo"
```

**DFS:** ao terminar cada vértice, coloque-o no **início** da lista (ordem decrescente de $f$).

Ambos $O(V + E)$. Exemplo: A→B, A→C, B→D, C→D → `A B C D` ou `A C B D`.

Aplicações: dependências de build/pacotes, pré-requisitos de disciplinas, escalonamento de tarefas, avaliação de planilhas.

#### Componentes Conexos e Fortemente Conexos

- **Não dirigido:** cada BFS/DFS a partir de um vértice não visitado descobre um componente. $O(V + E)$. (Ou Union-Find.)
- **SCC (dirigido):** classes de equivalência de "mutuamente alcançáveis". Contrair cada SCC num vértice gera um **DAG** (grafo de componentes).

**Kosaraju-Sharir:**
1. DFS em $G$ e registrar vértices em ordem de término.
2. Construir $G^T$ (arestas invertidas).
3. DFS em $G^T$ processando vértices em ordem **decrescente** de término; cada árvore = um SCC.

Por que funciona: o vértice com maior $f$ está num SCC "fonte" do DAG de componentes; em $G^T$ ele vira "sumidouro", então a DFS não escapa do componente.

**Tarjan:** uma DFS; cada vértice recebe `idx` (ordem de descoberta) e `low` (menor `idx` alcançável pela subárvore usando no máximo uma aresta de retorno para um vértice na pilha). Vértices são empilhados ao descobrir; quando `low[v] == idx[v]`, desempilha até $v$: isso é um SCC.

Ambos $O(V + E)$. Aplicação: 2-SAT (SCC no grafo de implicação), análise de dependências circulares.

#### Caminhos Mínimos

**Relaxamento:**

```
function relax(u, v, w) {
  if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; parent[v] = u; }
}
```

**Subestrutura ótima:** subcaminhos de caminhos mínimos são mínimos. Caminhos mínimos não existem se houver **ciclo negativo alcançável**.

| Situação | Algoritmo | Tempo |
| :---- | :---- | :---- |
| Sem pesos | BFS | $O(V + E)$ |
| DAG (qualquer peso) | topológica + relaxar | $O(V + E)$ |
| Pesos $\ge 0$ | Dijkstra | $O((V + E)\log V)$ |
| Pesos negativos | Bellman-Ford | $O(VE)$ |
| Todos os pares | Floyd-Warshall | $\Theta(V^3)$ |
| Todos os pares, esparso, pesos $\ge 0$ | $V \times$ Dijkstra | $O(V(V+E)\log V)$ |

**Dijkstra (guloso):**

```
dist[*] = ∞; dist[s] = 0
pq = minHeap([(0, s)])
while pq:
  (d, u) = pq.pop()
  if d > dist[u]: continue          // entrada obsoleta
  for (v, w) in adj[u]:
    if dist[u] + w < dist[v]:
      dist[v] = dist[u] + w
      pq.push((dist[v], v))
```

- Heap binário: $O((V + E)\log V)$. Array simples: $O(V^2)$ (melhor para densos). Fibonacci: $O(E + V\log V)$.
- **Invariante:** quando $u$ sai da fila, $dist[u]$ é definitivo. Isso depende de **pesos não negativos**: com peso negativo um caminho "mais longo" poderia ficar mais barato depois.
- Contraexemplo: A→B (2), A→C (3), C→B (−2), B→D (1). Na versão clássica (vértice extraído nunca é reprocessado), Dijkstra finaliza B = 2 e D = 3; o correto é B = 1 e D = 2.
- Exemplo (quiz Q31): A–B 4, A–C 1, C–B 2, B–D 1, C–D 5 → A=0, C=1, B=3, D=4.

**Bellman-Ford (programação dinâmica):**

```
dist[*] = ∞; dist[s] = 0
repeat V - 1 times:
  for each (u, v, w) in E: relax(u, v, w)
for each (u, v, w) in E:
  if dist[u] + w < dist[v]: "ciclo negativo alcançável"
```

- $O(VE)$. Após $i$ rodadas, `dist` está correta para caminhos com até $i$ arestas; caminho simples tem $\le V-1$ arestas.
- Funciona com pesos negativos; **detecta** ciclo negativo. Base do roteamento por vetor de distância (RIP). Pode parar cedo se uma rodada não alterar nada.

**Floyd-Warshall (DP, todos os pares):**

```
dist = matriz de pesos (∞ sem aresta, 0 na diagonal)
for k in V:
  for i in V:
    for j in V:
      dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
```

- $d^{(k)}[i][j]$ = menor caminho de $i$ a $j$ usando só intermediários $\{1..k\}$. **$k$ precisa ser o laço externo.**
- $\Theta(V^3)$ tempo, $\Theta(V^2)$ memória. Aceita pesos negativos; ciclo negativo ⇔ algum $dist[i][i] < 0$.
- Variante: fecho transitivo (Warshall) com `or`/`and`.

#### Árvore Geradora Mínima (MST)

Subconjunto de $V - 1$ arestas que conecta todos os vértices sem ciclos e com peso total mínimo (grafo **não dirigido, conexo, ponderado**).

- **Propriedade do corte:** a aresta mais leve que cruza qualquer corte pertence a alguma MST.
- **Propriedade do ciclo:** a aresta mais pesada de um ciclo (estritamente) não pertence a nenhuma MST.
- Pesos distintos → MST única. Pesos negativos são OK.
- MST ≠ árvore de caminhos mínimos (otimizam coisas diferentes).

**Kruskal (por arestas, global):**

```
sort(E by weight)
for (u, v, w) in E:
  if find(u) != find(v):
    union(u, v); mst.push((u, v, w))
  if mst.length == V - 1: break
```

$O(E \log E) = O(E \log V)$ (dominado pela ordenação). Bom para esparsos. Funciona em grafo desconexo (gera floresta geradora mínima).

**Prim (por vértices, cresce uma árvore):**

```
key[*] = ∞; key[r] = 0; pq = minHeap(V by key)
while pq:
  u = pq.extractMin(); inMST[u] = true
  for (v, w) in adj[u]:
    if !inMST[v] && w < key[v]:
      key[v] = w; parent[v] = u; pq.decreaseKey(v, w)
```

Heap binário $O(E \log V)$; array $O(V^2)$ (bom para densos); Fibonacci $O(E + V\log V)$.

**Exemplo:** A–B 1, B–C 2, A–C 3, C–D 4, B–D 5. Kruskal: AB ✓, BC ✓, AC ✗ (ciclo), CD ✓ → peso 7.

**Prim vs Dijkstra:** mesma estrutura; Prim usa `key[v] = w(u,v)`, Dijkstra usa `dist[v] = dist[u] + w(u,v)`.

---

### 2.4 NP-Completude

#### Problemas de decisão

A teoria é formulada para problemas **sim/não** ("existe rota de custo $\le K$?"). Otimização é pelo menos tão difícil quanto a decisão correspondente.

#### Classes

- **P:** decidível em tempo $O(n^k)$ por máquina de Turing determinística.
- **NP:** para toda instância "sim" existe um **certificado** de tamanho polinomial verificável em tempo polinomial. Equivalente: decidível em tempo polinomial por máquina **não-determinística**. NP **não** significa "não polinomial".
- **co-NP:** instâncias "não" têm certificado curto (ex.: "a fórmula é insatisfatível?").
- **NP-difícil:** todo $L \in$ NP satisfaz $L \le_p X$. Não precisa estar em NP (pode nem ser decidível — Problema da Parada).
- **NP-completo:** NP ∩ NP-difícil.

```
            NP-difícil
         ┌──────────────┐
   NP    │  NP-completo │
┌────────┼──────┐       │
│   P    │      │       │
└────────┴──────┘───────┘
     (supondo P ≠ NP)
```

Fatos: P ⊆ NP. Se **algum** NP-completo está em P, então P = NP. Se P ≠ NP, há problemas em NP que não são nem P nem NPC (Ladner) — candidatos: fatoração, isomorfismo de grafos.

#### Reduções

$A \le_p B$: função $f$ computável em tempo polinomial com $x \in A \Leftrightarrow f(x) \in B$.

- Lê-se: "$A$ não é mais difícil que $B$" / "$B$ é pelo menos tão difícil quanto $A$".
- Se $B \in$ P e $A \le_p B$ → $A \in$ P.
- Se $A$ é NP-difícil e $A \le_p B$ → $B$ é NP-difícil.
- Transitiva.

**Receita para provar que $X$ é NP-completo:**
1. Mostrar $X \in$ NP (descrever o certificado e o verificador polinomial).
2. Escolher NPC conhecido $A$ e construir $f$ com $A \le_p X$ (**do conhecido para o novo**).
3. Provar as duas direções: $x \in A \Rightarrow f(x) \in X$ e $f(x) \in X \Rightarrow x \in A$.
4. Mostrar que $f$ é polinomial.

**Cook-Levin (1971):** SAT é NP-completo — a computação de qualquer máquina não-determinística polinomial pode ser codificada como uma fórmula booleana.

**Cadeia clássica (Karp, 1972):**

```
SAT → 3-SAT → Clique → Conjunto Independente ↔ Vertex Cover → Set Cover
            → 3-Coloração
            → Ciclo Hamiltoniano → TSP
            → Subset Sum → Mochila / Partição
```

**Exemplos de redução:**
- **Independent Set ↔ Vertex Cover:** $S$ independente ⇔ $V \setminus S$ é cobertura. Logo IS com $k$ ⇔ VC com $V - k$.
- **Clique ↔ Independent Set:** clique em $G$ ⇔ independente no complemento $\bar G$.
- **Ham. Cycle → TSP:** dê peso 1 às arestas de $G$ e 2 às inexistentes; $G$ tem ciclo hamiltoniano ⇔ há rota de custo $\le V$.
- **3-SAT → Clique:** um vértice por literal em cada cláusula; arestas entre literais de cláusulas diferentes que não são contraditórios; fórmula satisfatível ⇔ clique de tamanho = nº de cláusulas.

#### Catálogo

| NP-completos | Em P (parecidos, mas fáceis) |
| :---- | :---- |
| 3-SAT, SAT, CNF-SAT | 2-SAT, Horn-SAT |
| Ciclo/caminho Hamiltoniano | Ciclo Euleriano |
| TSP (decisão) | Caminho mínimo |
| Caminho **mais longo** simples | Caminho mais curto |
| 3-Coloração | 2-Coloração (bipartido) |
| Clique, Independent Set, Vertex Cover | Vertex Cover em grafo bipartido, Emparelhamento máximo |
| Subset Sum, Partição, Mochila 0/1 | Mochila fracionária (guloso) |
| Set Cover | MST |
| Programação Inteira | Programação Linear |

#### Lidando com NP-difíceis na prática

- **Pseudo-polinomial:** Mochila 0/1 com DP $O(nW)$; Subset Sum $O(nS)$. Polinomial no **valor** numérico, exponencial no **tamanho em bits** ($W$ ocupa $\log W$ bits). Problemas assim são "fracamente" NP-completos.
- **Aproximação** com garantia:
  - Vertex Cover: emparelhamento maximal → 2-aprox.
  - TSP **métrico**: MST duplicada → 2-aprox; Christofides → 1,5-aprox. TSP **geral** não admite aproximação constante (a menos que P = NP).
  - Set Cover: guloso → $O(\ln n)$-aprox.
- **Heurísticas/metaheurísticas:** guloso, busca local, simulated annealing, algoritmos genéticos (sem garantia).
- **Exatos exponenciais:** backtracking, **Branch & Bound**, DP sobre subconjuntos (TSP Held-Karp $O(n^2 2^n)$), solvers SAT/ILP.
- **Casos especiais polinomiais:** árvores, grafos bipartidos, parâmetro pequeno (FPT).

---

### 2.5 Autômatos Finitos e Expressões Regulares

#### Definições

- **Alfabeto** $\Sigma$: conjunto finito de símbolos. **Cadeia/palavra**: sequência finita; $\epsilon$ = cadeia vazia. $\Sigma^*$ = todas as cadeias.
- **Linguagem**: subconjunto de $\Sigma^*$.
- Operações: união $L_1 \cup L_2$; concatenação $L_1L_2$; **fecho de Kleene** $L^* = \bigcup_{i \ge 0} L^i$ (inclui $\epsilon$); $L^+ = LL^*$.

#### AFD

$M = (Q, \Sigma, \delta, q_0, F)$ com $\delta: Q \times \Sigma \to Q$ **total** (exatamente uma transição por estado e símbolo; se faltar, há um estado "morto" implícito).

Função estendida: $\hat\delta(q, \epsilon) = q$; $\hat\delta(q, wa) = \delta(\hat\delta(q, w), a)$. Aceita $w$ ⇔ $\hat\delta(q_0, w) \in F$. Reconhece em $O(|w|)$ tempo e $O(1)$ memória.

**Exemplo 1 — número par de `a` sobre $\{a, b\}$:** estados `par` (inicial, final) e `ímpar`; `a` alterna, `b` mantém.

**Exemplo 2 — binários divisíveis por 3:** estados = resto $r \in \{0, 1, 2\}$; ler bit $b$ faz $r \leftarrow (2r + b) \bmod 3$.

| | 0 | 1 |
| :---- | :---- | :---- |
| →\*$r_0$ | $r_0$ | $r_1$ |
| $r_1$ | $r_2$ | $r_0$ |
| $r_2$ | $r_1$ | $r_2$ |

`110` (6): $r_0 \xrightarrow{1} r_1 \xrightarrow{1} r_0 \xrightarrow{0} r_0$ ✓.

**Exemplo 3 — termina em `01`:** estados por "sufixo relevante lido": $q_0$ (nada), $q_1$ (termina em 0), $q_2$ (termina em 01, final).

#### AFN e AFN-ε

$\delta: Q \times (\Sigma \cup \{\epsilon\}) \to \mathcal{P}(Q)$. Pode ter 0, 1 ou várias transições por símbolo, e transições ε. **Aceita se existe algum caminho** que consome a entrada e termina em $F$.

- **ε-fecho**$(q)$: estados alcançáveis de $q$ só com ε (inclui $q$).
- AFN-ε → AFN: eliminar ε usando ε-fechos.
- Simular um AFN diretamente: manter o **conjunto** de estados ativos → $O(|w| \cdot |Q|^2)$.

#### Equivalências (Teorema de Kleene)

Regex ⇔ AFN-ε ⇔ AFN ⇔ AFD ⇔ **linguagens regulares**.

**Construção de subconjuntos (AFN → AFD):**
- Estado do AFD = conjunto de estados do AFN.
- Inicial: ε-fecho($q_0$).
- $\Delta(S, a) = \epsilon\text{-fecho}\left(\bigcup_{q \in S} \delta(q, a)\right)$.
- Final: todo $S$ que contém algum estado final do AFN.
- Gere só os alcançáveis. Pior caso $2^n$ estados — **atingível** (ex.: "o $n$-ésimo símbolo a partir do fim é 1" tem AFN com $n+1$ estados e AFD mínimo com $2^n$).

**Exemplo — termina em `ab`:** AFN $q_0 \xrightarrow{a,b} q_0$, $q_0 \xrightarrow{a} q_1$, $q_1 \xrightarrow{b} q_2$.

| Estado AFD | a | b |
| :---- | :---- | :---- |
| →$\{q_0\}$ | $\{q_0, q_1\}$ | $\{q_0\}$ |
| $\{q_0, q_1\}$ | $\{q_0, q_1\}$ | $\{q_0, q_2\}$ |
| \*$\{q_0, q_2\}$ | $\{q_0, q_1\}$ | $\{q_0\}$ |

**Thompson (Regex → AFN-ε):** por indução na estrutura:
- Símbolo $a$: $s \xrightarrow{a} f$.
- $R \mid S$: novo início com ε para os inícios de $R$ e $S$; finais de ambos com ε para novo final.
- $RS$: final de $R$ ligado por ε ao início de $S$.
- $R^*$: novo início e final; ε para $R$ e direto para o final; final de $R$ com ε de volta ao início de $R$ e para o final.

Resultado com $O(m)$ estados para regex de tamanho $m$. Usado em `grep`, lex/flex.

**AFD → Regex:** eliminação de estados (rótulos das arestas viram regex) ou método de Arden / Kleene ($R_{ij}^{(k)}$).

**Pipeline de um analisador léxico:** regex dos tokens → Thompson → AFN → subconjuntos → AFD → minimização → tabela.

#### Expressões Regulares

Formais: $\emptyset$, $\epsilon$, $a \in \Sigma$, $R \mid S$, $RS$, $R^*$. Precedência: `*` > concatenação > `|`.

| Linguagem | Regex |
| :---- | :---- |
| Contém `11` | `(0\|1)*11(0\|1)*` |
| Termina em `01` | `(0\|1)*01` |
| Começa e termina com o mesmo símbolo | `0(0\|1)*0 \| 1(0\|1)*1 \| 0 \| 1` |
| Número par de `a` (sobre $\{a,b\}$) | `b*(ab*ab*)*` |
| Sem `11` | `(0\|10)*(1\|ε)` |
| Tamanho par | `((0\|1)(0\|1))*` |

Identidades: $(R^*)^* = R^*$; $\epsilon^* = \epsilon$; $\emptyset^* = \epsilon$; $R\emptyset = \emptyset$; $(a \mid b)^* = (a^*b^*)^*$.

**Regex prática** (das suas notas): `[0-9]` classe; `{n}` repetição; `^`/`$` âncoras; `( )` grupos de captura; `$1`/`\1` referências. ⚠️ **Retroreferências** (`(a*)b\1`) vão **além** das linguagens regulares — motores modernos não são AFDs puros.

#### Propriedades de fechamento

Regulares são fechadas sob: união, concatenação, estrela, **complemento** (inverter finais de um AFD completo), **interseção** (autômato produto $Q_1 \times Q_2$), diferença, reverso, homomorfismo.

Uso em prova: se $L \cap a^*b^* = \{a^nb^n\}$ e $a^*b^*$ é regular, então $L$ não pode ser regular.

#### Lema do Bombeamento

Se $L$ é regular, existe $p \ge 1$ (nº de estados de um AFD) tal que toda $w \in L$ com $|w| \ge p$ pode ser escrita $w = xyz$ com:
1. $|xy| \le p$,
2. $|y| \ge 1$,
3. $xy^iz \in L$ para todo $i \ge 0$.

**Intuição:** lendo $p$ símbolos, o AFD visita $p+1$ estados → repete algum (casa dos pombos) → o trecho $y$ entre as repetições é um laço que pode ser percorrido 0, 1, 2… vezes.

**Prova de que $L = \{a^nb^n \mid n \ge 0\}$ não é regular:**
1. Suponha $L$ regular com constante $p$.
2. Escolha $w = a^pb^p \in L$, $|w| = 2p \ge p$.
3. Como $|xy| \le p$, $x$ e $y$ só contêm `a`; $y = a^k$ com $k \ge 1$.
4. Bombeando $i = 2$: $xy^2z = a^{p+k}b^p \notin L$.
5. Contradição → $L$ não é regular. ∎

O jogo: **você** escolhe $w$ (esperta) e $i$; o **adversário** escolhe $p$ e a divisão $xyz$ (você precisa vencer para **toda** divisão válida).

**Não regulares clássicas:** $a^nb^n$, parênteses balanceados, palíndromos, $\{ww\}$, $\{a^{n^2}\}$, $\{a^p : p \text{ primo}\}$, $\{w : \#_a(w) = \#_b(w)\}$.

**Cuidado:** o lema é condição **necessária**, não suficiente — há linguagens não regulares que satisfazem o lema. Para caracterização exata use **Myhill-Nerode** ($L$ regular ⇔ número finito de classes de equivalência).

#### Minimização de AFD

1. Remover estados inalcançáveis.
2. Marcar pares (final, não final) como distinguíveis.
3. Repetir: marcar $(p, q)$ se para algum $a$, $(\delta(p,a), \delta(q,a))$ está marcado.
4. Pares não marcados são equivalentes → fundir.

O AFD mínimo é **único** a menos de renomeação. Algoritmo de Hopcroft: $O(n \log n)$.

#### Hierarquia de Chomsky

| Tipo | Linguagem | Máquina | Gramática | Exemplo |
| :---- | :---- | :---- | :---- | :---- |
| 3 | Regular | Autômato finito | $A \to aB \mid a$ | $a^*b^*$ |
| 2 | Livre de contexto | **Autômato de pilha** (não-det.) | $A \to \alpha$ | $a^nb^n$, parênteses |
| 1 | Sensível ao contexto | Autômato linearmente limitado | $\alpha A \beta \to \alpha\gamma\beta$ | $a^nb^nc^n$ |
| 0 | Recursivamente enumerável | Máquina de Turing | irrestrita | Problema da Parada (aceitação) |

Regular ⊊ LLC ⊊ LSC ⊊ RE. LLC **não** são fechadas sob interseção nem complemento ($a^nb^nc^m \cap a^mb^nc^n = a^nb^nc^n$).
