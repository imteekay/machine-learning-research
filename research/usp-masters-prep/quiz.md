# Quiz & Flashcards

> **Como usar até 17/11:** leia a pergunta, responda em voz alta ou no papel e só então abra a resposta. Marque com ❌ as que errar e repita **só essas** no dia seguinte, depois em 3 dias, depois em 7. Faça o **Simulado** (final do arquivo) cronometrado: 20 questões em ~1h30.
>
> Formato: **Flashcards** (resposta curta) e **Múltipla escolha** (estilo prova, com justificativa).

---

## 1. Arrays, Listas, Pilhas e Filas

### Flashcards

**F1.** Por que o acesso por índice em array é $O(1)$?
<details><summary>Resposta</summary>

Memória contígua: endereço = base + índice × tamanho do elemento. Uma conta, independente de $n$.

</details>

**F2.** Custo de inserir no meio de um array de $n$ elementos?
<details><summary>Resposta</summary>

$O(n)$ — é preciso deslocar (shift) os elementos à direita.

</details>

**F3.** Custo de `append` em array dinâmico que dobra a capacidade?
<details><summary>Resposta</summary>

$O(1)$ **amortizado**; uma operação isolada pode custar $O(n)$ (realocação + cópia).

</details>

**F4.** Se o array dinâmico crescesse +10 posições a cada realocação (em vez de dobrar), qual seria o custo amortizado do append?
<details><summary>Resposta</summary>

$O(n)$ amortizado ($O(n^2)$ total para $n$ appends). Por isso se usa crescimento **geométrico**.

</details>

**F5.** Lista ligada simples com ponteiro `head` e `tail`: custo de inserir no fim? E de remover do fim?
<details><summary>Resposta</summary>

Inserir no fim: $O(1)$. Remover do fim: $O(n)$ — é preciso achar o penúltimo nó. Em lista **duplamente** ligada, remover do fim é $O(1)$.

</details>

**F6.** Por que busca binária não é eficiente em lista ligada?
<details><summary>Resposta</summary>

Não há acesso aleatório: chegar ao meio custa $O(n)$.

</details>

**F7.** Como detectar ciclo em lista ligada com $O(1)$ de memória?
<details><summary>Resposta</summary>

Algoritmo de Floyd (tartaruga e lebre): um ponteiro anda 1 passo, outro anda 2. Se se encontrarem, há ciclo. $O(n)$ tempo.

</details>

**F8.** Política da pilha e operações principais?
<details><summary>Resposta</summary>

LIFO. `push`, `pop`, `peek/top` — todas $O(1)$.

</details>

**F9.** Política da fila e operações principais?
<details><summary>Resposta</summary>

FIFO. `enqueue` (fim), `dequeue` (início), `front` — todas $O(1)$ com lista ligada ou array circular.

</details>

**F10.** Em uma fila em array circular de tamanho $m$, como avança o índice `rear`?
<details><summary>Resposta</summary>

`rear = (rear + 1) mod m`. Fila cheia quando `(rear + 1) mod m == front` (desperdiçando uma posição) ou usando um contador.

</details>

**F11.** O que é um deque?
<details><summary>Resposta</summary>

Fila de duas pontas: inserção e remoção em $O(1)$ no início e no fim.

</details>

**F12.** Como implementar uma fila com duas pilhas? Custo?
<details><summary>Resposta</summary>

Pilha de entrada recebe `push`; ao fazer `dequeue`, se a pilha de saída estiver vazia, despeja toda a entrada nela. $O(1)$ **amortizado** por operação.

</details>

**F13.** Converta `A + B * C` para pós-fixa.
<details><summary>Resposta</summary>

`A B C * +` (multiplicação tem precedência).

</details>

**F14.** Converta `(A + B) * C` para pós-fixa.
<details><summary>Resposta</summary>

`A B + C *`

</details>

**F15.** Avalie a pós-fixa `3 4 + 2 *`.
<details><summary>Resposta</summary>

Empilha 3, 4 → `+` → 7; empilha 2 → `*` → **14**.

</details>

**F16.** Qual estrutura verifica se parênteses estão balanceados?
<details><summary>Resposta</summary>

Pilha: empilha cada abertura; em cada fechamento, desempilha e confere o tipo. Balanceado se a pilha termina vazia.

</details>

**F17.** Qual estrutura a BFS usa? E a DFS?
<details><summary>Resposta</summary>

BFS: **fila**. DFS: **pilha** (explícita ou a pilha de chamadas da recursão).

</details>

### Múltipla escolha

**Q1.** Uma sequência de $n$ operações `append` em um array dinâmico que dobra de tamanho quando cheio custa, no total:

- (A) $O(n^2)$
- (B) $O(n \log n)$
- (C) $O(n)$
- (D) $O(\log n)$
- (E) $O(1)$

<details><summary>Resposta</summary>

**(C)**. Cópias: $1 + 2 + 4 + \dots + n < 2n$. Somando os $n$ appends, total $< 3n = O(n)$ → $O(1)$ amortizado por operação.

</details>

**Q2.** Em uma lista **simplesmente** ligada que mantém ponteiros para cabeça e cauda, qual operação **não** é $O(1)$?

- (A) Inserir no início
- (B) Inserir no fim
- (C) Remover do início
- (D) Remover do fim
- (E) Ler o primeiro elemento

<details><summary>Resposta</summary>

**(D)**. Para remover o último, precisamos atualizar o `next` do penúltimo, que só é achado percorrendo a lista.

</details>

**Q3.** Empilhe 1, 2, 3, 4 em uma pilha P. Desempilhe dois elementos e enfileire-os (na ordem desempilhada) em uma fila F vazia. Em seguida, faça `dequeue` em F. Qual valor sai?

- (A) 1
- (B) 2
- (C) 3
- (D) 4
- (E) A fila está vazia

<details><summary>Resposta</summary>

**(D)**. Pop → 4, pop → 3. Fila F = [4, 3]. Dequeue → 4.

</details>

**Q4.** Qual aplicação **não** é tipicamente resolvida com uma pilha?

- (A) Avaliação de expressão pós-fixa
- (B) Undo em editor
- (C) Busca em largura
- (D) Verificação de parênteses
- (E) Chamadas de função

<details><summary>Resposta</summary>

**(C)**. BFS usa fila.

</details>

---

## 2. Listas de Prioridade (Heaps)

### Flashcards

**F18.** Propriedade de um max-heap?
<details><summary>Resposta</summary>

Árvore binária **completa** em que cada pai $\ge$ seus filhos. Raiz = máximo.

</details>

**F19.** Índices de filhos e pai em heap com base 0?
<details><summary>Resposta</summary>

Filhos: $2i+1$ e $2i+2$. Pai: $\lfloor (i-1)/2 \rfloor$. (Base 1: filhos $2i$, $2i+1$; pai $\lfloor i/2 \rfloor$.)

</details>

**F20.** Passos do `insert` em heap?
<details><summary>Resposta</summary>

Coloca na última posição do array e faz **sift-up** (troca com o pai enquanto violar a propriedade). $O(\log n)$.

</details>

**F21.** Passos do `extract-max` em heap?
<details><summary>Resposta</summary>

Guarda a raiz, move o último elemento para a raiz e faz **sift-down** (troca com o **maior** filho enquanto violar). $O(\log n)$.

</details>

**F22.** Custo de construir um heap a partir de um array de $n$ elementos (build-heap)?
<details><summary>Resposta</summary>

$O(n)$ — sift-down de $\lfloor n/2 \rfloor - 1$ até 0. A maioria dos nós está perto das folhas e desce pouco.

</details>

**F23.** Custo de buscar um elemento arbitrário em heap?
<details><summary>Resposta</summary>

$O(n)$ — o heap não tem ordem entre irmãos/subárvores; é preciso varrer.

</details>

**F24.** Em um heap de $n$ elementos, em que índices estão as folhas (base 0)?
<details><summary>Resposta</summary>

De $\lfloor n/2 \rfloor$ até $n-1$.

</details>

**F25.** Altura de um heap com $n$ elementos?
<details><summary>Resposta</summary>

$\lfloor \log_2 n \rfloor$.

</details>

**F26.** Como achar os $k$ maiores de $n$ elementos com heap? Custo?
<details><summary>Resposta</summary>

Min-heap de tamanho $k$: para cada elemento, se maior que o topo, troca o topo. $O(n \log k)$.

</details>

### Múltipla escolha

**Q5.** Insira, nesta ordem, 5, 3, 8, 1, 9 em um **min-heap** vazio (array, base 0). O array resultante é:

- (A) [1, 3, 5, 8, 9]
- (B) [1, 3, 8, 5, 9]
- (C) [1, 5, 3, 8, 9]
- (D) [3, 1, 8, 5, 9]
- (E) [1, 3, 9, 5, 8]

<details><summary>Resposta</summary>

**(B)**.
- 5 → [5]
- 3 → [5,3] → sift-up → [3,5]
- 8 → [3,5,8]
- 1 → [3,5,8,1] → pai de idx 3 é idx 1 (5) → [3,1,8,5] → pai idx 0 (3) → [1,3,8,5]
- 9 → [1,3,8,5,9] (pai idx 1 = 3 < 9, ok)

</details>

**Q6.** Aplicando `extract-min` no heap [1, 3, 8, 5, 9], o array resultante é:

- (A) [3, 5, 8, 9]
- (B) [3, 9, 8, 5]
- (C) [3, 8, 5, 9]
- (D) [5, 3, 8, 9]
- (E) [9, 3, 8, 5]

<details><summary>Resposta</summary>

**(A)**. Remove 1, coloca 9 na raiz → [9,3,8,5]. Sift-down: menor filho de 9 é 3 → [3,9,8,5]. Filho de idx 1 é idx 3 (5) → [3,5,8,9].

</details>

**Q7.** Qual vetor **é** um max-heap válido?

- (A) [9, 7, 8, 3, 8, 2]
- (B) [9, 8, 7, 9, 1, 2]
- (C) [9, 7, 8, 3, 1, 2]
- (D) [8, 9, 7, 3, 1, 2]
- (E) [9, 7, 8, 3, 1, 9]

<details><summary>Resposta</summary>

**(C)**. Filhos de 9 (idx 0): 7 e 8 ✓. Filhos de 7 (idx 1): 3 e 1 ✓. Filho de 8 (idx 2): 2 ✓.
- (A): filho de 7 é 8 ✗. (B): filho de 8 é 9 ✗. (D): raiz 8 < filho 9 ✗. (E): filho de 8 (idx 5) é 9 ✗.

Dica: para cada idx $i$, compare com $2i+1$ e $2i+2$.

</details>

**Q8.** Qual o custo de transformar um array arbitrário de $n$ elementos em heap, usando o algoritmo de build-heap bottom-up?

- (A) $O(\log n)$
- (B) $O(n)$
- (C) $O(n \log n)$
- (D) $O(n^2)$
- (E) $O(n \log^2 n)$

<details><summary>Resposta</summary>

**(B)**. $\sum_h \frac{n}{2^{h+1}} \cdot O(h) = O(n)$.

</details>

---

## 3. Recursão

### Flashcards

**F27.** Dois componentes obrigatórios de uma função recursiva?
<details><summary>Resposta</summary>

Caso base (parada) e caso recursivo (reduz o problema em direção ao caso base).

</details>

**F28.** O que causa *stack overflow* em recursão?
<details><summary>Resposta</summary>

Profundidade de chamadas grande demais (ou infinita, sem caso base): cada chamada empilha um quadro na call stack.

</details>

**F29.** `return n * fat(n-1)` é recursão de cauda?
<details><summary>Resposta</summary>

**Não**: depois que `fat(n-1)` retorna ainda falta multiplicar por `n`. Versão de cauda: `fat(n, acc) = fat(n-1, acc*n)`.

</details>

**F30.** O que a otimização de chamada de cauda (TCO) faz com a memória?
<details><summary>Resposta</summary>

Reaproveita o quadro atual da pilha: memória de $O(n)$ para $O(1)$, equivalente a um laço.

</details>

**F31.** Recursão direta vs. indireta?
<details><summary>Resposta</summary>

Direta: `f` chama `f`. Indireta (mútua): `f` chama `g`, que chama `f` (ex.: `par(n)` / `impar(n)`).

</details>

**F32.** Complexidade de Fibonacci recursivo ingênuo? E com memoização?
<details><summary>Resposta</summary>

Ingênuo: $O(2^n)$ (mais precisamente $\Theta(\varphi^n)$, $\varphi \approx 1{,}618$). Memoizado: $O(n)$ tempo e $O(n)$ memória.

</details>

**F33.** Quantos movimentos a Torre de Hanói com $n$ discos exige?
<details><summary>Resposta</summary>

$2^n - 1$. Recorrência $T(n) = 2T(n-1) + 1$.

</details>

**F34.** O que é backtracking?
<details><summary>Resposta</summary>

Busca recursiva que constrói a solução passo a passo e **desfaz** a última escolha quando ela leva a um beco sem saída (podando ramos inválidos). Ex.: N-Rainhas, Sudoku, permutações.

</details>

**F35.** Memória da busca binária recursiva vs. iterativa?
<details><summary>Resposta</summary>

Recursiva: $O(\log n)$ (pilha). Iterativa: $O(1)$.

</details>

### Múltipla escolha

**Q9.** Considere:
```
f(n):
  if n <= 1: return 1
  return f(n-1) + f(n-1)
```
A complexidade de tempo é:

- (A) $O(n)$
- (B) $O(n \log n)$
- (C) $O(n^2)$
- (D) $O(2^n)$
- (E) $O(n!)$

<details><summary>Resposta</summary>

**(D)**. $T(n) = 2T(n-1) + O(1) = \Theta(2^n)$. Se fosse `2 * f(n-1)` (uma chamada), seria $O(n)$.

</details>

**Q10.** Qual afirmação sobre recursão é **falsa**?

- (A) Todo algoritmo recursivo pode ser escrito iterativamente com uma pilha explícita.
- (B) Recursão de cauda pode ser convertida em laço pelo compilador.
- (C) A profundidade máxima de recursão determina a memória usada na pilha.
- (D) Recursão sempre é mais rápida que a versão iterativa equivalente.
- (E) Sem caso base alcançável, a recursão não termina.

<details><summary>Resposta</summary>

**(D)**. Recursão tem overhead de chamadas; normalmente é igual ou mais lenta.

</details>

---

## 4. Tabelas de Espalhamento (Hash)

### Flashcards

**F36.** O que é fator de carga $\alpha$?
<details><summary>Resposta</summary>

$\alpha = n / m$ (elementos / posições). Controla quando fazer rehash.

</details>

**F37.** Custo esperado de busca com encadeamento?
<details><summary>Resposta</summary>

$\Theta(1 + \alpha)$ (hash uniforme simples). Pior caso $O(n)$ (todas as chaves no mesmo bucket).

</details>

**F38.** Endereçamento aberto permite $\alpha > 1$?
<details><summary>Resposta</summary>

**Não** — todas as chaves ficam no próprio array, então $\alpha \le 1$. Encadeamento permite $\alpha > 1$.

</details>

**F39.** Problema da sondagem linear?
<details><summary>Resposta</summary>

**Agrupamento primário**: blocos contíguos ocupados crescem e aumentam o número de sondagens.

</details>

**F40.** Fórmula do double hashing?
<details><summary>Resposta</summary>

$h(k, i) = (h_1(k) + i \cdot h_2(k)) \bmod m$, com $h_2(k) \ne 0$ e coprimo com $m$.

</details>

**F41.** Por que remoção em endereçamento aberto precisa de *tombstone*?
<details><summary>Resposta</summary>

Se a posição virar "vazia", buscas por chaves que sondaram além dela parariam cedo e retornariam "não encontrado" erroneamente.

</details>

**F42.** No método da divisão $h(k) = k \bmod m$, como escolher $m$?
<details><summary>Resposta</summary>

Um **primo** não próximo de potência de 2 (evita depender só dos bits menos significativos).

</details>

**F43.** O que é hash universal?
<details><summary>Resposta</summary>

Escolher a função hash aleatoriamente de uma família $H$ tal que, para $x \ne y$, $\Pr[h(x) = h(y)] \le 1/m$. Garante bom caso médio contra entradas adversárias.

</details>

**F44.** Hash table mantém as chaves ordenadas?
<details><summary>Resposta</summary>

Não. Para ordem/consultas de intervalo, use BST balanceada.

</details>

**F45.** Que estruturas compõem um LRU Cache com operações $O(1)$?
<details><summary>Resposta</summary>

Hash table (chave → nó) + lista **duplamente** ligada (ordem de uso).

</details>

### Múltipla escolha

**Q11.** Tabela de tamanho 7, $h(k) = k \bmod 7$, sondagem **linear**. Insira 10, 17, 24, 3. Em que índice fica o 3?

- (A) 3
- (B) 4
- (C) 5
- (D) 6
- (E) 0

<details><summary>Resposta</summary>

**(D)**. 10→3. 17→3 ocupado→4. 24→3,4 ocupados→5. 3→3,4,5 ocupados→**6**.

</details>

**Q12.** Mesma tabela, mas com **encadeamento**. Quantos elementos ficam no bucket 3?

- (A) 1
- (B) 2
- (C) 3
- (D) 4
- (E) 0

<details><summary>Resposta</summary>

**(D)**. 10, 17, 24 e 3 têm resto 3 por 7.

</details>

**Q13.** Sobre hash tables, é correto afirmar:

- (A) Busca é $O(1)$ no pior caso.
- (B) Com endereçamento aberto, o fator de carga pode exceder 1.
- (C) Rehash custa $O(n)$, mas mantém custo amortizado $O(1)$ por inserção.
- (D) Sondagem quadrática elimina todo tipo de agrupamento.
- (E) São ideais para consultas por intervalo de chaves.

<details><summary>Resposta</summary>

**(C)**. (A) pior caso é $O(n)$; (B) falso; (D) ainda há agrupamento secundário; (E) hash não preserva ordem.

</details>

---

## 5. Árvores Binárias, BST, AVL, Rubro-Negra

### Flashcards

**F46.** Ordem das travessias pré, em e pós-ordem?
<details><summary>Resposta</summary>

Pré: Raiz-Esq-Dir. Em: Esq-Raiz-Dir. Pós: Esq-Dir-Raiz.

</details>

**F47.** Qual travessia de uma BST produz os valores ordenados?
<details><summary>Resposta</summary>

Em-ordem.

</details>

**F48.** Qual estrutura auxiliar a travessia por nível usa?
<details><summary>Resposta</summary>

Fila (é uma BFS).

</details>

**F49.** Três casos de remoção em BST?
<details><summary>Resposta</summary>

0 filhos: remove. 1 filho: o filho ocupa o lugar. 2 filhos: substitui pelo **sucessor em-ordem** (mínimo da subárvore direita) — ou predecessor — e remove este.

</details>

**F50.** Onde está o mínimo de uma BST?
<details><summary>Resposta</summary>

No nó mais à esquerda (segue `left` até `null`).

</details>

**F51.** Quando a BST degenera em lista?
<details><summary>Resposta</summary>

Ao inserir chaves em ordem crescente ou decrescente → altura $n - 1$, operações $O(n)$.

</details>

**F52.** Número máximo de nós em árvore binária de altura $h$?
<details><summary>Resposta</summary>

$2^{h+1} - 1$. (Nível $i$ tem até $2^i$ nós.)

</details>

**F53.** Fator de balanceamento no AVL?
<details><summary>Resposta</summary>

$FB = h(\text{esq}) - h(\text{dir}) \in \{-1, 0, +1\}$.

</details>

**F54.** Rotação para caso LL e para caso RR no AVL?
<details><summary>Resposta</summary>

LL (pesado à esquerda-esquerda): rotação simples à **direita**. RR: rotação simples à **esquerda**.

</details>

**F55.** Rotação para caso LR no AVL?
<details><summary>Resposta</summary>

Dupla: rotação à esquerda no filho esquerdo, depois rotação à direita no nó desbalanceado.

</details>

**F56.** As 4 regras da árvore rubro-negra?
<details><summary>Resposta</summary>

(1) Raiz preta. (2) Folhas NIL pretas. (3) Nó vermelho não tem filho vermelho. (4) Todo caminho de um nó até suas folhas tem o mesmo número de nós pretos.

</details>

**F57.** AVL vs Rubro-Negra: qual busca mais rápido? Qual atualiza mais rápido?
<details><summary>Resposta</summary>

AVL é mais rigidamente balanceada (altura ≤ ~1,44 log n) → busca mais rápida. RB tem altura ≤ 2 log(n+1) e faz menos rotações → inserção/remoção mais rápidas.

</details>

**F58.** Pré-ordem + em-ordem determinam a árvore unicamente?
<details><summary>Resposta</summary>

Sim (com chaves distintas). Pré + pós-ordem, em geral, **não**.

</details>

**F59.** Em uma árvore binária **cheia** (todo nó tem 0 ou 2 filhos), relação entre folhas $L$ e nós internos $I$?
<details><summary>Resposta</summary>

$L = I + 1$.

</details>

### Múltipla escolha

**Q14.** Insira 50, 30, 70, 20, 40, 60, 80 em uma BST vazia. A travessia **pós-ordem** é:

- (A) 20 30 40 50 60 70 80
- (B) 50 30 20 40 70 60 80
- (C) 20 40 30 60 80 70 50
- (D) 50 30 70 20 40 60 80
- (E) 80 70 60 50 40 30 20

<details><summary>Resposta</summary>

**(C)**. (A) é em-ordem, (B) pré-ordem, (D) por nível.

</details>

**Q15.** Na mesma BST, ao remover 50 usando o sucessor em-ordem, a nova raiz é:

- (A) 40
- (B) 60
- (C) 70
- (D) 30
- (E) 80

<details><summary>Resposta</summary>

**(B)**. Sucessor = mínimo da subárvore direita (70 → 60).

</details>

**Q16.** Inserindo 1, 2, 3 em uma AVL vazia, que rotação ocorre e qual é a raiz final?

- (A) Rotação à direita; raiz 1
- (B) Rotação à esquerda; raiz 2
- (C) Rotação dupla esquerda-direita; raiz 2
- (D) Nenhuma rotação; raiz 1
- (E) Rotação à direita; raiz 3

<details><summary>Resposta</summary>

**(B)**. Caso RR no nó 1 (FB = −2) → rotação simples à esquerda → 2 vira raiz com filhos 1 e 3.

</details>

**Q17.** Inserindo 3, 1, 2 em uma AVL vazia:

- (A) Rotação simples à direita; raiz 1
- (B) Rotação simples à esquerda; raiz 3
- (C) Rotação dupla (LR); raiz 2
- (D) Rotação dupla (RL); raiz 2
- (E) Nenhuma rotação

<details><summary>Resposta</summary>

**(C)**. 3 tem filho esquerdo 1, que tem filho direito 2 → caso LR: rotação à esquerda em 1, depois à direita em 3.

</details>

**Q18.** O pior caso de busca em uma BST **não balanceada** com $n$ nós é:

- (A) $O(1)$
- (B) $O(\log n)$
- (C) $O(\sqrt{n})$
- (D) $O(n)$
- (E) $O(n \log n)$

<details><summary>Resposta</summary>

**(D)**. Árvore degenerada (altura $n - 1$).

</details>

---

## 6. Union-Find

### Flashcards

**F60.** Três operações do Union-Find?
<details><summary>Resposta</summary>

`make_set(x)`, `find(x)` (retorna o representante/raiz), `union(x, y)` (liga as raízes).

</details>

**F61.** O que faz a compressão de caminho?
<details><summary>Resposta</summary>

Durante `find(x)`, faz todos os nós do caminho apontarem diretamente para a raiz.

</details>

**F62.** O que faz a união por rank?
<details><summary>Resposta</summary>

Anexa a raiz de menor rank (≈ altura) sob a de maior rank. O rank só aumenta (em 1) quando os ranks são iguais.

</details>

**F63.** Complexidade com ambas as otimizações?
<details><summary>Resposta</summary>

$O(\alpha(n))$ amortizado por operação — inversa de Ackermann, ≤ 4 na prática.

</details>

**F64.** Complexidade só com união por rank?
<details><summary>Resposta</summary>

$O(\log n)$ por operação (altura limitada a $\log n$).

</details>

**F65.** Como detectar ciclo em grafo **não direcionado** com Union-Find?
<details><summary>Resposta</summary>

Para cada aresta $(u, v)$: se `find(u) == find(v)` → ciclo; senão `union(u, v)`.

</details>

**F66.** Como contar componentes conexos com Union-Find?
<details><summary>Resposta</summary>

Começa com $n$ conjuntos; cada `union` bem-sucedida (raízes diferentes) reduz 1. Ou conte quantos $x$ têm `find(x) == x`.

</details>

### Múltipla escolha

**Q19.** Union-Find com 6 elementos {1..6}. Execute: union(1,2), union(3,4), union(2,4), union(5,6). Quantos conjuntos restam e `find(1) == find(3)`?

- (A) 2 conjuntos; sim
- (B) 3 conjuntos; sim
- (C) 3 conjuntos; não
- (D) 4 conjuntos; não
- (E) 2 conjuntos; não

<details><summary>Resposta</summary>

**(A)**. {1,2,3,4} e {5,6}.

</details>

**Q20.** Qual algoritmo de grafos usa Union-Find como componente central?

- (A) Dijkstra
- (B) Prim
- (C) Kruskal
- (D) Floyd-Warshall
- (E) BFS

<details><summary>Resposta</summary>

**(C)**.

</details>

---

## 7. Complexidade e Notação Assintótica

### Flashcards

**F67.** Definição formal de $f(n) = O(g(n))$?
<details><summary>Resposta</summary>

$\exists c > 0, n_0$ tais que $0 \le f(n) \le c \cdot g(n)$ para todo $n \ge n_0$.

</details>

**F68.** Diferença entre $O$ e $o$?
<details><summary>Resposta</summary>

$O$: existe **algum** $c$. $o$: vale para **todo** $c > 0$ (estritamente menor). $n = O(n)$, mas $n \ne o(n)$; $n = o(n^2)$.

</details>

**F69.** "$\Omega$ é o melhor caso" — verdadeiro?
<details><summary>Resposta</summary>

**Falso**. $\Omega$ é limite inferior de uma função. Pode-se falar de $\Omega$ do pior caso (ex.: pior caso de Quick Sort é $\Omega(n^2)$).

</details>

**F70.** Ordene: $n \log n$, $2^n$, $\sqrt{n}$, $n!$, $\log n$, $n^2$.
<details><summary>Resposta</summary>

$\log n < \sqrt{n} < n \log n < n^2 < 2^n < n!$

</details>

**F71.** $\log_2 n$ vs $\log_{10} n$ assintoticamente?
<details><summary>Resposta</summary>

Mesma classe: diferem por constante ($\log_a n = \log_b n / \log_b a$).

</details>

**F72.** $2^{n+1}$ é $O(2^n)$? E $2^{2n}$?
<details><summary>Resposta</summary>

$2^{n+1} = 2 \cdot 2^n$ → **sim**. $2^{2n} = 4^n$ → **não**.

</details>

**F73.** Complexidade de dois laços aninhados `for i in 1..n: for j in 1..i`?
<details><summary>Resposta</summary>

$\sum i = n(n+1)/2 = \Theta(n^2)$.

</details>

**F74.** Complexidade de `i = n; while i > 1: i = i / 2`?
<details><summary>Resposta</summary>

$\Theta(\log n)$.

</details>

**F75.** Complexidade de `for i in 1..n: j = 1; while j < n: j *= 2`?
<details><summary>Resposta</summary>

$\Theta(n \log n)$.

</details>

**F76.** Teorema Mestre, caso 2: quando ocorre e resultado?
<details><summary>Resposta</summary>

$f(n) = \Theta(n^{\log_b a})$ → $T(n) = \Theta(n^{\log_b a} \log n)$.

</details>

**F77.** $T(n) = 2T(n/2) + n$?
<details><summary>Resposta</summary>

$\Theta(n \log n)$ (caso 2).

</details>

**F78.** $T(n) = 8T(n/2) + n^2$?
<details><summary>Resposta</summary>

$n^{\log_2 8} = n^3$ domina → $\Theta(n^3)$ (caso 1).

</details>

**F79.** $T(n) = 2T(n/2) + n^2$?
<details><summary>Resposta</summary>

$n^{\log_2 2} = n$; $f = n^2$ domina → $\Theta(n^2)$ (caso 3).

</details>

**F80.** $T(n) = T(n-1) + n$?
<details><summary>Resposta</summary>

$\Theta(n^2)$ (soma $1 + 2 + \dots + n$). Não se aplica Mestre.

</details>

**F81.** $T(n) = 3T(n/3) + 1$?
<details><summary>Resposta</summary>

$n^{\log_3 3} = n$ domina → $\Theta(n)$.

</details>

**F82.** $\log(n!)$ é $\Theta$ de quê?
<details><summary>Resposta</summary>

$\Theta(n \log n)$ (Stirling).

</details>

### Múltipla escolha

**Q21.** Qual afirmação é **verdadeira**?

- (A) $n^2 = O(n \log n)$
- (B) $n \log n = \Omega(n^2)$
- (C) $2^n = \Theta(3^n)$
- (D) $n^3 = \Omega(n^2)$
- (E) $\log n = \Theta(\sqrt{n})$

<details><summary>Resposta</summary>

**(D)**. $n^3$ cresce pelo menos tanto quanto $n^2$.

</details>

**Q22.** A recorrência $T(n) = 4T(n/2) + n^2$ resolve-se em:

- (A) $\Theta(n^2)$
- (B) $\Theta(n^2 \log n)$
- (C) $\Theta(n^3)$
- (D) $\Theta(n \log n)$
- (E) $\Theta(n^4)$

<details><summary>Resposta</summary>

**(B)**. $n^{\log_2 4} = n^2 = f(n)$ → caso 2.

</details>

**Q23.** Complexidade do trecho:
```
for i = 1 to n:
  for j = i to n:
    for k = 1 to 10:
      x++
```

- (A) $\Theta(n)$
- (B) $\Theta(n \log n)$
- (C) $\Theta(n^2)$
- (D) $\Theta(n^3)$
- (E) $\Theta(10n)$

<details><summary>Resposta</summary>

**(C)**. Laço interno é constante (10); os dois externos somam $n(n+1)/2$.

</details>

**Q24.** Sobre Insertion Sort, qual afirmação é verdadeira?

- (A) Seu tempo é $\Theta(n^2)$ para qualquer entrada.
- (B) Seu tempo é $O(n^2)$ e $\Omega(n)$.
- (C) Seu tempo é $\Theta(n \log n)$ no caso médio.
- (D) Seu tempo é $O(n)$ no pior caso.
- (E) Seu tempo é $\Omega(n^2)$ para qualquer entrada.

<details><summary>Resposta</summary>

**(B)**. Melhor caso (já ordenado) $\Theta(n)$, pior caso $\Theta(n^2)$.

</details>

---

## 8. Ordenação e Seleção

### Flashcards

**F83.** O que é ordenação **estável**?
<details><summary>Resposta</summary>

Elementos com chaves iguais mantêm a ordem relativa original.

</details>

**F84.** Quais dos clássicos são estáveis?
<details><summary>Resposta</summary>

Bubble, Insertion, Merge, Counting, Radix. **Não estáveis**: Selection, Quick, Heap.

</details>

**F85.** Quais são in-place?
<details><summary>Resposta</summary>

Bubble, Selection, Insertion, Quick (pilha $O(\log n)$), Heap. **Não**: Merge (arrays), Counting, Radix, Bucket.

</details>

**F86.** Melhor caso do Insertion Sort e quando ocorre?
<details><summary>Resposta</summary>

$\Theta(n)$, array já ordenado.

</details>

**F87.** Quando o Quick Sort atinge $O(n^2)$?
<details><summary>Resposta</summary>

Quando o pivô é sempre o menor/maior (ex.: array já ordenado com pivô no início/fim). Recorrência $T(n) = T(n-1) + n$.

</details>

**F88.** Como evitar o pior caso do Quick Sort na prática?
<details><summary>Resposta</summary>

Pivô aleatório, mediana de três; ou Introsort (cai para Heap Sort quando a recursão fica profunda).

</details>

**F89.** Memória extra do Quick Sort?
<details><summary>Resposta</summary>

$O(\log n)$ médio (pilha de recursão); $O(n)$ no pior caso (ou $O(\log n)$ recursando primeiro na menor parte).

</details>

**F90.** Por que Merge Sort é bom para listas ligadas?
<details><summary>Resposta</summary>

Não precisa de acesso aleatório e o merge de listas é feito religando ponteiros, com $O(1)$ de memória extra.

</details>

**F91.** Passos do Heap Sort?
<details><summary>Resposta</summary>

Build max-heap $O(n)$; repetir $n-1$ vezes: troca raiz com último da parte heap, reduz o heap, sift-down $O(\log n)$. Total $O(n \log n)$, in-place, não estável.

</details>

**F92.** Selection Sort: quantas trocas faz?
<details><summary>Resposta</summary>

$O(n)$ (no máximo $n-1$). Mas sempre $\Theta(n^2)$ comparações.

</details>

**F93.** Número de trocas do Insertion/Bubble Sort é igual a quê?
<details><summary>Resposta</summary>

Número de **inversões** do array (pares $i < j$ com $A[i] > A[j]$).

</details>

**F94.** Por que nenhuma ordenação por comparação é $o(n \log n)$?
<details><summary>Resposta</summary>

Árvore de decisão precisa de $n!$ folhas → altura $\ge \log_2(n!) = \Omega(n \log n)$.

</details>

**F95.** Counting Sort: complexidade e quando usar?
<details><summary>Resposta</summary>

$O(n + k)$ tempo e memória, $k$ = intervalo dos valores. Use quando $k = O(n)$ (inteiros pequenos).

</details>

**F96.** Radix Sort LSD: requisito da sub-rotina?
<details><summary>Resposta</summary>

Precisa ser **estável** (geralmente Counting Sort por dígito). Tempo $O(d(n + k))$.

</details>

**F97.** Bucket Sort: hipótese para $O(n)$ médio?
<details><summary>Resposta</summary>

Entrada **uniformemente distribuída** (ex.: em $[0, 1)$). Pior caso $O(n^2)$ (tudo num bucket).

</details>

**F98.** Quickselect: o que faz e complexidade?
<details><summary>Resposta</summary>

Acha o $k$-ésimo menor: particiona e recursa **só** no lado que contém $k$. $O(n)$ médio, $O(n^2)$ pior.

</details>

**F99.** Algoritmo de seleção $O(n)$ no pior caso?
<details><summary>Resposta</summary>

Mediana das medianas (BFPRT): grupos de 5, mediana das medianas como pivô.

</details>

**F100.** Após a passada $i$ do Bubble Sort, o que se garante?
<details><summary>Resposta</summary>

Os $i$ maiores elementos estão nas posições finais corretas.

</details>

### Múltipla escolha

**Q25.** Usando partição de **Lomuto** (pivô = último elemento) sobre `[3, 7, 8, 5, 2, 1, 9, 5, 4]`, o array após a primeira partição é:

- (A) [3, 2, 1, 4, 7, 8, 9, 5, 5]
- (B) [1, 2, 3, 4, 5, 5, 7, 8, 9]
- (C) [3, 2, 1, 4, 5, 7, 8, 9, 5]
- (D) [2, 1, 3, 4, 7, 8, 9, 5, 5]
- (E) [3, 1, 2, 4, 7, 8, 9, 5, 5]

<details><summary>Resposta</summary>

**(A)**. Pivô 4. `i` marca o fim da região ≤ 4:
- 3 ≤ 4 → troca(0,0)
- 2 ≤ 4 → troca(1,4) → [3,2,8,5,7,1,9,5,4]
- 1 ≤ 4 → troca(2,5) → [3,2,1,5,7,8,9,5,4]
- final: troca pivô com posição 3 → [3,2,1,**4**,7,8,9,5,5]

</details>

**Q26.** Qual algoritmo é estável **e** tem pior caso $O(n \log n)$?

- (A) Quick Sort
- (B) Heap Sort
- (C) Merge Sort
- (D) Selection Sort
- (E) Insertion Sort

<details><summary>Resposta</summary>

**(C)**.

</details>

**Q27.** Para ordenar um array **quase ordenado** de 1 milhão de elementos, a melhor escolha entre as opções é:

- (A) Selection Sort
- (B) Insertion Sort
- (C) Heap Sort
- (D) Quick Sort com pivô no primeiro elemento
- (E) Bubble Sort sem flag de parada

<details><summary>Resposta</summary>

**(B)**. $O(n + \text{inversões})$ ≈ $O(n)$. (D) é o pior caso do Quick Sort.

</details>

**Q28.** Deseja-se ordenar $n$ inteiros no intervalo $[0, n^2 - 1]$. Qual complexidade é possível?

- (A) $O(n)$ com Radix Sort na base $n$
- (B) $O(n^2)$ no mínimo, por causa do intervalo
- (C) $O(n \log n)$ no mínimo, por ser ordenação
- (D) $O(n^2)$ com Counting Sort
- (E) $O(\log n)$

<details><summary>Resposta</summary>

**(A)**. Na base $n$ cada número tem 2 dígitos → $d = 2$, $k = n$ → $O(2(n + n)) = O(n)$. Counting Sort direto seria $O(n + n^2)$.

</details>

**Q29.** Qual afirmação é **falsa**?

- (A) Heap Sort é in-place.
- (B) Merge Sort precisa de $\Theta(n)$ memória extra em arrays.
- (C) Counting Sort faz comparações entre elementos.
- (D) Quick Sort é, em média, $O(n \log n)$.
- (E) Selection Sort faz $\Theta(n^2)$ comparações em qualquer caso.

<details><summary>Resposta</summary>

**(C)**. Counting Sort não compara; usa os valores como índices.

</details>

**Q30.** Quickselect com pivô aleatório tem complexidade esperada:

- (A) $O(\log n)$
- (B) $O(n)$
- (C) $O(n \log n)$
- (D) $O(n^2)$
- (E) $O(k \log n)$

<details><summary>Resposta</summary>

**(B)**. $T(n) = T(n/2) + O(n)$ em média → $O(n)$.

</details>

---

## 9. Grafos

### Flashcards

**F101.** Memória de matriz vs lista de adjacência?
<details><summary>Resposta</summary>

Matriz $\Theta(V^2)$; lista $\Theta(V + E)$.

</details>

**F102.** Custo de verificar se existe aresta $(u, v)$ em matriz e em lista?
<details><summary>Resposta</summary>

Matriz $O(1)$; lista $O(\deg(u))$.

</details>

**F103.** Soma dos graus de um grafo não direcionado?
<details><summary>Resposta</summary>

$2E$ (cada aresta conta duas vezes).

</details>

**F104.** BFS encontra caminho mínimo em que tipo de grafo?
<details><summary>Resposta</summary>

**Não ponderado** (ou todos os pesos iguais) — mínimo número de arestas.

</details>

**F105.** Complexidade de BFS/DFS com lista e com matriz?
<details><summary>Resposta</summary>

Lista: $O(V + E)$. Matriz: $O(V^2)$.

</details>

**F106.** Como DFS detecta ciclo em grafo dirigido?
<details><summary>Resposta</summary>

Encontrando uma **aresta de retorno** (back edge) para um vértice ainda na pilha de recursão (cinza).

</details>

**F107.** Tipos de aresta na DFS de grafo dirigido?
<details><summary>Resposta</summary>

Árvore, retorno (back), avanço (forward), cruzamento (cross). Em grafo **não** dirigido só há árvore e retorno.

</details>

**F108.** Ordenação topológica pelo algoritmo de Kahn?
<details><summary>Resposta</summary>

Enfileira vértices com grau de entrada 0; ao remover um, decrementa o grau dos vizinhos e enfileira os que zerarem. Se sobrarem vértices → há ciclo.

</details>

**F109.** Ordenação topológica via DFS?
<details><summary>Resposta</summary>

Ordem **decrescente de tempo de término** (pós-ordem invertida).

</details>

**F110.** O que é componente fortemente conexo (SCC)?
<details><summary>Resposta</summary>

Conjunto maximal de vértices em grafo **dirigido** em que todo par $u, v$ tem caminho $u \to v$ e $v \to u$.

</details>

**F111.** Passos de Kosaraju?
<details><summary>Resposta</summary>

(1) DFS em $G$ registrando ordem de término. (2) Transpor $G$. (3) DFS em $G^T$ em ordem decrescente de término; cada árvore = um SCC. $O(V + E)$.

</details>

**F112.** O que Tarjan usa para achar SCCs em uma DFS?
<details><summary>Resposta</summary>

Índice de descoberta, valor *low-link* e uma pilha. Quando `low[v] == idx[v]`, desempilha até `v` → um SCC.

</details>

**F113.** Por que Dijkstra falha com pesos negativos?
<details><summary>Resposta</summary>

Ele finaliza a distância de um vértice ao extraí-lo da fila; uma aresta negativa encontrada depois poderia diminuir essa distância "definitiva".

</details>

**F114.** Complexidade de Dijkstra com heap binário? Com array?
<details><summary>Resposta</summary>

Heap: $O((V + E)\log V)$. Array: $O(V^2)$ (melhor para grafos densos). Heap de Fibonacci: $O(E + V\log V)$.

</details>

**F115.** Quantas rodadas o Bellman-Ford faz e por quê?
<details><summary>Resposta</summary>

$V - 1$: um caminho mínimo simples tem no máximo $V - 1$ arestas. Uma $V$-ésima rodada que ainda relaxa indica **ciclo negativo**.

</details>

**F116.** Recorrência de Floyd-Warshall?
<details><summary>Resposta</summary>

$d[i][j] \leftarrow \min(d[i][j],\ d[i][k] + d[k][j])$, com $k$ no laço **externo**. $\Theta(V^3)$.

</details>

**F117.** Como Floyd-Warshall indica ciclo negativo?
<details><summary>Resposta</summary>

Algum $d[i][i] < 0$ no final.

</details>

**F118.** Quantas arestas tem uma MST de grafo conexo com $V$ vértices?
<details><summary>Resposta</summary>

$V - 1$.

</details>

**F119.** Quando a MST é única?
<details><summary>Resposta</summary>

Quando todos os pesos de aresta são distintos (condição suficiente).

</details>

**F120.** Propriedade do corte?
<details><summary>Resposta</summary>

Para qualquer corte $(S, V \setminus S)$, a aresta de menor peso que o cruza pertence a alguma MST. Justifica Prim e Kruskal.

</details>

**F121.** Diferença entre Prim e Dijkstra?
<details><summary>Resposta</summary>

Mesmo esqueleto (fila de prioridade). Prim usa como chave o **peso da aresta** que liga à árvore; Dijkstra usa a **distância acumulada** desde a origem.

</details>

**F122.** Caminho mínimo em DAG com pesos negativos: como e quanto custa?
<details><summary>Resposta</summary>

Ordenação topológica + relaxar arestas nessa ordem. $O(V + E)$.

</details>

**F123.** Como verificar se um grafo é bipartido?
<details><summary>Resposta</summary>

BFS/DFS colorindo com 2 cores; se uma aresta liga vértices da mesma cor → não bipartido (tem ciclo ímpar). $O(V + E)$.

</details>

### Múltipla escolha

**Q31.** Grafo não dirigido com arestas: A–B (4), A–C (1), C–B (2), B–D (1), C–D (5). Dijkstra a partir de A. As distâncias finais para B e D são:

- (A) B = 4, D = 5
- (B) B = 3, D = 4
- (C) B = 3, D = 6
- (D) B = 4, D = 6
- (E) B = 2, D = 3

<details><summary>Resposta</summary>

**(B)**. Extrai A(0): C=1, B=4. Extrai C(1): B=min(4, 1+2)=3, D=1+5=6. Extrai B(3): D=min(6, 3+1)=4. Extrai D(4).

</details>

**Q32.** Mesmo grafo. Qual o peso total da MST?

- (A) 4
- (B) 5
- (C) 6
- (D) 7
- (E) 8

<details><summary>Resposta</summary>

**(A)**. Kruskal: A–C(1) ✓, B–D(1) ✓, C–B(2) ✓ → 3 arestas = $V-1$. Total 1+1+2 = **4**.

</details>

**Q33.** DAG com arestas A→B, A→C, B→D, C→D, D→E. Qual **não** é ordenação topológica válida?

- (A) A, B, C, D, E
- (B) A, C, B, D, E
- (C) A, B, D, C, E
- (D) Todas as anteriores são válidas
- (E) A e B são válidas

<details><summary>Resposta</summary>

**(C)**. D aparece antes de C, mas existe C→D.

</details>

**Q34.** Para encontrar o caminho mínimo de uma origem em um grafo com pesos **negativos** e sem ciclos negativos, usa-se:

- (A) BFS
- (B) Dijkstra
- (C) Prim
- (D) Bellman-Ford
- (E) Kruskal

<details><summary>Resposta</summary>

**(D)**.

</details>

**Q35.** Um grafo esparso tem $V = 10^6$ e $E = 3 \times 10^6$. Qual representação é adequada?

- (A) Matriz de adjacência, por permitir checagem de aresta em $O(1)$
- (B) Lista de adjacência, por usar $O(V + E)$ de memória
- (C) Matriz de incidência
- (D) Qualquer uma, a memória é a mesma
- (E) Matriz de adjacência, por ser mais rápida para BFS

<details><summary>Resposta</summary>

**(B)**. Matriz exigiria $10^{12}$ posições.

</details>

**Q36.** Qual algoritmo calcula caminhos mínimos entre **todos** os pares em $\Theta(V^3)$ usando programação dinâmica?

- (A) Dijkstra
- (B) Bellman-Ford
- (C) Floyd-Warshall
- (D) Kosaraju
- (E) Kahn

<details><summary>Resposta</summary>

**(C)**.

</details>

**Q37.** Sobre Kruskal e Prim, é correto:

- (A) Kruskal funciona apenas em grafos dirigidos.
- (B) Prim requer Union-Find.
- (C) Ambos são gulosos e produzem uma MST.
- (D) Kruskal é sempre mais rápido que Prim.
- (E) Prim falha com pesos negativos.

<details><summary>Resposta</summary>

**(C)**. MST aceita pesos negativos (ao contrário de Dijkstra).

</details>

**Q38.** Em uma BFS a partir de $s$ em grafo não ponderado, a distância de $s$ a $v$ é:

- (A) O tempo de término de $v$
- (B) O nível de $v$ na árvore BFS
- (C) O grau de $v$
- (D) O número de vértices visitados antes de $v$
- (E) O tempo de descoberta da DFS

<details><summary>Resposta</summary>

**(B)**.

</details>

---

## 10. NP-Completude

### Flashcards

**F124.** Definição de P?
<details><summary>Resposta</summary>

Problemas de decisão resolvíveis em tempo polinomial por máquina determinística.

</details>

**F125.** Definição de NP?
<details><summary>Resposta</summary>

Problemas de decisão cujas respostas "sim" têm um **certificado verificável em tempo polinomial** (equivalente: resolvíveis em tempo polinomial por máquina não-determinística).

</details>

**F126.** P ⊆ NP? P = NP?
<details><summary>Resposta</summary>

P ⊆ NP: **sim** (resolver já verifica). P = NP: **problema em aberto**.

</details>

**F127.** O que significa $A \le_p B$?
<details><summary>Resposta</summary>

Existe função computável em tempo polinomial que transforma instâncias de $A$ em instâncias de $B$ preservando a resposta. $B$ é **pelo menos tão difícil** quanto $A$.

</details>

**F128.** Para provar que $X$ é NP-completo?
<details><summary>Resposta</summary>

(1) $X \in$ NP (verificador polinomial). (2) Algum NP-completo conhecido $A \le_p X$.

</details>

**F129.** Qual foi o primeiro problema provado NP-completo?
<details><summary>Resposta</summary>

SAT (Teorema de Cook-Levin, 1971).

</details>

**F130.** Um problema NP-difícil precisa estar em NP?
<details><summary>Resposta</summary>

**Não**. Ex.: Problema da Parada é NP-difícil e indecidível. Versão de otimização do TSP também é NP-difícil.

</details>

**F131.** Se um problema NP-completo tiver algoritmo polinomial, o que acontece?
<details><summary>Resposta</summary>

P = NP.

</details>

**F132.** 2-SAT e 3-SAT: classes?
<details><summary>Resposta</summary>

2-SAT ∈ P (grafo de implicação + SCC). 3-SAT é NP-completo.

</details>

**F133.** Ciclo Euleriano vs Hamiltoniano?
<details><summary>Resposta</summary>

Euleriano (cada **aresta** uma vez): P — existe sse conexo e todos os graus pares. Hamiltoniano (cada **vértice** uma vez): NP-completo.

</details>

**F134.** Por que a DP $O(nW)$ da mochila não prova P = NP?
<details><summary>Resposta</summary>

É **pseudo-polinomial**: polinomial no **valor** de $W$, mas $W$ é codificado com $\log W$ bits → exponencial no tamanho da entrada.

</details>

**F135.** Relação entre Clique, Conjunto Independente e Vertex Cover?
<details><summary>Resposta</summary>

$S$ é conjunto independente em $G$ ⇔ $S$ é clique no complemento $\bar G$ ⇔ $V \setminus S$ é cobertura de vértices em $G$.

</details>

**F136.** Aproximação clássica para Vertex Cover?
<details><summary>Resposta</summary>

Pegar um emparelhamento maximal e incluir os dois extremos de cada aresta → 2-aproximação.

</details>

**F137.** Cite 5 problemas NP-completos.
<details><summary>Resposta</summary>

SAT, 3-SAT, Clique, Vertex Cover, Ciclo Hamiltoniano, TSP (decisão), Subset Sum, Mochila (decisão), 3-Coloração, Set Cover, Conjunto Independente.

</details>

### Múltipla escolha

**Q39.** Sabendo que 3-SAT é NP-completo, para provar que um problema $X$ é NP-difícil, deve-se:

- (A) Reduzir $X$ a 3-SAT em tempo polinomial
- (B) Reduzir 3-SAT a $X$ em tempo polinomial
- (C) Mostrar um verificador polinomial para $X$
- (D) Mostrar um algoritmo exponencial para $X$
- (E) Mostrar que $X \in$ P

<details><summary>Resposta</summary>

**(B)**. (A) só mostraria que $X$ não é mais difícil que 3-SAT. (C) mostra $X \in$ NP.

</details>

**Q40.** Qual afirmação é **verdadeira**?

- (A) NP significa "não polinomial".
- (B) Todo problema em NP exige tempo exponencial.
- (C) Se P ≠ NP, nenhum problema NP-completo está em P.
- (D) Todo problema NP-difícil está em NP.
- (E) O problema do caminho mínimo é NP-completo.

<details><summary>Resposta</summary>

**(C)**.

</details>

**Q41.** Qual problema está em P?

- (A) Ciclo Hamiltoniano
- (B) 3-Coloração
- (C) 2-SAT
- (D) Clique
- (E) Subset Sum

<details><summary>Resposta</summary>

**(C)**.

</details>

**Q42.** Suponha $A \le_p B$ e $B \in$ P. Então:

- (A) $A \in$ P
- (B) $A$ é NP-completo
- (C) $B$ é NP-difícil
- (D) Nada se pode concluir sobre $A$
- (E) $A \notin$ NP

<details><summary>Resposta</summary>

**(A)**. Transforma a instância de $A$ (polinomial) e resolve $B$ (polinomial).

</details>

**Q43.** A versão de **decisão** do TSP ("existe rota de custo $\le K$?") é:

- (A) Em P
- (B) NP-completa
- (C) Indecidível
- (D) NP-difícil, mas não em NP
- (E) Resolvível por Dijkstra

<details><summary>Resposta</summary>

**(B)**. O certificado (a rota) é verificável em tempo polinomial.

</details>

---

## 11. Autômatos Finitos e Expressões Regulares

### Flashcards

**F138.** 5-tupla de um AFD?
<details><summary>Resposta</summary>

$(Q, \Sigma, \delta, q_0, F)$: estados, alfabeto, transição $\delta: Q \times \Sigma \to Q$, estado inicial, estados finais.

</details>

**F139.** Diferença de $\delta$ entre AFD e AFN-ε?
<details><summary>Resposta</summary>

AFD: $Q \times \Sigma \to Q$. AFN-ε: $Q \times (\Sigma \cup \{\epsilon\}) \to \mathcal{P}(Q)$ (conjunto de estados, incluindo transições vazias).

</details>

**F140.** AFN reconhece mais linguagens que AFD?
<details><summary>Resposta</summary>

**Não** — exatamente as mesmas (linguagens regulares).

</details>

**F141.** Quantos estados pode ter o AFD obtido de um AFN com $n$ estados?
<details><summary>Resposta</summary>

Até $2^n$ (construção de subconjuntos). Esse limite é atingível.

</details>

**F142.** O que é ε-fecho de um estado?
<details><summary>Resposta</summary>

Conjunto de estados alcançáveis a partir dele usando só transições ε (incluindo ele próprio).

</details>

**F143.** Algoritmo que converte regex em AFN-ε?
<details><summary>Resposta</summary>

Construção de **Thompson** (um fragmento por operador: união, concatenação, estrela).

</details>

**F144.** Teorema de Kleene?
<details><summary>Resposta</summary>

Uma linguagem é regular ⇔ é descrita por expressão regular ⇔ é aceita por autômato finito.

</details>

**F145.** Linguagens regulares são fechadas sob complemento? E interseção?
<details><summary>Resposta</summary>

Sim, ambos. (Complemento: inverter finais do AFD. Interseção: autômato produto.)

</details>

**F146.** Enunciado do Lema do Bombeamento?
<details><summary>Resposta</summary>

Se $L$ é regular, $\exists p$ tal que $\forall w \in L$ com $|w| \ge p$, $w = xyz$, $|xy| \le p$, $|y| \ge 1$ e $\forall i \ge 0$: $xy^iz \in L$.

</details>

**F147.** Para que serve o Lema do Bombeamento?
<details><summary>Resposta</summary>

Provar (por contradição) que uma linguagem **não** é regular. Não serve para provar que é regular.

</details>

**F148.** $\{a^n b^n \mid n \ge 0\}$ é regular?
<details><summary>Resposta</summary>

Não. É livre de contexto (autômato de pilha).

</details>

**F149.** $a^* b^*$ é regular?
<details><summary>Resposta</summary>

Sim (é uma regex!). Não exige mesma quantidade de $a$ e $b$.

</details>

**F150.** Toda linguagem finita é regular?
<details><summary>Resposta</summary>

Sim (união finita de strings).

</details>

**F151.** Precedência dos operadores em regex?
<details><summary>Resposta</summary>

Estrela `*` > concatenação > união `|`. Ex.: `ab*|c` = `(a(b*))|c`.

</details>

**F152.** Qual máquina reconhece linguagens livres de contexto?
<details><summary>Resposta</summary>

Autômato de pilha (AP).

</details>

**F153.** AFD para binários divisíveis por 3: quantos estados e transição?
<details><summary>Resposta</summary>

3 estados (resto 0, 1, 2). $\delta(r, b) = (2r + b) \bmod 3$. Inicial e final: resto 0.

</details>

**F154.** O AFD mínimo é único?
<details><summary>Resposta</summary>

Sim, a menos de renomeação dos estados (Myhill-Nerode).

</details>

### Múltipla escolha

**Q44.** Qual linguagem sobre $\{a, b\}$ **não** é regular?

- (A) Strings com número par de $a$
- (B) Strings que terminam em $ab$
- (C) $\{a^n b^m \mid n, m \ge 0\}$
- (D) $\{a^n b^n \mid n \ge 0\}$
- (E) $\{a, ab, abb\}$

<details><summary>Resposta</summary>

**(D)**. (E) é finita → regular.

</details>

**Q45.** O AFN abaixo aceita strings sobre $\{a,b\}$ que terminam em `ab`:
- $q_0 \xrightarrow{a,b} q_0$, $q_0 \xrightarrow{a} q_1$, $q_1 \xrightarrow{b} q_2$, $F = \{q_2\}$.

Quantos estados **alcançáveis** tem o AFD obtido pela construção de subconjuntos?

- (A) 2
- (B) 3
- (C) 4
- (D) 7
- (E) 8

<details><summary>Resposta</summary>

**(B)**. $\{q_0\}$, $\{q_0, q_1\}$, $\{q_0, q_2\}$.
- $\{q_0\}$: a→$\{q_0,q_1\}$, b→$\{q_0\}$
- $\{q_0,q_1\}$: a→$\{q_0,q_1\}$, b→$\{q_0,q_2\}$
- $\{q_0,q_2\}$: a→$\{q_0,q_1\}$, b→$\{q_0\}$

</details>

**Q46.** Qual regex descreve strings sobre $\{0,1\}$ que contêm `11` como substring?

- (A) `(0|1)*11`
- (B) `11(0|1)*`
- (C) `(0|1)*11(0|1)*`
- (D) `(01)*11(10)*`
- (E) `1*11*`

<details><summary>Resposta</summary>

**(C)**. (A) exige terminar em 11; (B) exige começar com 11.

</details>

**Q47.** Linguagens regulares **não** são fechadas sob:

- (A) União
- (B) Complemento
- (C) Interseção
- (D) Estrela de Kleene
- (E) Nenhuma das anteriores — são fechadas sob todas

<details><summary>Resposta</summary>

**(E)**.

</details>

**Q48.** Sobre o Lema do Bombeamento para linguagens regulares:

- (A) Se $L$ satisfaz o lema, então $L$ é regular.
- (B) É usado para provar que uma linguagem não é regular.
- (C) Exige $|y| = 0$.
- (D) Vale apenas para linguagens finitas.
- (E) Prova que todo AFN tem AFD equivalente.

<details><summary>Resposta</summary>

**(B)**.

</details>

**Q49.** A string `aab` é aceita pela regex `a*b|b*a`?

- (A) Sim, pela primeira alternativa
- (B) Sim, pela segunda alternativa
- (C) Não
- (D) Só se `|` tiver precedência sobre concatenação
- (E) Depende do autômato

<details><summary>Resposta</summary>

**(A)**. `a*b` casa `aa` + `b`.

</details>

**Q50.** Qual afirmação é **falsa**?

- (A) Todo AFN-ε pode ser convertido em AFN sem ε.
- (B) A conversão AFN → AFD pode gerar número exponencial de estados.
- (C) Parênteses balanceados formam uma linguagem regular.
- (D) Analisadores léxicos usam autômatos finitos.
- (E) Expressões regulares e AFDs têm o mesmo poder expressivo.

<details><summary>Resposta</summary>

**(C)**. Exige memória ilimitada (contagem) → livre de contexto.

</details>

---

# Simulado (20 questões — 1h30)

> Faça sem consultar. Gabarito comentado no final.

**S1.** Qual estrutura oferece inserção $O(\log n)$ e acesso ao mínimo em $O(1)$?
(A) BST não balanceada (B) Min-heap (C) Hash table (D) Lista ordenada (E) Pilha

**S2.** $T(n) = 3T(n/2) + n$ é:
(A) $\Theta(n)$ (B) $\Theta(n \log n)$ (C) $\Theta(n^{\log_2 3})$ (D) $\Theta(n^2)$ (E) $\Theta(n^{1,5}\log n)$

**S3.** Qual algoritmo de ordenação **não** é estável?
(A) Merge Sort (B) Insertion Sort (C) Counting Sort (D) Heap Sort (E) Bubble Sort

**S4.** Em uma hash table com encadeamento, $n = 2000$ chaves e $m = 500$ buckets, o tamanho esperado de cada lista é:
(A) 0,25 (B) 1 (C) 4 (D) 500 (E) 2000

**S5.** Qual sequência é a pré-ordem da BST obtida inserindo 8, 3, 10, 1, 6, 14?
(A) 1 3 6 8 10 14 (B) 8 3 1 6 10 14 (C) 1 6 3 14 10 8 (D) 8 3 10 1 6 14 (E) 8 10 14 3 6 1

**S6.** Em Union-Find, compressão de caminho **e** união por rank juntas dão custo amortizado:
(A) $O(1)$ exato (B) $O(\alpha(n))$ (C) $O(\log n)$ (D) $O(\log \log n)$ (E) $O(n)$

**S7.** Qual problema é NP-completo?
(A) Ordenar $n$ números (B) MST (C) Ciclo Euleriano (D) Vertex Cover (decisão) (E) 2-SAT

**S8.** O número mínimo de comparações no pior caso de qualquer ordenação por comparação é:
(A) $\Theta(n)$ (B) $\Theta(n \log n)$ (C) $\Theta(n^2)$ (D) $\Theta(\log n)$ (E) $\Theta(n!)$

**S9.** Grafo dirigido: A→B (1), A→C (4), B→C (−2), C→D (3). Distância mínima de A a D:
(A) 7 (B) 4 (C) 2 (D) 5 (E) 1

**S10.** Qual algoritmo é adequado para S9 com garantia de correção?
(A) Dijkstra (B) BFS (C) Bellman-Ford (D) Prim (E) Kruskal

**S11.** Uma pilha é usada para converter `A * (B + C)` em pós-fixa. O resultado é:
(A) `A B C + *` (B) `A B * C +` (C) `* A + B C` (D) `A B C * +` (E) `A + B C *`

**S12.** Qual linguagem é regular?
(A) $\{ww \mid w \in \{a,b\}^*\}$ (B) Palíndromos (C) $\{a^n b^n\}$ (D) $\{a^{2n} \mid n \ge 0\}$ (E) $\{a^p \mid p \text{ primo}\}$

**S13.** Remover a raiz de uma BST com dois filhos, usando sucessor em-ordem, exige encontrar:
(A) o máximo da subárvore esquerda (B) o mínimo da subárvore direita (C) o filho direito (D) a folha mais profunda (E) o pai da raiz

**S14.** Qual operação de um array **ordenado** é $O(\log n)$?
(A) Inserção (B) Remoção (C) Busca (D) Inverter (E) Concatenar

**S15.** Quick Sort com pivô sempre no **último** elemento, aplicado a um array já ordenado, executa em:
(A) $\Theta(n)$ (B) $\Theta(n \log n)$ (C) $\Theta(n^2)$ (D) $\Theta(n^3)$ (E) $\Theta(\log n)$

**S16.** Ordenação topológica existe se e somente se o grafo:
(A) é conexo (B) é árvore (C) é dirigido e acíclico (D) é bipartido (E) tem pesos positivos

**S17.** Um AFN com 4 estados é convertido em AFD. O número máximo de estados do AFD é:
(A) 4 (B) 8 (C) 16 (D) 24 (E) 256

**S18.** Recursão com `return f(n-1) + 1` e caso base $f(0) = 0$ usa memória de pilha:
(A) $O(1)$ (B) $O(\log n)$ (C) $O(n)$ (D) $O(n^2)$ (E) $O(2^n)$

**S19.** Se um algoritmo polinomial for encontrado para 3-SAT:
(A) Apenas 3-SAT fica em P (B) P = NP (C) NP-difícil = P (D) Problema da Parada fica decidível (E) Nada muda

**S20.** Qual afirmação sobre BFS e DFS é **falsa**?
(A) Ambas são $O(V+E)$ com listas de adjacência.
(B) BFS acha caminhos com menos arestas.
(C) DFS pode detectar ciclos.
(D) BFS usa pilha.
(E) DFS é usada em Kosaraju.

<details><summary>Gabarito comentado</summary>

| # | Resp. | Justificativa |
| :-- | :-- | :-- |
| S1 | B | Heap: topo $O(1)$, insert $O(\log n)$. |
| S2 | C | $n^{\log_2 3} \approx n^{1,58}$ domina $n$ → caso 1. |
| S3 | D | Heap Sort troca elementos distantes. |
| S4 | C | $\alpha = 2000/500 = 4$. |
| S5 | B | Raiz, subárvore esquerda (3,1,6), direita (10,14). |
| S6 | B | Inversa de Ackermann. |
| S7 | D | Os demais estão em P. |
| S8 | B | Árvore de decisão: $\log_2 n!$. |
| S9 | C | A→B→C→D = 1 − 2 + 3 = 2. |
| S10 | C | Há peso negativo; Dijkstra não garante. |
| S11 | A | Parênteses primeiro: `B C +`, depois `* ` com A. |
| S12 | D | Strings de $a$ de tamanho par: `(aa)*`. |
| S13 | B | Sucessor = mínimo à direita. |
| S14 | C | Busca binária. |
| S15 | C | Partições de tamanho $n-1$ e 0. |
| S16 | C | DAG. |
| S17 | C | $2^4 = 16$. |
| S18 | C | Não é cauda (há `+ 1` pendente); profundidade $n$. |
| S19 | B | 3-SAT é NP-completo. |
| S20 | D | BFS usa fila. |

**Nota:** ≥ 12 acertos = 6,0 (mínimo para aprovação). Mire em 16+.

</details>
