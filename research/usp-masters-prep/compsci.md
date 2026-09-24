# Computer Science

## Mock Tests

- [Flashcards](https://gemini.google.com/share/d/1aornDEUH8iBDNt8NwXka1HZ8NXLTmMro?usp=sharing)
- [20 Quizzes Test](https://gemini.google.com/share/d/1S7BW--7cQobPc1Q2Dt1ugwxmwHAJSUJB?usp=sharing)
- [51 Quizzes Test](https://gemini.google.com/share/d/1xVI2j3eqVCTqyT1G0bvw_qdXbIilvN4R?usp=sharing)

## Prep

Topics to study

- [X] Arrays
- [X] Listas Ligadas
- [X] Pilhas
- [X] Filas
- [X] Listas de Prioridade
- [X] Recursão
- [X] Hash Tables
- [X] BST
- [X] Union-Find
- [X] Complexidade Algorítmica e Notação Assintótica
- [X] Algoritmos de Ordenação e Seleção
- [X] Algoritmos para Problemas em Grafos
- [X] P, NP, NP-Completo
- [X] Autômatos Finitos e Expressões Regulares

## 1. Estruturas de Dados

### 1.1 Arrays, Listas Ligadas, Pilhas, Filas e Listas de Prioridade

#### Arrays

O array aloca um bloco contíguo de memória onde cada elemento ocupa uma posição indexada. O acesso por índice é direto — o hardware calcula o endereço de memória em tempo constante. A troca é que inserções e remoções fora do final exigem deslocar os elementos adjacentes, e arrays dinâmicos podem precisar realocar todo o bloco quando a capacidade é esgotada.

**Conceito:** Sequência de elementos em memória contígua, acessados por índice.

* **Operações:**  
  * Busca por índice: $O(1)$  
  * Busca por elemento: $O(n)$  
  * Busca por elemento (array ordenado, busca binária): $O(\log n)$  
  * Adicionar no final (array estático, com memória disponível): $O(1)$  
  * Adicionar no final (array dinâmico, worst case — resize): $O(n)$  
  * Adicionar em posição que não é o final (estático ou dinâmico): $O(n)$ — shift de posição  
  * Remover o último elemento: $O(1)$  
  * Remover elemento que não é o último: $O(n)$ — shift de posição  
* **Memória:** $O(n)$.  
* **Aplicação:** Base de heaps, buffers, cache de CPU, representação de vetores e matrizes.

#### Listas Ligadas (Linked Lists)

Diferente de um array, que aloca um bloco contíguo de memória, a lista ligada espalha seus elementos (nós) pela memória. Cada nó guarda o dado e um ponteiro indicando o próximo elemento. Inserção e remoção no início são $O(1)$, mas qualquer operação no meio ou no final exige percorrer a lista até a posição desejada — $O(n)$. Você também perde o acesso aleatório: para ler o 50º elemento, precisa percorrer os 49 anteriores.

**Conceito:** Sequência de nós onde cada elemento aponta para o próximo na memória. Não ocupa espaço contíguo.

* **Operações:**  
  * Busca por elemento: $O(n)$  
  * Adicionar no início: $O(1)$  
  * Adicionar no meio: $O(n)$  
  * Adicionar no final: $O(n)$  
  * Remover no início: $O(1)$  
  * Remover no meio: $O(n)$  
  * Remover no final: $O(n)$  
* **Memória:** $O(n)$. Gasta mais que arrays devido aos ponteiros extras.  
* **Aplicação:** Implementação de pilhas, filas e gerenciamento de memória dinâmica.

#### Pilhas (Stacks)

A pilha é uma abstração focada em restrição de acesso. Imagine uma pilha de pratos: você só pode colocar um prato no topo (`push`) e tirar o prato do topo (`pop`). O sistema operacional usa isso exaustivamente na *Call Stack* (pilha de chamadas): quando uma função chama outra, o estado da função atual é "empilhado" até que a nova função termine e retorne, momento em que o estado anterior é "desempilhado" e a execução continua.

**Conceito:** Segue a política LIFO (*Last In, First Out*). O último a entrar é o primeiro a sair.

* **Operações:**  
  * Busca por elemento: $O(n)$  
  * Peek (ver o topo): $O(1)$  
  * Push (inserir no topo): $O(1)$  
  * Pop (remover do topo): $O(1)$  
* **Memória:** $O(n)$.  
* **Aplicação:** Desfazer (Undo), avaliação de expressões matemáticas, chamadas de funções (Call Stack).

#### Filas (Queues)

A fila espelha o comportamento de uma fila de banco: o primeiro a entrar é o primeiro a sair. Em sistemas reais, filas são usadas como *buffers*. Se um servidor web recebe mais requisições do que consegue processar instantaneamente, ele coloca essas requisições em uma fila. Processos assíncronos (como o envio de milhares de e-mails) retiram tarefas dessa fila uma a uma, garantindo que nada se perca e que o sistema não trave.

**Conceito:** Segue a política FIFO (*First In, First Out*). O primeiro a entrar é o primeiro a sair.

* **Operações:**  
  * Busca por elemento: $O(n)$  
  * Pegar o primeiro (front): $O(1)$  
  * Enqueue (adicionar no fim): $O(1)$  
  * Dequeue (remover do início): $O(1)$  
* **Memória:** $O(n)$.
* **Aplicação:** Escalonamento de processos, buffers de impressão, IO de disco.

#### Listas de Prioridade (Heaps)

[Resource](https://www.youtube.com/watch?v=5EJKm_u6P1o)

Uma lista de prioridade garante que o elemento de maior (ou menor) relevância esteja sempre acessível imediatamente. A forma mais eficiente de construir isso é através de um *Heap*, que é uma árvore binária completa representada matematicamente dentro de um array simples. Quando você insere um elemento, ele vai para o final do array e "flutua" para cima (`sift-up`) trocando de lugar com os "pais" até chegar à posição correta. Isso custa apenas $O(\log n)$ e é o motor por trás de algoritmos de roteamento de GPS (como o Dijkstra).

**Conceito:** Estrutura onde cada elemento tem uma prioridade. Geralmente implementada como um **Heap** (árvore binária completa num array).

**Insert**:

- Adiciona o elemento na última posição do array
- Compara com o nó pai: verifica se é mais prioritário (ex: em MaxHeap, mais prioritário significa número maior que o valor do nó pai)
- Se for mais prioritário, faz o swap
- Continua esse processo até que chegue no inicio da árvore (index 0 do array) ou se o nó pai for mais prioritário

**Extract**:

- Remove o item mais prioritário (primeiro elemento da lista)
- Coloca o último item da lista na primeira posição da lista
- Compara com o nó filha da esquerda e direita e verifica se var fazer o swap e com qual nó fará
- Continua esse processo até que o elemento esteja na posição certa (não tenha mais itens prioritário) ou chegue no nó da árvore

**Sumário**:

* **Operações:**  
  * Busca do elemento mais prioritário (max/min): $O(1)$  
  * Busca por elemento: $O(\log n)$  
  * Insert (adicionar): $O(\log n)$  
  * Extract / remover prioritário: $O(\log n)$  
* **Memória:** $O(n)$.  
* **Aplicação:** Algoritmo de Dijkstra, escalonamento de tarefas por prioridade.

### 1.2 Recursão

A recursão ocorre quando uma função resolve um problema chamando a si mesma com uma entrada ligeiramente menor. A mecânica exige um "caso base" para interromper o loop infinito. Sem o caso base, a função continua empilhando chamadas na memória até causar um *Stack Overflow* (estouro de pilha).

A otimização de **recursão de cauda** (*tail recursion*) é um truque de compiladores modernos: se a chamada recursiva for a absoluta última instrução da função, o compilador não cria um novo quadro na memória, mas sim reaproveita o atual, transformando a recursão em um loop iterativo altamente eficiente por baixo dos panos.

- O gatilho (Zero pendências): A otimização só acontece se a função filha for a última coisa a ser executada. Se houver qualquer operação pendente (como n * f(n-1)), o sistema é forçado a empilhar para guardar o estado atual até a volta do cálculo.
- O que o TCO faz: Quando não há nada pendente, o sistema descarta o estado atual e sobrescreve o mesmo espaço na pilha, em vez de empilhar uma nova chamada por cima da outra.
- O problema que resolve: Elimina o acúmulo de memória e previne o estouro da pilha (Stack Overflow) em recursões profundas.
- O resultado: Reduz o consumo de memória de $O(N)$ (que cresce a cada chamada) para $O(1)$ (constante), fazendo a recursão rodar com a mesma eficiência de um loop while.

**Conceito:** Uma função que chama a si mesma para resolver subproblemas menores.

* **Mecânica:** Exige um **Caso Base** (parada) e um **Caso Recursivo**.  
* **Recursão de Cauda (Tail Call):** Ocorre quando a chamada recursiva é a última ação. Compiladores podem otimizar para usar espaço de pilha $O(1).
* **Tempo/Memória:** Depende do problema (ex: Fatorial é $O(n)$ tempo e $O(n)$ memória na pilha).

### 1.3 Tabelas de Espalhamento (Hash Tables)

Uma Hash Table é a estrutura definitiva para buscas rápidas. Você passa uma chave (como uma string) por uma **função matemática de hash**, que cospe um número inteiro. Esse número é usado como o índice exato de um array onde o valor será guardado.

O grande desafio arquitetural aqui são as **colisões**: quando duas chaves diferentes geram o mesmo número de hash. Para resolver isso, usamos o **Encadeamento** (cada posição do array guarda uma lista ligada de itens que colidiram) ou o **Endereçamento Aberto** (se a posição 5 estiver ocupada, o algoritmo tenta a 6, depois a 7, até achar um espaço vazio). Quando a tabela fica muito cheia (alto fator de carga), ela sofre um *rehash*, onde um array maior é criado e todos os itens são recalculados, garantindo que o tempo médio de busca continue sendo $O(1)$.

**Conceito:** Mapeia chaves para valores usando uma função Hash.

* **Operações:**  
  * Busca por elemento: $O(1)$  
  * Busca por elemento (com colisão, encadeamento): $O(n)$ — lista ligada no bucket  
  * Adicionar elemento: $O(1)$  
  * Remover elemento: $O(1)$  
* **Colisões:** Resolvidas por **Encadeamento** (listas nos índices) ou **Endereçamento Aberto** (busca o próximo índice vazio).  
* **Memória:** $O(n)$ (proporcional ao número de chaves + tamanho do array).  
* **Aplicação:** Bancos de dados, Caches, Implementação de Sets e Mapas.

### 1.4 Árvores Binárias e de Busca (BST)

Uma Árvore Binária de Busca organiza dados de forma que, a partir de qualquer nó, todos os valores à esquerda sejam menores e todos à direita sejam maiores. Isso permite descartar metade dos dados a cada passo de uma busca, imitando a busca binária.

O problema prático é o **desbalanceamento**. Se você inserir dados já ordenados (1, 2, 3, 4, 5) em uma BST simples, ela crescerá apenas para a direita, virando uma lista ligada e degradando o tempo de busca para $O(n)$. É por isso que bancos de dados utilizam árvores auto-balanceadas (como as árvores AVL ou Red-Black). Elas aplicam "rotações" matemáticas nos nós logo após a inserção para garantir que a árvore permaneça simétrica, fixando o tempo de busca em $O(\log n)$.

**Conceito:** Cada nó tem no máximo dois filhos. Na **BST**, o filho à esquerda é menor e o à direita é maior que o pai.

* **Operações (BST Balanceada):** Busca/Inserção/Remoção: $O(\log n)$.
* **Operações (BST Desbalanceada):** Busca: $O(n)$.
* **Roteamento em Árvores Rubro-Negras:**  
  * É uma BST que se auto-balanceia usando uma "cor" (Red/Black) para cada nó.  
  * **Regras:** A raiz é preta; folhas nulas são pretas; um nó vermelho não tem filhos vermelhos; todo caminho da raiz às folhas tem o mesmo número de nós pretos.  
  * **Mecânica:** Se uma inserção viola as regras, a árvore executa **Rotações** (Esquerda ou Direita) e **Recoloração** para manter a altura $O(\log n)$.

### 1.5 Union-Find (Conjuntos Disjuntos)

Esta é uma estrutura de nicho, mas extremamente poderosa para rastrear conexões. Imagine uma rede social onde você quer saber rapidamente se a Pessoa A tem alguma conexão indireta com a Pessoa B. O Union-Find faz isso elegendo um "nó representante" para cada grupo.

- Union: une dois conjuntos (se precisar)
  - Identifica qual os representantes de cada conjunto
  - Ao unir os dois conjuntos, apenas um dos representantes será o representante do novo conjunto
- Find: retornar o representante do conjunto
  - Explora recursivamente o conjunto: procura pelo pai (parent) do nó atual e verifica se é o pai é ele próprio
  - Se ele próprio, é o nó representante; Se não for, continua a exploração recusiva

Aplicações:

- Dois elementos estão no mesmo subset?
- Quantos subsets existem?

**Otimizações:**

- **União por Rank (Union by Rank):** ao unir dois conjuntos, a raiz da árvore de menor rank é anexada como filha da raiz de maior rank, evitando que a estrutura fique desbalanceada como uma lista encadeada.

```
Antes da União:
   1 (Rank 1)       5 (Rank 0)
  /
 2

Após union(1, 5):
     1 (Rank 1)  <-- Permanece com Rank 1 porque 1 > 0
    / \
   2   5
```

- **Compressão de Caminho (Path Compression):** toda vez que você busca o representante de um nó, a estrutura religa todos os nós que que ligam o nó diretamente ao representante principal, acelerando buscas futuras.

```
       1 (Root)
      / \
     2   3
    /
   4
  /
 5
```

`find(5)`: Liga todos os nós que ligam 5 até 1 (nó representante) ao representante, nesse caso, os nós 4 e 5 (2 já está ligado)

```
       1 (Root)
    / / \ \
   5 4   2 3
```

Combinadas, as duas otimizações reduzem a complexidade a $O(\alpha(n))$, a função inversa de Ackermann — que, na prática, é $\le 4$ para qualquer entrada realista.

**Conceito:** Gerencia conjuntos disjuntos. Determina rapidamente se dois elementos pertencem ao mesmo grupo.

* **Operações:**  
  * Union: une dois conjuntos.  
  * Find: identifica e retorna o representante do conjunto (raiz), percorrendo recursivamente até o pai raiz.
* **Otimizações:** **Compressão de Caminho** e **União por Rank**.  
* **Tempo (sem otimizações):**  
  * Union: $O(n)$  
  * Find: $O(n)$  
* **Tempo (com Union by Rank):**  
  * Union: $O(\log n)$  
  * Find: $O(\log n)$  
* **Tempo (com Path Compression):**  
  * Union: $O(\log n)$  
  * Find: $O(\log n)$  
* **Tempo (com ambas as otimizações):**  
  * Union: $O(\alpha(n))$  
  * Find: $O(\alpha(n))$  
* **Memória:** $O(n)$.

**Find**

```
function find(index) {
  if (parent[index] != index) {
    return find(parent[index]);
  }

  return index;
}
```

**Union**

```
function union(a, b) {
  const repA = find(a);
  const repB = find(b);
  parent[repB] = repA;
}
```

---

## 2. Algoritmos e Linguagens Formais

### 2.1 Complexidade Algorítmica e Notação Assintótica

Esta é a régua com a qual medimos a escalabilidade do código. Não medimos o tempo em segundos (pois isso depende do hardware), mas sim como o número de operações cresce conforme a entrada ($n$) aumenta.

* $O(1)$: Constante. Não importa se há 10 ou 1 bilhão de itens, leva o mesmo tempo.
* $O(\log n)$: Logarítmico. Extremamente eficiente. Se os dados dobrarem, o algoritmo só faz uma operação a mais.
* $O(n^2)$: Quadrático. O pesadelo da escalabilidade. Um loop dentro de outro loop. Se os dados dobrarem, o tempo aumenta em 4 vezes. Se aumentarem 10 vezes, o tempo aumenta 100 vezes.

**Conceito:** Mede como o tempo ou espaço cresce com o tamanho da entrada ($n$).

* **Notações:**  
  * $O(g(n))$ (Big-O): Limite superior (pior caso).  
  * $Ω(g(n))$ (Omega): Limite inferior (melhor caso).  
  * $ϴ(g(n))$ (Theta): Limite justo (comportamento exato).  
* **Hierarquia comum:** $O(1) < O(\log n) < O(n) < O(n \log n) < O(n²) < O(2ᴺ) < O(n!)$.

### 2.2 Algoritmos de Ordenação e Seleção

A escolha do algoritmo de ordenação depende de recursos de memória e da natureza dos dados.

* **Quick Sort:** Elege um elemento como "pivô" e reorganiza o array original movendo os valores menores para a esquerda e os maiores para a direita.
  * Loop: Escolhe um pivô
    * Valor da esquerda: valor maior que o pivô
    * Valor da direita: valor menor que o pivô
    * Faz o swap da esquerda e da direita, deixando valores menores na esquerda e maiores na direita
    * Se o index da esquerda for maior que o da direita, acaba os swaps para esse pivô, faz um último swap entre o pivô e o valor da esquerda (maior elemento, mantendo valores menores na esquerda)
    * Para o pivô, todos os elementos da esquerda são menores que ele, e todos os da direita são maiores
* **Merge Sort:** Corta o array pela metade repetidamente até os elementos ficarem isolados, para então ordená-los e juntá-los novamente.
* **Heap Sort:** Constrói uma estrutura de árvore (max-heap) com os dados para extrair repetidamente o maior elemento e colocá-lo no final da lista. 
  * Loop: max-heap → swap primeiro (max) e último (min) → extrai o novo último (max) → coloca no final da lista
  * $O(n \log n)$: heapify is $O(\log n)$, called n - 1 times
* **Selection Sort:** Percorre o array repetidas vezes para encontrar o menor elemento da parte não ordenada e trocá-lo para a posição correta.
* **Counting Sort:** Ordena os dados sem fazer comparações diretas, agrupando os números por seus valores literais em "baldes" de contagem.
  * Conta quantos vezes o número aparece na lista
  * Faz o cumulativo e define em qual index cada número inicia na lista
  * Itera sobre a lista e adiciona cada número no index. Ao adicionar o número, incrementa o index na lista auxiliar

| Algoritmo | Tempo (Médio) | Tempo (Pior) |
| :---- | :---- | :---- |
| Quick Sort | $O(n \log n)$ | $O(n^2)$ |
| Merge Sort | $O(n \log n)$ | $O(n \log n)$ |
| Heap Sort | $O(n \log n)$ | $O(n \log n)$ |
| Selection Sort | $O(n^2)$ | $O(n^2)$ |
| Counting Sort | $O(n + k)$* | $O(n + k)$* |

\* $k$ = intervalo (range) dos valores de entrada.

Resources:

- [Heap Sort](https://www.youtube.com/watch?v=2DmK_H7IdTo)

### 2.3 Algoritmos para Problemas em Grafos

Grafos mapeiam redes (estradas, roteadores, conexões sociais).

* **BFS (Largura):** Explora tudo ao seu redor primeiro antes de ir mais fundo. 
  * Usa uma fila para rastrear os próximos passos. 
  * Usa um HashSet para lidar com nodes já explorados, ou seja, quebra o loop de grafos em ciclos
  * Aplicação: É o algoritmo exato que o seu GPS usa para encontrar a rota com o menor número de ruas (caminho mínimo sem pesos).
  * Complexidade: $O(V + E)$
* **DFS (Profundidade):** Explora o máximo um caminho: segue um caminho único até o fim, batendo num beco sem saída e voltando. 
  * Usa recursão/pilha. 
  * Usa um HashSet para lidar com nodes já explorados, ou seja, quebra o loop de grafos em ciclos
  * É perfeito para resolver labirintos ou checar dependências de código (qual biblioteca deve ser compilada primeiro - Ordenação Topológica).
  * Aplicação: Detecção de ciclos.
  * Complexidade: $O(V + E)$

**Representação:** Matriz de Adjacência ($O(V²)$ memória) ou Lista de Adjacência ($O(V + E)$ memória).

#### Aplicações de BFS e DFS

* Componentes Conexos (Grafos Não-Direcionados): Identificados diretamente usando tanto BFS quanto DFS
* Componentes Fortemente Conexos (Grafos Direcionados):
  * Algoritmo de Kosaraju-Sharir: Usa duas passagens de DFS (uma no grafo original e outra no grafo transposto).
  * Algoritmo de Tarjan: Encontra os componentes em uma única passagem de DFS usando uma pilha.
* Ordenação Topológica:
  * Algoritmo de Kahn: Baseado no grau de entrada dos vértices (usa uma fila/BFS).
  * DFS com Pós-Ordem Inversa: Utiliza a propriedade de tempo de término da DFS.

#### Caminhos Mínimos (Shortest Paths)

* **Algoritmo de Dijkstra:** A evolução do BFS ([resource](https://www.youtube.com/watch?v=_lHSawdgXpI)).
  * Objetivo: encontrar o caminho mais curto baseado nos pesos
  * Utiliza uma Lista de Prioridade para decidir qual o próximo cruzamento explorar, priorizando sempre as ruas mais rápidas/curtas.
  * Para cada vértice, atualiza a distância, escolhendo a menor distância navegada até este momento
* **Bellman-Ford:** ([resource](https://www.youtube.com/watch?v=obWXjtg0L64))
  * Funciona com pesos negativos. Detecta ciclos negativos. $O(V · E)$.

#### Árvore Geradora Mínima (MST) (minimum spanning trees)

([resource](https://www.youtube.com/watch?v=JZBQLXgSGfs))

É um subgrafo que **conecta todos os vértices de um grafo conexo** e ponderado **sem formar ciclos**, de modo que a **soma dos pesos de suas arestas seja a menor possível**

* **Kruskal:** Usa Union-Find. Ordena arestas e as adiciona sem criar ciclos. $O(E \log E)$
  * Ordena a lista de arestas do menor peso até o maior
  * Union-find: para cada aresta na lista ordenada:
    * Se os 2 nós estiverem unidos (mesmo grupo), não adicione a aresta no grupo (é ignorado para não criar ciclos)
    * Se os 2 nós não estiverem unidos, adicione a aresta e una os 2 nós
* **Prim:** Usa Lista de Prioridade. Cresce a árvore a partir de um nó inicial. $O(E \log V)$.

### 2.4 NP-Completude

Este é o limite da computação moderna. Problemas **P** são fáceis de resolver. Problemas **NP** são aqueles em que é impossível achar a resposta rapidamente, mas se alguém te der a resposta pronta, é fácil verificar se está certa.

Problemas **NP-Completos** (como o do Caixeiro Viajante: qual a rota mais curta passando por 50 cidades?) são tão difíceis que tentar todas as combinações demoraria mais que a idade do universo para rodar, mesmo no melhor supercomputador do mundo. Quando um engenheiro prova que seu problema é NP-Completo, ele para de tentar achar a resposta perfeita e passa a usar heurísticas e Inteligência Artificial para achar uma resposta "boa o suficiente".

**Conceito:** Classificação de problemas quanto à dificuldade de resolução.

* **P:** Problemas que podem ser resolvidos em tempo polinomial ($O(nᴷ)$).
* **NP:** Problemas cuja solução pode ser **verificada** em tempo polinomial.  
* **NP-Completo:** Os problemas mais difíceis de NP. Se um for resolvido em tempo polinomial, todos em NP também serão ($P = NP$).  
* **Exemplos:** Caixeiro Viajante, Problema da Mochila, Satisfatibilidade Booleana (SAT).

### 2.5 Autômatos Finitos e Expressões Regulares

É a ciência de reconhecer padrões em textos (como validar se um e-mail é válido).

* **Expressões Regulares (Regex):** Uma sintaxe matemática para definir formatos de texto.
* **Autômatos Finitos (AFD e AFN):** São máquinas de estado teóricas. Imagine um fluxograma com círculos (estados) e setas (transições ligadas a letras). Se a máquina terminar em um estado de "sucesso" ao ler sua string letra por letra, o texto é válido.
  * **Autômatos Finitos Determinísticos (AFD):** Para cada estado e entrada, há exatamente um próximo estado.  
  * **Autômatos Finitos Não-Determinísticos (AFN):** Pode haver múltiplos caminhos para uma entrada.  
  * **Equivalência:** Todo AFN pode ser convertido em um AFD. Ambos reconhecem as **Linguagens Regulares**.  
* **O Lema do Bombeamento:** É uma prova matemática de limite. Ele prova que Regex e Autômatos **não têm memória**. Você não consegue escrever um Regex que valide se um código fonte tem parênteses perfeitamente balanceados, porque a máquina não consegue se lembrar de quantos parênteses abertos viu no passado. Para isso, precisamos de um nível computacional acima, as Máquinas de Pilha.
  * **Lema do Bombeamento (Pumping Lemma):** Usado para provar que uma linguagem **não** é regular (ex: strings com parênteses balanceados).

**Conceito:** Modelos matemáticos de computação com memória finita.

* **Expressões Regulares (Regex):** Linguagem para descrever padrões de strings.
  * Aplicações: Facilitar buscas, Validações, Substituições
  * Elementos básicos de busca:
    * Colchetes []: Definem um conjunto de caracteres permitido. Por exemplo, [0-9] encontra qualquer número de 0 a 9.
    * Quantificadores {}: Multiplicam a instrução anterior. [0-9]{6} busca exatamente seis dígitos numéricos.
    * Âncoras de linha: O caractere ^ indica o início da linha, enquanto o $ indica o fim da linha.
    * Intervalos: Comandos como [a-z] buscam qualquer letra de 'a' a 'z'.
  * Substituição avançada:
    * Parênteses (): Criam grupos de captura. Isso permite extrair partes de um padrão e reutilizá-las no destino.
    * Referência de grupo: utiliza-se o $ seguido do número do grupo (ex: \$1, \$2) para reorganizar ou substituir partes do texto, como alterar o formato de uma data.
