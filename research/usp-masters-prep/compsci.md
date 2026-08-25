# Computer Science

## Mock Tests

- [Flashcards](https://gemini.google.com/share/d/1aornDEUH8iBDNt8NwXka1HZ8NXLTmMro?usp=sharing)
- [Test](https://gemini.google.com/share/d/1S7BW--7cQobPc1Q2Dt1ugwxmwHAJSUJB?usp=sharing)

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
- [ ] Algoritmos para Problemas em Grafos
- [ ] NP-Completude
- [ ] Autômatos Finitos e Expressões Regulares

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
* **Recursão de Cauda (Tail Call):** Ocorre quando a chamada recursiva é a última ação. Compiladores podem otimizar para usar espaço de pilha ![][image1].  
* **Tempo/Memória:** Depende do problema (ex: Fatorial é ![][image2] tempo e ![][image2] memória na pilha).

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

* **Operações (BST Balanceada):** Busca/Inserção/Remoção: ![][image3].  
* **Operações (BST Desbalanceada):** Busca: ![][image2].  
* **Roteamento em Árvores Rubro-Negras:**  
  * É uma BST que se auto-balanceia usando uma "cor" (Red/Black) para cada nó.  
  * **Regras:** A raiz é preta; folhas nulas são pretas; um nó vermelho não tem filhos vermelhos; todo caminho da raiz às folhas tem o mesmo número de nós pretos.  
  * **Mecânica:** Se uma inserção viola as regras, a árvore executa **Rotações** (Esquerda ou Direita) e **Recoloração** para manter a altura ![][image3].

### 1.5 Union-Find (Conjuntos Disjuntos)

Esta é uma estrutura de nicho, mas extremamente poderosa para rastrear conexões. Imagine uma rede social onde você quer saber rapidamente se a Pessoa A tem alguma conexão indireta com a Pessoa B.

O Union-Find faz isso elegendo um "nó representante" para cada grupo. 

**Otimizações:**

- **União por Rank (Union by Rank):** ao unir dois conjuntos, a raiz da árvore de menor rank é anexada como filha da raiz de maior rank, evitando que a estrutura fique desbalanceada como uma lista encadeada.
- **Compressão de Caminho (Path Compression):** toda vez que você busca o representante de um nó, a estrutura religa esse nó diretamente ao representante principal, acelerando buscas futuras.

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

**Conceito:** Mede como o tempo ou espaço cresce com o tamanho da entrada (![][image6]).

* **Notações:**  
  * **![][image7]** (Big-O): Limite superior (pior caso).  
  * ![][image8] (Omega): Limite inferior (melhor caso).  
  * ![][image9] (Theta): Limite justo (comportamento exato).  
* **Hierarquia comum:** ![][image10].

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

* **BFS (Largura):** Explora tudo ao seu redor primeiro antes de ir mais fundo. Usa uma fila para rastrear os próximos passos. É o algoritmo exato que o seu GPS usa para encontrar a rota com o menor número de ruas (caminho mínimo sem pesos).
* **DFS (Profundidade):** Segue um caminho único até o fim, batendo num beco sem saída e voltando. Usa recursão/pilha. É perfeito para resolver labirintos ou checar dependências de código (qual biblioteca deve ser compilada primeiro - Ordenação Topológica).
* **Algoritmo de Dijkstra:** A evolução do BFS. Ele usa uma Lista de Prioridade para decidir qual o próximo cruzamento explorar, priorizando sempre as ruas mais rápidas/curtas.

**Representação:** Matriz de Adjacência (![][image16] memória) ou Lista de Adjacência (![][image17] memória).

#### **Busca em Largura (BFS) e Profundidade (DFS)**

* **BFS (Fila):** Explora nível por nível. Aplicação: Caminho mínimo em grafos sem peso. ![][image17].  
* **DFS (Pilha/Recursão):** Explora o máximo um caminho. Aplicação: Detecção de ciclos. ![][image17].

#### **Árvore Geradora Mínima (MST)**

* **Kruskal:** Usa Union-Find. Ordena arestas e as adiciona sem criar ciclos. ![][image18].  
* **Prim:** Usa Lista de Prioridade. Cresce a árvore a partir de um nó inicial. ![][image19].

#### **Caminhos Mínimos**

* **Dijkstra:** Encontra o caminho mais curto de um nó para todos os outros (pesos não-negativos). Usa Heap. ![][image19].  
* **Bellman-Ford:** Funciona com pesos negativos. Detecta ciclos negativos. ![][image20].

### 2.4 NP-Completude

Este é o limite da computação moderna. Problemas **P** são fáceis de resolver. Problemas **NP** são aqueles em que é impossível achar a resposta rapidamente, mas se alguém te der a resposta pronta, é fácil verificar se está certa.

Problemas **NP-Completos** (como o do Caixeiro Viajante: qual a rota mais curta passando por 50 cidades?) são tão difíceis que tentar todas as combinações demoraria mais que a idade do universo para rodar, mesmo no melhor supercomputador do mundo. Quando um engenheiro prova que seu problema é NP-Completo, ele para de tentar achar a resposta perfeita e passa a usar heurísticas e Inteligência Artificial para achar uma resposta "boa o suficiente".

**Conceito:** Classificação de problemas quanto à dificuldade de resolução.

* **P:** Problemas que podem ser resolvidos em tempo polinomial (![][image21]).  
* **NP:** Problemas cuja solução pode ser **verificada** em tempo polinomial.  
* **NP-Completo:** Os problemas mais difíceis de NP. Se um for resolvido em tempo polinomial, todos em NP também serão (![][image22]).  
* **Exemplos:** Caixeiro Viajante, Problema da Mochila, Satisfatibilidade Booleana (SAT).

### 2.5 Autômatos Finitos e Expressões Regulares

É a ciência de reconhecer padrões em textos (como validar se um e-mail é válido).

* **Expressões Regulares (Regex):** Uma sintaxe matemática para definir formatos de texto.
* **Autômatos Finitos (AFD e AFN):** São máquinas de estado teóricas. Imagine um fluxograma com círculos (estados) e setas (transições ligadas a letras). Se a máquina terminar em um estado de "sucesso" ao ler sua string letra por letra, o texto é válido.
* **O Lema do Bombeamento:** É uma prova matemática de limite. Ele prova que Regex e Autômatos **não têm memória**. Você não consegue escrever um Regex que valide se um código fonte tem parênteses perfeitamente balanceados, porque a máquina não consegue se lembrar de quantos parênteses abertos viu no passado. Para isso, precisamos de um nível computacional acima, as Máquinas de Pilha.

**Conceito:** Modelos matemáticos de computação com memória finita.

* **Expressões Regulares (Regex):** Linguagem para descrever padrões de strings.  
* **Autômatos Finitos Determinísticos (AFD):** Para cada estado e entrada, há exatamente um próximo estado.  
* **Autômatos Finitos Não-Determinísticos (AFN):** Pode haver múltiplos caminhos para uma entrada.  
* **Equivalência:** Todo AFN pode ser convertido em um AFD. Ambos reconhecem as **Linguagens Regulares**.  
* **Lema do Bombeamento (Pumping Lemma):** Usado para provar que uma linguagem **não** é regular (ex: strings com parênteses balanceados).

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAXCAYAAABnGz2mAAACGElEQVR4Xu2WSytEYRjH3S8LYkGpmTlzq6nZTiIbuYQPoFiILNTslSYfQSnJwnfwEWzYSFZiI8oCWWBBKTRu/+d4X97znzNzXoZY+NWTOb/ncp6ZM+eMiop/fplYLDbErhSO43TgTxX7kuAkTiQSWUXzSjKZbOY8E41Gs4gc+yAw/4WdLxi+JMVYalKO4/F4BMcXiHuu1SQSiTDy5+w1mNlSbIF0Ol2H3DN7kyq10CYnBOQeiw2QPpy8wXT4lNsw60RyOsy8CXJbiEX2Lqr5mL0GuX61+AD5HsSD6ZigxUC1bx7yzDfhxf1EEWumxHE+6LtlsZhbgzc9+C4wtFc1bnyUFYKmVlV3bXpxoVCo0XSM5WIH2GXHFHlp4u8Ig8Um1Al2tUulUk1BJxQsF5v31Ng0CVj8UL2BrHY47rPptTkH5o6/1+A2b7dpEvzqMGyanR9+vQxmdZk17t2AuDOLGFzGUanjRwmGTQWdULBZLBwOd3pqbJqK1WCxbj/PFOs3wawxXuymVJN+SGYymVqfnHunsmcsF8sV1IhAYs8j3/wlIs/eRHrlZ4W9ic1iyO87xh1vJq7UgG3EnVq2i+sY1TPLXoC/RpwjTlXIa89zUKPON8z+y+ByzmHoLftPUimLsSwbNbSGvS3oX0cssy8bXIIRDD5ib4n8Bj+x/DYwfAGXdYZ9ED+6lEYeuOxKgYdqAv+31bP/s7wCFbu/VDeQFXUAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAACgklEQVR4Xu2VXYuMYRjH1y5CkS1RM2PeNTUHUnOwOCCUnDhBOBA5UOtYXj4BtSdWCd+A8gUkB5w5FGeUA0kbaldhadfL75quW/f859mZZ4ZRMr+6mn3+/+vlfu69556RkSH/KKVSab9qaSgWi9tU6wsWUMjn87cKhcL1arW6Tn2FwZPEJdXTwpwfqqWGwdPWgAWftOdyuZzn+S3xRXMDlUplM/4b1XuB+hIxq3o3Rn2xj9Qw8BaJ76obVsfLrlK9V+jzifkHVV8SG0y8VD2At9dfap/oO4mvsdYvuVxui81QPRESX6dIbv4niLuxyPPC75xlxWZkMpk1qrfAwN2+mIfqxbDD4543F+umsUOrYy2Ad5s4FJ7pcZS4w8ztcV6Mz5hSvQXbKUvsdiYZdsIbPglarVZba1qcF0D/5p9Wc5iYY8Z6e0Gfd1xrDPT7hW7HzZsmDo6h2XMfNhk0nvck1drCyLvmOdZ/MfZdexBrAb9m23r+gqtqY9pFJ+WxsNOqGdzxW/kY43OT+XpGvde5WAuw6ItJPWPGvMG8GjE0OmJ5eh2y6FOdBuDdUJ/nHa4ti/UA3nmtacMX3TFpqRz7QiXpAa9bEG2mUw0bc7OT34SED52SaPLK/EajsSLBa94oqgfMI+dKgnY5/B17Bt499M+qt2HF7NrTBP2d7pRitfV6faXqnOMN5ul16AsdZd4EcTb2go8+rXoiJL+3AuIxMe/FE5qneE3bl4raY77AFtCemc6OXlXPcG9c9T8KAy4w6KPq/ZDNZnNJLzoQfNBy1XuFPrOF6Bd0oHAUDjDsheq94Hf6jOoDhYFTHJUzqqflrx0LxX5sVEsDu7xLtSFD/hd+Att/xzzV+CFTAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEIAAAAXCAYAAAC/F5msAAADqUlEQVR4Xu1WSWgUURCVJG64i3FgyMzPTAYiAfGQQ8SD4oJ686IoKAYPQlTw4oJeXBBBBEExahQUUfRgUDzoSYxxiyJEXEBEQYhKcEMMRKJGo69mqprqyp9MnEQYIQ+K7nr1qv7/1b9/97BhQxjCP0UikVhouUJEeXn5CstlBRbl4vF4g3OuPpVKjbdxCxSvg23TXDQanQJumuYKAVhXEut6bPkQMPGDEP2GeDX5yWQyDv8D7JvVCioqKmKIt4uPJs6mGmxNWlsowDoPwY5bnlDEDbhpAwTEfsJ6LE+gPBQd5eMLtREEmp/lZNKvLC9AbB43ar7hZ8G+a07wHzTirNOvCJy33u6Ekd4xsEZNwu+2Z4Og0BuB135CsG4sYg5PuDksCwM7YRLrvmieuLKystGaE2RrBPQp8HdhLXIWWdAkEdtJZxYO65G41kDfCTtstQTUnI7YqUgkMob8ysrKccjZRflwi4w8AM1RbrrJ8b3jGpjUKl7YI+FosKCQB75GwL8Pe6n8A84cxJjLPsotLS0diwVO5jrnwC/yjUeNAn8VtpK1ezDf0xSjZvpyBEGME7MKBZjEC9LhWicc/Ll95XLtoBHIXefTs+6O8Tcrv9GXJ0DsOl/TjcA4uyXWn4dFn72pPGhWocCnw4BrLKfBOUEjfDWYb9I869Yr/6QvT4DYFr4+szr4GyyngVgXXYt50LSTDdheS0lnP61oRG2OQaj2DeP30qPuZeaLyXeZT3WrxHH/FdYZJGQB128xXIdvTAHVlhvv5DSyadCImT5ewHk5GwGuRfOo+wB+G6yHc9q0PhtIi9zFHm675jSCcV2OjuFpvaZ4dXX1cE8s/SWxvIBiehfBb/XpHR/Yyu+lyQUsdrnNw9jLhONDt17HCaEc7toTFRf+I03S8hqUW1VVNcLyBIo5809PHCbYID5+x2cQR59LpaGd8Bl2D3YNdh7zWyJxH6B5GlpUhrsiHK7vdExgc4j4RKTLfN666B6D14REHnDOJs1hoQvAvYe9gbXDOnQc/hnOI6PPcYmJX1LxkGmdhsucK6F/DNrFkoe1TNQxAv1z9FXzr4BFb3X9OMj6C9Q75kzjBLyoE5bPF7QzUe+i5fMGdzX0VPMFaj2E3bY8AXwbnuwFy+eLQdsNAjqpnfpbHChQ65dZcAm4ZtgPxQ0I2A07UG+v5QcMFN2P4mstny9QLwE7iobcwvVILBaLWk2+wAEdQc3nlh80YNK1litE4IFttNwQgD+0fleONUA/1gAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEoAAAAaCAYAAAAQXsqGAAAD5klEQVR4Xu2YW4hNYRiGxzEK5WKYzN577dkzTA13c+FwI+SQKKcLOUtqUg6l5JIUIlKIUi6IG0kpiahxNyklN+MQOUzkkMgwwzi9757vr987a629Z802Ufuprz3r+f7/+w9r7bXWnoqKMmX+aWpqauao6wu5XG6CumIIgmCJOkep58iCQSaTOYlBj9XV1Y3SvJLNZpsQO9UnBeP+Ulcs6LsUcVo9aWhoGIrcR/W9Bos9wklik9bwGGc1g+M3iE5t66itrU0j/1J9UlDrNU5OpfregHXcRp0V6gn8YsQd9cUy0DboliYIct8RP9UT9sPEhqlPAk8QT4z6JHBe6hzM4VszVn1B2BHxRL0DuZm2kbPET0N89V1fsE2vUp8E1HqKOKqewG9AdKiPBR3a4nbfyF9xiAu+xHFXqe5N6XR6XBHzKBqc1AVx9eJyPcAip9sGNGvOB4OOtnYffE+XSqWG+06xvqcRB+PaIn8G8U094aJ5/3THtqmH4ff57ZS4zbD1zFcfChp2sUOhewwmtMoK33Wuvr5+ZNxECPKfArsKOYaNNZfHjY2NQ1B3mdf2B+KcO3agzUTEDuROWf/LiF3M4fhq3BxszlPVE/hvqHtNfShWKHIgByb20CbZ5ByOZ8T1tdqXfGevHPk++Lzo56z9Ht+Zb7fP/EahxmyXw98LC80Bc16vniB3H/FMfQ/wWB9jk4scyBHWjhNQ54C/E5YLvM3F5w/JhS4KfpPLB/LVxPGFsHEcVnO/ehIUuBp9BtngsXd/fj3YLiOvDpjA2qiBrG6XetTIWe6cPp5tUet852Nz2KsO8dp3PpY/pJ7AX4mafw+sUGzjqDZY1JQwT+iz3s3X4V3FLZqzPrvVE/gq5isrK0d42p3o0HsQYR6bu0U9Qa4V8V59KGj4MWqxBIM8Z5433pBc/kmontgCVqp3ffDUmqQ563NWPUG/EzoWjo87h42c5z8YvDasOU09ge9Ev+vqI2ExdLgX4t8GIV8fH/bl76cQ/wLxwHcYYxvcY/bBolZXV1en+Jj3+pzn5P0+DlvwHzkctyO+uLyfc0R5whzmtEh9LOj0zibTguiwIpO1nWJ9tqsn8Dctz2hz96SsPQQC+X0In6X3nYNerxjX3voM8HOE/ymIqkficiUn0/1+k398lwLbkNHqk4Barai1WT3BHi9H/pX6v4qdmcHqk4CFbUW9F+qTEHfFMJct0W/KosGA8zDwI/VJQa3vfOtX3xtQ4wbnpZ4E3e9yzer7BQx8AFfDRvVJibsaChF0/6fjvHqDP+4/q+xXcAbXqusDA/ADerzKYgjsTT6MIOSVpUyZMmX+Z34DqwBFqA2NDjgAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAaCAYAAABsONZfAAAAsElEQVR4XmNgGAVDEsjJybkC8QYgzkOXwwCysrIm8vLy/xUUFBxAfCC7CsSHyQPZ0+CKQUBGRkYXpABICyGLg8SAeDWU/RdZDib5AkUQIv4Pars5kI6GS4CcA5VwR1IPBkB/PYIaCHcmGICsxxCEAqD4NagmSRQJoA0NeDRdxCUH9hMwEFTRxO4B8VqYJiDdhywPCj0hoOBfqFP+A/0yAyYH5N8HiQFdFI+sZxQMUwAAyPY0r8FNze8AAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAbCAYAAABIpm7EAAAAoElEQVR4XmNgGAVDF8jLy38H4k9AfBjKfwrEJ4D4PxCfRlEsJyeXpqioqA6kbaAK/sHkZGVllUFiyOpBpv0B0UAN06GSjDA5oJgWhgYYgJr+C01sPSENx9DEPuLVoKCg4IFFrBJZDAyAguHoJgHdHwoTk5GREQKyp8AlgZzL6BqA/C0wMSD9AlkOHFJAPBlZzNjYmBWkAeosAWS5UTAEAQB35TTi2flf7gAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAaCAYAAADloEE2AAADv0lEQVR4Xu2YO2gVQRiF8/KNLxQvee5N7oVgQEGCGG18omlsjKiFGETEiIKFIEELKyOIogYRK0sV7LTUQhsVbUQbUbQQ8REFFZSoiY9zkn/C5GR2c+9NIinuB8PdOeef/5+d3Z3dpKSkSJFJQX19/UbV8qXQHFEUVeKnTHVHOp1erVrBYJJRXV3dJRS9kM1m56ivoHgHWqfq+TDWHJjrX9Uc8LqRu131vECCcyyChdnFfkNDQx36PWg/NNaRyWRq4b9VPR/GI0dTU9NU5PijugPeZ7TFqudCmS3KXTUIvP64whyHRZ2uej6MRw6CPPfQzqhOcG7zWUf1UeEgtFeqO+Cts8VbL/oqtJ++li/jkcOjPGkBeIHR2lSPBcFvkhIaA3cW2nVfRL9vLPsEGY8cPnYRN6hO4J3O+UJwF7eTvqOej7sl0b74OrWampoZvqY0NzdPwfiziD2mHknKAe8q2hbXR55taNcw7xY/zgfxz+A/VJ3Aq2Q91YMgsI/Boz3vmNBOW5zHTmtsbJw9WiH4J9B+4bCiurp6AY5/+2OScjDWflm3De0L5jmPC2lz3qFjCLyjcTlJkjcMKzxqMCby3CbU4TT01yaNhbdHffSf+lpcDi4Aap23GM6x3/dNu+1rDi5aKKfDxtarPgy8PhdZYGwiRygOk9itmo+NGbrTPO2D68flwHfWUvyU4zdFv6qqaqbvW57DvuZAzhWhnA56iGlVXRnY2dF61fDBI7WVcfqaR4H2uElAP2C5l4lO7aDrJ+Ug8C6qj/5K00p93VFbW7tcx/jQw7lsVn0ENtnYRCQuBifWEtIJ9JvqeVd06KSSchCr3Sfa+6QxyLk9yadnd2YyCPyalAgr/Jo+3zgBL/ajCl6XeujfVy0pB6GHmJMBrcsd+x7B4nSGdId5FaoHYTASPgnoHyO5agrH8rNddUKPbyM7vsI+Wk8oLpQD+8xCevqat5Mr452Itt/3zOemP2yvc9gbM3bhgmDAJ5v8A7ReHrO4xik2Jrgx4u+yufAeob3knWc5QycTzBH3eET2xuO3k3rE6mxSndgd/V31CQHFjqDYN9WVaPBPhBEnSnLNkSOlcXUIvD7ciUtUnzBsMkPPMIvrBKPBR3Tgoy6E5igU5LmF1q06SaVSs3ReEw5u4VYUfeH60eCjNDQJm3BwU3dojgLh339JF+Ad9pwa1SccFD6Fx2Ov6+P4EE74BvTL+F3jhcaiOfIlaWGQ9zjmsU/1/0Z6rP9pKyk8Bz78MtlsdprqDm7uqhUpUqTIZOEfFaZFJqzK3sMAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEYAAAAaCAYAAAAKYioIAAAED0lEQVR4Xu2YSWgUURCGs6m4ozhGYzI9kwSjUVAI4kUQQdGDB/WiB714EEQNwQU0qBcRRfCggRBBUFHw4HLwIBrRuICIIBgimJuKG3oJEpcYSeJfmXpOzT/d7fQQzWU+KKbfX6+qXr/X/bp7iooKFBg14vH4Mtaikkgk1rGWC4jbxJrD87zZ+ClhPRINDQ1jkGgnrAXF1hTlmBD9v+OnmPUoIEd3RUXFDNZzAYtSjfhO1h3wDbGWE+Xl5RMR3AfrTSaT5aJhYhaj/RHWX1lZOZ1jHOj3ELac9SjgxI6hzkHWo4AxnIKdYV2or68fi/yDrIeCVZogMyqTwj4B+gvxo+g8H18y79VIUzICOYYJywPfY9hJ1gNB56+SEFeKxz6HTlxWUWg9sPWsRwET3i6rzXo+YCwXveBbqtTvHAIJOmmL6VPKum3ng+YoYz0fqqurp4aNSXy4bVey7kuUibGbI9pb/xYn4GpYin6X/fYh3TR9c9TU1MzUuIRKZWjvRkyb7Im2r0Xy1dbWTmFdgK8b6Z6y7kuUibGaFID23moM/F9gLXp8FjZg8+D4BOd1QH+HTX+O1t4Au6/6iqAYQXwY2wHWBfiaw2IzQMer0hkrsZZ9gsy+Du6j1dEehD2ymsVLPeUyJk7zXHJt1HzgN1Boh1A3ZibmLvllvL7vTeLDxFxgXYC+ya9eIFr8DeuCFNFkGftA2AAw6Ebx03vJ8OYHX70TpKZfXWg79beVTwTxC1TL2O8c8P2APWFd0Ns694lBQFNQgE7ADT8dgzzPuiA+zof2Ph/tFey11Sya55fVUPM257HA9w32jHWhqqpqSVjsH/CIXiQrIMcIGPB7ZJtEJei7xeh9ciu4tkVPSN6GrdbLg0L8LS/kxUv6Y1GOsyaTYzWL1r7OuoBcG3kMgaDIOfnFbC5E0BXrQ3sH/Lv0WJ5C842vE/Y23TuNDq7DR7tqNQz0VNBA4ZslvlgsNslp+qQaqqurmyxtHHelI1JonWbWBeTcH1QvC03kbIB8963f+jBh21hzQD9ifV5qo5aVHr46HXjvmBuSo4V9aO91Gn5vyred9asudaaxLsDXBXvOui+SyFjGZY32S+u3PvVnaQ749sB64qnvoKz9xaF61seq1s46CWj9EiNXOPv0m8+3jiA+XDWrWR9xpBBOfBXrjKefHawL0D/DjrKeDxhLG3JdY10pDhrDiJNIfYHzVdYB+0marFST1Rx4Vxk/UgMOywPfHdhp1v8ZKPZBJsi0hzy9BWST1PbjdEQ28N9Dju2sRwFXy2Ev+MqTL/iM/fO/YIvqfd6KE22Xgcp/IbZvEOj7yT6BoiD/ISG+m3XHqEyKA8VXsBYVrPpm1nIBcY2sObBJ1+DzYhzrBQoUKDBa/Aaa91RFFCmw8QAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAaCAYAAADloEE2AAADzklEQVR4Xu2YS2gTQRjHtVZBVMRHjDZNN0mrlSooFEHESyl68ODrpOhJTz4uijf1IKiVQoVWEA+CevJx8uKroNS7oujBiqAtVIsK0h5KbavG/5d8007+md1kYyo95AfD7v6/x3wzszPZdtasChVmBIlEYjdrYSk1RzKZ3M6aRXVtbe1SFkNTU1Oz3PO8yyjyHNpatvuBmF6JZT0M/5KjqalpHuKHWTfANtTY2LiI9aJAcBItjQm5hccq0XB/QjS0PeSeQ11dXRt8zrAehnLkkDrRXrJukLGwVhAUtk0CcW1lm8y2TtB1tilVJXWaSzlyZJA82GJR1gWM7yLsL1gPRAf/kXWDJk27Xnm8Xd1onayHoRw5DKjzMNoo6wZdhNmsO0FRdyUAB9ZqttnoBP526bhUsx6GcuSwCXoLZQxo51l3ooP2TWZw+eGNSrHmAguQgO9NXPazLShHfX39CthuS7xK1Xg+iZhr0Wh0ge1ro7XuYF2QWNh+se7EHjSu46jjnt2gfbD9sJ83WLHtfgMzwN6L1qP3h7zsyk3GBOWAPoA3OqZ977XytPjFCLCNYxKesC7AtiUoNgft2ExO5p6b2sb0ucXEooDnxu4Ctvdof0iTHAPm2S8HtLMNDQ0Ra3Kekl1+QLbamsHLLkg/6wIWfKWrPyfasZmAIYe9S68ZP3nVLVt/QBEbxT9BH3Wa56j17MwB7bher5r6DJiUdarNsXUDbI84xibIloMWW9DZ5YfnT2h9tmaA/o394/H4JtUmfy2Ccgja74StyZbh3DawPShg97XlAMdT4owV3sU2A+yrtMhXto4iH3u0bQzqz5OZt6JBOQSt7RJrfmeKAPs7tB+sC/INxDUEAufRoADY+lx2FN3p0gUZsEe/CuKL9t3WgnKY8yESiSw0mv6Cpc2fArh/OxWRBdpPxHazLkDf7NefLwiYkElgHSt0Q5LhuoRtqVRqjV9H8G+1bbgf1sk5ZvsF5YB+hW2evul6/7C5uXmubVc976wzuHIWBYK6NPEbs6+9Ap/b2lHmbzEGZ8xO2D8j1x3rvMnDL4eX3R6vHfq4xCDnerYJfv0IsE2gnjbWpwUve/BeYJ2R19yv6GJzFAP62Ydcg6wb/GqYFvAdMp87xHOHQ5Oted/WDK4cpSJ55KxiXYDttOwK1qcVdPoMnR6xntNoY3IvZwLuR9C+TEXkwzlKwct+OfewbijXAoQGHX+1flXkXxDtem51xGKxZTnOPlCOsEifIywaYBvE4b+Y9f8GJuMga2EpNQcGf4A1g2xbfALEWa9QoUKFmcJfkq5av2gCNzIAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,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>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAXCAYAAACcTMh5AAAEL0lEQVR4Xu1XSWhUQRCNSZQouItKlunJIoEIXnLQ5CIuaMCTu6AoCmo8KYgieNUIXlQiLhePbnE5iJ48KIJ6kOByENzANaKCEYwxJC7vzVQP9Wt6JjEZTAh5UPyuV0tX1+/f//+8vBGMYFihvLx8ieWGGQpLS0unWDIINMPFYrFTzrnjVVVVE6zdIh6PN0L2Wb6fKMD81ZYcCkA/2qurq8dbPgU04Sic/qB5G6lXVFTEoH+C/LS+HpWVlWWwf7B8f8ACOT/F2oYKMtWWL427bQ0EbD2Q35YnGIfGF1m+v+DNy1TkUADqa0J9DyKk3PVXEVIBtoXS4EWGr4d0aW6gwBxrhnIDiUh9UN71oeDEDoW0aBJ6dw7PvgSQc0Uf6hlUoL5fiQEWP18acyvqEgV2xWTxa9c8ObyZxmrOA7ZzbIbXubMg5zHnPO1ngZjlmRoIfg+kDXK8trZ2tLUTzA/7Wcy1kzrnhP4b5/lE60vA/zR8tiq9AXIR3Grtp8EXbGKAxN0strczDAEbpIEPPce3UZaFJu6QxKyEtGOOSWy2zLfOxni4QAPLysqKJVe5UP6JWK/9oP+AXOYYczTSh43GtQNr2KZ9xf81LqMk10FIF2qc5blUowxgq/cDBgaboIFintGPRXkO+oJQLJsEv2Piw/w92i7cTc1puEADZTEnNQe9RvzyxedAKM7ZA18BtkdypV8otltzHljfTH5+TA8FhhDyQ5LNliPwDTcnL/ktN4P24uLicdouuXZrTsOZBmKepdRDjyB52E/LuMXWI3N1as6D+Vgjx/TDDVms7RJ7TXMWBdkm8EDiVTJB5BMHhW+yBWvAdsLaodcJN0rzGs40EOMz1ENnnq6/pKRkKnVexewf82UqJA08m22dWFsROdhmaz4NMkHGJhCZfOLJwzqN95C4yCMA/WO2GMKl78BdksuffykI/1LpX4T7yiti12r/EOD31NYE/ZLlNPzOpeO3bI64A29oD939mLyZLe9BG3wOBbgmP9Y2Dy7a2PwhH3ns/fxODnQe/pkO/WyQHPcC3F0Zd2gbwc2TUugM4rGye/6zy3CIejC2pqZmjOVx7k2jzX7ikMMlH/PNhezQNg/y4lfgOd4I4VJwyV/MF4bjwlshdyA3IM381dQ+FhKzMMDV8ZxEPRe0TezNlvBb/z6kk2MuMuIUgMSkvRACuygBcE/IoyFHrI2ArQPyHvLWJR/3q96GmFroXTIn69uuY4FCl/zlTNiNtBrfBPznkeXBXSFvz30P18vG6jMwwV4k+275wQAXjKbOtzyegpJQkwaCnOaTZIWW/9+QBk6yPJHLBSPX/tCR128gWQOSPrf8/4b/3HLq7ySe/FLohu289h0IcnkzUkDSwzH1PzmYQC1b0LiLuF7nbrH2gQD52kIf9DkBit5kueEEflH09kYfwT/gLz05dxAKfVU+AAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAXCAYAAACbDhZsAAAC00lEQVR4Xu2WyYsTURDGZ1FxwWVADISkOxsEcvASZUY9iAriWUVFxA2E8eJlQNSDNxe86ICIHjyoBwXRv8CDHoQBEVwuLuBBRETnoKCMMuPyVaZeUv3NS9K0rSD4gyJdXy1d7+WlOz09/0mHMAwfwH7C7nPsr1AsFjeyFgcMPG6ur8ImbbxQKOywflcwSBgEwUU0Ol+pVBZxnMENhmFHWI+D7DhqT6vbK76NY44StMdW84Im56QYBbvFL5VKAfz3sK+c6yiXy3nE37KehHw+v5KHFzDXKOwS644+HfoeBwTEpmA/WBd05+ayngQ5MrCdrAu+RTWQAOwV6w7E1uviNpC+GvbNaklBn8udzjfi10I+PhDetF1Vi8Y3A7tpRfiTSc+6BX32o8+QXGOD1nBcwBFeHJkTBWt1qLuttJmg4YDmfbS6aLlcbp7VHIhdh212Pnpsg91wQzpwzlfIDxa2F/EDqHlu4xa5X/MBIjsnQrczi6a7dPhHTqtWqwtFs3kO6N/1U2q2yKJxjyWyUL1f83hoTsRanaJo7dGm0ynZgYIXWjjsNPjrfLUyIPJGNUf6T9m4anesFhed4Yo84pbFHd6Xhyb7WBPwjliOj358ZiSezWbn27j2GrFaXFA3ARuT635tNMFJFhyZrZLHj1EMv8c3vAOxCxyHv0q1XqvHBbVfYA+dI8O3HUBol4Phh3y6Q+sir3r47zrVdEN73nbOp07NsNuvJV6v12d7Yo0nEOsOiSHnlEc76a5tLA46/LGIgF18YnKc/iGknWOktlarzWEd53ypxPgxqgP34X6DsIM2Fgdd/ACL47qqsXD6RyELGowkedCaGT8+1G7XQSNAe6oDnOVYNzKZzAJfz8RgiMNo+Jn1P0Ew/S/3Fuu/he7GLNbTJtVdd+CIbELjl6ynCXb9OO5xgvVUQOMz8r+E9TTQl90z1lNFXlqspQE25RBr/zS/AFw25G1vBLboAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAAAXCAYAAACoNQllAAACa0lEQVR4Xu2WP2tUQRTFY2JEgwYFV2HZ/7uwsIXNClEbUSGksjBiLIKSIhBrUfwECjaaIGJpp+BXsNDOUrRTsBCRoBYGhChJNOeEO3Bz8rJvkexbfbwfXF7m3Jk3c0/mzc7AQEbGP0W1Wh1XLbWg2HKpVHpULpcfNBqNUc0rlUplDnFL9dSBIu/DlD8w5wrbtVqthPYXxE/tG6jX60XkP6ueNgbNmJeaIMitIn6rTjgOxu5VPVWwSMQH1QPInTUDz4l+CvHLa72i3W4PY64bqvccTPqJxasubOwwxDMvor2S1NnDXZq4QZj0tBX+QnMe7JxD1u+716kVCoV9Xgsg9wRxIbTxjkuIp5jzhO/XLblcbn/iBnEHsMi4MwSFTZtBr4PWbDYPUPP9AtDX7MkxkzQWcxykmTbfZR0Th82XuEEsILJIDwp6Z4XNBQ3tM1FjaQL6zVsfvn/V50177rVuSNwg/Dwf6dagqH4wYUY1gjvUMTyG8DzKfD6fH/F5e9d1rynFYvG4hh0HC6ozdPxOMWSLXdaEB5/XRfbTKwAWfDXKoAByDzWP9knTdnldQdHnNTDfFNbwWHWGjt8xzKBtiyTb9eFhG6UHbNyKaIudxnQi8U+MYMKlTgvGf+wj87yDROQ2ftlUDzCHPncitNvhb5+Loy8GES4Uu+FNhP5Vd4DCsa1Wa4/qOHcOM6dXADNlEPONIa75XBx9M4hg4m9cPOIVYtlMG9N+io3ZcuDyvDAzNgHtLXXsonuai6OvBv0tKPQmFv1D9V7wXxpEbKfsVj3DwOc0AZPeq57hgEF38bnNqp7h4MVRtYwUsw6Ncr6VU6kZxAAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAXCAYAAACI2VaYAAACjklEQVR4Xu2WS2tTURDHa32DCiIRNY+blwQCroJUBPGJ+gEEXYilC6F7RYofoaCIFqlL3YkfwY1uRFyJbkSlCxEXVURBqJJW/c9xJpn777lJ2ip04Q+G0/nP48y9Jznp0NB/VgmVSuUka/1IkmQ3lmHWe4KNklKpNI3iqXq9vo3jTLlcHodNsC7UarWd6FNh3UDsF2tRsMENScZgF8SvVqsl+LOw75xrYPMi4h9Yh3YZtiD9YNc5bjSbzQ2I/2TdM6xDPeaAgNh8VgOpw0NtYl3AW89JXFaOeZDzBHaN9YA+3QzrBmLHdPjjpB+E/fCaB7HbUsd6hLXRPIjvo4E04c3CHngRfjvrsyZoTb/eAX34Ex0BjQ9rg0fdtMWgaLvmffG6aIVCYbPXPLrhtPnFYvGQ9z3IfYV5nnmhLQ2yPjMGGp7X4Z6b1mg0torm8zwYZI/EZRUff8/gQfZhvQeb4nxoV1P9dMPMDQwM/1ofYtw0+Ed71SL3jsWxzqp2RPucTWeH2LlOP71/BhoulodmY6x5rCahayiXy23xvoF+I75f+IbA5nwSgyM9I3l8zaDZ6ADDvdV1Pp/P7+AcD45/f6qfFmZuIGTlYLgDMV1AbJfE5HTUDx8LzvPIUfNwX3sV4W29k3ir1VofiYVvMOsC9Fs+ho3vmy/HJ6fRze7kTCzqJwICL1LiH/0jrM26R2rl5yeih58t89H/rvlYP3czu0B/mbjbwAc+STHsKWxOBx7hPEZrLsV01I+RFq4tXCd7vW5ozSnWlw2O5wqafmN9GayR4VhcMdp0HetLAT0ewm6yvmJwFKfR+A3rS0B+txdY/Gug+SSO+CLrg/BPBzPwBkdZ6wcu3hr+19vI+qrmN5oP0lUNMu80AAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAbCAYAAACqenW9AAAAw0lEQVR4XmNgGMJAXl7+KhD/AeL/MjIynOjyGACocC9IMbo4VgBSSJJiOTm56ejiGEBWVlYKpFhBQUECXQ4DABXOQnYC0IY2IP8pVmchuxdo+iEVFRU+IL0Kn+K5QAWXjI2NWaFi84D4HrpCSZjJQJyDIokOgKZFQBWCIgZE70FXAwdAyesgRUh8kIYpyGrgACp5DZkPtG0llP0RoRIqCQyqMGQ+EGcDmYxA+hhcobKyshiyE0AAGEF+UA0fkMVHwSAEAEjqQDr+M9TuAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAaCAYAAAANIPQdAAAC6klEQVR4Xu2Xy2tTURDGa32g+EIwBkJyb14QzDaKVReigvoHuBARX6h05UYp6sKVgrhRwYVduFCXov4LVhQKIqgrq9CFigvtQkFpJUX9pplTJp/30d54U4T8YCDnmzkzc8695yTp6+uxsPi+/xz2G/aMfQtKqVTaw1oSsLAJ8/kurGn9xWLxgB0nBg37nufdQoGb1Wp1DfsZFB6EnWM9CfIEkeuKDhfJ2PrRVxnaK6vNCyS/LkmR6LCMy+Wyh/Fn2BTHOiqVSgH+T1bD+Iu+bs6mYFttDGqM2BiMr1m/UCgUNouPdfR5AzbMehz9WmiEHQJ807BfrAsyDwWXsy7oAqZZd8D3FHP3su6Avwk7yLoQtPhItJlx1h3w7dJN2E36NthPq1k0b2AzmUxmFXwfWHfAdzvq/MF/z5/ra4vAj2GNGGaeNOy+FTFuRp3FqEX60U/4OPIOyGds7Hb2CzhKa8Nyt4FEO7SRx+yzoNA6jftqddHy+fwKq1nCFol8p1D7POsCzuEmuXhgRxF3EvPHOMYhuWMvRr/1zoeeKQeKHdKGXzqtVqutDlqAxdcLKEAPPN+C2xhrHOMQX9hmzRKXxIFEbzXhoNMw3hk3F/5HWmOj0cZyudx6G5cU7ekO67Pg6t8w10UGxSH5MdYY+M/q3BMyxqtdxbwnHJcU5J2EjbJuWawNTLLDgld1v8Tx1wuaPRK3SMQMSIzb7bj4+YJ8P2AvWG9DFxlZOCzGLYB1wt3K44gfhu3jgE7Q3A9ZbwMB36IaxdN7L/5Go7E0wDdz47LOuE3y6Wb+F2jeC6z/hQRih18H6HIztv04ZmRuvV5fxrrFLRIf+9nXKZJXNpv1QBA8oc2M+q3DLAvfwnGMzjnDukVjLrHeKdlsdqVuXrpgF4dQ6Dvr3cBr/Ut6wHoq6G4uYT1tuvIUHXJjouA71tMET/Eial5mPVVQ8Kr8zmQ9DfCHPot6b1jvCvLjgLU0wGaeZq1Hjx7/B38A6EnwvgcWShQAAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFoAAAAaCAYAAAA38EtuAAADpklEQVR4Xu2ZX4hMURzHd9efyJ+ItdmdmTuzOzXZ4mUVthAKJVKUF9mUPxt58ie8eCBJQpvEi3j24MWDR+TBozYvqH1AFApFu5oN3x+/s/32O+feuTszO7XT/dSvuef7+53zO+d37z1zpmlqSkhIiCCXy21mrRHJZrPrWasYFC3IZDK3giC4kc/n57OfQfJ+2GnWGxHUZABr7WN9QmCA6xjoD4q8T9qdnZ0ZtD/BRjjW0dXVlYb/g9XQ/izjGBuBrbExyPHExqB9zfrrAfKepXmGWmtr61zT7ytsmR0rLi0ymCyeHQJ8o7DfrAvSDzdoFuuCTnKUdQd8z9B3C+v1xhWTdaGnp2cG+1CnhazFQhMNse6Ab6PEIMEm0nthv6xmiVqAPCHwvWM9LlqAk6xXgs7T+yAJvjVIPGwX66Eg+L1vIOLfEw+7b0W0i1F7s/bxjh1EPOlxkLcoqEGh0+n0Sp3nVaujfdNcl6wB2pUg4iEbh3yDapLH7LO4VwX2zeqipVKp2VazaJ+SSWK8Q8h9hvWJoG9E1YXGGA9ljpjPAqfpeu+YmG3u2mhLfWvzgsCiJvHusQ4k3qtFe+G0QqEwr1yiQL8UPXroaxoXzV+LQsu6xHpx2lqHzyPSlpMXxzK+tXlxSVhncCNeSxw++52G9oZyfeF/oDnGvqFx/aq9vX2xjauEGhd6FA/TxeD/0W2w3Loc2jfH+jhwLFuigWUH9cVhQvtZY+A/oX0PSBvbTB79nnJcOWQfZdNtb4B1Me4fBsZYJfPD5yWr07qaZSsx7TG071bWmWlahGF2WJBkt8Tx0Q8J+soVGjGrdTL3pF0uPgwUbwcbxtyDOd1lXYz7h4H+j2RO9owsQDtvrl9an0Xrsp31ErTQkYsPi3FFZJ1wp5UhxN+Oc/fjUoutI2xthmb4v7PokL7Yy1ewXoIMEpUId+ut+OXM6vHFOrS7xQR0YqmWGhY69Jgpcw4ifgHq+qez7kWC8aQNenQ5MRRZt0jf7u7umaxbXKFx2cK+aqi20FjzTp3buPOzgO1nrdwAnbeXjo6ORVF+L+jwRZM+hw3LtXxRcByjfY6zbtGYC6xXS6WFxpt4TIsov+xkbmyii38EdpT7O/SU8pP1SQHJTiHZD9brQaWFrhXIXcQpajnrk4Y8BU1x96kGoa2tbY6uu37ISQJJ37DeyGC9H7FHp1ifdJD4MraRg6w3IljnOTxch1mvG9lq/3WYIsgPJdYSEhISEqYmfwHAp0OGKQV4EAAAAABJRU5ErkJggg==>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAaCAYAAABM1ImiAAAEzklEQVR4Xu2ZW2hdRRSGa1vFeleI0ZCc2blgJA+2GEErUvGCFFG8VkS8PggBS18UqQpeHixFqlhsQR9URISK6IOigi+t+GCLVNo+qHgpqEXQWkxr2qamF/8/Z0268mf2PidNdxp0fzDMnn+tNbNmZu85OzuzZlVUVJRIZ2fnDar9n8iy7G7VpgwWNdRqtVdDCGt6enrOUruCJAZQlqt+PGhrazsNfS9QfaaB9erCem1V/ZjAhF9GZ0fQ6f1sd3V11dD+A2VYfSPd3d0dsP+mujGb/TVZVmkwtH+iXW1lgvGuTOSXLFirm2Ic1m81ymu+r8kyumDo9HM1ENgOohxWnTAOg5+qugc+n9APT1eL2oj1cavqBLbttKs+HWDcPUVjW97nqObbk8J2d7vqEdiupQ826jrReecc8FoK6z83Qd4A7e3t56lOEPdNUWyZNMo7NXdob4djOaIQtKNoMCMeMe95Ee2RZn4bLPaQaGvd9QZnGgf639ZEfmUwx/L+2otov+6ux82J4Dg/e9L5YpJX22Ab1ObBHXuu+Q16nRru5HleUzo6Oi6z2Be97pPVJ80Dv62pifE4hP5GqB9dT6k9gr7vRFmHcj3b8N2C8rv6KfB5jOMi7mbRx04OXN/hbRHGNfOSMwYCRhjU6IxHMvfSj5OIWm9v75mpBVLg83GcEOqFqO9CPYj6TfVNERIbgXwHvIa3vEvYbmlpOSNqXAjL+Ua2UW+OMVbPjr4p4LPb8r6KBWM+iPZB1A+or8I4+D2hei6WaMPFRKffW+cDUUP7mmZi4xiYzArUL6F8xDbfyNQ3RZCN4BNofV7o/ZDbu94P18Oan+Ux9pZThM87q79Nrtf+8qAfYt5SPQleO8+Pg6lNSflhoIdUS2GxI6r5Nu843/YE2Qhcf6nxBE/FIhtrdINSOZv2pNdymGu+G72o/eFmusi3I/Dbr7FFxB+j/Wrw8IylX01ebfmIamIKfK6gD+qVXoe2I16j3yVo3+7tnjBxI5jzhHExxgKzjS402ssl7rZUXIoYi3qx16Gtd9drYb/A2yOw7UXZrHoueZPy5PnERVbdA5/P6MPfE7VFGvURJm7EX6kYfmKhjo29z9qtaH9h+e9jnck7fx7wHUqN4Smy25gfqJ5LsB8k1SOY1C+09/f3n5ywjb5Jqe6xhHJ9YHsF/axT3QOfb30fWMxbrD3HudFvlffD9Tt5d2wjGuWNnDdl7vdSsfhmjsCjMAidbkvoO4Oc7Qpj+/r6TlGdULeEJvxFDq0T5aeiyUbg86v6of0zyi4nxWN2aRSwWA+b9hWPVczxQ9TP5OUb4eZZ3Li/HwiesvnQBzUfhXbeqKo3BIF/2uAbQ/2HhptzufopFvNoQuc3okNm13I41D+ZHED5UWM9sO8K9T84WXZm7uMf2o+4Pocw8T4fa4sW7Vru8b4E2qWh/jrP/NSfJeY9XPQUt7a2nk5/1UsFCT2OQYdUnwnYYpykOjbzuTIXqlb/av2+6qVjk5qr+omE363yFhu2njzb8aDMvgvBHbYYg/+g+okm1L/4/o38Lo4a7tZlXKhawSeVqYB+n0b/z6s+bWDwF/jjqPpMABvxLPL7lOd6VsZ/0Qx7Xf5O9Wkna+IbzH8ZPm2qVVRUVFRUTIV/AdJ91uDXheWbAAAAAElFTkSuQmCC>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGMAAAAaCAYAAACjFuKcAAAEs0lEQVR4Xu2YW4hVVRjHvRVdJYNpai5nnbnExDyUMIEmUXQhIhJTmhAxhx6CoOglKqmHkkBCLISKeqqnwuglutPTiA8KIaiEgqnghUht0Gjy0qT2/8/5Vn39z9pnn5nOVsH9g8Xe6/9931rfWmvvddbZM2aUlJRcAHp6eh5S7XKiWq3OV60lYGJDpVL5IITwbn9//1y1K0jkGZTVqreCjo6OawobaIvBfJ1Xbdpg0BvYIBZiFeu9vb0V1I+inFbfSF9fXzfsP6tuzGJ7TZb1Ggztz2hXWxGgn18lp/H29vZrxeew98FcDTtbD8px7z8dJicNDW9SA4HtL5RzqhPGYRGvUt0Dn2/oh7esTW3E2nhMdQLbftpVLwruCjbRo2qLwDbOB1V1AtsfmMfFqjeNdb5f9Qhs99MHnTwg+iKUM15LYe1nTigfgq6urhtVJ4jb1Si2CCzf5Li6u7vvRL6fqx7BOG6ddr7BXjvVhbjdfOZF1Cea+a2w2LOivefuR53pP6D9nU3k11Is32SfWbqHPvytU70hGOi91vGo2jx4EuaZ3wmvU8OTcLXXFD5JFvuW1/2g9I3zwG9HagK4NUL/MNS2sVfVHkHbj6NsRHmQdfhuRzmifh7Lt65PaJ+gLFJdsfh1qjcEARMMzNvzMZCV1sH2qA0MDFyfSliBz9f04z6K6124PoHrCVw/Ut8UIbEYPL15Dfv87ay3tbVdFzWeAi3nR1jHdVuMseus6KtYnPbJxT/qtSzg+33I2OYySXWaAo3voR8nIWqo39dMbOwDk78W17dRvmQ96wdQCbIYfBOtzVu8H3L71Pvh/rTmZ3k86rUU8DueiG16ciu1vwW5c/MPOJLeZIPKDUr5YfBPqZbCYidU83Ukf7eve4IsBu63aDzB23GP9TW5SKmcTXvFayng8xV9sfD9rNvO8Ib6ZQH/l7XvPGZbcqfU4OGeS7+KHHuxGCN5HcJnIX1wfdPr0A7He7Q7jPoyb/eE+sVgznX9oo/5ZpucbNRXS9zSVFyKGMsHjvVm4yLwf3GqMZkD82T5xIlW3VOt7Z3n+fuitkheG6F+Meq2EMLPMdSxuE9avR31zZb/SV6Rzw0al4JvqsXxgPAj4m5Wn0Yg/v1Ujg1BwG+NgtDoQdqHhoauSNgmT1iqe2xAmT6wvYN2Nqrugc9u3wYmZonVZzs3+q33frj/eKqT6Ii7Bv+Rb1FjHhjTd4g7qXou7BRJ70zox4Ls9QpjBwcHr1SdULcB1f1zD7XPBvv85GUBn0Pqh/oBlDEnxcl7LgqYkKdN+4FbLMb4Ba6vZeWrWGxufikYh/42qN4U4d9vMltRTlljC9RPsZgXEjq/KZ2NA5JyLtQ+r5xB2auxHtjHQu1PKcuxqvtgiPqzrs1xTPSgj8U2dYeza1nhfVPQr2L/TaaKxc5TvVDQ4UucCNUvBTghuMxUHQu6xmyF0NnZ2VVk+w2xjueofjHhd66sCeFxNcvWCkLtgJF5OiwUPGkPo/OfVL/YhNqX4t+R321Rw5v8vG0hmZ9f/g92gvtF9QsKEljHH0zVLwWwGK8jv295asP9crW3kiLfuCmBgY6odjnBLwCqlZSUlJSUtJq/AfgyvXBzdQvEAAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAaCAYAAAAg0tunAAADaElEQVR4Xu2XXYhNURTHp/ERhSIXTffeM3fm1s28jmKkhEJKIaTIJB/N+0i8S5KvQvEkT9Q8eBDiaciDFyUvvqcwUUgzRXPn23/NrK01/3PuvuQedXR+tTp3/9fae+29ztln31NXl5LyX1EoFNazliRqOn8MFuTz+ctBEFwsFovz2M80NjZ2wI6yniRaWlpmYr39rP8RKMJ5DDKO4u2VdlNTUx7tz7Ayxzqam5tz8H+0GtpfZBxjZVibjUGOBzYG7XPW/5fUU36fnXad8Hsr7Ikd6HeZSCiLYocA3whsjHVB+qHws1gXdIIjrDvge4S+G1ivFRj/jswBuyjDPkHnvoU17MDFVquKLrSHdQd8a7XA60hfCRu0mkXHHWddyGQyc+D7wHot8eUX5IHJZrMLrIb4/bABq3lBcK8vieK2RJcV0R72vft8Cwg8T2at0PyjpF0yv7uN6xeV5hwCi1+tSbrZZ8Gdmq9xfVYXDXdwttUs2ic0GYx3CLmPsV5LcrncMs1/xup2PryjHNpvE+shEDQswZXeYQ4k2qODPnVaqVSaG1UcS6CHSYQe+T6tJchxW3Jj7ptxbcN1J659uF7lWAZxQ4i7x3oILUpogQwK/EoL3eE0tNdU6wv/Tc2x1GgvGxoaFtq4OHBrQyFO4HoWdkva8s+CYxnEvYC9Y30K+PuxyCVhHxMVh2LuY42B/7D2PSBtbPci+j3kuGqgf5eMg9OxxL5KaN5h1mwbxV1l2w7E3eXYKKZpEu+JgyTbJY7/4qAQ7dWSIGaFxOB6TdrV4iuBfte1gAH7ojB5T1odWq/7jfXsQHub9TsC3f6sh9ACegMrxbhJsk6407sH8VdgGzkgDpDnvuSV9zT7HL65w/cc9o31EAjq9w2Eu/Re/K2trTMifBMnM+uMuwEBneBx4nKy7oDvAuZ/g3UH/GW5CaxHIokQ/CxClxN0yjuEkb7yDcm6xSymnn1xoN+0kjN00kMrwN76iitoTaZ8oXhBh6+a9DFsQAdYznGM9ulk3aIxx1mPA+QZgo1qTraxYPKTdBD2hvtaJJ61WMA2OIJk31lPMnhwdmFNn1iPDb1b01lPKrIeFHEJ67EhJyuSvmY9iQSTHwfdrMcOkp7Cdj7IesKQv1w/WPxn4ElsZy1JoHi7WUtJSUlJieYnVycxYBAu9H8AAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAaCAYAAAD8K6+QAAAC5ElEQVR4Xu2WO2hUQRSGY6LiA0XxsWZ33TcsLmizRdRGVBAbGyNqIQYLMdaCCBZ2CiL4IAZ7CwVLsbPQSkvRRhQERVSMYAQlysbo/4czYfJn9u69Zlct9oPDvfOfc+bMzJ079/b0dOk8+Xx+FPZL9f+CYrG4R7W4pNPptVETKxQKO1T7IzDIfC6Xu4FiI5VKZaX6FRQehp1RPS6oc531VHfAfw39D6keGyRf4cqhyFG2S6VSDu2PsO8a6yiXyxvhf6d6ElgT1p/NZpfi+gj2OhDzGbZJ9Vb02oQeqoPANwmbUp0wDwuyRPUksI9MJrOG2xn3Z2FjGoOxrWac6pHYir1S3QHfLpv4btG3w374WlKw1ddZ/ZvqU7i4sEHVgyDwbYyVmH6isDu+iHZjPu8WQR8jsFG+y63GAf+lWAvJ08YG/EB9Pm4bwMZ9nRrfC19zwHcLtt+10cdB2G3U3CpxU9A22P30xHC968c4oPe7mEgQ1GBgq3cEAzpiE3vitGq1uqJZEeg/7cqcQdg4aqyyw4H1DnuxM33g/h7suWuHaFZzFla4ZSAG8sIGNOw0tHeGcjl4xF21GPY/6ftNu+9rSbD8ouoz4JheH3dioTgM/phqBCfbFlz6cE3Rj4/vMt9vfZ3ytSQwH7X3qu7TZ0Um1OGDbXiAcfopQOdDoYk58oHfJLS3mbbA15NgY9mn+ixsYk0HR5rF8BAI6Q7La4j2ISonDsy3XdEcBH2JKoSVeUN/vV5fFPBFfjDpQ8yFgHbe3fu+uFjeQtXnwECs/tOAPqYrrjC3VqstVt391OqnwAbVi3oDsJO+Lw78O0m0IAj+xATYY9iETXZA4xTLmXMQIPdQaADQnlHHU7usvjjwaSP/m+ptB4VOo9BX1TsFajWwCzar3hHsybTe8/MklUotD+2CjsFvCgq+VL3doMZ7vGNZ1TsKil7EtjyuertA3+ewgCdU/yvwg61au+BhpFqXLl3+Db8Bay/h8h3K2qsAAAAASUVORK5CYII=>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEwAAAAaCAYAAAAdQLrBAAACcUlEQVR4Xu2XT0hUURTGhXBRtDIGZP7dGWibLQZq0SoIRERoV0uZhS5ctHDRItq0CdpUBkGF4Z+F4MZ9ELSJIHAVBEE7UdpoSqZipX3Hd66cPt4dfUrPQe4PDjPvfOfe+e55l/vedHREIpHIKaFWqz1xzq0hdjU2ESs2h5pZHpcn8PAF8dP4GSS9bvxL7JTL5S7RUHsb19+Ntq3Xq1KnuXk736HwE3K+XhcveybfsJY3zjSNNUGaUSwWz3FeCI1rNBqdqq2w1hId9IHzQujH8gQ37nK1Wr0LH9/EC25gN9e08qhr+MV5IfP6dNuKiV7WsLXPZp7wP4Dfn8PHmUqlUlQ/i1wD/x85J0hzZQwa/pA1IfP6UPw5NECMajNvspYn1l/aAtGMIeT6bM6D/HOpLxQK51nDuDvazMesBUkzoPnr2qynrOUNfKya7031/NLklvx3JrQ+rKum2lvWWuInFFMueYJs6fWnUql0gevTQO0rxHRawNck7uAEvr9GjGttP88RAudXD+pHbU79/bPrrG7xtS558ssbwIZeL+HIucT1LfHnF6LJWrvg9PzinPreazw+31vd488vxBRrR8Il7zjBu9MOhPxpI+RdagQ7+AbrAhr2QuvqrB0JnSzVUBYwxwPEo8MGFjjAc4RA/TrnBJccIbuIZdY8qh97ffvohF853y5gh1yFv/ucF3D+XDyoIQfpmcBE92QymBpmrR3Qd8DfiDHWPNqQd5wXsIuvqT7DWiYwwTPED5c8EeWpIX85drjuJIGfBZc81Zad/u/jGgH5ETxFr9gcNsAtsz4Zv474I3+DbF0kEolEIpFIJHKq+As4Sfcz7zCyCQAAAABJRU5ErkJggg==>