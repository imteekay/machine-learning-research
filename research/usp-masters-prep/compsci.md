# Computer Science

## Mock Tests

- [Flashcards](https://gemini.google.com/share/d/1aornDEUH8iBDNt8NwXka1HZ8NXLTmMro?usp=sharing)
- [Test](https://gemini.google.com/share/d/1S7BW--7cQobPc1Q2Dt1ugwxmwHAJSUJB?usp=sharing)

## Prep

Topics to study

- [ ] Listas Ligadas
- [ ] Pilhas
- [ ] Filas
- [ ] Listas de Prioridade
- [ ] Recursão
- [ ] Hash Tables
- [ ] BST
- [ ] Union-Find
- [ ] Complexidade Algorítmica e Notação Assintótica
- [ ] Algoritmos de Ordenação e Seleção
- [ ] Algoritmos para Problemas em Grafos
- [ ] NP-Completude
- [ ] Autômatos Finitos e Expressões Regulares

## 1. Estruturas de Dados

### 1.1 Listas Ligadas, Pilhas, Filas e Listas de Prioridade

#### Listas Ligadas (Linked Lists)

Diferente de um array, que aloca um bloco contíguo de memória, a lista ligada espalha seus elementos (nós) pela memória. Cada nó guarda o dado e um ponteiro de memória indicando onde está o próximo elemento. Isso torna a inserção e remoção no início ou no meio (se você já tiver o ponteiro) extremamente rápidas, operando em tempo $O(1)$. No entanto, você perde o acesso aleatório: para ler o 50º elemento, você obrigatoriamente precisa percorrer os 49 anteriores, resultando em uma busca de tempo $O(n)$.

#### Pilhas (Stacks)

A pilha é uma abstração focada em restrição de acesso. Imagine uma pilha de pratos: você só pode colocar um prato no topo (`push`) e tirar o prato do topo (`pop`). O sistema operacional usa isso exaustivamente na *Call Stack* (pilha de chamadas): quando uma função chama outra, o estado da função atual é "empilhado" até que a nova função termine e retorne, momento em que o estado anterior é "desempilhado" e a execução continua.

#### Filas (Queues)

A fila espelha o comportamento de uma fila de banco: o primeiro a entrar é o primeiro a sair. Em sistemas reais, filas são usadas como *buffers*. Se um servidor web recebe mais requisições do que consegue processar instantaneamente, ele coloca essas requisições em uma fila. Processos assíncronos (como o envio de milhares de e-mails) retiram tarefas dessa fila uma a uma, garantindo que nada se perca e que o sistema não trave.

#### Listas de Prioridade (Heaps)

Uma lista de prioridade garante que o elemento de maior (ou menor) relevância esteja sempre acessível imediatamente. A forma mais eficiente de construir isso é através de um *Heap*, que é uma árvore binária completa representada matematicamente dentro de um array simples. Quando você insere um elemento, ele vai para o final do array e "flutua" para cima (`sift-up`) trocando de lugar com os "pais" até chegar à posição correta. Isso custa apenas $O(\log n)$ e é o motor por trás de algoritmos de roteamento de GPS (como o Dijkstra).

### 1.2 Recursão

A recursão ocorre quando uma função resolve um problema chamando a si mesma com uma entrada ligeiramente menor. A mecânica exige um "caso base" para interromper o loop infinito. Sem o caso base, a função continua empilhando chamadas na memória até causar um *Stack Overflow* (estouro de pilha).

A otimização de **recursão de cauda** (*tail recursion*) é um truque de compiladores modernos: se a chamada recursiva for a absoluta última instrução da função, o compilador não cria um novo quadro na memória, mas sim reaproveita o atual, transformando a recursão em um loop iterativo altamente eficiente por baixo dos panos.

### 1.3 Tabelas de Espalhamento (Hash Tables)

Uma Hash Table é a estrutura definitiva para buscas rápidas. Você passa uma chave (como uma string) por uma **função matemática de hash**, que cospe um número inteiro. Esse número é usado como o índice exato de um array onde o valor será guardado.

O grande desafio arquitetural aqui são as **colisões**: quando duas chaves diferentes geram o mesmo número de hash. Para resolver isso, usamos o **Encadeamento** (cada posição do array guarda uma lista ligada de itens que colidiram) ou o **Endereçamento Aberto** (se a posição 5 estiver ocupada, o algoritmo tenta a 6, depois a 7, até achar um espaço vazio). Quando a tabela fica muito cheia (alto fator de carga), ela sofre um *rehash*, onde um array maior é criado e todos os itens são recalculados, garantindo que o tempo médio de busca continue sendo $O(1)$.

### 1.4 Árvores Binárias e de Busca (BST)

Uma Árvore Binária de Busca organiza dados de forma que, a partir de qualquer nó, todos os valores à esquerda sejam menores e todos à direita sejam maiores. Isso permite descartar metade dos dados a cada passo de uma busca, imitando a busca binária.

O problema prático é o **desbalanceamento**. Se você inserir dados já ordenados (1, 2, 3, 4, 5) em uma BST simples, ela crescerá apenas para a direita, virando uma lista ligada e degradando o tempo de busca para $O(n)$. É por isso que bancos de dados utilizam árvores auto-balanceadas (como as árvores AVL ou Red-Black). Elas aplicam "rotações" matemáticas nos nós logo após a inserção para garantir que a árvore permaneça simétrica, fixando o tempo de busca em $O(\log n)$.

### 1.5 Union-Find (Conjuntos Disjuntos)

Esta é uma estrutura de nicho, mas extremamente poderosa para rastrear conexões. Imagine uma rede social onde você quer saber rapidamente se a Pessoa A tem alguma conexão indireta com a Pessoa B.

O Union-Find faz isso elegendo um "nó representante" para cada grupo. A otimização que torna isso mágico é a **Compressão de Caminho**: toda vez que você busca o representante do grupo de um nó, a estrutura religa esse nó diretamente ao representante principal. Nas buscas futuras, o acesso é quase instantâneo. A complexidade de tempo se torna a função inversa de Ackermann, $\alpha(n)$, que para propósitos práticos no universo computacional, é $\le 4$ (essencialmente $O(1)$).

---

## 2. Algoritmos e Linguagens Formais

### 2.1 Complexidade Algorítmica e Notação Assintótica

Esta é a régua com a qual medimos a escalabilidade do código. Não medimos o tempo em segundos (pois isso depende do hardware), mas sim como o número de operações cresce conforme a entrada ($n$) aumenta.

* $O(1)$: Constante. Não importa se há 10 ou 1 bilhão de itens, leva o mesmo tempo.
* $O(\log n)$: Logarítmico. Extremamente eficiente. Se os dados dobrarem, o algoritmo só faz uma operação a mais.
* $O(n^2)$: Quadrático. O pesadelo da escalabilidade. Um loop dentro de outro loop. Se os dados dobrarem, o tempo aumenta em 4 vezes. Se aumentarem 10 vezes, o tempo aumenta 100 vezes.

### 2.2 Algoritmos de Ordenação e Seleção

A escolha do algoritmo de ordenação depende de recursos de memória e da natureza dos dados.

* **Merge Sort:** Corta o array pela metade repetidamente, ordena as metades e as junta. É brilhante e previsível ($O(n \log n)$ sempre), mas tem um custo oculto: exige criar novos arrays na memória.
* **Quick Sort:** Elege um "pivô" e joga os menores para a esquerda e maiores para a direita. Modifica o array original (não gasta memória extra), sendo o mais rápido na prática. Porém, se você escolher um pivô ruim constantemente, ele quebra e roda em $O(n^2)$.
* **Radix/Counting Sort:** Não comparam os itens. Eles agrupam os números por seus dígitos ou valores literais em "baldes". Podem ordenar dados na velocidade da luz (tempo $O(n)$), mas só funcionam se os dados forem inteiros dentro de um limite conhecido.

### 2.3 Algoritmos para Problemas em Grafos

Grafos mapeiam redes (estradas, roteadores, conexões sociais).

* **BFS (Largura):** Explora tudo ao seu redor primeiro antes de ir mais fundo. Usa uma fila para rastrear os próximos passos. É o algoritmo exato que o seu GPS usa para encontrar a rota com o menor número de ruas (caminho mínimo sem pesos).
* **DFS (Profundidade):** Segue um caminho único até o fim, batendo num beco sem saída e voltando. Usa recursão/pilha. É perfeito para resolver labirintos ou checar dependências de código (qual biblioteca deve ser compilada primeiro - Ordenação Topológica).
* **Algoritmo de Dijkstra:** A evolução do BFS. Ele usa uma Lista de Prioridade para decidir qual o próximo cruzamento explorar, priorizando sempre as ruas mais rápidas/curtas.

### 2.4 NP-Completude

Este é o limite da computação moderna. Problemas **P** são fáceis de resolver. Problemas **NP** são aqueles em que é impossível achar a resposta rapidamente, mas se alguém te der a resposta pronta, é fácil verificar se está certa.

Problemas **NP-Completos** (como o do Caixeiro Viajante: qual a rota mais curta passando por 50 cidades?) são tão difíceis que tentar todas as combinações demoraria mais que a idade do universo para rodar, mesmo no melhor supercomputador do mundo. Quando um engenheiro prova que seu problema é NP-Completo, ele para de tentar achar a resposta perfeita e passa a usar heurísticas e Inteligência Artificial para achar uma resposta "boa o suficiente".

### 2.5 Autômatos Finitos e Expressões Regulares

É a ciência de reconhecer padrões em textos (como validar se um e-mail é válido).

* **Expressões Regulares (Regex):** Uma sintaxe matemática para definir formatos de texto.
* **Autômatos Finitos (AFD e AFN):** São máquinas de estado teóricas. Imagine um fluxograma com círculos (estados) e setas (transições ligadas a letras). Se a máquina terminar em um estado de "sucesso" ao ler sua string letra por letra, o texto é válido.
* **O Lema do Bombeamento:** É uma prova matemática de limite. Ele prova que Regex e Autômatos **não têm memória**. Você não consegue escrever um Regex que valide se um código fonte tem parênteses perfeitamente balanceados, porque a máquina não consegue se lembrar de quantos parênteses abertos viu no passado. Para isso, precisamos de um nível computacional acima, as Máquinas de Pilha.