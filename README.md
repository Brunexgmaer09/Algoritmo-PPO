# Algoritmo PPO em JavaScript Puro

Implementação completa e robusta de **Proximal Policy Optimization (PPO)** em JavaScript puro, sem dependências, projetada para treinar redes neurais a jogar jogos no navegador.

Funciona com qualquer jogo onde você consiga descrever o estado como vetor de números e as ações como índices discretos (cima/baixo/esquerda/direita/pular/etc).

## Características

- **PPO Clipped Objective** (Schulman et al. 2017)
- **Generalized Advantage Estimation (GAE)**
- **Actor-Critic** com redes separadas
- **Otimizador Adam** com bias correction
- **Inicialização ortogonal** dos pesos (Gram-Schmidt)
- **Normalização de observações** (Welford running stats)
- **Normalização de recompensas** por desvio padrão dos retornos
- **Value function clipping** (PPO2)
- **Early stopping por divergência KL**
- **Decay linear de learning rate** opcional
- **Gradient clipping** por norma global
- **Ambientes vetorizados** (rodar N jogos em paralelo compartilhando a mesma rede)
- **Salvar/carregar modelo** em JSON
- **Zero dependências**, funciona direto no browser

## Estrutura

```
.
├── modulos/
│   ├── matriz.js          ops basicas (mat-vec, outer product)
│   ├── ativacoes.js       tanh, relu, softmax e derivadas
│   ├── inicializador.js   inicializacao ortogonal
│   ├── rede.js            MLP com gradientes acumulaveis
│   ├── adam.js            otimizador Adam
│   ├── normalizador.js    Welford pra observacoes + filtro pra recompensa
│   ├── distribuicoes.js   politica categorica (acoes discretas)
│   └── buffer.js          rollout buffer + GAE com N trajetorias
├── ppo.js                 orquestrador principal
└── exemplo-snake.html     exemplo completo treinando Snake com 16 envs
```

## Instalação

Não tem instalação. Baixa os arquivos e inclui no seu HTML:

```html
<script src="modulos/matriz.js"></script>
<script src="modulos/ativacoes.js"></script>
<script src="modulos/inicializador.js"></script>
<script src="modulos/rede.js"></script>
<script src="modulos/adam.js"></script>
<script src="modulos/normalizador.js"></script>
<script src="modulos/distribuicoes.js"></script>
<script src="modulos/buffer.js"></script>
<script src="ppo.js"></script>
```

A ordem importa (cada módulo depende dos anteriores).

## Como integrar com seu jogo (3 passos)

### Passo 1: Crie a função `estado()` do seu jogo

A rede precisa receber um vetor de números descrevendo a situação atual. Boas práticas:

- Normalize valores entre -1 e 1 quando possível (ex: `posX / larguraTela`)
- Use sinais relativos ao agente (ex: `Math.sign(alvoX - jogadorX)`)
- Inclua perigos imediatos (ex: distância ao próximo obstáculo)
- Inclua direção/velocidade do próprio jogador

Exemplo (jogo de desviar blocos):

```js
function estado() {
  const obstProx = obstaculos[0];
  return [
    jogador.x / largura,
    jogador.y / altura,
    jogador.velY / 20,
    (obstProx.x - jogador.x) / largura,
    (obstProx.y - jogador.y) / altura,
    obstProx.velocidade / 20,
    podePular ? 1 : 0
  ];
}
```

### Passo 2: Defina recompensas

Princípios:

- **Recompensa densa** quando possível (sinal a cada frame, não só no final)
- **Recompensa pequena por progresso** + **grande por sucesso/falha**
- **Penalidade por morrer** (geralmente -1)
- Evite recompensas gigantes, prefira valores entre -1 e +1

Exemplos por tipo de jogo:

| Jogo | Recompensa por frame | Eventos |
|---|---|---|
| Snake | `0` (ou +0.05 ao se aproximar da comida) | +1 ao comer, -1 ao morrer |
| Dino | `+0.01` (vivo) | -1 ao bater |
| Desviar blocos | `+0.01` (vivo) | -1 ao bater, +0.5 a cada onda passada |
| Rocket | `-distancia/maxDist` | +10 ao pousar, -10 ao explodir |
| Corrida | `+velocidade/100` | -1 fora da pista, +50 ao terminar volta |

### Passo 3: Loop de treinamento

```js
const N_AMBS = 16;
const PASSOS_POR_AMB = 128;

const ag = new PPO({
  nEst: 7,           // tamanho do vetor de estado
  nAcoes: 3,         // numero de acoes possiveis
  nHid: 64,          // neuronios por camada oculta
  nAmbientes: N_AMBS,
  lr: 3e-4,
  gama: 0.99,
  lam: 0.95,
  eps: 0.2,
  entCoef: 0.01,
  epocas: 10,
  tamLote: 128,
  klAlvo: 0.02,
  normObs: true,
  normRec: true
});

const envs = Array.from({ length: N_AMBS }, () => new MeuJogo());
let ests = envs.map(e => e.reset());

function loop() {
  const treinando = true;
  const res = ag.agirLote(ests, treinando);

  for (let i = 0; i < N_AMBS; i++) {
    const r = envs[i].passo(res[i].acao);
    ag.lembrar(i, res[i].estProc, res[i].acao, res[i].logP, res[i].val, r.rec, r.fim);
    ests[i] = r.fim ? envs[i].reset() : r.est;
  }

  if (ag.bufferCheio(PASSOS_POR_AMB)) {
    const valsFinal = new Float32Array(N_AMBS);
    const resF = ag.agirLote(ests, false);
    for (let i = 0; i < N_AMBS; i++) valsFinal[i] = resF[i].val;
    const stats = ag.treinar(valsFinal);
    console.log("Update:", ag.atualizacao, "KL:", stats.klMed.toFixed(4));
  }

  requestAnimationFrame(loop);
}

loop();
```

Pronto. Sua IA está aprendendo.

## API resumida

### `new PPO(cfg)`

Parâmetros principais:

| Param | Padrão | Descrição |
|---|---|---|
| `nEst` | obrigatório | Tamanho do vetor de estado |
| `nAcoes` | obrigatório | Número de ações discretas |
| `nHid` | 64 | Neurônios por camada oculta |
| `nAmbientes` | 1 | Quantos ambientes paralelos |
| `lr` | 3e-4 | Learning rate |
| `gama` | 0.99 | Fator de desconto |
| `lam` | 0.95 | Lambda do GAE |
| `eps` | 0.2 | Epsilon do clipping |
| `entCoef` | 0.01 | Coef de bônus de entropia (exploração) |
| `vCoef` | 0.5 | Coef da loss do crítico |
| `epocas` | 10 | Épocas de treino por update |
| `tamLote` | 64 | Tamanho do minibatch |
| `klAlvo` | 0.02 | KL alvo (early stop se passar de 1.5x) |
| `normObs` | true | Normalizar observações |
| `normRec` | true | Normalizar recompensas |

### Métodos

```js
ag.agirLote(estados, treinando)       // forward em N estados, retorna array
ag.lembrar(idAmb, est, acao, logP, val, rec, fim)
ag.bufferCheio(passosPorAmb)          // quando chamar treinar
ag.treinar(valsFinal)                 // executa update PPO
ag.salvar()                           // retorna JSON do modelo
ag.carregar(json)                     // restaura modelo

// modo de avaliação (sem exploração, sempre escolhe melhor ação)
const r = ag.agirLote(ests, false);
```

## Ajustando hiperparâmetros

Se a rede **não aprende**:
- Verifique se o estado tem informação suficiente pra resolver o jogo
- Aumente `entCoef` pra 0.05 (mais exploração)
- Verifique a escala da recompensa (deve ficar perto de [-1, 1])
- Aumente `nHid` pra 128

Se a rede **aprende e depois piora**:
- Diminua `lr` pra 1e-4
- Diminua `klAlvo` pra 0.01 (cortes mais agressivos)
- Diminua `epocas` pra 4

Se está **muito lento**:
- Aumente `nAmbientes` pra 32 (mais paralelismo)
- Aumente `tamLote` pra 256
- Reduza `nHid` pra 32

## Exemplo: Snake

Abra [exemplo-snake.html](exemplo-snake.html) no navegador. Em alguns minutos no modo rápido a cobra aprende a jogar com média de 30+ corpos.

Botões:
- **Modo rapido**: desativa render e roda 20 passos por frame
- **Modo normal**: render normal a 60fps
- **Avaliar (greedy)**: usa sempre a melhor ação (sem exploração)
- **Salvar modelo**: baixa JSON com pesos da rede

## Limitações conhecidas

- **Apenas ações discretas** (controle analógico tipo "ângulo de direção" exigiria adicionar política Gaussiana)
- **MLP apenas** (sem CNN, então input precisa ser features extraídas, não pixels)
- **JS puro** é ~10-50x mais lento que PyTorch — adequado pra jogos pequenos/médios

## Referências

- [Proximal Policy Optimization Algorithms (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [The 37 Implementation Details of PPO (ICLR 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

## Licença

MIT
