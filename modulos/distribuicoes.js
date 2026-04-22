"use strict";

class Categorica {
  constructor(nAcoes) { this.nAcoes = nAcoes; }

  amostrar(probs) {
    const r = Math.random();
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) return i;
    }
    return probs.length - 1;
  }

  logProb(probs, acao) {
    return Math.log(probs[acao] + 1e-10);
  }

  entropia(probs) {
    let h = 0;
    for (let i = 0; i < probs.length; i++) h -= probs[i] * Math.log(probs[i] + 1e-10);
    return h;
  }

  klAprox(probsVelho, probsNovo) {
    let kl = 0;
    for (let i = 0; i < probsVelho.length; i++) {
      const lv = Math.log(probsVelho[i] + 1e-10);
      const ln = Math.log(probsNovo[i] + 1e-10);
      kl += probsVelho[i] * (lv - ln);
    }
    return kl;
  }

  ativacaoSaida(z, dest) {
    return Ativacoes.softmax(z, dest);
  }

  gradLogits(probs, acao, escala) {
    const g = new Float32Array(probs.length);
    for (let i = 0; i < probs.length; i++) {
      const oh = i === acao ? 1 : 0;
      g[i] = escala * (oh - probs[i]);
    }
    return g;
  }

  gradEntLogits(probs) {
    const h = this.entropia(probs);
    const g = new Float32Array(probs.length);
    for (let i = 0; i < probs.length; i++) {
      g[i] = -probs[i] * (Math.log(probs[i] + 1e-10) + h);
    }
    return g;
  }
}

if (typeof window !== "undefined") window.Categorica = Categorica;
if (typeof module !== "undefined") module.exports = { Categorica };
