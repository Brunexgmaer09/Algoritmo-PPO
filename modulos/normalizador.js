"use strict";

class Normalizador {
  constructor(dim, clipMax) {
    this.dim = dim;
    this.media = new Float32Array(dim);
    this.varM2 = new Float32Array(dim);
    this.cont = 1e-4;
    this.clipMax = clipMax || 10;
  }

  atualizar(x) {
    this.cont++;
    for (let i = 0; i < this.dim; i++) {
      const delta = x[i] - this.media[i];
      this.media[i] += delta / this.cont;
      const delta2 = x[i] - this.media[i];
      this.varM2[i] += delta * delta2;
    }
  }

  variancia(i) {
    return this.cont > 1 ? this.varM2[i] / (this.cont - 1) : 1;
  }

  normalizar(x, dest) {
    const r = dest || new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      const dp = Math.sqrt(this.variancia(i)) + 1e-8;
      let v = (x[i] - this.media[i]) / dp;
      if (v > this.clipMax) v = this.clipMax;
      else if (v < -this.clipMax) v = -this.clipMax;
      r[i] = v;
    }
    return r;
  }

  serializar() {
    return { dim: this.dim, media: Array.from(this.media), varM2: Array.from(this.varM2), cont: this.cont, clipMax: this.clipMax };
  }

  carregar(o) {
    this.dim = o.dim;
    this.media = Float32Array.from(o.media);
    this.varM2 = Float32Array.from(o.varM2);
    this.cont = o.cont;
    this.clipMax = o.clipMax;
  }
}

class NormalizadorRecompensa {
  constructor(gama) {
    this.gama = gama;
    this.retornoEma = 0;
    this.norm = new Normalizador(1, 1e9);
  }

  filtrar(rec, terminou) {
    this.retornoEma = this.retornoEma * this.gama + rec;
    if (terminou) this.retornoEma = 0;
    this.norm.atualizar([this.retornoEma]);
    const dp = Math.sqrt(this.norm.variancia(0)) + 1e-8;
    return rec / dp;
  }
}

if (typeof window !== "undefined") { window.Normalizador = Normalizador; window.NormalizadorRecompensa = NormalizadorRecompensa; }
if (typeof module !== "undefined") module.exports = { Normalizador, NormalizadorRecompensa };
