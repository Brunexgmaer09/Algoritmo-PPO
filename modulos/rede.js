"use strict";

class Rede {
  constructor(cfg) {
    this.nIn = cfg.nIn;
    this.nHid = cfg.nHid;
    this.nOut = cfg.nOut;
    this.ativ = cfg.ativ || "tanh";
    this.ganhoSaida = cfg.ganhoSaida != null ? cfg.ganhoSaida : 1.0;
    const ganhoOcul = this.ativ === "relu" ? Math.SQRT2 : Math.SQRT2;

    this.w1 = Inicializador.ortogonal(this.nHid, this.nIn, ganhoOcul);
    this.b1 = new Float32Array(this.nHid);
    this.w2 = Inicializador.ortogonal(this.nHid, this.nHid, ganhoOcul);
    this.b2 = new Float32Array(this.nHid);
    this.w3 = Inicializador.ortogonal(this.nOut, this.nHid, this.ganhoSaida);
    this.b3 = new Float32Array(this.nOut);

    this.zerarGrads();
  }

  zerarGrads() {
    this.gW1 = Matriz.matZeros(this.nHid, this.nIn);
    this.gB1 = new Float32Array(this.nHid);
    this.gW2 = Matriz.matZeros(this.nHid, this.nHid);
    this.gB2 = new Float32Array(this.nHid);
    this.gW3 = Matriz.matZeros(this.nOut, this.nHid);
    this.gB3 = new Float32Array(this.nOut);
  }

  fwd(x) {
    const z1 = Matriz.matVec(this.w1, x, this.b1);
    const a1 = this.ativ === "relu" ? Ativacoes.relu(z1) : Ativacoes.tanh(z1);
    const z2 = Matriz.matVec(this.w2, a1, this.b2);
    const a2 = this.ativ === "relu" ? Ativacoes.relu(z2) : Ativacoes.tanh(z2);
    const z3 = Matriz.matVec(this.w3, a2, this.b3);
    return { x, z1, a1, z2, a2, z3 };
  }

  acumGrad(cache, gradSaida) {
    const { x, z1, a1, z2, a2 } = cache;

    for (let i = 0; i < this.nOut; i++) this.gB3[i] += gradSaida[i];
    Matriz.acumOuter(this.gW3, gradSaida, a2, 1);

    const gradA2 = Matriz.matVecT(this.w3, gradSaida);
    const gradZ2 = new Float32Array(this.nHid);
    if (this.ativ === "relu") {
      for (let j = 0; j < this.nHid; j++) gradZ2[j] = gradA2[j] * (z2[j] > 0 ? 1 : 0);
    } else {
      for (let j = 0; j < this.nHid; j++) gradZ2[j] = gradA2[j] * (1 - a2[j] * a2[j]);
    }

    for (let i = 0; i < this.nHid; i++) this.gB2[i] += gradZ2[i];
    Matriz.acumOuter(this.gW2, gradZ2, a1, 1);

    const gradA1 = Matriz.matVecT(this.w2, gradZ2);
    const gradZ1 = new Float32Array(this.nHid);
    if (this.ativ === "relu") {
      for (let j = 0; j < this.nHid; j++) gradZ1[j] = gradA1[j] * (z1[j] > 0 ? 1 : 0);
    } else {
      for (let j = 0; j < this.nHid; j++) gradZ1[j] = gradA1[j] * (1 - a1[j] * a1[j]);
    }

    for (let i = 0; i < this.nHid; i++) this.gB1[i] += gradZ1[i];
    Matriz.acumOuter(this.gW1, gradZ1, x, 1);
  }

  escalarGrads(f) {
    for (let i = 0; i < this.nHid; i++) {
      this.gB1[i] *= f;
      const lw = this.gW1[i];
      for (let j = 0; j < this.nIn; j++) lw[j] *= f;
    }
    for (let i = 0; i < this.nHid; i++) {
      this.gB2[i] *= f;
      const lw = this.gW2[i];
      for (let j = 0; j < this.nHid; j++) lw[j] *= f;
    }
    for (let i = 0; i < this.nOut; i++) {
      this.gB3[i] *= f;
      const lw = this.gW3[i];
      for (let j = 0; j < this.nHid; j++) lw[j] *= f;
    }
  }

  normaGrad() {
    let s = 0;
    for (let i = 0; i < this.nHid; i++) {
      s += this.gB1[i] * this.gB1[i];
      const lw = this.gW1[i];
      for (let j = 0; j < this.nIn; j++) s += lw[j] * lw[j];
    }
    for (let i = 0; i < this.nHid; i++) {
      s += this.gB2[i] * this.gB2[i];
      const lw = this.gW2[i];
      for (let j = 0; j < this.nHid; j++) s += lw[j] * lw[j];
    }
    for (let i = 0; i < this.nOut; i++) {
      s += this.gB3[i] * this.gB3[i];
      const lw = this.gW3[i];
      for (let j = 0; j < this.nHid; j++) s += lw[j] * lw[j];
    }
    return Math.sqrt(s);
  }

  clipGrad(maxNorma) {
    const n = this.normaGrad();
    if (n > maxNorma) this.escalarGrads(maxNorma / n);
  }

  paramsArr() {
    return [
      { tipo: "mat", ref: this.w1, grad: this.gW1 },
      { tipo: "vec", ref: this.b1, grad: this.gB1 },
      { tipo: "mat", ref: this.w2, grad: this.gW2 },
      { tipo: "vec", ref: this.b2, grad: this.gB2 },
      { tipo: "mat", ref: this.w3, grad: this.gW3 },
      { tipo: "vec", ref: this.b3, grad: this.gB3 }
    ];
  }

  serializar() {
    return {
      nIn: this.nIn, nHid: this.nHid, nOut: this.nOut, ativ: this.ativ, ganhoSaida: this.ganhoSaida,
      w1: this.w1.map(l => Array.from(l)), b1: Array.from(this.b1),
      w2: this.w2.map(l => Array.from(l)), b2: Array.from(this.b2),
      w3: this.w3.map(l => Array.from(l)), b3: Array.from(this.b3)
    };
  }

  carregarPesos(o) {
    this.w1 = o.w1.map(l => Float32Array.from(l));
    this.b1 = Float32Array.from(o.b1);
    this.w2 = o.w2.map(l => Float32Array.from(l));
    this.b2 = Float32Array.from(o.b2);
    this.w3 = o.w3.map(l => Float32Array.from(l));
    this.b3 = Float32Array.from(o.b3);
    this.zerarGrads();
  }
}

if (typeof window !== "undefined") window.Rede = Rede;
if (typeof module !== "undefined") module.exports = Rede;
