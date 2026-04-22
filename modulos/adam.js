"use strict";

class Adam {
  constructor(rede, cfg) {
    cfg = cfg || {};
    this.lr = cfg.lr != null ? cfg.lr : 3e-4;
    this.beta1 = cfg.beta1 != null ? cfg.beta1 : 0.9;
    this.beta2 = cfg.beta2 != null ? cfg.beta2 : 0.999;
    this.eps = cfg.eps != null ? cfg.eps : 1e-8;
    this.t = 0;
    this.estados = rede.paramsArr().map(p => {
      if (p.tipo === "mat") {
        return { m: Matriz.matZeros(p.ref.length, p.ref[0].length), v: Matriz.matZeros(p.ref.length, p.ref[0].length) };
      } else {
        return { m: new Float32Array(p.ref.length), v: new Float32Array(p.ref.length) };
      }
    });
    this.rede = rede;
  }

  passo() {
    this.t++;
    const params = this.rede.paramsArr();
    const cb1 = 1 - Math.pow(this.beta1, this.t);
    const cb2 = 1 - Math.pow(this.beta2, this.t);
    const lrEf = this.lr * Math.sqrt(cb2) / cb1;

    for (let pi = 0; pi < params.length; pi++) {
      const p = params[pi], st = this.estados[pi];
      if (p.tipo === "mat") {
        for (let i = 0; i < p.ref.length; i++) {
          const lw = p.ref[i], lg = p.grad[i], lm = st.m[i], lv = st.v[i];
          for (let j = 0; j < lw.length; j++) {
            const g = lg[j];
            lm[j] = this.beta1 * lm[j] + (1 - this.beta1) * g;
            lv[j] = this.beta2 * lv[j] + (1 - this.beta2) * g * g;
            lw[j] -= lrEf * lm[j] / (Math.sqrt(lv[j]) + this.eps);
          }
        }
      } else {
        for (let i = 0; i < p.ref.length; i++) {
          const g = p.grad[i];
          st.m[i] = this.beta1 * st.m[i] + (1 - this.beta1) * g;
          st.v[i] = this.beta2 * st.v[i] + (1 - this.beta2) * g * g;
          p.ref[i] -= lrEf * st.m[i] / (Math.sqrt(st.v[i]) + this.eps);
        }
      }
    }
  }

  setLr(lr) { this.lr = lr; }
}

if (typeof window !== "undefined") window.Adam = Adam;
if (typeof module !== "undefined") module.exports = Adam;
