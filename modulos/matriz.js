"use strict";

const Matriz = {
  zeros(n) { return new Float32Array(n); },

  matZeros(linhas, cols) {
    const m = new Array(linhas);
    for (let i = 0; i < linhas; i++) m[i] = new Float32Array(cols);
    return m;
  },

  matVec(m, v, b, dest) {
    const r = dest || new Float32Array(m.length);
    for (let i = 0; i < m.length; i++) {
      const lin = m[i];
      let s = b[i];
      for (let j = 0; j < v.length; j++) s += lin[j] * v[j];
      r[i] = s;
    }
    return r;
  },

  matVecT(m, v, dest) {
    const cols = m[0].length;
    const r = dest || new Float32Array(cols);
    for (let j = 0; j < cols; j++) r[j] = 0;
    for (let i = 0; i < m.length; i++) {
      const lin = m[i], vi = v[i];
      for (let j = 0; j < cols; j++) r[j] += lin[j] * vi;
    }
    return r;
  },

  acumOuter(m, a, b, escala) {
    for (let i = 0; i < a.length; i++) {
      const lin = m[i], ai = a[i] * escala;
      for (let j = 0; j < b.length; j++) lin[j] += ai * b[j];
    }
  },

  acumVec(d, s, escala) {
    for (let i = 0; i < d.length; i++) d[i] += s[i] * escala;
  },

  zerarMat(m) {
    for (let i = 0; i < m.length; i++) m[i].fill(0);
  },

  copiarMat(m) {
    const r = new Array(m.length);
    for (let i = 0; i < m.length; i++) r[i] = Float32Array.from(m[i]);
    return r;
  }
};

if (typeof window !== "undefined") window.Matriz = Matriz;
if (typeof module !== "undefined") module.exports = Matriz;
