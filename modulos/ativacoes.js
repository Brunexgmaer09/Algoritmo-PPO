"use strict";

const Ativacoes = {
  tanh(v, dest) {
    const r = dest || new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) r[i] = Math.tanh(v[i]);
    return r;
  },

  derivTanh(a, dest) {
    const r = dest || new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) r[i] = 1 - a[i] * a[i];
    return r;
  },

  relu(v, dest) {
    const r = dest || new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) r[i] = v[i] > 0 ? v[i] : 0;
    return r;
  },

  derivRelu(z, dest) {
    const r = dest || new Float32Array(z.length);
    for (let i = 0; i < z.length; i++) r[i] = z[i] > 0 ? 1 : 0;
    return r;
  },

  softmax(v, dest) {
    const r = dest || new Float32Array(v.length);
    let mx = -Infinity;
    for (let i = 0; i < v.length; i++) if (v[i] > mx) mx = v[i];
    let s = 0;
    for (let i = 0; i < v.length; i++) { r[i] = Math.exp(v[i] - mx); s += r[i]; }
    const inv = 1 / s;
    for (let i = 0; i < v.length; i++) r[i] *= inv;
    return r;
  }
};

if (typeof window !== "undefined") window.Ativacoes = Ativacoes;
if (typeof module !== "undefined") module.exports = Ativacoes;
