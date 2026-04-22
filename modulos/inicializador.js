"use strict";

const Inicializador = {
  randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  },

  ortogonal(linhas, cols, ganho) {
    const m = new Array(linhas);
    for (let i = 0; i < linhas; i++) m[i] = new Float32Array(cols);
    for (let i = 0; i < linhas; i++)
      for (let j = 0; j < cols; j++)
        m[i][j] = Inicializador.randn();
    Inicializador.gramSchmidt(m);
    for (let i = 0; i < linhas; i++)
      for (let j = 0; j < cols; j++)
        m[i][j] *= ganho;
    return m;
  },

  gramSchmidt(m) {
    const linhas = m.length;
    for (let i = 0; i < linhas; i++) {
      for (let k = 0; k < i; k++) {
        let prod = 0;
        for (let j = 0; j < m[i].length; j++) prod += m[i][j] * m[k][j];
        for (let j = 0; j < m[i].length; j++) m[i][j] -= prod * m[k][j];
      }
      let nrm = 0;
      for (let j = 0; j < m[i].length; j++) nrm += m[i][j] * m[i][j];
      nrm = Math.sqrt(nrm) + 1e-8;
      for (let j = 0; j < m[i].length; j++) m[i][j] /= nrm;
    }
  }
};

if (typeof window !== "undefined") window.Inicializador = Inicializador;
if (typeof module !== "undefined") module.exports = Inicializador;
