"use strict";

class Buffer {
  constructor(cfg) {
    this.gama = cfg.gama;
    this.lam = cfg.lam;
    this.nAmbs = cfg.nAmbs || 1;
    this.limpar();
  }

  limpar() {
    this.trajs = new Array(this.nAmbs);
    for (let i = 0; i < this.nAmbs; i++) {
      this.trajs[i] = { est: [], acao: [], logP: [], val: [], rec: [], term: [] };
    }
    this.fEst = null;
    this.fAcao = null;
    this.fLogP = null;
    this.fVal = null;
    this.fAdv = null;
    this.fRet = null;
  }

  add(idAmb, est, acao, logP, val, rec, terminou) {
    const t = this.trajs[idAmb];
    t.est.push(est);
    t.acao.push(acao);
    t.logP.push(logP);
    t.val.push(val);
    t.rec.push(rec);
    t.term.push(terminou ? 1 : 0);
  }

  tam() {
    let s = 0;
    for (let i = 0; i < this.nAmbs; i++) s += this.trajs[i].rec.length;
    return s;
  }

  tamPorAmb() {
    return this.trajs[0].rec.length;
  }

  calcGae(valsFinal) {
    const total = this.tam();
    this.fEst = new Array(total);
    this.fAcao = new Int32Array(total);
    this.fLogP = new Float32Array(total);
    this.fVal = new Float32Array(total);
    this.fAdv = new Float32Array(total);
    this.fRet = new Float32Array(total);

    let pos = 0;
    for (let a = 0; a < this.nAmbs; a++) {
      const t = this.trajs[a];
      const n = t.rec.length;
      const advL = new Float32Array(n);
      const retL = new Float32Array(n);
      let gae = 0;
      for (let tt = n - 1; tt >= 0; tt--) {
        const proxV = tt === n - 1 ? valsFinal[a] : t.val[tt + 1];
        const naoTerm = 1 - t.term[tt];
        const delta = t.rec[tt] + this.gama * proxV * naoTerm - t.val[tt];
        gae = delta + this.gama * this.lam * naoTerm * gae;
        advL[tt] = gae;
        retL[tt] = gae + t.val[tt];
      }
      for (let tt = 0; tt < n; tt++) {
        this.fEst[pos] = t.est[tt];
        this.fAcao[pos] = t.acao[tt];
        this.fLogP[pos] = t.logP[tt];
        this.fVal[pos] = t.val[tt];
        this.fAdv[pos] = advL[tt];
        this.fRet[pos] = retL[tt];
        pos++;
      }
    }

    let med = 0;
    for (let i = 0; i < total; i++) med += this.fAdv[i];
    med /= total;
    let vrn = 0;
    for (let i = 0; i < total; i++) vrn += (this.fAdv[i] - med) ** 2;
    const dp = Math.sqrt(vrn / total) + 1e-8;
    for (let i = 0; i < total; i++) this.fAdv[i] = (this.fAdv[i] - med) / dp;
  }

  embaralhar() {
    const n = this.fEst.length;
    const idx = new Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    return idx;
  }
}

if (typeof window !== "undefined") window.Buffer = Buffer;
if (typeof module !== "undefined") module.exports = Buffer;
