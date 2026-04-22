"use strict";

class PPO {
  constructor(cfg) {
    this.nEst = cfg.nEst;
    this.nAcoes = cfg.nAcoes;
    this.nHid = cfg.nHid || 64;
    this.nAmbs = cfg.nAmbientes || 1;
    this.lr = cfg.lr != null ? cfg.lr : 3e-4;
    this.lrFinal = cfg.lrFinal != null ? cfg.lrFinal : this.lr;
    this.gama = cfg.gama || 0.99;
    this.lam = cfg.lam || 0.95;
    this.eps = cfg.eps || 0.2;
    this.entCoef = cfg.entCoef != null ? cfg.entCoef : 0.01;
    this.vCoef = cfg.vCoef != null ? cfg.vCoef : 0.5;
    this.epocas = cfg.epocas || 10;
    this.tamLote = cfg.tamLote || 64;
    this.maxNormaGrad = cfg.maxNormaGrad || 0.5;
    this.klAlvo = cfg.klAlvo != null ? cfg.klAlvo : 0.02;
    this.normObs = cfg.normObs !== false;
    this.normRec = cfg.normRec !== false;
    this.totalAtualizacoes = cfg.totalAtualizacoes || 0;

    this.ator = new Rede({ nIn: this.nEst, nHid: this.nHid, nOut: this.nAcoes, ativ: "tanh", ganhoSaida: 0.01 });
    this.critico = new Rede({ nIn: this.nEst, nHid: this.nHid, nOut: 1, ativ: "tanh", ganhoSaida: 1.0 });
    this.optAtor = new Adam(this.ator, { lr: this.lr });
    this.optCrit = new Adam(this.critico, { lr: this.lr });
    this.dist = new Categorica(this.nAcoes);
    this.buf = new Buffer({ gama: this.gama, lam: this.lam, nAmbs: this.nAmbs });

    this.normEst = this.normObs ? new Normalizador(this.nEst, 10) : null;
    this.normRecF = this.normRec ? Array.from({ length: this.nAmbs }, () => new NormalizadorRecompensa(this.gama)) : null;

    this.atualizacao = 0;
  }

  prepEst(estado, atualiza) {
    const x = estado instanceof Float32Array ? estado : Float32Array.from(estado);
    if (!this.normEst) return x;
    if (atualiza) this.normEst.atualizar(x);
    return this.normEst.normalizar(x);
  }

  agir(estado, treinando) {
    const x = this.prepEst(estado, treinando !== false);
    const cAtor = this.ator.fwd(x);
    const probs = this.dist.ativacaoSaida(cAtor.z3);
    const cCrit = this.critico.fwd(x);
    const acao = treinando === false ? PPO.argmax(probs) : this.dist.amostrar(probs);
    const logP = this.dist.logProb(probs, acao);
    return { acao, logP, val: cCrit.z3[0], probs, estProc: x };
  }

  agirLote(estados, treinando) {
    const n = estados.length;
    const res = new Array(n);
    for (let i = 0; i < n; i++) res[i] = this.agir(estados[i], treinando);
    return res;
  }

  static argmax(v) {
    let mx = -Infinity, idx = 0;
    for (let i = 0; i < v.length; i++) if (v[i] > mx) { mx = v[i]; idx = i; }
    return idx;
  }

  lembrar(idAmb, estProc, acao, logP, val, rec, terminou) {
    let recF = rec;
    if (this.normRecF) recF = this.normRecF[idAmb].filtrar(rec, terminou);
    this.buf.add(idAmb, estProc, acao, logP, val, recF, terminou);
  }

  bufferCheio(passosPorAmb) {
    return this.buf.tamPorAmb() >= passosPorAmb;
  }

  treinar(valsFinal) {
    if (!Array.isArray(valsFinal) && !(valsFinal instanceof Float32Array)) {
      const v = valsFinal != null ? valsFinal : 0;
      valsFinal = new Float32Array(this.nAmbs);
      for (let i = 0; i < this.nAmbs; i++) valsFinal[i] = v;
    }
    const nTotal = this.buf.tam();
    if (nTotal === 0) return null;
    this.buf.calcGae(valsFinal);
    this.atualizacao++;

    if (this.totalAtualizacoes > 0) {
      const frac = 1 - (this.atualizacao - 1) / this.totalAtualizacoes;
      const lrAt = Math.max(this.lrFinal, this.lr * Math.max(frac, 0));
      this.optAtor.setLr(lrAt);
      this.optCrit.setLr(lrAt);
    }

    const stats = { klMed: 0, lossA: 0, lossC: 0, ent: 0, epocas: 0 };
    let parou = false;
    let totalLotes = 0;
    const n = nTotal;

    for (let ep = 0; ep < this.epocas && !parou; ep++) {
      const idxs = this.buf.embaralhar();
      let klEpoca = 0, contLote = 0;

      for (let ini = 0; ini < n; ini += this.tamLote) {
        const fim = Math.min(ini + this.tamLote, n);
        const tam = fim - ini;

        this.ator.zerarGrads();
        this.critico.zerarGrads();

        let lossAcumA = 0, lossAcumC = 0, entAcum = 0, klAcum = 0;

        for (let k = ini; k < fim; k++) {
          const i = idxs[k];
          const est = this.buf.fEst[i];
          const acao = this.buf.fAcao[i];
          const logPVel = this.buf.fLogP[i];
          const adv = this.buf.fAdv[i];
          const ret = this.buf.fRet[i];
          const valVel = this.buf.fVal[i];

          const cA = this.ator.fwd(est);
          const probs = this.dist.ativacaoSaida(cA.z3);
          const logPNov = this.dist.logProb(probs, acao);
          const ratio = Math.exp(logPNov - logPVel);
          const ent = this.dist.entropia(probs);

          const gradLog = new Float32Array(this.nAcoes);
          const ratioClip = Math.max(1 - this.eps, Math.min(1 + this.eps, ratio));
          const obj1 = ratio * adv;
          const obj2 = ratioClip * adv;
          const usaUnclip = obj1 <= obj2;
          if (usaUnclip) {
            const fator = -adv * ratio / tam;
            for (let j = 0; j < this.nAcoes; j++) {
              const oh = j === acao ? 1 : 0;
              gradLog[j] = fator * (oh - probs[j]);
            }
          }

          const gradEnt = this.dist.gradEntLogits(probs);
          for (let j = 0; j < this.nAcoes; j++) {
            gradLog[j] += this.entCoef * gradEnt[j] / tam;
          }
          this.ator.acumGrad(cA, gradLog);

          const cC = this.critico.fwd(est);
          const v = cC.z3[0];
          const vClip = valVel + Math.max(-this.eps, Math.min(this.eps, v - valVel));
          const ls1 = (v - ret) ** 2;
          const ls2 = (vClip - ret) ** 2;
          const gradV = new Float32Array(1);
          if (ls1 >= ls2) gradV[0] = 2 * (v - ret) * this.vCoef / tam;
          else {
            const dentro = (v - valVel) > -this.eps && (v - valVel) < this.eps;
            gradV[0] = dentro ? 2 * (vClip - ret) * this.vCoef / tam : 0;
          }
          this.critico.acumGrad(cC, gradV);

          lossAcumA += -Math.min(obj1, obj2);
          lossAcumC += 0.5 * Math.max(ls1, ls2);
          entAcum += ent;
          klAcum += (logPVel - logPNov);
        }

        this.ator.clipGrad(this.maxNormaGrad);
        this.critico.clipGrad(this.maxNormaGrad);
        this.optAtor.passo();
        this.optCrit.passo();

        stats.lossA += lossAcumA / tam;
        stats.lossC += lossAcumC / tam;
        stats.ent += entAcum / tam;
        klEpoca += klAcum / tam;
        contLote++;
        totalLotes++;
      }

      const klMed = klEpoca / Math.max(contLote, 1);
      stats.klMed = klMed;
      stats.epocas = ep + 1;
      if (this.klAlvo > 0 && klMed > 1.5 * this.klAlvo) parou = true;
    }

    if (totalLotes > 0) {
      stats.lossA /= totalLotes;
      stats.lossC /= totalLotes;
      stats.ent /= totalLotes;
    }
    this.buf.limpar();
    return stats;
  }

  salvar() {
    return JSON.stringify({
      cfg: { nEst: this.nEst, nAcoes: this.nAcoes, nHid: this.nHid },
      ator: this.ator.serializar(),
      critico: this.critico.serializar(),
      normEst: this.normEst ? this.normEst.serializar() : null
    });
  }

  carregar(json) {
    const o = typeof json === "string" ? JSON.parse(json) : json;
    this.ator.carregarPesos(o.ator);
    this.critico.carregarPesos(o.critico);
    this.optAtor = new Adam(this.ator, { lr: this.lr });
    this.optCrit = new Adam(this.critico, { lr: this.lr });
    if (o.normEst && this.normEst) this.normEst.carregar(o.normEst);
  }
}

if (typeof window !== "undefined") window.PPO = PPO;
if (typeof module !== "undefined") module.exports = PPO;
