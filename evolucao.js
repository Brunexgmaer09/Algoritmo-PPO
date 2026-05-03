(function(glb) {
const RedeNeural = (typeof module !== "undefined" && module.exports)
    ? require("./rede_neural.js").RedeNeural
    : glb.RedeNeural;

class Resultado {
    constructor(rede, score, desempate = 0) {
        this.rede = rede;
        this.score = score;
        this.desempate = desempate;
    }

    cmp(outro) {
        if (this.score > outro.score) return 1;
        if (this.score < outro.score) return -1;
        if (this.desempate > outro.desempate) return 1;
        if (this.desempate < outro.desempate) return -1;
        return 0;
    }
}

class Entidade {
    constructor() {
        this.nome = "";
        this.params = {};
        this.shape = [];
        this.qtdGeracoes = 0;
        this.scoreMax = 0;
        this.rede = null;
    }

    incGeracao() {
        this.qtdGeracoes += 1;
    }

    getRedeAleatoria() {
        const r = new RedeNeural(this.shape);
        r.setPesosAleatorios();
        return r;
    }

    setRedeDeResultado(res) {
        this.rede = res.rede;
        this.scoreMax = res.score;
    }

    getParam(chave, padrao) {
        return this.params[chave] ?? padrao;
    }

    setParam(chave, vlr) {
        this.params[chave] = vlr;
    }

    getParametrosSave() {
        return {
            name: this.nome,
            shape: this.shape.slice(),
            max_score: this.scoreMax,
            gen_count: this.qtdGeracoes,
            params: { ...this.params }
        };
    }

    setParametrosDeDict(par) {
        this.nome = par.name ?? this.nome;
        this.shape = par.shape ?? this.shape;
        this.qtdGeracoes = par.gen_count ?? this.qtdGeracoes;
        this.scoreMax = par.max_score ?? this.scoreMax;
        this.params = par.params ?? this.params;
    }

    toJSON() {
        return {
            settings: this.getParametrosSave(),
            weights: this.rede ? this.rede.weights.map(m => m.map(r => r.slice())) : []
        };
    }

    static fromJSON(obj) {
        const e = new Entidade();
        e.setParametrosDeDict(obj.settings);
        e.rede = new RedeNeural(obj.settings.shape);
        e.rede.setPesos(obj.weights);
        return e;
    }
}

function indexLoop(idx, len) {
    return idx >= len ? idx % len : idx;
}

class Evolucao {
    constructor() {
        this.melhorResultado = new Resultado(null, -1, 0);
        this.taxaMutacao = 0;
        this.taxaMutacaoInicial = 0.8;
        this.taxaMutacaoMin = 0.3;
        this.geracaoEstabilizacao = 300;
    }

    calcTaxaMutacao(ger) {
        if (ger <= 1) return this.taxaMutacaoInicial;
        if (ger >= this.geracaoEstabilizacao) return this.taxaMutacaoMin;
        const p = (ger - 1) / (this.geracaoEstabilizacao - 1);
        return this.taxaMutacaoInicial - (this.taxaMutacaoInicial - this.taxaMutacaoMin) * p;
    }

    carregarGeracao(rede, populacao) {
        return this.novaGeracao([rede], populacao);
    }

    novaGeracao(redes, populacao, geracao) {
        const txMut = geracao != null ? this.calcTaxaMutacao(geracao) : this.taxaMutacao;
        const out = new Array(populacao);
        for (let i = 0; i < populacao; i++) {
            out[i] = redes[indexLoop(i, redes.length)].reproduzir(txMut);
        }
        return out;
    }

    novaGeracaoDeResultados(resultados, populacao, qtdMelhores = 3, geracao) {
        const ord = resultados.slice().sort((a, b) => b.cmp(a));
        const qtd = Math.min(qtdMelhores, ord.length);
        const melhores = [];
        for (let i = 0; i < qtd; i++) melhores.push(ord[i].rede);
        return this.novaGeracao(melhores, populacao, geracao);
    }

    acharMelhorResultado(resultados) {
        let atual = resultados[0];
        for (let i = 1; i < resultados.length; i++) {
            if (resultados[i].cmp(atual) === 1) atual = resultados[i];
        }
        if (atual.cmp(this.melhorResultado) === 1) this.melhorResultado = atual;
        return this.melhorResultado;
    }
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = { Resultado, Entidade, Evolucao, indexLoop };
} else {
    glb.Resultado = Resultado;
    glb.Entidade = Entidade;
    glb.Evolucao = Evolucao;
    glb.indexLoop = indexLoop;
}
})(typeof window !== "undefined" ? window : globalThis);
