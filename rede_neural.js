(function(glb) {
function sig(x) {
    return 1 / (1 + Math.exp(-x));
}

function rndUni() {
    return Math.random() - 0.5;
}

function criaMtx(lin, col, fn) {
    const m = new Array(lin);
    for (let i = 0; i < lin; i++) {
        const row = new Array(col);
        for (let j = 0; j < col; j++) row[j] = fn ? fn() : 0;
        m[i] = row;
    }
    return m;
}

function dotVecMtx(vec, mtx) {
    const lin = mtx.length;
    const col = mtx[0].length;
    const res = new Array(col);
    for (let j = 0; j < col; j++) {
        let s = 0;
        for (let i = 0; i < lin; i++) s += vec[i] * mtx[i][j];
        res[j] = s;
    }
    return res;
}

class RedeNeural {
    constructor(shape) {
        this.shape = shape.slice();
        this.size = shape.length;
        this.weights = [];
        for (let i = 1; i < this.size; i++) {
            this.weights.push(criaMtx(shape[i - 1], shape[i]));
        }
    }

    setPesosAleatorios() {
        for (let i = 1; i < this.size; i++) {
            this.weights[i - 1] = criaMtx(this.shape[i - 1], this.shape[i], rndUni);
        }
    }

    setPesos(pesos) {
        for (let i = 0; i < pesos.length; i++) {
            this.weights[i] = pesos[i].map(row => row.slice());
        }
    }

    forward(inp) {
        let lyr = inp;
        for (let i = 0; i < this.size - 1; i++) {
            lyr = dotVecMtx(lyr, this.weights[i]);
        }
        for (let k = 0; k < lyr.length; k++) lyr[k] = sig(lyr[k]);
        return lyr;
    }

    clone() {
        const n = new RedeNeural(this.shape);
        for (let i = 0; i < this.size - 1; i++) {
            n.weights[i] = this.weights[i].map(row => row.slice());
        }
        return n;
    }

    reproduzir(taxaMutacao) {
        const n = new RedeNeural(this.shape);
        for (let i = 1; i < this.size; i++) {
            const lin = this.shape[i - 1];
            const col = this.shape[i];
            const novo = criaMtx(lin, col);
            const orig = this.weights[i - 1];
            for (let a = 0; a < lin; a++) {
                for (let b = 0; b < col; b++) {
                    novo[a][b] = orig[a][b] + rndUni() * taxaMutacao;
                }
            }
            n.weights[i - 1] = novo;
        }
        return n;
    }

    toJSON() {
        return {
            shape: this.shape.slice(),
            weights: this.weights.map(m => m.map(r => r.slice()))
        };
    }

    static fromJSON(obj) {
        const n = new RedeNeural(obj.shape);
        n.setPesos(obj.weights);
        return n;
    }

    static calcTaxaMutacao(ger, inicial = 0.8, min = 0.3, estabiliza = 300) {
        if (ger <= 1) return inicial;
        if (ger >= estabiliza) return min;
        const p = (ger - 1) / (estabiliza - 1);
        return inicial - (inicial - min) * p;
    }
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = { RedeNeural, sig };
} else {
    glb.RedeNeural = RedeNeural;
    glb.sig = sig;
}
})(typeof window !== "undefined" ? window : globalThis);
