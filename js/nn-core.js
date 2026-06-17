// ========= NN Core =========
const Activations = {
  relu: { f: (x) => Math.max(0, x), df: (y) => (y > 0 ? 1 : 0) },
  sigmoid: { f: (x) => 1 / (1 + Math.exp(-x)), df: (y) => y * (1 - y) },
  tanh: { f: (x) => Math.tanh(x), df: (y) => 1 - y * y },
  linear: { f: (x) => x, df: (_) => 1 },
};

function zeros(r, c) {
  return Array.from({ length: r }, () => Array(c).fill(0));
}

function randn(r, c, rand) {
  const m = zeros(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      let u = 0;
      let v = 0;
      while (u === 0) u = rand();
      while (v === 0) v = rand();
      const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      const scale = Math.sqrt(2 / r); // He initialization, buona per ReLU e accettabile didatticamente
      m[i][j] = z * scale;
    }
  }
  return m;
}

function dot(a, b) {
  const r = a.length;
  const c = b[0].length;
  const n = b.length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += a[i][k] * b[k][j];
      m[i][j] = s;
    }
  }
  return m;
}

function addBias(a, b) {
  const r = a.length;
  const c = a[0].length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) m[i][j] = a[i][j] + b[0][j];
  }
  return m;
}

function applyActivation(a, act) {
  const r = a.length;
  const c = a[0].length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) m[i][j] = act.f(a[i][j]);
  }
  return m;
}

function hadamard(a, b) {
  const r = a.length;
  const c = a[0].length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) m[i][j] = a[i][j] * b[i][j];
  }
  return m;
}

function transpose(a) {
  const r = a.length;
  const c = a[0].length;
  const m = zeros(c, r);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) m[j][i] = a[i][j];
  }
  return m;
}

function bce(pred, target) {
  const r = pred.length;
  const c = pred[0].length;
  let loss = 0;
  const grad = zeros(r, c);
  const eps = 1e-7;

  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      const p = clamp(pred[i][j], eps, 1 - eps);
      const targetValue = target[i][j];
      loss += -(
        targetValue * Math.log(p) +
        (1 - targetValue) * Math.log(1 - p)
      );
      grad[i][j] = p - targetValue; // da usare con output sigmoid: dZ = p - t
    }
  }

  return [loss / r, grad];
}

class DenseLayer {
  constructor(
    inputSize,
    outputSize,
    activation = "relu",
    useBias = true,
    rand = Math.random,
  ) {
    this.in = inputSize;
    this.out = outputSize;
    this.activation = activation;
    this.useBias = useBias;
    this.W = randn(inputSize, outputSize, rand);
    this.b = useBias ? zeros(1, outputSize) : null;
  }

  forward(X) {
    this.X = X;
    const Zlin = addBias(
      dot(X, this.W),
      this.useBias ? this.b : zeros(1, this.out),
    );
    this.Z = Zlin;
    this.A = applyActivation(Zlin, Activations[this.activation]);
    return this.A;
  }

  backward(dA, lr, isLastLayer = false) {
    const act = Activations[this.activation];
    const r = this.A.length;
    const c = this.A[0].length;

    const isSigmoidOutput = this.activation === "sigmoid" && isLastLayer;

    const dAct = zeros(r, c);
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < c; j++) dAct[i][j] = act.df(this.A[i][j]);
    }

    const dZ = isSigmoidOutput ? dA : hadamard(dA, dAct);

    const Xt = transpose(this.X);
    const dW = dot(Xt, dZ);

    const dB = this.useBias
      ? [
          Array.from({ length: c }, (_, j) =>
            dZ.reduce((s, row) => s + row[j], 0),
          ),
        ]
      : null;

    const Wt = transpose(this.W);
    const dX = dot(dZ, Wt);

    const CLIP = 1.0;

    for (let i = 0; i < this.W.length; i++) {
      for (let j = 0; j < this.W[0].length; j++) {
        let g = dW[i][j] / r;
        g = clamp(g, -CLIP, CLIP);
        this.W[i][j] -= lr * g;
      }
    }

    if (this.useBias) {
      for (let j = 0; j < this.b[0].length; j++) {
        let g = dB[0][j] / r;
        g = clamp(g, -CLIP, CLIP);
        this.b[0][j] -= lr * g;
      }
    }

    return dX;
  }
}

class Network {
  constructor() {
    this.layers = [];
  }

  add(L) {
    this.layers.push(L);
  }

  forward(X) {
    let A = X;
    for (const L of this.layers) A = L.forward(A);
    return A;
  }

  backward(dA, lr) {
    let grad = dA;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const isLastLayer = i === this.layers.length - 1;
      grad = this.layers[i].backward(grad, lr, isLastLayer);
    }
  }
}

