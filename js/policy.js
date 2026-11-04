// Policy runners. Interface: await init(); reset() (LSTM state zero at every episode reset); await act(obs Float32Array) ->
// Float32Array(action_dim) (deterministic mean action, before the [-1, 1] clamp of action.js).
//
// OnnxPolicy: onnxruntime-web (wasm backend, single thread: no SharedArrayBuffer / COOP-COEP needed, so it runs on GitHub Pages
// and in a claude.ai Artifact). ONNX contract (PLAN.md / agent B): inputs obs [1, obs_dim], h [layers, 1, hidden],
// c [layers, 1, hidden] -> outputs action [1, action_dim], h' and c' (names from policy/manifest.json, else the session's
// input / output order). Large models ship as an .onnx graph plus external-data chunks (manifest variant.external_data).

export class ZeroPolicy {
  constructor(dim = 22) { this.dim = dim; this.name = "zero (hold)"; this.ms = 0; }
  async init() {}
  reset() {}
  async act() { return new Float32Array(this.dim); }
}

/** Smooth random actions (an Ornstein-Uhlenbeck walk), for exercising the pipeline without a network. */
export class RandomPolicy {
  constructor(dim = 22, seed = 7) { this.dim = dim; this.name = "random (OU)"; this.x = new Float32Array(dim); this.s = seed >>> 0; this.ms = 0; }
  _r() { this.s = (this.s * 1664525 + 1013904223) >>> 0; return this.s / 4294967296 - 0.5; }
  async init() {}
  reset() { this.x.fill(0); }
  async act() {
    for (let i = 0; i < this.dim; i++) this.x[i] = 0.9 * this.x[i] + 0.35 * this._r();
    return Float32Array.from(this.x);
  }
}

export class OnnxPolicy {
  /**
   * ort: the onnxruntime-web module; manifest: resolved policy manifest (loader.resolvePolicy) with base URL, chosen
   * variant {onnx, external_data[]}, names, dims.
   */
  constructor(ort, manifest, fetchBinary) {
    this.ort = ort;
    this.man = manifest;
    this.fetchBinary = fetchBinary;
    this.name = `ONNX ${manifest.label || ""} (${manifest.variantName})`.trim();
    this.ms = 0;
  }

  async init(onBytes) {
    const { ort, man } = this;
    const v = man.variant;
    // man.fetchFile: by original name, decoding base64 text parts where the host cannot serve binaries (Artifact build)
    const get = man.fetchFile || ((name, cb) => this.fetchBinary(man.url(name), cb));
    const modelBuf = await get(v.onnx, onBytes);
    const ext = [];
    for (const f of v.external_data || []) {
      const data = new Uint8Array(await get(f.file || f, onBytes));
      ext.push({ path: f.path || f.file || f, data });
    }
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    if (ext.length) opts.externalData = ext;
    this.session = await ort.InferenceSession.create(new Uint8Array(modelBuf), opts);
    const ins = this.session.inputNames, outs = this.session.outputNames;
    const io = man.io || {};
    this.inObs = io.obs || ins[0];
    this.inH = io.h || ins[1];
    this.inC = io.c || ins[2];
    this.outA = io.action || outs[0];
    this.outH = io.h_out || outs[1];
    this.outC = io.c_out || outs[2];
    this.obsDim = man.obs_dim;
    this.actDim = man.action_dim;
    this.stateShape = man.state_shape || [1, 1, man.hidden || 1024];
    this.stateless = !this.inH;
    this.reset();
    // warm up (first run allocates)
    await this.act(new Float32Array(this.obsDim));
    this.reset();
  }

  reset() {
    const n = this.stateShape.reduce((a, b) => a * b, 1);
    this.h = new this.ort.Tensor("float32", new Float32Array(n), this.stateShape);
    this.c = new this.ort.Tensor("float32", new Float32Array(n), this.stateShape);
  }

  async act(obs) {
    const t0 = performance.now();
    const feeds = { [this.inObs]: new this.ort.Tensor("float32", Float32Array.from(obs), [1, this.obsDim]) };
    if (!this.stateless) { feeds[this.inH] = this.h; feeds[this.inC] = this.c; }
    const out = await this.session.run(feeds);
    if (!this.stateless) { this.h = out[this.outH]; this.c = out[this.outC]; }
    const a = Float32Array.from(out[this.outA].data);
    const dt = performance.now() - t0;
    this.ms = this.ms ? 0.9 * this.ms + 0.1 * dt : dt;
    return a;
  }
}
