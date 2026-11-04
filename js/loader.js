// Loading: app config, MuJoCo WASM, onnxruntime-web, the scene manifest (agent A's mjcf/manifest.json) with
// lazily loaded objects (files written into a MuJoCo virtual FS, then compiled), per-object task data, and the policy manifest
// (agent B's policy/manifest.json). Every URL is relative to the page (GitHub Pages project site / claude.ai Artifact).

const VENDOR = new URL("../vendor/", import.meta.url);

export async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  if (!res.body || !onBytes) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    n += value.length;
    onBytes(value.length);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

/** Decode base64 text bytes (Uint8Array.fromBase64 where available, else atob). */
function decodeBase64(buf) {
  const text = new TextDecoder().decode(buf).replace(/\s+/g, "");
  if (typeof Uint8Array.fromBase64 === "function") return Uint8Array.fromBase64(text);
  const bin = atob(text);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * Fetch a manifest-listed file by its original name. served (written by tools/build.py for the Artifact target, whose host serves
 * only some extensions): {name: {parts: [paths], encoding: raw | base64}}; the parts are fetched, decoded and concatenated, so the
 * caller gets the original bytes. url: resolves a path relative to the manifest.
 */
export async function fetchServed(url, served, name, onBytes) {
  const s = served && served[name];
  if (!s) return fetchWithProgress(url(name), onBytes);
  const bufs = await Promise.all(s.parts.map((p) => fetchWithProgress(url(p), onBytes)));
  const arrs = bufs.map((b) => (s.encoding === "base64" ? decodeBase64(b) : new Uint8Array(b)));
  const out = new Uint8Array(arrs.reduce((n, x) => n + x.length, 0));
  let o = 0;
  for (const x of arrs) { out.set(x, o); o += x.length; }
  return out.buffer;
}

export async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

async function exists(url) {
  try { const r = await fetch(url, { method: "GET" }); return r.ok; } catch { return false; }
}

/**
 * App config: app_config.json next to the page (written by tools/build.sh), overridden by URL parameters
 * ?scene=<manifest url>&policy=<manifest url | none | zero | random>&variant=<fp16|fp32|...>&object=<name>.
 * Defaults: assets/mjcf/manifest.json (agent A's scene; tools/serve.py maps it onto ../mjcf) and assets/policy/manifest.json.
 */
export async function loadAppConfig() {
  let cfg = {};
  try { cfg = await fetchJson("app_config.json"); } catch { /* optional */ }
  const q = new URLSearchParams(location.search);
  for (const k of ["scene", "policy", "variant", "object", "conditions", "seed", "object_data"]) if (q.get(k)) cfg[k] = q.get(k);
  if (!cfg.scene) cfg.scene = "assets/mjcf/manifest.json";
  if (!cfg.policy) {
    cfg.policy = (await exists("assets/policy/manifest.json")) ? "assets/policy/manifest.json" : "random";
  }
  return cfg;
}

export async function loadMujoco(onBytes) {
  const wasmUrl = new URL("mujoco/mujoco.wasm", VENDOR).href;
  const [{ default: load }, wasm] = await Promise.all([
    import(new URL("mujoco/mujoco.js", VENDOR).href),
    fetchWithProgress(wasmUrl, onBytes),
  ]);
  return load({ wasmBinary: new Uint8Array(wasm), locateFile: (f) => new URL(`mujoco/${f}`, VENDOR).href });
}

export async function loadOrt() {
  const ort = await import(new URL("ort/ort.wasm.min.mjs", VENDOR).href);
  ort.env.wasm.wasmPaths = new URL("ort/", VENDOR).href;
  ort.env.wasm.numThreads = 1; // no SharedArrayBuffer without COOP/COEP (GitHub Pages, Artifacts)
  ort.env.wasm.proxy = false;
  return ort;
}

// Defaults from STATE_SCHEMA.md section 1 (agent A names the MJCF bodies after the URDF links).
const DEFAULT_NAMES = {
  object_body: "object", palm_body: "link6",
  fingertip_bodies: ["link_3_0", "link_7_0", "link_11_0", "link_15_0"],
  hand_link_bodies: ["link_1_0", "link_2_0", "link_3_0", "link_5_0", "link_6_0", "link_7_0", "link_9_0", "link_10_0", "link_11_0",
    "link_12_0", "link_14_0", "link_15_0"],
  table_surface_site: "table_surface", goal_mocap: "goal_marker", robot_base_body: "link_base",
};
const DEFAULT_JOINTS = [...Array.from({ length: 6 }, (_, i) => `joint${i + 1}`), ...Array.from({ length: 16 }, (_, i) => `joint_${i}_0`)];

/**
 * Normalize a scene manifest into {base, names, joint_names, actuator_names, home_keyframe, render, task, actuation, objects[],
 * common_files[], default_object}. Two formats: agent A's mjcf/manifest.json (robot_files + objects[{name, scene, visual}];
 * actuators named like the joints, mocap body "goal") and a generic form (common_files + objects[{name, scene, files, task}]). Object task data
 * (points, mesh, hull, rest poses) come from <objectDataBase><name>.json unless an object lists its own "task" file.
 */
export function normalizeScene(json, url, objectDataBase = "assets/objdata/") {
  const base = new URL(".", new URL(url, location.href)).href;
  const fromA = Array.isArray(json.robot_files);
  const names = Object.assign({}, DEFAULT_NAMES, fromA ? { goal_mocap: "goal", table_surface_site: "table_top" } : {}, json.names || {});
  const dataBase = new URL(objectDataBase, location.href).href;
  const objects = (json.objects || []).map((o) => (typeof o === "string" ? { name: o } : o)).map((o) => ({
    name: o.name, label: o.label || prettyName(o.name), scene: o.scene || `scene_${o.name}.xml`,
    files: o.files || (o.visual ? [o.visual] : []), split: o.split || "", mass_kg: o.mass_kg,
    taskUrl: o.task ? new URL(o.task, base).href : new URL(`${o.name}.json`, dataBase).href,
  }));
  const task = Object.assign({ control_hz: 30, table_surface_z: 0.53, goal_box: { mins: [-0.35, -0.2, 0.53], maxs: [0.35, 0.2, 0.95] },
    // tolerance_m: 2 cm page default (surface64 metric, ref radius 0.1; user 11/02, was the r321 checkpoint 5 cm); from the r321 18.42B env_state:
    // rot_cap_rad = _c3s_rot_cap (the goal-rotation curriculum's cap; the SOFT check); precision_m: the stricter 2 cm readout
    clearance_m: 0.005, delta_m: 0.1, rot_cap_rad: 1.5, tolerance_m: 0.02, precision_m: 0.02, surface64_ref_radius: 0.1 }, json.task || {});
  const joints = json.joint_names || DEFAULT_JOINTS;
  return {
    raw: json, base, url: (f) => new URL(f, base).href, names, fromA,
    joint_names: joints,
    actuator_names: json.actuator_names || (fromA ? joints : joints.map((n) => `act_${n}`)),
    home_keyframe: json.home_keyframe ?? "home",
    render: Object.assign({ visual_groups: [0, 1, 2], hidden_groups: [3, 4, 5] }, json.render || {}),
    task, actuation: json.actuation || { mode: "position" },
    common_files: json.common_files || json.robot_files || [], objects,
    default_object: json.default_object || (objects.find((o) => o.split === "train") || objects[0])?.name,
    sizes: json.files && !Array.isArray(json.files) ? json.files : json.sizes || {},
    served: json.served || null,
  };
}

function prettyName(n) {
  return n.replace(/^vx_/, "").replace(/^\d+_/, "").replace(/_/g, " ");
}

/**
 * Agent A's params.json (optional, next to the scene manifest): the Isaac-matched hand PD table for the 'isaac_pd' hand drive,
 * the object spawn xy and the table top. Returns {handPD: {K, D, effort, vmax}, spawnXY, tableTopZ, handMode} or null.
 */
export async function loadSceneParams(spec) {
  let p;
  try { p = await fetchJson(spec.url(spec.raw.params || "params.json")); } catch { return null; }
  const joints = p.robot?.joints || {};
  const hand = spec.joint_names.slice(6);
  const ok = hand.every((n) => joints[n]);
  return {
    raw: p,
    handPD: ok ? { K: hand.map((n) => joints[n].stiffness), D: hand.map((n) => joints[n].damping),
      effort: joints[hand[0]].effort_limit_eval ?? 0.17, vmax: hand.map((n) => joints[n].velocity_limit ?? 1e9) } : null,
    spawnXY: p.object?.spawn_xy || null, tableTopZ: p.table?.top_z ?? null, handMode: p.hand_actuator_mode || "position",
  };
}

/** Restrict the manifest's objects to a subset (app_config.json "objects", e.g. a public build). */
export function filterObjects(spec, names) {
  if (!names || !names.length) return spec;
  const keep = new Set(names);
  spec.objects = spec.objects.filter((o) => keep.has(o.name));
  if (!keep.has(spec.default_object)) spec.default_object = spec.objects[0]?.name;
  return spec;
}

/** Loads object scenes lazily into one MuJoCo VFS; compiled models are cached per object (deleted on eviction). */
export class SceneLoader {
  constructor(mj, spec) {
    this.mj = mj;
    this.spec = spec;
    this.vfs = new mj.MjVFS();
    this.added = new Set();
    this.dataCache = new Map();
  }

  async _addFiles(files, onBytes) {
    const todo = files.filter((f) => !this.added.has(f));
    const bufs = await Promise.all(todo.map((f) => fetchServed(this.spec.url, this.spec.served, f, onBytes)));
    todo.forEach((f, i) => { this.vfs.addBuffer(f, new Uint8Array(bufs[i])); this.added.add(f); });
  }

  /** Bytes still to download for an object (for the progress bar), when the manifest lists sizes. */
  pendingBytes(obj) {
    const sizes = this.spec.sizes || {};
    return [...this.spec.common_files, ...obj.files, obj.scene].filter((f) => !this.added.has(f)).reduce((s, f) => s + (sizes[f] || 0), 0);
  }

  async loadObject(name, onBytes) {
    const obj = this.spec.objects.find((o) => o.name === name);
    if (!obj) throw new Error(`object '${name}' not in the scene manifest`);
    await this._addFiles([...this.spec.common_files, ...obj.files, obj.scene], onBytes);
    let m;
    try {
      m = this.mj.MjModel.from_xml_path(obj.scene, this.vfs);
    } catch (e) {
      const msg = typeof e === "number" && this.mj.getExceptionMessage ? this.mj.getExceptionMessage(e) : e?.message || String(e);
      throw new Error(`MuJoCo could not compile ${obj.scene}: ${msg}`);
    }
    const data = await this.objectData(obj);
    return { m, obj, data };
  }

  /** Per-object task data: points64, pool_p / pool_n (512 + normals), hull, centroid, mesh {verts, faces}, rest_quats.
   * TODO(B): read policy/objects/<name>/ (export_object_data.py) when it exists. */
  async objectData(obj) {
    if (this.dataCache.has(obj.name)) return this.dataCache.get(obj.name);
    let d = { name: obj.name, label: obj.label };
    Object.assign(d, await fetchJson(obj.taskUrl));
    d.label = obj.label || d.label;
    if (d.mesh && Array.isArray(d.mesh.faces) && Array.isArray(d.mesh.faces[0])) d.mesh.faces = Int32Array.from(d.mesh.faces.flat());
    if (!d.points64 && d.pool_p) d.points64 = d.pool_p.slice(0, 64);
    this.dataCache.set(obj.name, d);
    return d;
  }
}

/**
 * Resolve a policy manifest. Formats: agent B's policy/onnx[_fp32]/manifest.json ({onnx, external_data[], inputs {obs,h,c},
 * outputs {action,h_out,c_out}, lstm_hidden}) or a combined manifest {variants: {fp16: "fp16/manifest.json" | {...}},
 * default_variant} written by tools/build.py (GitHub Pages ships fp32, the Artifact build fp16 chunks).
 */
export async function resolvePolicy(url, variantName) {
  const top = await fetchJson(url);
  let man = top, manUrl = new URL(url, location.href).href, name = "default";
  if (top.variants) {
    const keys = Object.keys(top.variants);
    name = variantName && top.variants[variantName] ? variantName : top.default_variant && top.variants[top.default_variant] ? top.default_variant : keys[0];
    const v = top.variants[name];
    if (typeof v === "string") { manUrl = new URL(v, manUrl).href; man = await fetchJson(manUrl); }
    else man = Object.assign({}, top, v);
  } else if (top.weights) name = top.weights;
  const base = new URL(".", manUrl).href;
  const ins = man.inputs || {}, outs = man.outputs || {};
  const inNames = Object.keys(ins), outNames = Object.keys(outs);
  const ext = (man.external_data || []).map((f) => (typeof f === "string" ? { file: f, path: f } : f));
  const sizes = {};
  for (const [f, info] of Object.entries(man.files || {})) sizes[f] = info.bytes ?? info;
  const urlOf = (f) => new URL(f, base).href;
  return {
    fetchFile: (name, onBytes) => fetchServed(urlOf, man.served || null, name, onBytes),
    raw: man, base, url: urlOf, label: man.source?.run ? `${man.source.run} ${(man.source.frame / 1e9).toFixed(2)}B` : man.label || "",
    variantName: name, variants: top.variants || null, variant: { onnx: man.onnx || "policy.onnx", external_data: ext },
    io: man.io || { obs: inNames[0] || "obs", h: inNames[1] || "h", c: inNames[2] || "c", action: outNames[0] || "action",
      h_out: outNames[1] || "h_out", c_out: outNames[2] || "c_out" },
    obs_dim: man.obs_dim || 601, action_dim: man.action_dim || 22, hidden: man.lstm_hidden || man.hidden || 1024,
    state_shape: ins.h || man.state_shape || null,
    constantsUrl: man !== top && man.constants ? new URL(man.constants, manUrl).href : top.constants ? new URL(top.constants, new URL(url, location.href)).href : null,
    bytes: [man.onnx || "policy.onnx", ...ext.map((e) => e.file)].reduce((s, f) => s + (sizes[f] || 0), 0),
  };
}
