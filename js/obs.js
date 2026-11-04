// JS port of agent B's policy/obs_ref.py (line by line; same names): the r321 actor observation (601) = f(state, episode memory),
// as obs_utils.build_observations builds it. Field order, frames and inputs: posedex/web_demo/STATE_SCHEMA.md. Constants:
// policy/constants.json (golden_dump.py, live env); DEFAULTS below are obs_ref.DEFAULTS. Checked against B's Isaac golden dumps by
// test/obs_golden.test.mjs. Quaternions are wxyz everywhere except the two xyzw observation fields.

import { rng as makeRng } from "./mathq.js";

export const AFFORDANCE_SIGMA_M = 0.02; // obs_utils.POINT_AFFORDANCE_SIGMA_M
export const NUM_POINTS = 64;

export const DEFAULTS = {
  palm_center_offset: [-0.000402822, -0.01270504, 0.126175931], // link6 frame
  palm_surface_centre: [-0.000678656, -0.020678878, 0.096772371], // link6 frame
  palmar_axis: [0.0, -1.0, 0.0], // link6 frame
  fingertip_offsets: [[0.000168834, -0.000886207, 0.040755136], [0.000168834, -0.000886207, 0.040755136],
    [0.000168834, -0.000886207, 0.040755136], [-0.000087614, -0.000522227, 0.051505136]],
  obs_z_shift: 0.53,
  clamp_abs_observations: 10.0,
  robot_root_pos: [0.0, 0.5, 0.493], robot_root_quat_wxyz: [Math.SQRT1_2, 0.0, 0.0, -Math.SQRT1_2],
  // task / control values (constants.json has the live ones)
  step_dt: 1 / 30, arm_moving_average: 0.1, hand_moving_average: 0.19, dof_speed_scale: 1.5,
  dwell_steps: 5, kept_grace_steps: 15, near_radius_m: 0.05, normal_tolerance_deg: 45.0, airborne_m: 0.01, success_steps: 5,
  max_episode_steps: 599,
  eval_real: { latency_steps: 6, hold_steps: 3, obs_delay_steps: 1, action_delay_steps: 1, pos_noise_m: 0.002, rot_noise_deg: 1.0 },
};

// xArm6 chain of xarm_allegro_v5_cal_sim.urdf (link_base -> link6): joint origin xyz, rpy; every axis is +z. Only used for the SIGN
// of the palm quaternion (palmQuatSignFix); the position / rotation themselves come from the simulator.
export const ARM_CHAIN = [
  [[-0.0007769996882416308, 0.0007711717044003308, 0.26595622301101685], [-0.000981912831775844, -0.0020678131841123104, 1.0152065215152106e-06]],
  [[-4.07262341468595e-05, -3.840348838934915e-08, -7.701249360447804e-12], [-1.5712535381317139, 0.002067566616460681, 0.0009407909237779677]],
  [[0.053162120282649994, -0.2844921350479126, 0.0004026298993267119], [0.001618124428205192, -0.0010822704061865807, -2.415801873212331e-06]],
  [[0.0777892991900444, 0.34164339303970337, -6.047571878298186e-05], [-1.57097327709198, 0.0020220461301505566, -1.7895548864999e-07]],
  [[-0.00013099999341648072, 0.0021800000686198473, -2.6697300799058424e-19], [1.5707963705062866, 0.0, 0.0]],
  [[0.07616599649190903, 0.09558600187301636, 0.00021499999274965376], [-1.570866346359253, 0.00012199999764561653, -0.00012799999967683107]],
];

// Actor observation layout (name, width), in order (obs.obs_list of r321).
export const LAYOUT = [["joint_pos", 22], ["joint_vel", 22], ["prev_action_targets", 22], ["palm_pos", 3], ["palm_rot", 4], ["object_rot", 4],
  ["fingertip_pos_rel_palm", 12], ["object_points_pair_aff", 448], ["affordance_vec_palm", 3], ["affordance_nearest_palm", 3],
  ["affordance_normal_palm", 3], ["affordance_normal_alignment", 1], ["affordance_point_enabled", 1], ["affordance_regrasp_mode", 1],
  ["hand_surface_vec", 51], ["table_height", 1]];
export const OBS_DIM = LAYOUT.reduce((s, [, w]) => s + w, 0); // 601
export const N_JOINTS = 22;

export function slices() {
  const out = {};
  let start = 0;
  for (const [name, width] of LAYOUT) { out[name] = [start, start + width]; start += width; }
  return out;
}

// ---- quaternion helpers (wxyz) ---------------------------------------------------------------------------------------------
export function quatMul(a, b) {
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw];
}
export const quatConj = (q) => [q[0], -q[1], -q[2], -q[3]];
/** R(q) v: v + w t + u x t, t = 2 u x v (Isaac Lab quat_apply). */
export function rotate(q, v) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}
export const rotateInv = (q, v) => rotate(quatConj(q), v);
/** URDF rpy (fixed axes x, then y, then z): q = qz(yaw) qy(pitch) qx(roll). */
export function quatFromRpy(roll, pitch, yaw) {
  const qx = [Math.cos(roll / 2), Math.sin(roll / 2), 0, 0], qy = [Math.cos(pitch / 2), 0, Math.sin(pitch / 2), 0];
  const qz = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
  return quatMul(qz, quatMul(qy, qx));
}
const CHAIN_Q = ARM_CHAIN.map(([, rpy]) => quatFromRpy(...rpy));
/** link6 orientation by URDF forward kinematics with the CONTINUOUS joint half-angles. */
export function armFkQuat(armQ, rootQuat) {
  let q = rootQuat.slice();
  for (let i = 0; i < 6; i++) {
    q = quatMul(q, CHAIN_Q[i]);
    q = quatMul(q, [Math.cos(armQ[i] / 2), 0, 0, Math.sin(armQ[i] / 2)]);
  }
  return q;
}
/** The palm quaternion with the sign the policy was trained on: flipped into the hemisphere of the FK quaternion. */
export function palmQuatSignFix(palmQuat, armQ, rootQuat) {
  const f = armFkQuat(armQ, rootQuat);
  const d = palmQuat[0] * f[0] + palmQuat[1] * f[1] + palmQuat[2] * f[2] + palmQuat[3] * f[3];
  return d >= 0 ? palmQuat.slice() : [-palmQuat[0], -palmQuat[1], -palmQuat[2], -palmQuat[3]];
}
const wxyzToXyzw = (q) => [q[1], q[2], q[3], q[0]];

// ---- geometry helpers --------------------------------------------------------------------------------------------------------
/** Farthest-point sampling: index 0 = start; each next index = the point with the largest squared distance to the chosen set
 * (ties: the lowest index). */
export function fpsIndices(points, start, count = NUM_POINTS) {
  const n = points.length;
  const nearest = new Float64Array(n).fill(Infinity);
  const index = [];
  let current = start;
  for (let it = 0; it < count; it++) {
    index.push(current);
    const c = points[current];
    let best = -1, bd = -Infinity;
    for (let j = 0; j < n; j++) {
      const p = points[j];
      const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2;
      if (d < nearest[j]) nearest[j] = d;
      if (nearest[j] > bd) { bd = nearest[j]; best = j; } // strict > keeps the lowest index on ties (np.argmax)
    }
    current = best;
  }
  return index;
}

const TINY = 2.2250738585072014e-308; // np.finfo(np.float64).tiny
const safe = (x) => (Math.abs(x) > TINY ? x : TINY);

/** Exact nearest point of a triangle mesh to p: Ericson's closest point on each triangle with the env's region order, then the
 * closest of those. tris: Float64Array (T * 9). */
export function closestPointOnMesh(p, tris) {
  let best = Infinity, bx = 0, by = 0, bz = 0;
  const T = tris.length / 9;
  for (let t = 0; t < T; t++) {
    const o = 9 * t;
    const ax = tris[o], ay = tris[o + 1], az = tris[o + 2], bxv = tris[o + 3], byv = tris[o + 4], bzv = tris[o + 5];
    const cx = tris[o + 6], cy = tris[o + 7], cz = tris[o + 8];
    const abx = bxv - ax, aby = byv - ay, abz = bzv - az, acx = cx - ax, acy = cy - ay, acz = cz - az;
    const apx = p[0] - ax, apy = p[1] - ay, apz = p[2] - az;
    const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
    const bpx = p[0] - bxv, bpy = p[1] - byv, bpz = p[2] - bzv;
    const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
    const cpx = p[0] - cx, cpy = p[1] - cy, cpz = p[2] - cz;
    const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
    const vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
    let qx, qy, qz;
    if (d1 <= 0 && d2 <= 0) { qx = ax; qy = ay; qz = az; } // vertex A
    else if (d3 >= 0 && d4 <= d3) { qx = bxv; qy = byv; qz = bzv; } // vertex B
    else if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / safe(d1 - d3); qx = ax + abx * v; qy = ay + aby * v; qz = az + abz * v; } // edge AB
    else if (d6 >= 0 && d5 <= d6) { qx = cx; qy = cy; qz = cz; } // vertex C
    else if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / safe(d2 - d6); qx = ax + acx * w; qy = ay + acy * w; qz = az + acz * w; } // edge AC
    else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { // edge BC
      const w = (d4 - d3) / safe((d4 - d3) + (d5 - d6));
      qx = bxv + (cx - bxv) * w; qy = byv + (cy - byv) * w; qz = bzv + (cz - bzv) * w;
    } else { // interior
      const denom = safe(va + vb + vc), v = vb / denom, w = vc / denom;
      qx = ax + abx * v + acx * w; qy = ay + aby * v + acy * w; qz = az + abz * v + acz * w;
    }
    const d = (qx - p[0]) ** 2 + (qy - p[1]) ** 2 + (qz - p[2]) ** 2;
    if (d < best) { best = d; bx = qx; by = qy; bz = qz; }
  }
  return [bx, by, bz];
}

// ---- constants, objects, memory ------------------------------------------------------------------------------------------------
/** obs_ref.load_constants: DEFAULTS updated by constants.json (joint_lower / joint_upper required for the real policy). */
export function loadConstants(json) {
  const c = JSON.parse(JSON.stringify(DEFAULTS));
  if (json) Object.assign(c, json);
  return c;
}

/** obs_ref.load_object from objects/<name>.json (points512, vertices, faces, face_normals); triangles flattened for the query. */
export function prepareObject(d) {
  const tris = new Float64Array(d.faces.length * 9);
  d.faces.forEach((f, i) => { for (let k = 0; k < 3; k++) for (let a = 0; a < 3; a++) tris[9 * i + 3 * k + a] = d.vertices[f[k]][a]; });
  return Object.assign({}, d, { triangles: tris });
}

/** Episode memory (STATE_SCHEMA section 2) at a reset: the 64 observed points by FPS from start_index, the commanded point. */
export function newMemory(obj, startIndex, pointP, pointN, { scaleMult = [1, 1, 1], pointEnabled = 1, regraspMode = 0, tableOffset = 0 } = {}) {
  return { obs_points: fpsIndices(obj.points512, startIndex).map((i) => obj.points512[i]), scale_mult: scaleMult.slice(),
    point_p: pointP.slice(), point_n: pointN.slice(), point_enabled: pointEnabled, regrasp_mode: regraspMode, table_offset: tableOffset };
}

// ---- the observation ------------------------------------------------------------------------------------------------------------
/**
 * (601) Float32Array actor observation. state: STATE_SCHEMA section 1 (env frame, wxyz). observedPose: [pos, quat] of the object as
 * the actor sees it ('real': the tracker pose); null = the true pose. terms (optional object) receives the per-term values.
 */
export function observation(state, mem, obj, c, observedPose = null, terms = null, out = new Float32Array(OBS_DIM)) {
  const lo = c.joint_lower, hi = c.joint_upper;
  const q = state.q, qd = state.qd;
  const palmPos = state.palm_pos, palmQuat = state.palm_quat;
  const [o, oq] = observedPose || [state.obj_pos, state.obj_quat];
  const goalPos = state.goal_pos, goalQuat = state.goal_quat;
  const scale = mem.scale_mult, enabled = mem.point_enabled;
  const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const mul3 = (a, b) => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];

  const cc = add3(palmPos, rotate(palmQuat, c.palm_center_offset)); // palm centre
  const s = add3(palmPos, rotate(palmQuat, c.palm_surface_centre)); // palm surface centre
  const pads = [0, 1, 2, 3].map((k) => add3(state.tip_pos[k], rotate(state.tip_quat[k], c.fingertip_offsets[k])));
  let k = 0;
  const put = (v) => { for (let i = 0; i < v.length; i++) out[k++] = v[i]; };
  const mark = terms ? (name, start) => { terms[name] = Array.from(out.subarray(start, k)); } : null;
  let st = 0;
  for (let j = 0; j < 22; j++) out[k++] = (2 * (q[j] - lo[j])) / (hi[j] - lo[j]) - 1;
  mark?.("joint_pos", st); st = k;
  for (let j = 0; j < 22; j++) out[k++] = j < 6 ? qd[j] : 0; // actor_hand_qd_zero
  mark?.("joint_vel", st); st = k;
  put(state.prev_targets);
  mark?.("prev_action_targets", st); st = k;
  put([cc[0], cc[1], cc[2] - c.obs_z_shift]);
  mark?.("palm_pos", st); st = k;
  put(wxyzToXyzw(palmQuatSignFix(palmQuat, q.slice(0, 6), c.robot_root_quat_wxyz)));
  mark?.("palm_rot", st); st = k;
  put(wxyzToXyzw(oq));
  mark?.("object_rot", st); st = k;
  for (const pad of pads) put(sub3(pad, cc));
  mark?.("fingertip_pos_rel_palm", st); st = k;
  // 64 point slots [x_i - c, y_i - x_i, g_i]
  const a = mul3(mem.point_p, scale);
  const inv = 1 / (2 * AFFORDANCE_SIGMA_M ** 2);
  for (let i = 0; i < NUM_POINTS; i++) {
    const P = mul3(mem.obs_points[i], scale);
    const x = add3(o, rotate(oq, P)), y = add3(goalPos, rotate(goalQuat, P));
    put(sub3(x, cc)); put(sub3(y, x));
    out[k++] = Math.exp(-((P[0] - a[0]) ** 2 + (P[1] - a[1]) ** 2 + (P[2] - a[2]) ** 2) * inv) * enabled;
  }
  mark?.("object_points_pair_aff", st); st = k;
  // palm-frame point fields; q_near = nearest point of the scaled mesh to s, object frame
  const sl = rotateInv(oq, sub3(s, o));
  const sLocal = [sl[0] / scale[0], sl[1] / scale[1], sl[2] / scale[2]];
  const qNear = mul3(closestPointOnMesh(sLocal, obj.triangles), scale);
  put(rotateInv(palmQuat, sub3(add3(o, rotate(oq, a)), s)).map((v) => v * enabled));
  mark?.("affordance_vec_palm", st); st = k;
  put(rotateInv(palmQuat, sub3(add3(o, rotate(oq, qNear)), s)));
  mark?.("affordance_nearest_palm", st); st = k;
  let n = [mem.point_n[0] / scale[0], mem.point_n[1] / scale[1], mem.point_n[2] / scale[2]];
  const nn = Math.hypot(n[0], n[1], n[2]);
  n = [n[0] / nn, n[1] / nn, n[2] / nn];
  const m = rotateInv(palmQuat, rotate(oq, n));
  put(m.map((v) => v * enabled));
  mark?.("affordance_normal_palm", st); st = k;
  const ax = c.palmar_axis;
  out[k++] = -(m[0] * ax[0] + m[1] * ax[1] + m[2] * ax[2]) * enabled;
  mark?.("affordance_normal_alignment", st); st = k;
  out[k++] = enabled;
  mark?.("affordance_point_enabled", st); st = k;
  out[k++] = mem.regrasp_mode;
  mark?.("affordance_regrasp_mode", st); st = k;
  // hand_surface_vec: 17 hand points -> nearest scaled surface sample (object frame search), world-axis vectors
  const hand = [cc];
  for (let f = 0; f < 4; f++) { for (let l = 0; l < 3; l++) hand.push(state.link_pos[3 * f + l]); hand.push(pads[f]); }
  const P512 = obj.points512;
  for (const h of hand) {
    const local = rotateInv(oq, sub3(h, o));
    let bj = 0, bd = Infinity;
    for (let j = 0; j < P512.length; j++) {
      const p = P512[j];
      const d = (local[0] - p[0] * scale[0]) ** 2 + (local[1] - p[1] * scale[1]) ** 2 + (local[2] - p[2] * scale[2]) ** 2;
      if (d < bd) { bd = d; bj = j; }
    }
    put(rotate(oq, sub3(mul3(P512[bj], scale), local)));
  }
  mark?.("hand_surface_vec", st); st = k;
  out[k++] = state.table_top_z - c.obs_z_shift + mem.table_offset;
  mark?.("table_height", st);
  if (k !== OBS_DIM) throw new Error(`obs layout: wrote ${k} of ${OBS_DIM}`);
  const clip = c.clamp_abs_observations;
  for (let i = 0; i < OBS_DIM; i++) out[i] = out[i] > clip ? clip : out[i] < -clip ? -clip : out[i];
  return out;
}

/** affordance_staged near(p) under the true pose (obs_ref.near_point). */
export function nearPoint(state, mem, c, airborne) {
  const s = rotate(state.palm_quat, c.palm_surface_centre).map((v, i) => v + state.palm_pos[i]);
  const xp = rotate(state.obj_quat, mem.point_p).map((v, i) => v + state.obj_pos[i]);
  const nw = rotate(state.obj_quat, mem.point_n);
  const np = rotate(state.palm_quat, c.palmar_axis);
  const d = [s[0] - xp[0], s[1] - xp[1], s[2] - xp[2]];
  return { near: !!(airborne && Math.hypot(...d) <= c.near_radius_m && -(np[0] * nw[0] + np[1] * nw[1] + np[2] * nw[2]) >= Math.cos((c.normal_tolerance_deg * Math.PI) / 180)
    && d[0] * nw[0] + d[1] * nw[1] + d[2] * nw[2] >= 0), dist: Math.hypot(...d), cos: -(np[0] * nw[0] + np[1] * nw[1] + np[2] * nw[2]) };
}

// ---- 'real' evaluation (utils/real_eval.py): tracker model + 1-step observation delay --------------------------------------------
/**
 * obs_ref.RealObservation: every step the true pose is pushed into a history (depth 16); at an update step the observed pose =
 * history[lag] (lag = latency_steps - obs_delay_steps = 5) + noise (position N(0, 2 mm) per axis; rotation vector N(0, 1 deg) per
 * axis, left-multiplied), held in between. Update when floor(2k / 3) != floor(2(k - 1) / 3), k = steps since the evaluation start.
 * Episode starts refill the history and read at once. Then the whole observation is delayed by obs_delay_steps (1).
 */
export class RealObservation {
  constructor(c, seed = 1) {
    const r = c.eval_real || DEFAULTS.eval_real;
    this.lag = r.latency_steps - r.obs_delay_steps; this.hold = r.hold_steps; this.factor = 2; this.depth = 16; this.delay = r.obs_delay_steps;
    this.posNoise = r.pos_noise_m; this.rotNoise = (r.rot_noise_deg * Math.PI) / 180;
    this.rand = makeRng(seed);
    this.k = 0; this.history = null; this.observed = null; this.queue = null;
  }

  measure(pose) {
    const r = this.rand;
    const pos = [pose[0] + r.normal() * this.posNoise, pose[1] + r.normal() * this.posNoise, pose[2] + r.normal() * this.posNoise];
    const v = [r.normal() * this.rotNoise, r.normal() * this.rotNoise, r.normal() * this.rotNoise];
    const angle = Math.hypot(...v);
    const axis = angle > 0 ? v.map((x) => x / angle) : [0, 0, 0];
    const dq = [Math.cos(angle / 2), ...axis.map((x) => Math.sin(angle / 2) * x)];
    let quat = quatMul(dq, pose.slice(3, 7));
    const qn = Math.hypot(...quat);
    quat = quat.map((x) => x / qn);
    return [...pos, ...quat];
  }

  tracker(state, episodeStart) {
    const pose = [...state.obj_pos, ...state.obj_quat];
    if (this.history === null) { this.history = Array.from({ length: this.depth }, () => pose.slice()); this.observed = pose.slice(); }
    this.history.pop();
    this.history.unshift(pose.slice());
    if (episodeStart) this.history = Array.from({ length: this.depth }, () => pose.slice());
    const k = this.k, f = this.factor;
    if (Math.floor((k * f) / this.hold) !== Math.floor(((k - 1) * f) / this.hold) || episodeStart) this.observed = this.measure(this.history[this.lag]);
    this.k += 1;
    return [this.observed.slice(0, 3), this.observed.slice(3, 7)];
  }

  /** observedPose: override the tracker (tests: the env's recorded tracker pose). Returns [obs (delayed), the pose used]. */
  step(state, mem, obj, c, episodeStart, observedPose = null) {
    const pose = observedPose || this.tracker(state, episodeStart);
    const obs = observation(state, mem, obj, c, pose);
    if (this.queue === null || episodeStart) this.queue = Array.from({ length: this.delay + 1 }, () => Float32Array.from(obs));
    this.queue.pop();
    this.queue.unshift(Float32Array.from(obs));
    return [this.queue[this.delay], pose];
  }
}
