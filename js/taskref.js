// JS port of agent D's sim2sim/task_ref.py (line by line, same names): r321's per-episode task logic in evaluation (the cal3 'point3'
// pass: stage 3, coupled switch goals) at 30 Hz. Reset (robot noise, table jitter, stable-pose spawn, first 6D goal, first point, FPS
// start), the phase machine (taskstate.js = B's obs_ref.TaskState), the 5-step surface64 success test, deadline re-draw, the stage-3
// switch (final rule + support-gap filter), coupled goal (contact / min_rot + table twist), delta goals, terminations and the
// 599-step inactivity truncation. The random draws use the page's seeded RNG (same distributions as numpy's, not the same numbers).
// User inputs (setPoint / setGoal) as in task_ref; the demo layer (task.js) adds live goals, point mode and the episode-end modes.

import { TaskState } from "./taskstate.js";

// ---- constants (r321 resolved_config.json at control_hz 30; utils sources in task_ref.py) ---------------------------------------
export const C = {
  dwell_steps: 5, kept_grace_steps: 15, drop_debounce_steps: 3, near_radius_m: 0.05, normal_tol_deg: 45.0, airborne_m: 0.01,
  timeout_steps: 300, switch_timeout_steps: 600, switch_prob: 0.3, switch_tries: 16, switch_min_dist_m: 0.045,
  switch_support_gap_max_m: 0.01, coupled_table_margin_m: 0.01,
  lift_threshold_m: 0.10,
  success_steps: 5, surface64_ref_radius: 0.10, max_successes: 50, inactivity_steps: 599, eval_steps: 1800, fall_z: 0.1, hand_far_m: 1.5,
  table_reset_z: 0.38, table_half_height: 0.15, table_reset_z_range: 0.01,
  table_centre_xy: [0.0, -0.05], table_half_xy: [0.55, 0.40],
  spawn_centre_xy: [0.05, 0.0], spawn_half_xy: [0.20, 0.15], spawn_clearance_m: 0.005, table_object_z_offset: 0.25,
  reset_dof_pos_interval: 0.1, reset_dof_vel: 0.5, wrist_noise_range: [-Math.PI, Math.PI],
  table_offset_m: 0.003,
  goal_volume_min: [-0.35, -0.2, 0.53], goal_volume_max: [0.35, 0.2, 0.95],
  first_goal_floor_above_table_z: 0.35, goal_clearance_m: 0.005, goal_tries: 16,
  delta_goal_distance: 0.1, delta_rotation_deg: 90.0,
  palm_surface_centre: [-0.000678656, -0.020678878, 0.096772371], palmar_axis: [0.0, -1.0, 0.0],
};
export const L_PHASE = 0, P1_PHASE = 1, A_PHASE = 2, G_PHASE = 3;
const DEG = Math.PI / 180;

// ---- quaternions (wxyz) ---------------------------------------------------------------------------------------------------------
export function qmul(a, b) {
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw];
}
export const qconj = (q) => [q[0], -q[1], -q[2], -q[3]];
export function qrot(q, v) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}
export const qaxis = (axis, angle) => [Math.cos(angle / 2), ...axis.map((a) => Math.sin(angle / 2) * a)];
export function qnorm(q) { const n = Math.hypot(...q); return q.map((v) => v / n); }
const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const vcross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const vnorm = (a) => Math.hypot(a[0], a[1], a[2]);
const vunit = (a) => { const n = vnorm(a); return [a[0] / n, a[1] / n, a[2] / n]; };

/** affordance_staged.cg_min_rotation: the minimal rotation taking unit u to unit v. */
export function minRotation(u, v, eps = 1e-6) {
  u = vunit(u); v = vunit(v);
  const c = vdot(u, v);
  if (1 + c < eps) {
    const a = u.map(Math.abs), e = [0, 0, 0];
    e[a.indexOf(Math.min(...a))] = 1;
    const axis = vunit(vcross(u, e));
    return [0, axis[0], axis[1], axis[2]];
  }
  return qnorm([1 + c, ...vcross(u, v)]);
}

/** Isaac Lab random_orientation: a normalised 4-D Gaussian (uniform on SO(3)). */
const randomQuat = (rng) => qnorm([rng.normal(), rng.normal(), rng.normal(), rng.normal()]);
const wrapDeg = (x) => 180 - ((((180 - x) % 360) + 360) % 360);
const uniform = (rng, lo, hi) => lo + (hi - lo) * rng();

// ---- object data ------------------------------------------------------------------------------------------------------------------
/** policy/objects/<name>.json (vertices, faces, face_normals, area_cdf, support_hull, points512) + sim2sim/task_data/<name>.json
 * (points64, centroid, mean_radius, stable_quats / stable_probs, mesh_hull, face_gap), merged by tools/export_object_data.py. */
export function prepareTaskObject(d) {
  const tri = d.faces.map((f) => [d.vertices[f[0]], d.vertices[f[1]], d.vertices[f[2]]]);
  const rr = Math.max(...d.support_hull.map((v) => vnorm(v)));
  return Object.assign({}, d, { triangles: tri, rotation_radius: rr });
}

export function lowestZ(hull, pos, quat) {
  let lo = Infinity;
  for (const v of hull) lo = Math.min(lo, qrot(quat, v)[2]);
  return pos[2] + lo;
}

/** reward.goal_metric surface64 with surface64_ref_radius (mesh_geometry.centred_point_error). */
export function goalMetric(obj, pos, quat, gpos, gquat, refRadius = C.surface64_ref_radius) {
  const q = qnorm(quat), qg = qnorm(gquat), c = obj.centroid;
  const shift = vnorm(vsub(vadd(pos, qrot(q, c)), vadd(gpos, qrot(qg, c))));
  let s = 0;
  for (const x of obj.points64) { const l = vsub(x, c); s += vnorm(vsub(qrot(q, l), qrot(qg, l))); }
  return shift + (refRadius / obj.mean_radius) * (s / obj.points64.length);
}

/** (point, oriented normal, face) area-uniform on the mesh (affordance_staged.sample_with_faces). */
export function sampleSurfacePoint(obj, rng) {
  const u = [rng(), rng(), rng()];
  const x = Math.min(Math.max(u[0], 1e-12), 1 - 1e-12);
  let lo = 0, hi = obj.area_cdf.length; // np.searchsorted(cdf, x) (side left)
  while (lo < hi) { const mid = (lo + hi) >> 1; if (obj.area_cdf[mid] < x) lo = mid + 1; else hi = mid; }
  const face = Math.min(Math.max(lo, 0), obj.faces.length - 1);
  const [a, b, c] = obj.triangles[face];
  const r1 = Math.sqrt(u[1]), r2 = u[2];
  const p = [0, 1, 2].map((k) => (1 - r1) * a[k] + r1 * (1 - r2) * b[k] + r1 * r2 * c[k]);
  return [p, obj.face_normals[face].slice(), face];
}

/** affordance_staged.point_terms in the object frame: [|s - p|, facing, near geometry]. */
export function pointTerms(localS, npalmLocal, p, n, radius = C.near_radius_m, tolDeg = C.normal_tol_deg) {
  const delta = vsub(localS, p);
  const dist = vnorm(delta);
  const facing = -vdot(npalmLocal, n) >= Math.cos(tolDeg * DEG) && vdot(delta, n) >= 0;
  return [dist, facing, facing && dist <= radius];
}

// ---- the task ---------------------------------------------------------------------------------------------------------------------
export class TaskRef {
  /** obj: prepareTaskObject(...); rng: mathq.rng (rng() uniform, rng.normal()); pc: obs.loadConstants() (policy constants). */
  constructor(obj, rng, pc, { tol = 0.05, constOverrides = null, coupled = true, switchProb = null } = {}) {
    this.obj = obj; this.rng = rng; this.tol = tol;
    this.c = Object.assign({}, C, constOverrides || {});
    this.pc = pc;
    this.coupled = coupled;
    this.switchProb = switchProb == null ? this.c.switch_prob : switchProb;
  }

  // -- reset ------------------------------------------------------------------------------------------------------------------------
  /** A new episode: {q, qd, obj_pos, obj_quat, table_top, goal_pos, goal_quat, point_p, point_n, fps_start, table_offset}. matched:
   * keys taken as given (the page: the user's goal / point kept over a reset). */
  reset(defaultQ, qLower, qUpper, matched = {}) {
    const c = this.c, rng = this.rng, obj = this.obj, m = matched || {};
    const out = {};
    if (defaultQ) [out.q, out.qd] = this.robotReset(defaultQ, qLower, qUpper);
    this.table_top = m.table_top != null ? m.table_top : c.table_reset_z + uniform(rng, -c.table_reset_z_range, c.table_reset_z_range) + c.table_half_height;
    this.table_z = this.table_top - c.table_half_height;
    if (m.obj_pos) { out.obj_pos = m.obj_pos.slice(); out.obj_quat = m.obj_quat.slice(); } else [out.obj_pos, out.obj_quat] = this.spawn();
    if (m.goal_pos) { this.goal_pos = m.goal_pos.slice(); this.goal_quat = m.goal_quat.slice(); } else [this.goal_pos, this.goal_quat] = this.firstGoal();
    if (m.point_p) { this.point_p = m.point_p.slice(); this.point_n = m.point_n.slice(); } else [this.point_p, this.point_n] = sampleSurfacePoint(obj, rng);
    out.fps_start = m.fps_start != null ? m.fps_start : Math.floor(rng() * obj.points512.length);
    out.table_offset = m.table_offset != null ? m.table_offset : (rng() * 2 - 1) * c.table_offset_m;
    Object.assign(out, { table_top: this.table_top, goal_pos: this.goal_pos.slice(), goal_quat: this.goal_quat.slice(),
      point_p: this.point_p.slice(), point_n: this.point_n.slice() });
    this._episodeState();
    return out;
  }

  /** affordance_staged.reset_episode + reset_env_state: the per-episode state (the goal, point and table stay). */
  _episodeState() {
    this.ts = new TaskState(this.pc);
    this.mem = { point_p: this.point_p.slice(), point_n: this.point_n.slice(), regrasp_mode: 0 };
    this.latch_age = -1; this.clock = 0; this.steps = 0;
    this.wait = 0; this.command = 0; this.bonus_paid = false; this.switched = false;
    this.cmd_dropped = false; this.sw_drop = false; this.cmd_active = false;
    this.near_goal_steps = 0; this.successes = 0;
    this.pending_goal = null;
    this.done = false; this.reason = "";
    this.counts = Object.fromEntries(["acquisitions", "reacquisitions", "deadlines", "switches", "inhand_reacq", "drops", "relifts", "losses",
      "g_entries", "setdown_acq", "g_steps", "a_steps", "coupled", "coupled_fallbacks"].map((k) => [k, 0]));
  }

  /** reset_utils._randomize_robot_dof_state: q = lerp(default, U(noise lo, hi), 0.1) clamped (joint4 / joint6 noise at +-pi), qd = U(+-0.5). */
  robotReset(defaultQ, lower, upper) {
    const c = this.c, rng = this.rng;
    const lo = lower.slice(), hi = upper.slice();
    [lo[3], hi[3]] = c.wrist_noise_range; [lo[5], hi[5]] = c.wrist_noise_range;
    const q = defaultQ.map((d, j) => { const s = lo[j] + (hi[j] - lo[j]) * rng(); return Math.min(upper[j], Math.max(lower[j], d + c.reset_dof_pos_interval * (s - d))); });
    const qd = defaultQ.map(() => uniform(rng, -c.reset_dof_vel, c.reset_dof_vel));
    return [q, qd];
  }

  /** mesh_geometry.spawn: a stable pose (by probability) x uniform yaw, xy in the spawn box kept on the table footprint, 5 mm above it. */
  spawn() {
    const c = this.c, rng = this.rng, obj = this.obj;
    const noise = [uniform(rng, -1, 1), uniform(rng, -1, 1), uniform(rng, -1, 1)];
    const sc = c.spawn_centre_xy, sh = c.spawn_half_xy;
    const pxy = [noise[0] * sh[0] + sc[0], noise[1] * sh[1] + sc[1]];
    const u = rng(), yaw = rng() * 2 * Math.PI;
    const cdf = []; let acc = 0;
    for (const p of obj.stable_probs) { acc += p; cdf.push(acc); }
    cdf[cdf.length - 1] = 1;
    const x = Math.min(Math.max(u, 1e-7), 1 - 1e-7);
    let k = cdf.findIndex((v) => v >= x); if (k < 0) k = cdf.length - 1;
    const q = qmul([Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], obj.stable_quats[k]);
    const tc = c.table_centre_xy, th = c.table_half_xy.map((v) => v - 0.005);
    const unit = [0, 1].map((i) => (pxy[i] - sc[i] + sh[i]) / (2 * sh[i]));
    const xy = obj.support_hull.map((v) => qrot(q, v));
    const minx = Math.min(...xy.map((v) => v[0])), miny = Math.min(...xy.map((v) => v[1]));
    const maxx = Math.max(...xy.map((v) => v[0])), maxy = Math.max(...xy.map((v) => v[1]));
    let lower = [Math.max(tc[0] - th[0] - minx, sc[0] - sh[0]), Math.max(tc[1] - th[1] - miny, sc[1] - sh[1])];
    let upper = [Math.min(tc[0] + th[0] - maxx, sc[0] + sh[0]), Math.min(tc[1] + th[1] - maxy, sc[1] + sh[1])];
    const room = th.map((v) => v - obj.rotation_radius);
    if (room[0] >= 0 && room[1] >= 0) {
      lower = [Math.max(lower[0], tc[0] - room[0]), Math.max(lower[1], tc[1] - room[1])];
      upper = [Math.min(upper[0], tc[0] + room[0]), Math.min(upper[1], tc[1] + room[1])];
    }
    const p = [0, 1].map((i) => lower[i] + (upper[i] - lower[i]) * Math.min(1, Math.max(0, unit[i])));
    const z = this.table_top + c.spawn_clearance_m - Math.min(...xy.map((v) => v[2]));
    return [[p[0], p[1], z], q];
  }

  /** goal_clearance.pick_clear_goals: the first candidate clearing the table + 5 mm, else candidate 0 raised to it. */
  _pickClear(cands) {
    const need = this.table_top + this.c.goal_clearance_m;
    const lows = cands.map(([p, q]) => lowestZ(this.obj.support_hull, p, q));
    for (let i = 0; i < cands.length; i++) if (lows[i] >= need) return cands[i];
    const [p, q] = cands[0];
    return [[p[0], p[1], p[2] + need - lows[0]], q];
  }

  /** The evaluation's first goal: 16 candidates uniform in the goal volume, uniform orientation, root z >= table centre + 0.35. */
  firstGoal() {
    const c = this.c, rng = this.rng, lo = c.goal_volume_min, hi = c.goal_volume_max;
    const cands = [];
    for (let i = 0; i < c.goal_tries; i++) {
      const p = [0, 1, 2].map((k) => lo[k] + (hi[k] - lo[k]) * rng());
      p[2] = Math.max(p[2], this.table_z + c.first_goal_floor_above_table_z);
      cands.push([p, randomQuat(rng)]);
    }
    return this._pickClear(cands);
  }

  /** The evaluation's delta goal around the goal just hit: +-10 cm per axis (clamped to the volume), q * Rot(positive-octant axis,
   * U(+-90 deg)), w >= 0; the clearance pick. */
  deltaGoal() {
    const c = this.c, rng = this.rng, lo = c.goal_volume_min, hi = c.goal_volume_max;
    const cands = [];
    for (let i = 0; i < c.goal_tries; i++) {
      const p = [0, 1, 2].map((k) => Math.min(hi[k], Math.max(lo[k], this.goal_pos[k] + (rng() * 2 - 1) * c.delta_goal_distance)));
      const axis = vunit([rng(), rng(), rng()]);
      const angle = (2 * rng() - 1) * c.delta_rotation_deg * DEG;
      const q = qmul(this.goal_quat, qaxis(axis, angle));
      cands.push([p, q[0] < 0 ? q.map((v) => -v) : q]);
    }
    return this._pickClear(cands);
  }

  // -- geometry -----------------------------------------------------------------------------------------------------------------------
  geometry(state) {
    const o = state.obj_pos, oq = state.obj_quat, pq = state.palm_quat;
    const s = vadd(state.palm_pos, qrot(pq, this.c.palm_surface_centre));
    const inv = qconj(oq);
    return [o, oq, qrot(inv, vsub(s, o)), qrot(inv, qrot(pq, this.c.palmar_axis))];
  }

  get phase() { return this.ts.phase; }
  get regrasp() { return this.ts.regrasp; }

  // -- one control step ---------------------------------------------------------------------------------------------------------------
  step(state) {
    const c = this.c;
    const ev = { success: false, acquire: false, first: false, drop: false, relift: false, lost: false, deadline: false, switch: false, done: false, reason: "" };
    if (this.done) { ev.done = true; ev.reason = this.reason; return ev; }
    this.steps += 1;
    this.clock += 1;
    const [o, oq, localS, npalmLocal] = this.geometry(state);
    const d6 = goalMetric(this.obj, o, oq, this.goal_pos, this.goal_quat);
    const ts = this.ts;
    const phaseBefore = ts.phase, droppedBefore = ts.dropped, regraspBefore = ts.regrasp;
    const f = ts.update(Object.assign({}, state, { table_top_z: this.table_top }), this.mem, this.obj);
    const { held, airborne: air, near, acquire, drop, lost } = f;
    const relift = held && droppedBefore;
    if (phaseBefore === L_PHASE && ts.phase !== L_PHASE) this.latch_age = this.steps;
    this.cmd_dropped = this.cmd_dropped || drop;
    this.sw_drop = this.sw_drop || drop;
    const switchDrop = drop && regraspBefore; // sd300: a drop inside a switch spell ends the episode
    const spellAcq = acquire && regraspBefore;
    const first = acquire && !this.bonus_paid;
    this.bonus_paid = this.bonus_paid || first;
    if (acquire) this.wait = 0;
    const inA = ts.phase === A_PHASE && held;
    this.wait += inA ? 1 : 0;
    this.cmd_active = this.cmd_active || inA || acquire;
    const nearGoal = d6 <= this.tol && held && ts.phase === G_PHASE;
    this.near_goal_steps = nearGoal ? this.near_goal_steps + 1 : 0;
    const success = this.near_goal_steps >= c.success_steps && ts.phase === G_PHASE;
    this.successes += success ? 1 : 0;
    if (success || first) this.clock = 0;
    let handFar = false;
    if (state.pads) {
      const pts = this.obj.points64.map((p) => vadd(o, qrot(oq, p)));
      handFar = Math.max(...state.pads.map((pad) => Math.min(...pts.map((p) => vnorm(vsub(pad, p)))))) > c.hand_far_m;
    }
    let reasons = [["fall", o[2] < c.fall_z], ["max_successes", this.successes >= c.max_successes], ["hand_far", handFar],
      ["switch_drop", switchDrop], ["timeout", this.clock >= c.inactivity_steps]].filter(([, flag]) => flag).map(([n]) => n);
    let done = reasons.length > 0;
    const joint = success && ts.phase === G_PHASE;
    const k = this.counts;
    k.acquisitions += first ? 1 : 0;
    k.reacquisitions += acquire && !first ? 1 : 0;
    k.inhand_reacq += first && this.switched && !this.sw_drop ? 1 : 0;
    k.setdown_acq += first && this.cmd_dropped ? 1 : 0;
    k.drops += drop ? 1 : 0;
    k.relifts += relift ? 1 : 0;
    k.losses += lost ? 1 : 0;
    k.g_entries += acquire ? 1 : 0;
    k.g_steps += ts.phase === G_PHASE ? 1 : 0;
    k.a_steps += inA ? 1 : 0;
    const limit = this.command === 0 ? c.timeout_steps : c.switch_timeout_steps;
    const deadline = !done && ts.phase === A_PHASE && !this.bonus_paid && this.wait >= limit;
    const sw = !done && joint && !deadline && this.rng() < this.switchProb;
    const oldP = this.point_p.slice(), oldN = this.point_n.slice();
    if (deadline) [this.point_p, this.point_n] = sampleSurfacePoint(this.obj, this.rng);
    if (sw) [this.point_p, this.point_n] = this.switchPoint(oldP, localS, npalmLocal, air);
    if (deadline || sw) {
      ts.newPoint(this.mem, this.point_p, this.point_n, sw); // switch: regrasp_mode 1, phase A
      this.command += 1;
      this.bonus_paid = false;
      this.wait = 0;
      this.cmd_active = this.cmd_dropped = false;
      this.switched = sw;
    }
    if (sw) {
      this.sw_drop = false;
      if (this.coupled) this.pending_goal = this.coupledGoal(o, oq, oldP, oldN, this.point_p, this.point_n);
    }
    k.deadlines += deadline ? 1 : 0;
    k.switches += sw ? 1 : 0;
    if (success) { // the next 6D goal of a success
      this.near_goal_steps = 0;
      if (this.pending_goal) { [this.goal_pos, this.goal_quat] = this.pending_goal; k.coupled += 1; }
      else {
        if (sw && this.coupled) k.coupled_fallbacks += 1;
        [this.goal_pos, this.goal_quat] = this.deltaGoal();
      }
      this.pending_goal = null;
    }
    if (this.steps >= c.eval_steps && !done) { reasons = ["cap"]; done = true; }
    this.done = done; this.reason = reasons.join("+");
    Object.assign(ev, { success, acquire, first, drop, relift, lost, deadline, switch: sw, spell_acq: spellAcq, held, airborne: air, near, d6,
      done, reason: this.reason, phase: ts.phase, dist: f.dist });
    return ev;
  }

  // -- switch point and coupled goal --------------------------------------------------------------------------------------------------
  /** affordance_staged._switch_candidates, evaluation (final rule). */
  switchPoint(oldP, localS, npalmLocal, airborne) {
    const c = this.c;
    const cands = Array.from({ length: c.switch_tries }, () => sampleSurfacePoint(this.obj, this.rng));
    const near = cands.map(([p, n]) => pointTerms(localS, npalmLocal, p, n)[2] && !!airborne);
    const far = cands.map(([p]) => vnorm(vsub(p, oldP)));
    const bad = cands.map(([, , f]) => this.obj.face_gap[f] > c.switch_support_gap_max_m);
    const allBad = bad.every(Boolean);
    const usable = bad.map((b) => !b || allBad);
    const failing = near.map((nr, i) => !nr && usable[i]);
    const good = failing.map((fl, i) => fl && far[i] >= c.switch_min_dist_m);
    let i = good.indexOf(true);
    if (i < 0 && failing.some(Boolean)) i = argmax(far.map((d, j) => (failing[j] ? d : -1)));
    if (i < 0) i = argmax(far.map((d, j) => (usable[j] ? d : -1)));
    return [cands[i][0].slice(), cands[i][1].slice()];
  }

  /** affordance_staged.coupled_goal_poses, contact mode, min_rot twist (+ antiparallel axis search, table twist); null if no twist
   * clears the table (the caller then draws the independent delta goal). */
  coupledGoal(o, oq, pOld, nOld, pNew, nNew) {
    const c = this.c;
    const un = vunit(nNew), uo = vunit(nOld);
    const angle = Math.acos(Math.min(1, Math.max(-1, vdot(un, uo)))) / DEG;
    const base = qnorm(qmul(oq, minRotation(nNew, nOld)));
    const m = vunit(qrot(oq, uo));
    const w = vadd(qrot(oq, pOld), o);
    const pose = (phiDeg) => { const q = qnorm(qmul(qaxis(m, phiDeg * DEG), base)); return [vsub(w, qrot(q, pNew)), q]; };
    let phi = 0;
    if (angle > 179) {
      let best = null;
      for (let b = 0; b < 18; b++) {
        const cand = wrapDeg(2 * 10 * b);
        const [pos, q] = pose(cand);
        const score = goalMetric(this.obj, o, oq, pos, q);
        if (best === null || score < best[0] - 1e-6) best = [score, cand];
      }
      phi = best[1];
    }
    const [pos, q] = pose(phi);
    const floor = this.table_top + c.coupled_table_margin_m;
    if (lowestZ(this.obj.mesh_hull, pos, q) >= floor) return [pos, q];
    const psis = [];
    for (let kk = 1; kk < 12; kk++) psis.push(kk * 15, -kk * 15);
    psis.push(180);
    for (const psi of psis) {
      const rot = qaxis(m, psi * DEG);
      const qt = qnorm(qmul(rot, q));
      const pt = vadd(w, qrot(rot, vsub(pos, w)));
      if (lowestZ(this.obj.mesh_hull, pt, qt) >= floor) return [pt, qt];
    }
    return null;
  }

  // -- user inputs (the page) ---------------------------------------------------------------------------------------------------------
  /** A user-chosen point and its oriented face normal: in hand (phase G) a stage-3 switch (regrasp_mode 1, phase A), else a plain new
   * command. Returns true for an in-hand switch. */
  setPoint(p, n) {
    const inHand = this.ts.phase === G_PHASE;
    this.point_p = p.slice(); this.point_n = n.slice();
    this.ts.newPoint(this.mem, this.point_p, this.point_n, inHand);
    this.command += 1;
    this.bonus_paid = false;
    this.wait = 0;
    this.cmd_active = this.cmd_dropped = false;
    this.switched = inHand;
    if (inHand) this.sw_drop = false;
    return inHand;
  }

  /** A user-chosen 6D goal: replaces the current goal; the success dwell restarts. */
  setGoal(pos, quat) {
    this.goal_pos = pos.slice(); this.goal_quat = qnorm(quat);
    this.near_goal_steps = 0;
    this.pending_goal = null;
  }

  /** The demo's memory refresh: the episode state restarts as at a reset, the goal, point and table stay. */
  refresh() { this._episodeState(); }

  record() {
    const k = this.counts;
    return { lifted: this.latch_age >= 0 ? 1 : 0, acquired: k.acquisitions > 0 ? 1 : 0, goals: this.successes, drops: k.drops,
      deadlines: k.deadlines, switches: k.switches, inhand_reacq: k.inhand_reacq, losses: k.losses, g_steps: k.g_steps, a_steps: k.a_steps,
      relifts: k.relifts, setdown_acq: k.setdown_acq, coupled: k.coupled, coupled_fallbacks: k.coupled_fallbacks, steps: this.steps,
      reason: this.reason };
  }
}

function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }
