// Goal geometry of the page's UI, in the env frame (= MuJoCo world; z up, metres): ports of posedex/sim_demo/demo_geometry.py
// (preview check, random draft goal, nudges), the live-goal clamp, the point-frame helper. The task's own geometry (goal metric,
// coupled goal, switch points) is taskref.js (agent D's task_ref.py).

import { DEG, add, sub, norm, qmul, qrot, qrotInv, qaxis, qangle, qnorm } from "./mathq.js";

// Nudge actions: world axes through the pivot (the object's centroid). (kind, axis, sign)
export const NUDGES = {
  "x+": ["t", 0, 1], "x-": ["t", 0, -1], "y+": ["t", 1, 1], "y-": ["t", 1, -1], "z+": ["t", 2, 1], "z-": ["t", 2, -1],
  "yaw+": ["r", 2, 1], "yaw-": ["r", 2, -1], "pitch+": ["r", 1, 1], "pitch-": ["r", 1, -1], "roll+": ["r", 0, 1], "roll-": ["r", 0, -1],
};

/** Root pose (p, q) after one nudge applied about the pivot (the centroid in world). */
export function nudge(pos, q, centroidLocal, action, stepM, stepDeg) {
  const [kind, axis, sign] = NUDGES[action];
  const pivot = add(pos, qrot(q, centroidLocal));
  if (kind === "t") {
    const p = pos.slice();
    p[axis] += sign * stepM;
    return [p, q];
  }
  const e = [0, 0, 0];
  e[axis] = 1;
  const q2 = qnorm(qmul(qaxis(e, sign * stepDeg * DEG), q));
  return [sub(pivot, qrot(q2, centroidLocal)), q2];
}

/** Lowest world z of object-frame points at the pose (pos, q). */
export function lowestZ(points, pos, q) {
  const [w, x, y, z] = q;
  const r0 = 2 * (x * z - w * y), r1 = 2 * (y * z + w * x), r2 = 1 - 2 * (x * x + y * y);
  let lo = Infinity;
  for (const v of points) lo = Math.min(lo, r0 * v[0] + r1 * v[1] + r2 * v[2]);
  return lo + pos[2];
}

/**
 * Preview check of a goal (object root pose, env frame) before it is sent (demo_geometry.goal_check).
 * HARD (refused): the lowest support-hull point below the table top + clearance, or the root outside the goal box.
 * SOFT (allowed after confirmation): a position step larger than the training delta on one axis, or a rotation larger than the
 * training rotation cap, from the active goal (refPos / refQuat).
 */
export function goalCheck(pos, q, ctx, refPos = null, refQuat = null) {
  const hard = [], soft = [];
  const low = lowestZ(ctx.hull, pos, q);
  const clear = ctx.clearance_m ?? 0.005;
  if (low < ctx.surface_z + clear - 1e-6) {
    hard.push(`lowest point ${(100 * (low - ctx.surface_z)).toFixed(1)} cm vs table + ${(100 * clear).toFixed(1)} cm`);
  }
  const lo = ctx.goal_box.mins, hi = ctx.goal_box.maxs, tol = 1e-3;
  const out = [];
  for (let k = 0; k < 3; k++) {
    if (!(lo[k] - tol <= pos[k] && pos[k] <= hi[k] + tol)) {
      out.push(`${"xyz"[k]} ${(100 * pos[k]).toFixed(1)} cm not in [${(100 * lo[k]).toFixed(0)}, ${(100 * hi[k]).toFixed(0)}]`);
    }
  }
  if (out.length) hard.push("outside the goal box: " + out.join("; "));
  let rotRef = null;
  if (refQuat) {
    rotRef = qangle(refQuat, q) / DEG;
    const cap = (ctx.rot_cap_rad ?? Math.PI) / DEG;
    if (rotRef > cap + 1e-6) soft.push(`rotation ${rotRef.toFixed(0)} deg from the active goal > training cap ${cap.toFixed(0)} deg`);
  }
  if (refPos) {
    const step = Math.max(...[0, 1, 2].map((k) => Math.abs(pos[k] - refPos[k])));
    const delta = ctx.delta_m ?? 0.1;
    if (step > delta + 1e-6) soft.push(`position step ${(100 * step).toFixed(1)} cm on one axis > training delta ${(100 * delta).toFixed(0)} cm`);
  }
  return { hard, soft, ok: hard.length === 0, clean: hard.length === 0 && soft.length === 0, lowestAboveSurface: low - ctx.surface_z, rotFromRefDeg: rotRef };
}

/**
 * Live-goal clamp: the nearest valid pose to a requested goal (orientation kept): the root clamped into the goal box per axis, then
 * raised until the lowest support-hull point clears the table + clearance. Returns {pos, quat, clamped, notes}.
 */
export function clampGoal(pos, q, ctx) {
  const lo = ctx.goal_box.mins, hi = ctx.goal_box.maxs;
  const notes = [];
  const p = pos.slice();
  for (let k = 0; k < 3; k++) {
    if (p[k] < lo[k] || p[k] > hi[k]) {
      notes.push(`${"xyz"[k]} ${(100 * p[k]).toFixed(1)} -> ${(100 * Math.min(hi[k], Math.max(lo[k], p[k]))).toFixed(1)} cm (goal box)`);
      p[k] = Math.min(hi[k], Math.max(lo[k], p[k]));
    }
  }
  const floor = ctx.surface_z + (ctx.clearance_m ?? 0.005);
  const low = lowestZ(ctx.hull, p, q);
  if (low < floor - 1e-9) {
    notes.push(`raised ${(100 * (floor - low)).toFixed(1)} cm to clear the table`);
    p[2] += floor - low;
  }
  return { pos: p, quat: q, clamped: notes.length > 0, notes };
}

/** A random draft goal passing the HARD check: root uniform in the goal box, the reference orientation rotated by a random axis
 * and an angle <= the rotation cap, raised to clear the table (demo_geometry.random_goal). */
export function randomGoal(rand, ctx, refQuat, tries = 64) {
  const lo = ctx.goal_box.mins, hi = ctx.goal_box.maxs;
  const cap = ctx.rot_cap_rad ?? Math.PI;
  const floor = ctx.surface_z + (ctx.clearance_m ?? 0.005) + 0.005;
  let best = null;
  for (let i = 0; i < tries; i++) {
    let p = [0, 1, 2].map((k) => lo[k] + rand() * (hi[k] - lo[k]));
    const axis = [rand.normal(), rand.normal(), rand.normal()];
    const q = qnorm(qmul(qaxis(axis, rand() * cap), refQuat));
    const low = lowestZ(ctx.hull, p, q);
    if (low < floor) p = [p[0], p[1], p[2] + floor - low];
    best = [p, q];
    if (goalCheck(p, q, ctx).ok) return best;
  }
  return best;
}

/** Index of the point nearest to x. */
export function nearestIndex(points, x) {
  let best = -1, bd = Infinity;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const d = (p[0] - x[0]) ** 2 + (p[1] - x[1]) ** 2 + (p[2] - x[2]) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

/** Centroid and mean radius of a point table (mesh_geometry.point_frame). */
export function pointFrame(points) {
  const c = [0, 0, 0];
  for (const p of points) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
  c[0] /= points.length; c[1] /= points.length; c[2] /= points.length;
  let r = 0;
  for (const p of points) r += norm(sub(p, c));
  return { centroid: c, meanRadius: r / points.length };
}

export { qrotInv };
