// Small vector / quaternion helpers. Quaternions are wxyz (Isaac Lab, MuJoCo); vectors are plain arrays or typed arrays.
// Mirrors posedex/sim_demo/demo_geometry.py so the UI geometry matches the deploy / Isaac demo.

export const DEG = Math.PI / 180;

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);
export const normalize = (a) => { const n = Math.max(norm(a), 1e-12); return [a[0] / n, a[1] / n, a[2] / n]; };
export const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);

export function qnorm(q) {
  const n = Math.max(Math.hypot(q[0], q[1], q[2], q[3]), 1e-12);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export function qmul(a, b) {
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw];
}

export const qconj = (q) => [q[0], -q[1], -q[2], -q[3]];

/** v rotated by q: R(q) v. */
export function qrot(q, v) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}

/** R(q)^T v. */
export const qrotInv = (q, v) => qrot(qconj(q), v);

export function qaxis(axis, angle) {
  const a = normalize(axis), s = Math.sin(angle / 2);
  return [Math.cos(angle / 2), a[0] * s, a[1] * s, a[2] * s];
}

/** Geodesic angle (rad) between two orientations. */
export function qangle(a, b) {
  const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]) /
    Math.max(Math.hypot(...a) * Math.hypot(...b), 1e-12);
  return 2 * Math.acos(Math.min(1, d));
}

/** Row-major 3x3 rotation matrix of q. */
export function qmatrix(q) {
  const [w, x, y, z] = qnorm(q);
  return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
}

/** Quaternion (wxyz) of a row-major 3x3 rotation matrix. */
export function qfromMat(m) {
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m;
  const tr = m00 + m11 + m22;
  let q;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s];
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    q = [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s];
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    q = [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s];
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s];
  }
  return qnorm(q);
}

/** Extrinsic x-y-z (roll, pitch, yaw) in degrees: R = Rz(yaw) Ry(pitch) Rx(roll) (demo_geometry.rpy_deg). */
export function rpyDeg(q) {
  const m = qmatrix(q);
  const pitch = Math.asin(clamp(-m[6], -1, 1));
  const roll = Math.atan2(m[7], m[8]);
  const yaw = Math.atan2(m[3], m[0]);
  return [roll / DEG, pitch / DEG, yaw / DEG];
}

export function quatFromRpyDeg(rpy) {
  const [r, p, y] = rpy.map((v) => v * DEG);
  return qnorm(qmul(qaxis([0, 0, 1], y), qmul(qaxis([0, 1, 0], p), qaxis([1, 0, 0], r))));
}

export function qslerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb = b;
  if (d < 0) { d = -d; bb = [-b[0], -b[1], -b[2], -b[3]]; }
  if (d > 0.9995) return qnorm(a.map((v, i) => v + t * (bb[i] - v)));
  const th = Math.acos(d), s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return qnorm(a.map((v, i) => wa * v + wb * bb[i]));
}

/** Rotation vector (axis * angle) -> quaternion. */
export function qfromRotvec(r) {
  const th = norm(r);
  if (th < 1e-12) return [1, 0, 0, 0];
  return qaxis(r, th);
}

/** Small seeded RNG (mulberry32) so a session is reproducible from its seed. */
export function rng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.normal = () => {
    let u = 0, v = 0;
    while (u === 0) u = next();
    while (v === 0) v = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return next;
}
