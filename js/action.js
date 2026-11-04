// JS port of agent B's policy/action_ref.py (line by line): policy output -> joint position targets (utils/action_utils at control_hz
// 30, evaluation mode); STATE_SCHEMA.md section 7. Checked against the Isaac golden dumps by test/obs_golden.test.mjs.
//   arm  (0:6):  raw = clamp(prev + dof_speed_scale * step_dt * a, lo, hi);  t = clamp(alpha * raw + (1 - alpha) * prev, lo, hi)
//   hand (6:22): raw = lo + 0.5 (a + 1) (hi - lo);                            t = clamp(beta * raw + (1 - beta) * prev, lo, hi)
// a = clamp(mu, -1, 1); lo / hi = constants.json joint_lower / joint_upper; 'real': the action applied one control step late.

export const NUM_ARM = 6;

export function actionToTargets(mu, prevTargets, c, clampAction = true, out = new Float64Array(prevTargets.length)) {
  const lo = c.joint_lower, hi = c.joint_upper;
  const cl = (x, j) => (x < lo[j] ? lo[j] : x > hi[j] ? hi[j] : x);
  const alpha = c.arm_moving_average ?? 0.1;
  const speed = (c.dof_speed_scale ?? 1.5) * (c.step_dt ?? 1 / 30);
  const beta = c.hand_moving_average ?? 0.19; // constants.json holds the converted (30 Hz) value
  for (let j = 0; j < prevTargets.length; j++) {
    let a = mu[j];
    if (clampAction) a = a < -1 ? -1 : a > 1 ? 1 : a;
    const prev = prevTargets[j];
    if (j < NUM_ARM) {
      const raw = cl(prev + speed * a, j);
      out[j] = cl(alpha * raw + (1 - alpha) * prev, j);
    } else {
      const raw = lo[j] + 0.5 * (a + 1) * (hi[j] - lo[j]);
      out[j] = cl(beta * raw + (1 - beta) * prev, j);
    }
  }
  return out;
}

/** utils/real_eval.delay_actions: a fixed n-step delay of the (clamped) action; an episode start fills the queue (applies at once). */
export class RealActionDelay {
  constructor(steps = 1) { this.steps = steps; this.queue = null; }
  step(action, episodeStart) {
    const a = Array.from(action, (v) => (v < -1 ? -1 : v > 1 ? 1 : v));
    if (this.queue === null || episodeStart) this.queue = Array.from({ length: this.steps + 1 }, () => a.slice());
    this.queue.pop();
    this.queue.unshift(a);
    return this.queue[this.steps];
  }
}
