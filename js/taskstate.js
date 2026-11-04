// Port of agent B's obs_ref.TaskState + bottom_height (line by line): the cal3 stage-3 episode flags behind regrasp_mode
// (affordance_staged.gate_near_goal + the lift latch of compute_rewards). Per control step, after the physics and before
// observation(): update(state, mem, obj, c) (writes mem.regrasp_mode); a new point: newPoint(mem, p, n, inHand) (inHand: a switch
// while the object is held = regrasp_mode 1 until the next acquisition; otherwise a plain new command, e.g. before the lift);
// reset() at an episode reset. Phases: 0 L (until the lift latch), 2 A (point not acquired), 3 G (acquired; 6D goal open).
// The 6D goal success logic is not here (task.js; TODO(D): sim2sim/task_ref.py).
// Checked against the Isaac golden dumps by test/obs_golden.test.mjs (phase / streak / regrasp / latch replay).

import { rotate, nearPoint } from "./obs.js";

/** lift_bottom.bottom_height: the lowest support-hull point (convex hull of the CoACD collision parts) above the table top (m). */
export function bottomHeight(obj, objPos, objQuat, tableTopZ) {
  let lo = Infinity;
  for (const v of obj.support_hull) lo = Math.min(lo, rotate(objQuat, v)[2]);
  return objPos[2] + lo - tableTopZ;
}

export class TaskState {
  static L = 0;
  static A = 2;
  static G = 3;

  constructor(c) {
    this.dwell = c.dwell_steps ?? 5; // acquisition streak (10 at 60 Hz -> 5)
    this.grace = c.kept_grace_steps ?? 15; // point kept = near within this many steps
    this.airM = c.airborne_m ?? 0.01;
    this.c = c;
    this.liftM = 0.10; // reward.lift_bottom_threshold_m (liftB)
    this.debounce = 3; // drop_debounce_steps (6 at 60 Hz -> 3)
    this.reset();
  }

  reset() {
    this.latch = false; this.armed = false; this.phase = TaskState.L; this.streak = 0; this.misses = this.grace + 1;
    this.regrasp = false; this.notheld = 0; this.dropped = false;
  }

  update(state, mem, obj) {
    const L = TaskState.L, A = TaskState.A, G = TaskState.G;
    const h = bottomHeight(obj, state.obj_pos, state.obj_quat, state.table_top_z);
    const air = h > this.airM;
    const held = this.latch && air; // the latch as of the previous step (env order)
    if (this.phase === L && this.latch) this.phase = A;
    this.notheld = this.latch && !held ? this.notheld + 1 : 0;
    const drop = this.latch && !this.dropped && this.notheld >= this.debounce;
    const relift = held && this.dropped;
    this.dropped = (this.dropped || drop) && !relift;
    const np = nearPoint(state, mem, this.c, air);
    const near = (this.phase === A || this.phase === G) && held && np.near;
    this.streak = near ? this.streak + 1 : 0;
    this.misses = near ? 0 : this.misses + 1;
    if (drop && this.phase === G) this.phase = A;
    const acquire = this.phase === A && this.streak >= this.dwell;
    if (acquire) { this.phase = G; this.regrasp = false; }
    const lost = this.phase === G && this.misses > this.grace && !acquire;
    if (lost) this.phase = A;
    this.armed = this.armed || this.latch || h <= this.liftM; // liftB latch with arming (lift_bottom.update_bottom_lift_state)
    this.latch = (this.latch || h > this.liftM) && this.armed;
    mem.regrasp_mode = this.regrasp ? 1 : 0;
    return { height: h, airborne: air, held, near, acquire, drop, relift, lost, dist: np.dist, cos: np.cos };
  }

  newPoint(mem, p, n, inHand) {
    mem.point_p = p.slice(); mem.point_n = n.slice();
    this.streak = 0; this.misses = this.grace + 1;
    if (inHand) { this.regrasp = true; this.phase = TaskState.A; }
    mem.regrasp_mode = this.regrasp ? 1 : 0;
  }
}
