// The demo's episode layer around agent D's task logic (taskref.js = port of sim2sim/task_ref.py, the evaluation's 'point3' pass):
// it writes TaskRef's resets into MuJoCo (robot noise, table height, stable-pose spawn), runs the observation pipeline ('ideal' /
// 'real' with the env's fresh rule), steps TaskRef after the physics, and adds the page's features on top: user goals (Send or live)
// and points (point mode) applied at once, the after-success modes, and the episode-end handling (reset like training, memory refresh,
// or none). TaskRef stays the single source of the phase / success / switch / termination logic.

import { sub, norm, qrot, qmul, qconj, qnorm, qangle, DEG, rng as makeRng } from "./mathq.js";
import { lowestZ, goalCheck, randomGoal, pointFrame } from "./geometry.js";
import { observation, RealObservation, newMemory, prepareObject } from "./obs.js";
import { actionToTargets, RealActionDelay } from "./action.js";
import { TaskRef, prepareTaskObject, goalMetric, G_PHASE } from "./taskref.js";

export const PHASE_NAMES = { 0: "L", 1: "P1", 2: "A", 3: "G" };

/** Object data (tools/export_object_data.py: B's objects + D's task_data + the click pool) -> the obs object, the TaskRef object and
 * the page's geometry (support hull for the goal checks, click pool, centroid for the gizmo pivot). */
export function makeGeo(od) {
  return {
    name: od.name, label: od.label || od.name, obj: prepareObject(od), tobj: prepareTaskObject(od), hull: od.support_hull,
    points512: od.points512, poolP: od.pool_p, poolN: od.pool_n, centroid: od.centroid || pointFrame(od.points64).centroid,
  };
}

export class Task {
  /** c: obs.loadConstants() (policy/constants.json: joint limits, init_joint_pos, offsets); taskCfg: the scene manifest's task block. */
  constructor(sim, geo, c, taskCfg, seed = 2027) {
    this.sim = sim;
    this.geo = geo;
    this.c = c;
    this.cfg = Object.assign({ tolerance_m: 0.05, clearance_m: 0.005, delta_m: 0.1, rot_cap_rad: 1.5 }, taskCfg || {});
    this.rand = makeRng(seed);
    // the evaluation's first-episode cap (60 s) is a measurement cut, not an episode end of the trained task: off in the demo
    this.ref = new TaskRef(geo.tobj, this.rand, c, { tol: this.cfg.tolerance_m, constOverrides: { eval_steps: Infinity }, coupled: true, switchProb: 0 });
    this.obsOut = new Float32Array(601);
    this.conditions = "ideal";
    this.afterSuccess = "auto"; // auto (user goals held, env goals: eval rule) | hold | next | eval_rule
    this.keepAfterReset = true;
    this.episodeEnd = "train"; // train | refresh | off (falls / hand far always reset)
    this.coupled = true; // coupled goal on a user's in-hand point change
    this.events = [];
    this.totals = { episodes: 0, goals: 0, goalsReached: 0, drops: 0, falls: 0, userGoals: 0, userGoalsReached: 0, switches: 0, deadlines: 0,
      acquisitions: 0, refreshes: 0, lifts: 0, relifts: 0, ends: {} };
    this.goal = null;
    this.point = null;
    this.userGoal = null;
    this.userPoint = null;
    this.T = 0;
    this.reset("start");
  }

  log(text) {
    this.events.push(`${this.T.toFixed(1).padStart(6)}s ${text}`);
    if (this.events.length > 40) this.events.shift();
  }

  get ts() { return this.ref.ts; }

  goalCtx() {
    return { hull: this.geo.hull, surface_z: this.sim.tableTopZ(), clearance_m: this.cfg.clearance_m, goal_box: this.cfg.goal_box,
      delta_m: this.cfg.delta_m, rot_cap_rad: this.cfg.rot_cap_rad };
  }

  // ---- episode -------------------------------------------------------------------------------------------------------------------
  reset(why = "manual") {
    const sim = this.sim, ref = this.ref, c = this.c;
    sim.reset();
    const matched = {};
    if (this.keepAfterReset && this.userGoal) { matched.goal_pos = this.userGoal.pos; matched.goal_quat = this.userGoal.quat; }
    const keepPoint = this.keepAfterReset && this.userPoint != null;
    if (keepPoint) { matched.point_p = this.geo.poolP[this.userPoint]; matched.point_n = this.geo.poolN[this.userPoint]; }
    matched.table_offset = 0; // the actor's table-height calibration error: 0 in the demo (STATE_SCHEMA 2)
    const init = ref.reset(c.init_joint_pos, c.joint_lower, c.joint_upper, matched);
    sim.setTableTop(init.table_top);
    if (init.q) sim.setRobotState(init.q, init.qd);
    sim.setObjectPose(init.obj_pos, init.obj_quat);
    this.prevTargets = Float64Array.from(init.q || sim.q());
    this.mem = newMemory(this.geo.obj, init.fps_start, ref.point_p, ref.point_n, { tableOffset: 0 });
    this._newGoalRecord(matched.goal_pos ? this.goal?.source === "live" ? "live" : "user" : "env", false);
    this.point = { index: keepPoint ? this.userPoint : null, p: ref.point_p, n: ref.point_n, source: keepPoint ? "user" : "env" };
    this.epSteps = 0;
    this.tEp = 0;
    this.forceFresh = true;
    this.latchLogged = false;
    this._configureConditions();
    this.totals.episodes++;
    this.log(`episode reset (${why})`);
    this.state = this._readState();
    this.flags = { held: false, airborne: false, near: false, dist: NaN };
    this.err = null;
    this.onReset?.(why);
  }

  setConditions(cond) { this.conditions = cond; this._configureConditions(); this.log(`conditions: ${cond}`); }

  _configureConditions() {
    const real = this.conditions === "real";
    this.sim.setHandDelay(real ? 2 : 0); // isaac_pd hand drive: hand_delay_pd n = 2 in the real pass, 0 in ideal
    // 'real' (utils/real_eval.py): tracker pose + 1-step obs delay, 1-step action delay; kept across episodes (k counts from the start)
    if (real && !this.realObs) {
      this.realObs = new RealObservation(this.c, (this.rand() * 1e9) | 0);
      this.realAct = new RealActionDelay(this.c.eval_real?.action_delay_steps ?? 1);
      this.forceFresh = true;
    }
    if (!real) { this.realObs = null; this.realAct = null; this.trackedPose = null; }
  }

  // ---- goal / point --------------------------------------------------------------------------------------------------------------
  /** A new goal record for the page (number, source, timing) from TaskRef's current goal. */
  _newGoalRecord(source, log = true) {
    const id = (this.goal?.id || 0) + 1;
    this.goal = { pos: this.ref.goal_pos.slice(), quat: this.ref.goal_quat.slice(), source, id, t0: this.T, reached: null, bestMetric: Infinity };
    this.sim.setGoalMarker(this.goal.pos, this.goal.quat);
    if (source !== "env") this.totals.userGoals++;
    if (log) this.log(`goal #${id} (${source})`);
  }

  /** The user's goal (Send). Returns the preview check; refused when HARD fails, or SOFT fails without confirm. */
  sendGoal(pos, quat, confirm = false) {
    const chk = goalCheck(pos, quat, this.goalCtx(), this.goal?.pos, this.goal?.quat);
    if (!chk.ok || (!chk.clean && !confirm)) return chk;
    this.userGoal = { pos: pos.slice(), quat: quat.slice() };
    this.ref.setGoal(pos, quat);
    this._newGoalRecord("user");
    return chk;
  }

  /** Live goal: applied at once (the caller clamps it); a new session starts a new goal number, else the goal moves in place. */
  setGoalLive(pos, quat, newSession) {
    this.userGoal = { pos: pos.slice(), quat: quat.slice() };
    this.ref.setGoal(pos, quat);
    if (newSession || !this.goal || this.goal.source !== "live") { this._newGoalRecord("live"); return; }
    const g = this.goal;
    g.pos = this.ref.goal_pos.slice(); g.quat = this.ref.goal_quat.slice();
    g.reached = null; g.bestMetric = Infinity; g.t0 = this.T;
    this.sim.setGoalMarker(g.pos, g.quat);
  }

  /** A user point (pool index k), applied at once through TaskRef.setPoint: in hand (phase G) an in-hand switch (regrasp), else a new
   * command. With the coupled goal on, an in-hand switch also re-targets the goal by r321's contact rule (TaskRef.coupledGoal). */
  sendPoint(k, { coupled = this.coupled, goalGiven = false } = {}) {
    const ref = this.ref, oldP = ref.point_p.slice(), oldN = ref.point_n.slice();
    const p = this.geo.poolP[k], n = this.geo.poolN[k];
    const latched = ref.ts.latch;
    this.userPoint = k;
    const inHand = ref.setPoint(p, n);
    this.point = { index: k, p: ref.point_p, n: ref.point_n, source: "user" };
    const kind = inHand ? "switch" : latched ? "command" : "before_lift";
    if (!inHand) { this.log(`point #${k} (${kind === "command" ? "new command" : "before lift"})`); return { kind }; }
    this.totals.switches++;
    this.log(`point #${k} (switch: re-target in hand)`);
    if (coupled && !goalGiven) {
      const [op, oq] = this.sim.objectPose();
      const cg = ref.coupledGoal(op, oq, oldP, oldN, p, n);
      if (cg) {
        this.userGoal = { pos: cg[0].slice(), quat: cg[1].slice() };
        ref.setGoal(cg[0], cg[1]);
        this._newGoalRecord("user", false);
        this.log(`coupled goal #${this.goal.id}`);
      } else this.log("coupled goal clips the table: goal kept");
    }
    return { kind };
  }

  randomGoal() {
    const [p, q] = randomGoal(this.rand, this.goalCtx(), this.goal ? this.goal.quat : this.sim.objectPose()[1]);
    return { pos: p, quat: q };
  }

  // ---- stepping ------------------------------------------------------------------------------------------------------------------
  _readState() {
    const st = this.sim.readState();
    st.goal_pos = this.ref.goal_pos; st.goal_quat = this.ref.goal_quat;
    st.prev_targets = this.prevTargets;
    st.pads = st.tip_pos.map((t, i) => qrot(st.tip_quat[i], this.c.fingertip_offsets[i]).map((v, k) => v + t[k]));
    return st;
  }

  /** real_eval._fresh (task_ref: inactivity clock 0 and no goal yet = an episode start or a first acquisition before the first goal),
   * plus the page's own refills (reset, refresh, teleport, switching to 'real'). */
  _fresh() { return this.forceFresh || (this.ref.clock === 0 && this.ref.successes === 0); }

  /** The actor observation for the current state ('real': the tracker pose and the 1-step observation delay). */
  observe() {
    const ref = this.ref, mem = this.mem;
    mem.point_p = ref.point_p; mem.point_n = ref.point_n; mem.regrasp_mode = ref.regrasp ? 1 : 0;
    this.stepFresh = this._fresh();
    if (this.realObs) {
      const [o, pose] = this.realObs.step(this.state, mem, this.geo.obj, this.c, this.stepFresh);
      this.trackedPose = pose;
      return o;
    }
    return observation(this.state, mem, this.geo.obj, this.c, null, null, this.obsOut);
  }

  /** Policy action -> targets (held for the next control step's substeps). */
  applyAction(action) {
    const a = this.realAct ? this.realAct.step(action, this.stepFresh) : action;
    const t = actionToTargets(a, this.prevTargets, this.c);
    this.sim.setTargets(t);
    this.prevTargets = t;
    this.forceFresh = false;
  }

  /** After the physics substeps of one control step: TaskRef.step, then the page's goal records, events and episode-end handling. */
  afterStep() {
    const sim = this.sim, ref = this.ref;
    this.T += sim.controlDt;
    this.tEp += sim.controlDt;
    this.epSteps++;
    const st = (this.state = this._readState());
    const userHeld = this.goal.source !== "env";
    const hold = this.afterSuccess === "hold" || (this.afterSuccess === "auto" && userHeld);
    ref.switchProb = hold || this.afterSuccess === "next" ? 0 : ref.c.switch_prob;
    const goalBefore = this.goal;
    const ev = ref.step(st);
    this.flags = ev;
    this.err = { metric: ev.d6 ?? goalMetric(this.geo.tobj, st.obj_pos, st.obj_quat, ref.goal_pos, ref.goal_quat),
      posErr: norm(sub(st.obj_pos, goalBefore.pos)), rotErr: qangle(st.obj_quat, goalBefore.quat) };
    goalBefore.bestMetric = Math.min(goalBefore.bestMetric, this.err.metric);
    if (ev.first) this.totals.acquisitions++;
    if (ev.acquire) this.log(ev.spell_acq ? "re-acquired IN HAND" : "acquired");
    if (ev.lost) this.log("point lost: G -> A");
    if (ev.drop) { this.totals.drops++; this.log("DROP"); }
    if (ev.relift) { this.totals.relifts++; this.log("re-lift"); }
    if (ref.ts.latch && !this.latchLogged) { this.latchLogged = true; this.totals.lifts++; this.log("lifted"); }
    if (ev.deadline) { this.totals.deadlines++; this.log("DEADLINE: new point"); }
    if (ev.switch) { this.totals.switches++; this.log("SWITCH (auto): new point"); }
    if (ev.deadline || ev.switch) { this.point = { index: null, p: ref.point_p, n: ref.point_n, source: "env" }; this.userPoint = null; }
    if (ev.success) {
      this.totals.goals++;
      if (goalBefore.reached == null) {
        goalBefore.reached = this.T - goalBefore.t0;
        this.totals.goalsReached++;
        if (goalBefore.source !== "env") this.totals.userGoalsReached++;
        this.log(`goal #${goalBefore.id} reached in ${goalBefore.reached.toFixed(1)} s`);
      }
      if (hold) ref.setGoal(goalBefore.pos, goalBefore.quat); // keep the goal: the dwell restarts, the next success needs 5 more steps
      else { this.userGoal = null; this._newGoalRecord("env"); }
    }
    if (ev.done) this._episodeEnd(ev.reason);
  }

  /** TaskRef's terminations (fall, 50 goals, hand far, switch-spell drop, 599-step inactivity): falls / hand far reset; the rest per the
   * page's mode. */
  _episodeEnd(reason) {
    const reasons = reason.split("+");
    const physical = reasons.includes("fall") || reasons.includes("hand_far");
    const mode = physical ? "train" : this.episodeEnd;
    if (mode === "off") { this.ref.done = false; this.ref.reason = ""; return; }
    for (const r of reasons) this.totals.ends[r] = (this.totals.ends[r] || 0) + 1;
    if (reasons.includes("fall")) this.totals.falls++;
    const why = reasons.join(", ");
    if (mode === "refresh") this.memoryRefresh(why);
    else this.reset(why);
    this.onEpisodeEnd?.(why, mode);
  }

  /**
   * Memory refresh: keep the physics state (object where it lies, robot where it is) and restart only the policy's memory as at an
   * episode start: LSTM state zero (onMemoryRefresh -> policy.reset()), prev targets = the current joint positions (also held as the
   * drive targets), the 'real' tracker history / delay queues refilled, TaskRef's episode state (phases, clocks, counters) restarted
   * with the current goal and point.
   */
  memoryRefresh(why = "manual") {
    const q = this.sim.q();
    this.prevTargets = q;
    this.sim.setTargets(q);
    this.ref.refresh();
    this.mem.regrasp_mode = 0;
    this.epSteps = 0;
    this.tEp = 0;
    this.latchLogged = false;
    this.forceFresh = true;
    this.state = this._readState();
    this.totals.refreshes++;
    this.log(`memory refresh (${why})`);
    this.onMemoryRefresh?.(why);
  }

  /** Teleport the object by (dx, dy) on the table (SimToolReal W/A/S/D), at rest. */
  teleportObject(dx, dy) {
    const [p, q] = this.sim.objectPose();
    const top = this.sim.tableTopZ();
    const np = [p[0] + dx, p[1] + dy, p[2]];
    const low = lowestZ(this.geo.hull, np, q);
    if (low < top + 0.001) np[2] += top + 0.001 - low;
    this.sim.setObjectPose(np, q);
    this.forceFresh = true; // tracker / delay queues refill with the teleported state
    this.state = this._readState();
  }

  /** A random velocity kick of the object (sim_demo push). */
  kick(speed = 0.3) {
    const a = this.rand() * 2 * Math.PI;
    let d = [Math.cos(a), Math.sin(a), 0.1 + 0.3 * this.rand()];
    const n = norm(d);
    d = d.map((v) => (v / n) * speed);
    this.sim.addObjectVelocity(d, [0, 1, 2].map(() => this.rand.normal() * 2 * speed));
    this.log(`push ${speed.toFixed(2)} m/s`);
  }

  readouts() {
    const e = this.err || { posErr: NaN, rotErr: NaN, metric: NaN };
    const f = this.flags, ref = this.ref;
    return {
      phase: PHASE_NAMES[ref.ts.phase] || "L", held: !!f.held, lifted: ref.ts.latch, airborne: !!f.airborne, regrasp: ref.regrasp,
      acquired: ref.ts.phase === G_PHASE, near: !!f.near, dPoint: f.dist, posErr: e.posErr, rotErrDeg: e.rotErr / DEG, metric: e.metric,
      tol: this.cfg.tolerance_m, dwell: ref.near_goal_steps, noProgress: ref.clock, maxSteps: ref.c.inactivity_steps, successes: ref.successes,
      goal: this.goal, point: this.point, totals: this.totals, T: this.T, tEp: this.tEp, events: this.events, record: ref.record(),
    };
  }
}

export { qmul, qconj, qnorm };
