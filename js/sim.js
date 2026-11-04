// MuJoCo side of the demo: one compiled scene (robot + table + one object), the control step (fixed physics substeps per
// 30 Hz control step), actuation, the mouse push (a spring on a grabbed point through xfrc_applied), teleports and the
// StepState read-back of STATE_SCHEMA.md section 1. Rendering lives in render.js, episode logic in task.js.

import { qrot, sub, cross, norm } from "./mathq.js";

let OBJ = null; // mjtObj values, read from the module
export function objTypes(mj) {
  const t = mj.mjtObj;
  return { BODY: t.mjOBJ_BODY.value, JOINT: t.mjOBJ_JOINT.value, GEOM: t.mjOBJ_GEOM.value, SITE: t.mjOBJ_SITE.value,
    ACTUATOR: t.mjOBJ_ACTUATOR.value, KEY: t.mjOBJ_KEY.value, MESH: t.mjOBJ_MESH.value };
}

export class Sim {
  /**
   * mj: the loaded @mujoco/mujoco module; m: MjModel; spec: normalized scene manifest (loader.normalizeScene) with
   * names, joint_names (canonical policy order), actuator_names, actuation, task.control_hz.
   */
  constructor(mj, m, spec) {
    this.mj = mj;
    this.m = m;
    this.d = new mj.MjData(m);
    this.spec = spec;
    OBJ = objTypes(mj);
    this.OBJ = OBJ;
    const id = (type, name, required = true) => {
      const i = name == null ? -1 : mj.mj_name2id(m, type, name);
      if (i < 0 && required) throw new Error(`scene has no ${Object.keys(OBJ).find((k) => OBJ[k] === type) || type} named '${name}'`);
      return i;
    };
    this.id = id;
    const N = spec.names;
    this.jointIds = spec.joint_names.map((n) => id(OBJ.JOINT, n));
    this.qadr = this.jointIds.map((j) => m.jnt_qposadr[j]);
    this.dadr = this.jointIds.map((j) => m.jnt_dofadr[j]);
    this.nJ = this.jointIds.length;
    this.actIds = (spec.actuator_names || []).map((n) => id(OBJ.ACTUATOR, n));
    this.objBody = id(OBJ.BODY, N.object_body);
    const oj = m.body_jntadr[this.objBody];
    if (oj < 0) throw new Error("object body has no free joint");
    this.objQ = m.jnt_qposadr[oj];
    this.objV = m.jnt_dofadr[oj];
    this.palmBody = id(OBJ.BODY, N.palm_body);
    this.tipBodies = N.fingertip_bodies.map((n) => id(OBJ.BODY, n));
    this.linkBodies = N.hand_link_bodies.map((n) => id(OBJ.BODY, n));
    this.tableSite = id(OBJ.SITE, N.table_surface_site, false);
    this.tableBody = id(OBJ.BODY, N.table_body || "table", false);
    this.tableHalfHeight = spec.task.table_half_height ?? 0.15;
    this.goalMocapBody = id(OBJ.BODY, N.goal_mocap, false);
    this.goalMocap = this.goalMocapBody >= 0 ? m.body_mocapid[this.goalMocapBody] : -1;
    this.homeKey = spec.home_keyframe ? id(OBJ.KEY, spec.home_keyframe, false) : -1;
    this.rootBody = id(OBJ.BODY, N.robot_base_body, false);

    this.controlDt = 1 / (spec.task.control_hz || 30);
    this.nSub = Math.max(1, Math.round(this.controlDt / m.opt.timestep));
    this.targets = new Float64Array(this.nJ);
    this.handPD = null; // Isaac delayed hand PD (setHandDrive('isaac_pd'))
    this.perturb = null;
    this.stats = { physMs: 0, steps: 0 };
    this.objMass = m.body_subtreemass[this.objBody];
    this.reset();
  }

  reset() {
    const { mj, m, d } = this;
    if (this.homeKey >= 0) mj.mj_resetDataKeyframe(m, d, this.homeKey);
    else mj.mj_resetData(m, d);
    d.xfrc_applied.fill(0);
    this.perturb = null;
    for (let j = 0; j < this.nJ; j++) this.targets[j] = d.qpos[this.qadr[j]];
    if (this.handPD) this.handPD.fresh = true;
    this._applyTargets(-1);
    mj.mj_forward(m, d);
  }

  /** Joint positions (canonical order). */
  q(out = new Float64Array(this.nJ)) { const qp = this.d.qpos; for (let j = 0; j < this.nJ; j++) out[j] = qp[this.qadr[j]]; return out; }
  qd(out = new Float64Array(this.nJ)) { const qv = this.d.qvel; for (let j = 0; j < this.nJ; j++) out[j] = qv[this.dadr[j]]; return out; }

  /** The table top height (per-episode jitter of the reset): moves the table body (its top site follows). */
  setTableTop(z) {
    if (this.tableBody < 0) return;
    this.m.body_pos[3 * this.tableBody + 2] = z - this.tableHalfHeight;
    this.mj.mj_forward(this.m, this.d);
  }

  /** Robot joint positions / velocities (canonical order) at a reset; the drive targets start at q. */
  setRobotState(q, qd) {
    for (let j = 0; j < this.nJ; j++) { this.d.qpos[this.qadr[j]] = q[j]; this.d.qvel[this.dadr[j]] = qd ? qd[j] : 0; this.targets[j] = q[j]; }
    if (this.handPD) this.handPD.fresh = true;
    this._applyTargets(-1);
    this.mj.mj_forward(this.m, this.d);
  }

  setJointQ(q) {
    for (let j = 0; j < this.nJ; j++) { this.d.qpos[this.qadr[j]] = q[j]; this.d.qvel[this.dadr[j]] = 0; this.targets[j] = q[j]; }
    this.mj.mj_forward(this.m, this.d);
  }

  bodyPos(b) { const x = this.d.xpos; return [x[3 * b], x[3 * b + 1], x[3 * b + 2]]; }
  bodyQuat(b) { const x = this.d.xquat; return [x[4 * b], x[4 * b + 1], x[4 * b + 2], x[4 * b + 3]]; }

  objectPose() {
    const q = this.d.qpos, a = this.objQ;
    return [[q[a], q[a + 1], q[a + 2]], [q[a + 3], q[a + 4], q[a + 5], q[a + 6]]];
  }

  /** Object linear velocity (world) and angular velocity (world; MuJoCo's free joint stores it in the body frame). */
  objectVel() {
    const v = this.d.qvel, a = this.objV;
    const [, q] = this.objectPose();
    return [[v[a], v[a + 1], v[a + 2]], qrot(q, [v[a + 3], v[a + 4], v[a + 5]])];
  }

  setObjectPose(pos, quat, zeroVel = true) {
    const q = this.d.qpos, a = this.objQ;
    for (let i = 0; i < 3; i++) q[a + i] = pos[i];
    for (let i = 0; i < 4; i++) q[a + 3 + i] = quat[i];
    if (zeroVel) for (let i = 0; i < 6; i++) this.d.qvel[this.objV + i] = 0;
    this.mj.mj_forward(this.m, this.d);
  }

  addObjectVelocity(lin, angWorld) {
    const v = this.d.qvel, a = this.objV;
    const [, q] = this.objectPose();
    const wb = qrot([q[0], -q[1], -q[2], -q[3]], angWorld);
    for (let i = 0; i < 3; i++) { v[a + i] += lin[i]; v[a + 3 + i] += wb[i]; }
  }

  setGoalMarker(pos, quat) {
    if (this.goalMocap < 0) return;
    const mp = this.d.mocap_pos, mq = this.d.mocap_quat, k = this.goalMocap;
    for (let i = 0; i < 3; i++) mp[3 * k + i] = pos[i];
    for (let i = 0; i < 4; i++) mq[4 * k + i] = quat[i];
  }

  tableTopZ() {
    if (this.tableSite >= 0) return this.d.site_xpos[3 * this.tableSite + 2];
    return this.spec.task.table_surface_z;
  }

  // ---- actuation -------------------------------------------------------------------------------------------------------
  setTargets(t) { for (let j = 0; j < this.nJ; j++) this.targets[j] = t[j]; }

  /**
   * Hand drive. 'servo': the scene's position actuators (ctrl = target; PD at every MuJoCo step, no delay). 'isaac_pd': r321's
   * explicit hand PD (Isaac utils/hand_delay_pd.py, port of agent A's mjcf/hand_pd.py): the 16 hand actuators become torque motors
   * (+-effort), tau = clip(K (q* - q[t-n]) - D qd[t-n], +-effort) computed once per Isaac substep (1/120 s) from a 3-deep ring
   * buffer and held; n = 0 'ideal', 2 'real'; hand joint speeds clipped to the PhysX velocity limit after every step.
   * cfg: {K[16], D[16], effort, vmax[16], nDelay}.
   */
  setHandDrive(mode, cfg = null) {
    const m = this.m, hand = this.actIds.slice(6);
    if (!this._servoBackup) {
      this._servoBackup = hand.map((a) => ({ gt: m.actuator_gaintype[a], gp: Array.from(m.actuator_gainprm.subarray(10 * a, 10 * a + 10)),
        bt: m.actuator_biastype[a], bp: Array.from(m.actuator_biasprm.subarray(10 * a, 10 * a + 10)), cl: m.actuator_ctrllimited[a],
        cr: Array.from(m.actuator_ctrlrange.subarray(2 * a, 2 * a + 2)), fl: m.actuator_forcelimited[a],
        fr: Array.from(m.actuator_forcerange.subarray(2 * a, 2 * a + 2)) }));
    }
    if (mode === "isaac_pd") {
      const e = cfg.effort;
      hand.forEach((a) => {
        m.actuator_gaintype[a] = 0; // mjGAIN_FIXED
        m.actuator_gainprm.fill(0, 10 * a, 10 * a + 10); m.actuator_gainprm[10 * a] = 1;
        m.actuator_biastype[a] = 0; // mjBIAS_NONE
        m.actuator_biasprm.fill(0, 10 * a, 10 * a + 10);
        m.actuator_ctrllimited[a] = 1; m.actuator_ctrlrange[2 * a] = -e; m.actuator_ctrlrange[2 * a + 1] = e;
        m.actuator_forcelimited[a] = 1; m.actuator_forcerange[2 * a] = -e; m.actuator_forcerange[2 * a + 1] = e;
      });
      const steps = Math.max(1, Math.round(1 / 120 / m.opt.timestep));
      this.handPD = { K: cfg.K, D: cfg.D, effort: e, vmax: cfg.vmax, n: cfg.nDelay || 0, steps, q: [], qd: [], fresh: true,
        tau: new Float64Array(16) };
    } else {
      hand.forEach((a, i) => {
        const b = this._servoBackup[i];
        m.actuator_gaintype[a] = b.gt; m.actuator_gainprm.set(b.gp, 10 * a); m.actuator_biastype[a] = b.bt; m.actuator_biasprm.set(b.bp, 10 * a);
        m.actuator_ctrllimited[a] = b.cl; m.actuator_ctrlrange.set(b.cr, 2 * a); m.actuator_forcelimited[a] = b.fl; m.actuator_forcerange.set(b.fr, 2 * a);
      });
      this.handPD = null;
    }
    this.handDrive = mode;
  }

  setHandDelay(n) { if (this.handPD) { this.handPD.n = n; this.handPD.fresh = true; } }

  _applyTargets(sub) {
    const ctrl = this.d.ctrl, pd = this.handPD;
    const nArm = pd ? 6 : this.nJ;
    for (let j = 0; j < nArm; j++) ctrl[this.actIds[j]] = this.targets[j];
    if (!pd) return;
    if (sub < 0) { pd.tau.fill(0); } else if (sub % pd.steps === 0) {
      const qp = this.d.qpos, qv = this.d.qvel;
      const q = new Float64Array(16), qd = new Float64Array(16);
      for (let i = 0; i < 16; i++) { q[i] = qp[this.qadr[6 + i]]; qd[i] = qv[this.dadr[6 + i]]; }
      if (pd.fresh) { pd.q = [q, q, q]; pd.qd = [qd, qd, qd]; pd.fresh = false; } else { pd.q.push(q); pd.qd.push(qd); pd.q.shift(); pd.qd.shift(); }
      const qs = pd.q[2 - pd.n], vs = pd.qd[2 - pd.n];
      for (let i = 0; i < 16; i++) {
        const t = pd.K[i] * (this.targets[6 + i] - qs[i]) - pd.D[i] * vs[i];
        pd.tau[i] = t > pd.effort ? pd.effort : t < -pd.effort ? -pd.effort : t;
      }
    }
    for (let i = 0; i < 16; i++) ctrl[this.actIds[6 + i]] = pd.tau[i];
  }

  _clampHandVel() {
    const pd = this.handPD, qv = this.d.qvel;
    for (let i = 0; i < 16; i++) {
      const a = this.dadr[6 + i], v = qv[a], vm = pd.vmax[i];
      if (v > vm) qv[a] = vm; else if (v < -vm) qv[a] = -vm;
    }
  }

  // ---- mouse push ------------------------------------------------------------------------------------------------------
  /** Start a push: a spring from a body-fixed point (world `point` at grab time) to a world target. */
  startPush(body, point) {
    const bp = this.bodyPos(body), bq = this.bodyQuat(body);
    const local = qrot([bq[0], -bq[1], -bq[2], -bq[3]], sub(point, bp));
    const mass = Math.max(this.m.body_subtreemass[body], 0.02);
    this.perturb = { body, local, target: point.slice(), prev: null, mass, k: 300 * mass, fmax: 25 * mass + 2, lastPt: point.slice() };
  }

  endPush() { this.perturb = null; this.d.xfrc_applied.fill(0); }

  _applyPush() {
    const xf = this.d.xfrc_applied;
    const p = this.perturb;
    if (!p) return;
    const b = p.body;
    const bp = this.bodyPos(b), bq = this.bodyQuat(b);
    const pt = qrot(bq, p.local).map((v, i) => v + bp[i]);
    const h = this.m.opt.timestep;
    const vel = p.prev ? pt.map((x, i) => (x - p.prev[i]) / h) : [0, 0, 0];
    p.prev = pt;
    p.lastPt = pt;
    const c = 2 * Math.sqrt(p.k * p.mass) * 0.8;
    let F = [0, 1, 2].map((i) => p.k * (p.target[i] - pt[i]) - c * vel[i]);
    // carry the body's weight while dragging (feels like lifting rather than towing)
    F[2] += p.mass * 9.81;
    const n = norm(F);
    if (n > p.fmax) F = F.map((v) => (v * p.fmax) / n);
    const xi = this.d.xipos;
    const com = [xi[3 * b], xi[3 * b + 1], xi[3 * b + 2]];
    const T = cross(sub(pt, com), F);
    for (let i = 0; i < 3; i++) { xf[6 * b + i] = F[i]; xf[6 * b + 3 + i] = T[i]; }
  }

  // ---- stepping ----------------------------------------------------------------------------------------------------------
  /** One control step: hold the targets for nSub physics substeps. */
  controlStep() {
    const t0 = performance.now();
    const { mj, m, d } = this;
    for (let s = 0; s < this.nSub; s++) {
      this._applyTargets(s);
      if (this.perturb) this._applyPush();
      mj.mj_step(m, d);
      if (this.handPD) this._clampHandVel();
    }
    const dt = performance.now() - t0;
    this.stats.physMs = this.stats.steps ? 0.9 * this.stats.physMs + 0.1 * dt : dt;
    this.stats.steps++;
  }

  /** StepState (STATE_SCHEMA section 1) without goal / prev_targets (task.js adds them). */
  readState() {
    const st = {
      q: this.q(), qd: this.qd(),
      palm_pos: this.bodyPos(this.palmBody), palm_quat: this.bodyQuat(this.palmBody),
      tip_pos: this.tipBodies.map((b) => this.bodyPos(b)), tip_quat: this.tipBodies.map((b) => this.bodyQuat(b)),
      link_pos: this.linkBodies.map((b) => this.bodyPos(b)),
      table_top_z: this.tableTopZ(),
    };
    const [op, oq] = this.objectPose();
    st.obj_pos = op; st.obj_quat = oq;
    return st;
  }

  rootQuat() { return this.rootBody >= 0 ? this.bodyQuat(this.rootBody) : [1, 0, 0, 0]; }

  dispose() { try { this.d.delete(); } catch { /* already freed */ } try { this.m.delete(); } catch { /* already freed */ } }
}

