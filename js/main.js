// The demo page: boot (MuJoCo WASM, scene, policy), the real-time loop (30 Hz control steps paced against wall time, rendering
// every display frame) and the UI: live goal (default; clamped, applied at the control rate) or draft + Send, point mode (hover
// preview, click / drag to apply), push, camera, key legend, readouts, episode-end banner.

import { loadAppConfig, loadMujoco, loadOrt, fetchJson, fetchWithProgress, normalizeScene, filterObjects, SceneLoader, resolvePolicy, loadSceneParams } from "./loader.js";
import { Sim } from "./sim.js";
import { Task, makeGeo } from "./task.js";
import { View } from "./render.js";
import { loadConstants } from "./obs.js";
import { ZeroPolicy, RandomPolicy, OnnxPolicy } from "./policy.js";
import { nudge, nearestIndex, goalCheck, clampGoal } from "./geometry.js";
import { add, sub, qrot, qrotInv, qnorm, rpyDeg, quatFromRpyDeg } from "./mathq.js";

const $ = (id) => document.getElementById(id);
const S = {
  cfg: null, mj: null, spec: null, loader: null, sim: null, task: null, view: null, policy: null, policies: {},
  consts: null, playing: true, speed: 1, acc: 0, busy: false,
  handDrive: "servo", draft: null, gizmoOn: false, gizmoMode: "translate", allowSoft: false, showBox: false,
  goalMode: "live", pendingLive: null, liveNote: null, liveGoalId: null, pointMode: false, hover: null, dragPush: false,
  episodeEnd: "train", banner: null,
  rt: [], lastUi: 0, objectName: null, switching: false, frames: 0, fps: 0, fpsT: 0, error: null,
};
window.__demo = S; // for the headless smoke test (test/browser_smoke.py); nothing on the page reads it

// ---- status / progress -------------------------------------------------------------------------------------------------
let bytesGot = 0, bytesTotal = 1;
function setStatus(msg, isError = false) { $("status").textContent = msg; $("status").classList.toggle("error", isError); }
function bump(n) { bytesGot += n; $("bar-fill").style.width = `${Math.min(100, (100 * bytesGot) / bytesTotal).toFixed(1)}%`; }

// Web fonts are added from script so an unreachable font host never blocks the page (a <link> in <head> would hold the module
// scripts until it loads or times out); system fonts show until they arrive.
function loadFonts() {
  const l = document.createElement("link");
  l.rel = "stylesheet";
  l.href = "https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Semi+Condensed:wght@600&family=JetBrains+Mono:wght@400;500&display=swap";
  document.head.append(l);
}

async function boot() {
  loadFonts();
  try {
    setStatus("Reading the configuration…");
    S.cfg = await loadAppConfig();
    if (S.cfg.hand_drive) S.handDrive = S.cfg.hand_drive;
    if (S.cfg.goal_mode === "send") S.goalMode = "send";
    if (["train", "refresh", "off"].includes(S.cfg.episode_end)) S.episodeEnd = S.cfg.episode_end;
    if (!S.cfg.scene) throw new Error("no scene manifest found (assets/mjcf/manifest.json); see app/README.md");
    S.spec = filterObjects(normalizeScene(await fetchJson(S.cfg.scene), S.cfg.scene, S.cfg.object_data), S.cfg.objects);
    if (!S.spec.objects.length) throw new Error("the scene manifest lists no objects");
    S.sceneParams = await loadSceneParams(S.spec);
    let pman = null;
    if (/\.json$/.test(S.cfg.policy || "")) {
      try { pman = await resolvePolicy(S.cfg.policy, S.cfg.variant); }
      catch (e) { console.warn("policy manifest unavailable, using the random policy:", e); }
    }
    let constJson = S.spec.raw.constants ? await fetchJson(S.spec.url(S.spec.raw.constants)).catch(() => null) : null;
    if (pman?.constantsUrl) constJson = Object.assign(constJson || {}, await fetchJson(pman.constantsUrl).catch(() => ({})));
    S.consts = loadConstants(constJson);
    S.constsFromFile = !!(constJson && constJson.joint_lower && constJson.joint_upper);
    const first0 = S.spec.objects.find((o) => o.name === S.cfg.object) || S.spec.objects.find((o) => o.name === S.spec.default_object);
    const sceneBytes = [...S.spec.common_files, first0.scene, ...first0.files].reduce((s, f) => s + (S.spec.sizes[f] || 0), 0) || 3e6;
    bytesTotal = 10.8e6 + sceneBytes + (pman ? pman.bytes || 18e6 : 0);

    setStatus("Loading MuJoCo (WASM)…");
    const ortP = pman ? loadOrt() : null;
    S.mj = await loadMujoco(bump);
    $("v-mj").textContent = S.mj.mj_versionString ? S.mj.mj_versionString() : "";
    S.loader = new SceneLoader(S.mj, S.spec);
    S.view = new View($("view"));
    wireTheme();

    // policies: the ONNX actor (if a manifest is configured), plus the two pipeline checks
    S.policies.random = new RandomPolicy(22);
    S.policies.zero = new ZeroPolicy(22);
    if (pman) {
      setStatus("Loading the policy (onnxruntime-web)…");
      try {
        const ort = await ortP;
        const p = new OnnxPolicy(ort, pman, fetchWithProgress);
        await p.init(bump);
        S.policies.onnx = p;
      } catch (e) {
        console.error(e);
        S.error = `policy failed to load: ${e.message}`;
      }
    }
    const want = S.cfg.policy === "zero" ? "zero" : S.policies.onnx ? "onnx" : "random";
    fillPolicySelect(want);
    S.policy = S.policies[want];

    fillObjectSelect();
    const first = S.cfg.object && S.spec.objects.some((o) => o.name === S.cfg.object) ? S.cfg.object : S.spec.default_object;
    await switchObject(first);
    if (S.cfg.conditions) { $("conditions").value = S.cfg.conditions; S.task.setConditions(S.cfg.conditions); }

    $("loading").hidden = true;
    $("hud").hidden = false;
    $("toolbar").hidden = false;
    for (const id of ["play", "step", "reset"]) $(id).disabled = false;
    wireUi();
    showLegend(true);
    S.legendTimer = setTimeout(() => showLegend(false), 7000); // shown briefly on load; H / ? brings it back
    if (S.error) S.task.log(S.error);
    requestAnimationFrame(frame);
  } catch (e) {
    console.error(e);
    setStatus(`Could not start: ${e.message}`, true);
  }
}

function fillObjectSelect() {
  const sel = $("object");
  sel.innerHTML = "";
  for (const o of S.spec.objects) {
    const opt = document.createElement("option");
    opt.value = o.name;
    opt.textContent = o.split === "held_out" ? `${o.label} (held-out)` : o.label;
    sel.append(opt);
  }
}

function fillPolicySelect(sel) {
  const el = $("policy");
  el.innerHTML = "";
  const names = { onnx: S.policies.onnx?.name || "ONNX", random: "random actions (pipeline check)", zero: "zero actions (hold)" };
  for (const k of Object.keys(S.policies)) {
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = names[k];
    el.append(opt);
  }
  el.value = sel;
  $("policy-name").textContent = `policy: ${names[sel]}`;
}

async function switchObject(name) {
  if (S.switching) return;
  S.switching = true;
  const was = S.playing;
  S.playing = false;
  try {
    setStatus(`Loading ${name}…`);
    while (S.busy) await new Promise((r) => setTimeout(r, 5));
    const { m, data } = await S.loader.loadObject(name, bump);
    const prev = S.task;
    if (S.sim) S.sim.dispose();
    S.sim = new Sim(S.mj, m, S.spec);
    if (!S.constsFromFile) {
      // no policy/constants.json yet: the canonical joint limits are the scene's joint ranges (agent A writes Isaac's prepped limits)
      S.consts.joint_lower = S.sim.jointIds.map((j) => m.jnt_range[2 * j]);
      S.consts.joint_upper = S.sim.jointIds.map((j) => m.jnt_range[2 * j + 1]);
    }
    const hp = S.sceneParams?.handPD;
    if (S.handDrive === "isaac_pd" && hp) S.sim.setHandDrive("isaac_pd", hp);
    const geo = makeGeo(data);
    const tcfg = Object.assign({}, S.spec.task, { goal_box: S.spec.task.goal_box });
    S.task = new Task(S.sim, geo, S.consts, tcfg, Number(S.cfg.seed) || 2027);
    if (prev) {
      S.task.conditions = prev.conditions; S.task.afterSuccess = prev.afterSuccess; S.task.keepAfterReset = prev.keepAfterReset;
      S.task.coupled = prev.coupled; S.task.totals = prev.totals; S.task.events = prev.events;
      S.task.userGoal = null; S.task.userPoint = null;
      S.task.reset(`object ${name}`);
    }
    S.task.episodeEnd = S.episodeEnd;
    S.task.onReset = () => { S.policy?.reset(); S.draft = null; S.pendingLive = null; };
    S.task.onMemoryRefresh = () => { S.policy?.reset(); };
    S.task.onEpisodeEnd = (why, mode) => showBanner(mode === "refresh" ? `Memory refresh: ${reasonText(why)}` : `Episode reset: ${reasonText(why)}`);
    S.policy?.reset();
    S.view.load(S.sim, S.spec);
    S.view.setGoalBox(S.spec.task.goal_box);
    S.view.onGizmoDrag = onGizmoDrag;
    S.view.onGizmoChange = onGizmoChange;
    S.objectName = name;
    $("object").value = name;
    S.draft = null; S.pendingLive = null; S.hover = null;
    S.acc = 0;
    S.task.log(`object: ${data.label || name}`);
  } catch (e) {
    console.error(e);
    S.task?.log(`could not load ${name}: ${e.message}`);
    setStatus(`Could not load ${name}: ${e.message}`, true);
  } finally {
    S.playing = was;
    S.switching = false;
  }
}

// ---- loop ------------------------------------------------------------------------------------------------------------------
async function controlStep() {
  applyPendingLive(); // the live goal is throttled to the control rate: at most one goal update per 30 Hz step
  const t = S.task;
  const obs = t.observe();
  const a = await S.policy.act(obs);
  t.applyAction(a);
  S.sim.controlStep();
  t.afterStep();
}

/** Run the control steps that wall time asks for, within a compute budget per display frame (slow devices fall behind into slow
 * motion instead of freezing the page). */
async function runSteps(budgetMs = 45) {
  const dt = S.sim.controlDt;
  const t0 = performance.now();
  while (S.acc >= dt && S.playing && !S.switching && performance.now() - t0 < budgetMs) {
    await controlStep();
    S.acc -= dt;
    S.simAdvanced += dt;
  }
  if (S.acc > 12 * dt) S.acc = 12 * dt; // keep at most a short backlog
}

let last = null;
function frame(t) {
  requestAnimationFrame(frame);
  if (!S.sim) return;
  const dtReal = last == null ? 0 : (t - last) / 1000;
  last = t;
  if (S.playing && !S.switching) {
    S.acc = Math.min(S.acc + dtReal * S.speed, 12 * S.sim.controlDt);
    if (!S.busy && S.acc >= S.sim.controlDt) {
      S.busy = true;
      // compute budget: half the display frame interval, 30-150 ms (slow renderers trade fps for sim speed)
      runSteps(Math.min(150, Math.max(30, 500 * dtReal))).catch((e) => { console.error(e); S.task.log(`step error: ${e.message}`); S.playing = false; }).finally(() => { S.busy = false; });
    }
  } else if (!S.busy) applyPendingLive(); // paused: the goal ghost still follows the gizmo
  S.rt.push([dtReal, S.simAdvanced || 0]);
  S.simAdvanced = 0;
  if (S.rt.length > 90) S.rt.shift();
  S.frames++;
  if (t - S.fpsT > 1000) { S.fps = (1000 * S.frames) / (t - S.fpsT); S.frames = 0; S.fpsT = t; }
  draw();
  if (t - S.lastUi > 110) { S.lastUi = t; updateUi(); }
}

function draw() {
  const task = S.task, st = task.state;
  const palm = add(st.palm_pos, qrot(st.palm_quat, S.consts.palm_surface_centre));
  const push = S.sim.perturb ? { from: S.sim.perturb.lastPt, to: S.sim.perturb.target } : null;
  if (S.gizmoOn && !S.view.gizmo.dragging) {
    const g = S.draft || task.goal;
    S.view.setGizmo(true, add(g.pos, qrot(g.quat, task.geo.centroid)), S.gizmoMode);
  }
  // the orange draft ghost: Send mode, or a live request that had to be clamped (shows what was asked for)
  const showDraft = S.draft && (S.goalMode === "send" || (S.liveNote && S.liveNote.kind === "hard"));
  S.view.draw({
    goal: task.goal, draft: showDraft ? S.draft : null, point: task.point, palm, push, showGoalBox: S.showBox || !!(S.liveNote && S.liveNote.kind === "hard"),
    previewPoint: S.pointMode && S.hover ? { p: task.geo.poolP[S.hover.index], n: task.geo.poolN[S.hover.index] } : null,
  });
}

// ---- goal: live (default) or draft + Send -------------------------------------------------------------------------------------
function ensureDraft() {
  const g = S.task.goal;
  // live: edits start from the active goal, unless the last live request was clamped (keep stepping from what was asked for)
  if (!S.draft || (S.goalMode === "live" && S.liveGoalId !== g.id)) S.draft = { pos: g.pos.slice(), quat: g.quat.slice() };
  return S.draft;
}

/** A draft edit. Live mode: queue it for the next control step (newSession: a new goal number, else the goal moves in place). */
function draftChanged(newSession) {
  if (S.goalMode !== "live" || !S.draft) return;
  S.pendingLive = { pos: S.draft.pos.slice(), quat: S.draft.quat.slice(), newSession: newSession || !!S.pendingLive?.newSession };
}

/** Apply a queued live goal: HARD violations are clamped to the nearest valid pose (red note), SOFT ones only noted (amber). */
function applyPendingLive() {
  const p = S.pendingLive;
  if (!p || !S.task) return;
  S.pendingLive = null;
  const t = S.task, ctx = t.goalCtx();
  if (p.newSession || !S.liveRef) S.liveRef = { pos: t.goal.pos.slice(), quat: t.goal.quat.slice() };
  const c = clampGoal(p.pos, p.quat, ctx);
  t.setGoalLive(c.pos, c.quat, p.newSession);
  S.liveGoalId = t.goal.id;
  const chk = goalCheck(c.pos, c.quat, ctx, S.liveRef.pos, S.liveRef.quat);
  S.liveNote = c.clamped ? { kind: "hard", text: `Clamped: ${c.notes.join("; ")}` }
    : !chk.clean ? { kind: "soft", text: `Out of training range: ${chk.soft.join("; ")}` } : null;
}

function applyNudge(action) {
  const d = ensureDraft();
  const [p, q] = nudge(d.pos, d.quat, S.task.geo.centroid, action, Number($("step-m").value), Number($("step-deg").value));
  S.draft = { pos: p, quat: q };
  draftChanged(true);
}

let dragStart = null;
function onGizmoDrag(on) {
  if (on) {
    const d = ensureDraft();
    const [c0] = S.view.pivotPose();
    dragStart = { pos: d.pos.slice(), quat: d.quat.slice(), c0, first: true };
  } else {
    dragStart = null;
    S.view.resetPivotRotation();
  }
}

function onGizmoChange() {
  if (!dragStart) return;
  const [c, r] = S.view.pivotPose();
  S.draft = { pos: add(c, qrot(r, sub(dragStart.pos, dragStart.c0))), quat: qnorm(mulq(r, dragStart.quat)) }; // root = c + r (p0 - c0)
  draftChanged(dragStart.first);
  dragStart.first = false;
}

function mulq(a, b) {
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw];
}

function draftFromFields() {
  const pos = ["gx", "gy", "gz"].map((id) => Number($(id).value) / 100);
  const quat = quatFromRpyDeg(["groll", "gpitch", "gyaw"].map((id) => Number($(id).value)));
  if (pos.some((v) => !Number.isFinite(v))) return;
  S.draft = { pos, quat };
  draftChanged(true);
}

function send() {
  if (S.goalMode !== "send" || !S.draft) return;
  const chk = S.task.sendGoal(S.draft.pos, S.draft.quat, S.allowSoft);
  if (!chk.ok) { flashCheck(`Refused: ${chk.hard.join("; ")}`, "hard"); return; }
  if (!chk.clean && !S.allowSoft) { flashCheck(`Needs confirmation: ${chk.soft.join("; ")}`, "soft"); return; }
  S.draft = null;
  flashCheck("Goal sent", "ok");
}

let flash = null;
function flashCheck(text, kind) { flash = { text, kind, until: performance.now() + 2500 }; }

function setGoalMode(mode) {
  S.goalMode = mode;
  S.draft = null; S.pendingLive = null; S.liveNote = null;
  $("goal-mode").value = mode;
  showBanner(mode === "live" ? "Live goal: the goal follows the gizmo and keys" : "Send mode: edit a draft, then Enter / Send", 1800);
}

function randomGoalNow() {
  const g = S.task.randomGoal();
  S.task.sendGoal(g.pos, g.quat, true);
  S.draft = null; S.pendingLive = null; S.liveNote = null;
}

function randomPointNow() {
  S.task.sendPoint(Math.floor(Math.random() * S.task.geo.poolP.length), { coupled: $("coupled").checked });
}

// ---- modes ------------------------------------------------------------------------------------------------------------------
function setPlaying(on) { S.playing = on; S.acc = 0; $("play").textContent = on ? "Pause" : "Resume"; }

async function singleStep() {
  if (S.playing || S.busy) return;
  S.busy = true;
  try { await controlStep(); S.simAdvanced += S.sim.controlDt; } finally { S.busy = false; }
}

function setGizmo(on, mode) {
  S.gizmoOn = on;
  if (mode) S.gizmoMode = mode;
  $("gizmo-mode").textContent = S.gizmoMode === "translate" ? "Move" : "Rotate";
  if (!on) S.view.setGizmo(false);
}

function setPointMode(on) {
  S.pointMode = on;
  S.hover = null;
  $("view").classList.toggle("point-mode", on);
}

function setPushMode(on) { S.dragPush = on; $("drag-push").checked = on; }

function setEpisodeEnd(mode) {
  S.episodeEnd = mode;
  S.task.episodeEnd = mode;
  $("episode-end").value = mode;
  const txt = { train: "reset like training (20 s without a goal, switch drop, 50 goals, fall)", refresh: "memory refresh on the same triggers (keeps the scene)", off: "none: one endless episode (falls still reset)" };
  showBanner(`Episode end: ${txt[mode]}`, 2200);
}

function cycleObject(step) {
  const names = S.spec.objects.map((o) => o.name);
  const i = (names.indexOf(S.objectName) + step + names.length) % names.length;
  switchObject(names[i]);
}

const REASONS = { timeout: "no goal for 20 s", switch_drop: "drop during an in-hand switch", max_successes: "50 goals reached", fall: "object fell", hand_far: "hand too far" };
const reasonText = (why) => why.split(", ").map((r) => REASONS[r] || r).join(", ");

function showBanner(text, ms = 2600) {
  const b = $("banner");
  b.textContent = text;
  b.hidden = false;
  clearTimeout(S.bannerTimer);
  S.bannerTimer = setTimeout(() => { b.hidden = true; }, ms);
}

function showLegend(on) {
  clearTimeout(S.legendTimer);
  $("legend").hidden = !on;
}

// ---- UI wiring --------------------------------------------------------------------------------------------------------------
function wireUi() {
  $("play").onclick = () => setPlaying(!S.playing);
  $("step").onclick = singleStep;
  $("reset").onclick = () => S.task.reset("manual");
  $("speed").onchange = (e) => { S.speed = Number(e.target.value); };
  $("conditions").onchange = (e) => S.task.setConditions(e.target.value);
  $("hand-drive").disabled = !S.sceneParams?.handPD;
  $("hand-drive").value = S.handDrive;
  $("hand-drive").onchange = (e) => {
    S.handDrive = e.target.value;
    S.sim.setHandDrive(S.handDrive, S.sceneParams.handPD);
    S.task.setConditions(S.task.conditions); // re-applies the real-pass hand delay (n = 2) to the new drive
    S.task.log(`hand drive: ${S.handDrive === "isaac_pd" ? "Isaac delayed PD" : "position servo"}`);
  };
  $("policy").onchange = (e) => {
    S.policy = S.policies[e.target.value];
    S.policy.reset();
    fillPolicySelect(e.target.value);
    S.task.log(`policy: ${S.policy.name}`);
  };
  $("object").onchange = (e) => switchObject(e.target.value);
  $("goal-mode").value = S.goalMode;
  $("goal-mode").onchange = (e) => setGoalMode(e.target.value);
  $("gizmo-toggle").onclick = () => setGizmo(!S.gizmoOn);
  $("gizmo-mode").onclick = () => setGizmo(true, S.gizmoMode === "translate" ? "rotate" : "translate");
  $("goal-random").onclick = randomGoalNow;
  for (const b of document.querySelectorAll(".nudges button")) b.onclick = () => applyNudge(b.dataset.n);
  for (const id of ["gx", "gy", "gz", "groll", "gpitch", "gyaw"]) $(id).addEventListener("change", draftFromFields);
  $("allow-soft").onchange = (e) => { S.allowSoft = e.target.checked; };
  $("show-box").onchange = (e) => { S.showBox = e.target.checked; };
  $("send").onclick = send;
  $("discard").onclick = () => { S.draft = null; };
  $("point-mode").onclick = () => setPointMode(!S.pointMode);
  $("point-random").onclick = randomPointNow;
  $("coupled").onchange = (e) => { S.task.coupled = e.target.checked; };
  $("after-success").onchange = (e) => { S.task.afterSuccess = e.target.value; };
  $("keep-reset").onchange = (e) => { S.task.keepAfterReset = e.target.checked; };
  $("episode-end").value = S.episodeEnd;
  $("episode-end").onchange = (e) => setEpisodeEnd(e.target.value);
  $("drag-push").onchange = (e) => setPushMode(e.target.checked);
  $("kick").onclick = () => S.task.kick(0.3);
  $("tb-live").onclick = () => setGoalMode(S.goalMode === "live" ? "send" : "live");
  $("tb-point").onclick = () => setPointMode(!S.pointMode);
  $("tb-push").onclick = () => setPushMode(!S.dragPush);
  $("tb-gizmo").onclick = () => setGizmo(!S.gizmoOn);
  $("tb-follow").onclick = () => { S.view.follow = !S.view.follow; };
  $("tb-home").onclick = () => { S.view.follow = false; S.view.resetCamera(); };
  $("tb-keys").onclick = () => showLegend($("legend").hidden);
  $("legend-close").onclick = () => showLegend(false);
  $("panel-toggle").onclick = () => {
    const c = $("app").classList.toggle("collapsed");
    $("panel-toggle").textContent = c ? "⟨" : "⟩";
    setTimeout(() => S.view.resize(), 0);
  };
  $("help-open").onclick = (e) => { e.preventDefault(); showLegend($("legend").hidden); };
  wirePointer();
  window.addEventListener("keydown", onKey);
}

function onKey(e) {
  const tag = e.target?.tagName;
  if ((tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") && e.key !== "Escape") return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  const nudgeKeys = { ArrowLeft: "x-", ArrowRight: "x+", ArrowUp: "z+", ArrowDown: "z-", "[": "y-", "]": "y+",
    q: "yaw-", e: "yaw+", i: "pitch+", k: "pitch-", j: "roll-", l: "roll+" };
  const lk = k.length === 1 ? k.toLowerCase() : k;
  let handled = true;
  if (k === " ") setPlaying(!S.playing);
  else if (k === ".") singleStep();
  else if (k === "Backspace") S.task.reset("manual");
  else if (nudgeKeys[lk]) applyNudge(nudgeKeys[lk]);
  else if (k === "Enter") send();
  else if (k === "Escape") { if (S.pointMode) setPointMode(false); else S.draft = null; showLegend(false); }
  else if (lk === "v") setGoalMode(S.goalMode === "live" ? "send" : "live");
  else if (lk === "p") { if (e.shiftKey) randomPointNow(); else setPointMode(!S.pointMode); }
  else if (lk === "r") randomGoalNow();
  else if (lk === "m") setPushMode(!S.dragPush);
  else if (lk === "b") S.task.kick(0.3);
  else if (lk === "z") setEpisodeEnd({ train: "refresh", refresh: "off", off: "train" }[S.episodeEnd]);
  else if (lk === "n") cycleObject(e.shiftKey ? -1 : 1);
  else if (k === "0" || k === "Home") { S.view.follow = false; S.view.resetCamera(); }
  else if (lk === "f") S.view.follow = !S.view.follow;
  else if (lk === "x") S.view.showAxes = !S.view.showAxes;
  else if (lk === "t") setGizmo(true, S.gizmoMode === "translate" ? "rotate" : "translate");
  else if (lk === "g") setGizmo(!S.gizmoOn);
  else if (lk === "c") { $("coupled").checked = !$("coupled").checked; S.task.coupled = $("coupled").checked; }
  else if (lk === "h" || k === "?") showLegend($("legend").hidden);
  else if (lk === "w") S.task.teleportObject(0, 0.02);
  else if (lk === "s") S.task.teleportObject(0, -0.02);
  else if (lk === "a") S.task.teleportObject(-0.02, 0);
  else if (lk === "d") S.task.teleportObject(0.02, 0);
  else handled = false;
  if (handled) e.preventDefault();
}

// ---- mouse: orbit (default), push (Shift / push mode), point mode (hover preview, click / drag applies) ------------------------
function pointUnder(x, y) {
  const hit = S.view.pick(x, y);
  if (!hit) return null;
  const [pos, quat] = hit.target === "object" ? S.sim.objectPose() : [S.task.goal.pos, S.task.goal.quat];
  return { index: nearestIndex(S.task.geo.poolP, qrotInv(quat, sub(hit.point, pos))), target: hit.target };
}

function applyPoint(index) {
  if (index == null || index === S.task.point?.index) return;
  S.task.sendPoint(index, { coupled: $("coupled").checked });
  S.lastPointApply = performance.now();
}

function wirePointer() {
  const canvas = $("view");
  let down = null;
  canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    if (S.view.gizmo?.enabled && S.view.gizmo.axis) return; // the gizmo takes this drag
    down = { x: e.clientX, y: e.clientY, kind: null };
    if (S.pointMode) {
      const h = pointUnder(e.clientX, e.clientY);
      if (h) { // a press on the object in point mode: slide the point; the camera stays put
        S.hover = { ...h, since: performance.now() };
        down.kind = "point";
        S.view.controls.enabled = false;
        canvas.setPointerCapture(e.pointerId);
        e.preventDefault();
        return;
      }
    }
    if (e.shiftKey || S.dragPush) {
      const hit = S.view.pick(e.clientX, e.clientY, { ghost: false });
      if (hit && hit.target === "object") {
        S.sim.startPush(S.sim.objBody, hit.point);
        down.kind = "push";
        S.view.controls.enabled = false;
        canvas.setPointerCapture(e.pointerId);
        e.preventDefault();
      }
    }
  }, { capture: true });
  canvas.addEventListener("pointermove", (e) => {
    if (down?.kind === "push" && S.sim.perturb) {
      const p = S.view.dragPoint(e.clientX, e.clientY, S.sim.perturb.target);
      if (p) S.sim.perturb.target = p;
      return;
    }
    if (!S.pointMode) return;
    const h = pointUnder(e.clientX, e.clientY);
    if (!h) { if (down?.kind !== "point") S.hover = null; return; }
    if (!S.hover || S.hover.index !== h.index) S.hover = { ...h, since: performance.now() };
    // sliding with the button down: apply at a low rate once the point under the cursor has been stable for 250 ms
    if (down?.kind === "point" && performance.now() - S.hover.since > 250 && performance.now() - (S.lastPointApply || 0) > 330) applyPoint(S.hover.index);
  });
  const up = (e) => {
    if (!down) return;
    const kind = down.kind;
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    down = null;
    if (kind === "push") { S.sim.endPush(); S.view.controls.enabled = true; try { canvas.releasePointerCapture(e.pointerId); } catch { /* released */ } return; }
    if (kind === "point") {
      S.view.controls.enabled = true;
      try { canvas.releasePointerCapture(e.pointerId); } catch { /* released */ }
      if (S.hover) applyPoint(S.hover.index); // a click, or the end of a slide
      return;
    }
    if (S.pointMode && moved <= 5) { const h = pointUnder(e.clientX, e.clientY); if (h) applyPoint(h.index); }
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("pointerleave", () => { if (!down) S.hover = null; });
}

function wireTheme() {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const apply = () => {
    const t = document.documentElement.getAttribute("data-theme");
    S.view.setTheme(t ? t === "dark" : mq.matches);
  };
  mq.addEventListener("change", apply);
  new MutationObserver(apply).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  apply();
}

// ---- readouts ----------------------------------------------------------------------------------------------------------------
const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : "–");

function updateUi() {
  const t = S.task, r = t.readouts();
  // HUD
  const ph = $("hud-phase");
  ph.textContent = `phase ${r.phase}${r.regrasp ? " (regrasp)" : ""}`;
  ph.className = `chip phase-${r.phase}`;
  $("hud-held").textContent = r.held ? "held" : r.lifted ? "lifted, not airborne" : "resting";
  $("hud-held").className = `chip ${r.held ? "good" : ""}`;
  const near = r.metric <= r.tol;
  $("hud-err").textContent = `${fmt(100 * r.posErr)} cm · ${fmt(r.rotErrDeg, 0)}° · metric ${fmt(100 * r.metric)} / ${fmt(100 * r.tol)} cm`;
  $("hud-err").className = `chip ${near ? "good" : ""}`;
  const pl = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  $("hud-succ").textContent = `${pl(r.totals.goalsReached, "goal")} reached · ${pl(r.totals.drops, "drop")}`;
  const note = $("hud-note");
  note.hidden = !(S.goalMode === "live" && S.liveNote);
  if (S.liveNote) { note.textContent = S.liveNote.text; note.className = `chip ${S.liveNote.kind === "hard" ? "bad" : "warn"}`; }
  const wall = S.rt.reduce((s, x) => s + x[0], 0), sim = S.rt.reduce((s, x) => s + x[1], 0);
  const rtf = wall > 0 ? sim / wall : 0;
  const slow = S.playing && rtf < 0.9 * S.speed && wall > 0.5;
  $("hud-rt").hidden = !slow;
  if (slow) $("hud-rt").textContent = `${rtf.toFixed(2)}× real time (device slower than real time)`;
  // toolbar states + mode hint
  const tb = { "tb-live": S.goalMode === "live", "tb-point": S.pointMode, "tb-push": S.dragPush, "tb-gizmo": S.gizmoOn, "tb-follow": S.view.follow,
    "tb-keys": !$("legend").hidden, "gizmo-toggle": S.gizmoOn, "point-mode": S.pointMode };
  for (const [id, on] of Object.entries(tb)) $(id).classList.toggle("on", on);
  $("tb-live").firstChild.textContent = S.goalMode === "live" ? "Live goal " : "Send mode ";
  const hint = $("mode-hint");
  hint.hidden = !S.pointMode && !S.dragPush;
  hint.textContent = S.pointMode ? "Point mode: hover the object, click to apply, drag to slide · Esc / P to leave"
    : "Push mode: drag the object to push it · M to leave";
  // goal fields show the draft (or the active goal) unless being edited
  const g = S.draft || t.goal;
  if (!["gx", "gy", "gz", "groll", "gpitch", "gyaw"].includes(document.activeElement?.id)) {
    const e = rpyDeg(g.quat);
    $("gx").value = (100 * g.pos[0]).toFixed(1); $("gy").value = (100 * g.pos[1]).toFixed(1); $("gz").value = (100 * g.pos[2]).toFixed(1);
    $("groll").value = e[0].toFixed(0); $("gpitch").value = e[1].toFixed(0); $("gyaw").value = e[2].toFixed(0);
  }
  // goal check / note
  const live = S.goalMode === "live";
  $("send-row").hidden = live;
  $("allow-soft-row").hidden = live;
  const chkEl = $("check");
  if (flash && performance.now() < flash.until) {
    chkEl.textContent = flash.text; chkEl.className = `check small ${flash.kind}`;
  } else if (live) {
    if (S.liveNote) { chkEl.textContent = S.liveNote.text; chkEl.className = `check small ${S.liveNote.kind}`; }
    else { chkEl.textContent = "Live goal: drag the gizmo (G) or use the keys; the policy follows at once."; chkEl.className = "check small muted"; }
  } else if (S.draft) {
    const c = goalCheck(S.draft.pos, S.draft.quat, t.goalCtx(), t.goal.pos, t.goal.quat);
    if (!c.ok) { chkEl.textContent = `HARD: ${c.hard.join("; ")}`; chkEl.className = "check small hard"; }
    else if (!c.clean) { chkEl.textContent = `SOFT: ${c.soft.join("; ")}${S.allowSoft ? " (allowed)" : ". Tick the box to send."}`; chkEl.className = "check small soft"; }
    else { chkEl.textContent = `OK: ${fmt(100 * c.lowestAboveSurface)} cm above the table, ${fmt(c.rotFromRefDeg, 0)}° from the active goal`; chkEl.className = "check small ok"; }
    $("send").disabled = !(c.ok && (c.clean || S.allowSoft));
  } else {
    chkEl.textContent = "Send mode: nudge, type or drag the gizmo to edit a draft, then Enter."; chkEl.className = "check small muted";
    $("send").disabled = true;
  }
  $("discard").disabled = !S.draft;
  const pt = r.point;
  const ptName = pt ? (pt.index != null ? `#${pt.index}` : "drawn") : "–";
  $("point-info").textContent = pt ? `${ptName} (${pt.source})${r.acquired ? " · acquired" : r.near ? " · near" : ""}` : "–";
  // readouts table
  const G = r.goal;
  const rows = [
    ["phase", `${r.phase}${r.regrasp ? " · regrasp" : ""}`, r.phase === "G" ? "good" : r.phase === "A" ? "warn" : ""],
    ["held / lifted", `${r.held ? "held" : "not held"} / ${r.lifted ? "lifted" : "not lifted"}`, r.held ? "good" : ""],
    ["point", pt ? `${ptName} ${pt.source} · ${r.acquired ? "acquired" : "not acquired"} · ${fmt(100 * r.dPoint)} cm` : "–", r.acquired ? "good" : ""],
    ["episode record", `lift ${r.record.lifted} · acq ${r.totals.acquisitions} · switches ${r.record.switches} · deadlines ${r.record.deadlines} · losses ${r.record.losses}`],
    ["goal", `#${G.id} ${G.source} · ${fmt(t.T - G.t0, 0)} s${G.reached != null ? ` · reached ${fmt(G.reached)} s` : ""}`, G.reached != null ? "good" : ""],
    ["position error", `${fmt(100 * r.posErr)} cm`],
    ["rotation error", `${fmt(r.rotErrDeg)}°`],
    ["metric / tol", `${fmt(100 * r.metric, 2)} / ${fmt(100 * r.tol)} cm (best ${fmt(100 * G.bestMetric, 2)})`, near ? "good" : ""],
    ["2 cm precision", r.metric <= (t.cfg.precision_m ?? 0.02) ? "within 2 cm" : "no", r.metric <= (t.cfg.precision_m ?? 0.02) ? "good" : ""],
    ["dwell", `${r.dwell} / ${t.cfg.success_steps}`],
    ["no-goal clock", `${fmt((r.noProgress * S.sim.controlDt), 1)} / ${fmt(r.maxSteps * S.sim.controlDt, 0)} s (${S.episodeEnd})`, r.noProgress > 0.75 * r.maxSteps ? "warn" : ""],
    ["goals reached", `${r.totals.goalsReached} (success events: episode ${r.successes}, session ${r.totals.goals})`],
    ["drops / falls", `${r.totals.drops} / ${r.totals.falls}`, r.totals.drops ? "warn" : ""],
    ["episodes / refreshes", `${r.totals.episodes} / ${r.totals.refreshes}`],
    ["episode time", `${fmt(r.tEp)} s`],
    ["sim time", `${fmt(r.T)} s`],
    ["real-time factor", `${fmt(rtf, 2)}× (speed ${S.speed}×)`, slow ? "warn" : ""],
    ["policy / physics", `${fmt(S.policy.ms, 1)} ms / ${fmt(S.sim.stats.physMs, 1)} ms per step (${S.sim.nSub} substeps)`],
    ["render", `${fmt(S.fps, 0)} fps`],
    ["conditions", t.conditions],
  ];
  $("readouts").innerHTML = rows.map(([k, v, c]) => `<tr><td>${k}</td><td class="${c || ""}">${v}</td></tr>`).join("");
  const ev = $("events");
  const txt = r.events.slice().reverse().join("\n");
  if (ev.textContent !== txt) ev.textContent = txt;
}

boot();
