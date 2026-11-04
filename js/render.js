// three.js view of the MuJoCo scene: one mesh per visible MuJoCo geom (geometry from the compiled model: mesh_vert / mesh_face,
// or a primitive), posed every frame from geom_xpos / geom_xmat. Plus the active goal ghost (green), the draft goal ghost
// (orange, TransformControls gizmo in world axes, pivot at the object centroid), the affordance point + normal, the palm surface
// centre, the goal box, axes and the push spring. World is z-up.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

const COL = { goal: 0x3cc85a, draft: 0xff9628, point: 0xff32be, normal: 0x1edcf0, palm: 0xffd71e, box: 0x7878ff, push: 0xe0402f };
const GEOM = { PLANE: 0, HFIELD: 1, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7 };

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export class View {
  constructor(canvas) {
    this.canvas = canvas;
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" }));
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFShadowMap;
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.0;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 30);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.95, -0.9, 1.22);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0.0, 0.12, 0.66);
    this.home = { pos: this.camera.position.clone(), target: this.controls.target.clone() };
    this.follow = false;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.minDistance = 0.15;
    this.controls.maxDistance = 4;
    this.controls.update();

    const pmrem = new THREE.PMREMGenerator(r);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.45;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8c93a0, 0.8));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(0.9, -0.7, 2.2);
    key.target.position.set(0, 0.1, 0.5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, { left: -0.9, right: 0.9, top: 0.9, bottom: -0.9, near: 0.3, far: 4 });
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.01;
    this.scene.add(key, key.target);
    const rim = new THREE.DirectionalLight(0xdfe8ff, 0.6);
    rim.position.set(-1.2, 1.0, 1.0);
    this.scene.add(rim);

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.meshes = [];
    this.raycaster = new THREE.Raycaster();
    this._makeOverlays();
    this.showAxes = false;

    this._resize = () => this.resize();
    new ResizeObserver(this._resize).observe(canvas);
    this.resize();
  }

  _makeOverlays() {
    const s = this.scene;
    const basic = (c, opts = {}) => new THREE.MeshBasicMaterial(Object.assign({ color: c, depthTest: false, transparent: true }, opts));
    this.pointDot = new THREE.Mesh(new THREE.SphereGeometry(0.006, 16, 12), basic(COL.point));
    this.pointDot.renderOrder = 20;
    this.normalArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.04, COL.normal, 0.012, 0.007);
    this.normalArrow.renderOrder = 20;
    for (const ch of this.normalArrow.children) { ch.material.depthTest = false; ch.material.transparent = true; ch.renderOrder = 20; }
    this.draftDot = new THREE.Mesh(new THREE.SphereGeometry(0.0055, 16, 12), basic(COL.draft));
    this.draftDot.renderOrder = 21;
    this.draftDot.visible = false;
    this.previewArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.04, COL.draft, 0.012, 0.007);
    this.previewArrow.renderOrder = 21;
    for (const ch of this.previewArrow.children) { ch.material.depthTest = false; ch.material.transparent = true; ch.renderOrder = 21; }
    this.previewArrow.visible = false;
    this.palmDot = new THREE.Mesh(new THREE.SphereGeometry(0.005, 12, 10), basic(COL.palm, { opacity: 0.9 }));
    this.palmDot.renderOrder = 19;
    s.add(this.pointDot, this.normalArrow, this.draftDot, this.previewArrow, this.palmDot);

    const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    this.pushLine = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: COL.push, depthTest: false, transparent: true }));
    this.pushLine.renderOrder = 22;
    this.pushDot = new THREE.Mesh(new THREE.SphereGeometry(0.006, 16, 12), basic(COL.push));
    this.pushDot.renderOrder = 22;
    this.pushLine.visible = this.pushDot.visible = false;
    s.add(this.pushLine, this.pushDot);

    this.boxHelper = new THREE.Box3Helper(new THREE.Box3(), COL.box);
    this.boxHelper.material.transparent = true;
    this.boxHelper.material.opacity = 0.55;
    this.boxHelper.visible = false;
    s.add(this.boxHelper);

    this.objAxes = new THREE.AxesHelper(0.08);
    this.goalAxes = new THREE.AxesHelper(0.08);
    for (const a of [this.objAxes, this.goalAxes]) { a.material.depthTest = false; a.renderOrder = 18; a.visible = false; s.add(a); }
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Build meshes for a compiled scene. */
  load(sim, spec) {
    for (const m of this.meshes) { m.geometry.dispose(); m.material.dispose(); this.root.remove(m); }
    this.meshes = [];
    for (const g of [this.goalGhost, this.draftGhost]) if (g) { this.scene.remove(g); g.traverse((o) => { o.geometry?.dispose(); }); }
    if (this.gizmo) { this.gizmo.detach(); this.scene.remove(this.gizmo.getHelper()); this.gizmo.dispose(); this.gizmo = null; }
    this.sim = sim;
    const m = sim.m;
    const vis = new Set(spec.render.visual_groups), hide = new Set(spec.render.hidden_groups);
    const geoCache = new Map();
    const objGeoms = [];
    for (let g = 0; g < m.ngeom; g++) {
      const grp = m.geom_group[g];
      if (hide.has(grp) || !vis.has(grp)) continue;
      const rgba = this._rgba(m, g);
      if (rgba[3] < 0.01) continue;
      const type = m.geom_type[g];
      let geo;
      if (type === GEOM.MESH) {
        const id = m.geom_dataid[g];
        if (!geoCache.has(id)) geoCache.set(id, this._meshGeometry(m, id));
        geo = geoCache.get(id);
      } else {
        geo = this._primitive(type, m.geom_size.subarray(3 * g, 3 * g + 3));
        if (!geo) continue;
      }
      const isPlane = type === GEOM.PLANE;
      const mat = isPlane
        ? new THREE.ShadowMaterial({ opacity: 0.16 })
        : new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace),
          roughness: 0.62, metalness: 0.05, transparent: rgba[3] < 1, opacity: rgba[3] });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = !isPlane;
      mesh.receiveShadow = true;
      const body = m.geom_bodyid[g];
      mesh.userData = { geom: g, body, isObj: body === sim.objBody };
      this.root.add(mesh);
      this.meshes.push(mesh);
      if (isPlane) {
        const grid = new THREE.GridHelper(4, 40, 0xc8ccd4, 0xdde0e6);
        grid.rotation.x = Math.PI / 2;
        grid.material.transparent = true;
        grid.material.opacity = 0.35;
        mesh.add(grid);
      }
      if (body === sim.objBody) objGeoms.push({ g, geo, rgba });
    }
    this.objMeshes = this.meshes.filter((x) => x.userData.isObj);
    this.goalGhost = this._ghost(m, objGeoms, COL.goal, 0.28);
    this.draftGhost = this._ghost(m, objGeoms, COL.draft, 0.32);
    this.draftGhost.visible = false;
    this.scene.add(this.goalGhost, this.draftGhost);
    // gizmo: a pivot at the object centroid; the draft ghost follows it
    this.pivot = new THREE.Object3D();
    this.scene.add(this.pivot);
    this.gizmo = new TransformControls(this.camera, this.canvas);
    this.gizmo.setSpace("world");
    this.gizmo.setSize(0.75);
    this.gizmo.attach(this.pivot);
    this.gizmo.getHelper().visible = false;
    this.gizmo.enabled = false;
    this.scene.add(this.gizmo.getHelper());
    this.gizmo.addEventListener("dragging-changed", (e) => { this.controls.enabled = !e.value; this.onGizmoDrag?.(e.value); });
    this.gizmo.addEventListener("objectChange", () => this.onGizmoChange?.());
  }

  _rgba(m, g) {
    const mat = m.geom_matid[g];
    const src = mat >= 0 ? m.mat_rgba.subarray(4 * mat, 4 * mat + 4) : m.geom_rgba.subarray(4 * g, 4 * g + 4);
    // default MuJoCo grey for geoms left at rgba 0.5 0.5 0.5 1
    return Array.from(src);
  }

  _meshGeometry(m, id) {
    const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
    const pos = m.mesh_vert.slice(3 * va, 3 * (va + vn));
    const idx = m.mesh_face.slice(3 * fa, 3 * (fa + fn));
    let g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pos), 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
    g = toCreasedNormals(g, Math.PI / 6);
    g.computeBoundingSphere();
    return g;
  }

  _primitive(type, s) {
    switch (type) {
      case GEOM.PLANE: return new THREE.PlaneGeometry(s[0] > 0 ? 2 * s[0] : 8, s[1] > 0 ? 2 * s[1] : 8);
      case GEOM.SPHERE: return new THREE.SphereGeometry(s[0], 32, 20);
      case GEOM.CAPSULE: return new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 24).rotateX(Math.PI / 2);
      case GEOM.ELLIPSOID: return new THREE.SphereGeometry(1, 32, 20).scale(s[0], s[1], s[2]);
      case GEOM.CYLINDER: return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 40).rotateX(Math.PI / 2);
      case GEOM.BOX: return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      default: return null;
    }
  }

  /** Translucent copy of the object's visual geoms, as children placed at their body-frame offsets (geom_pos / geom_quat). */
  _ghost(m, objGeoms, color, opacity) {
    const root = new THREE.Group();
    root.matrixAutoUpdate = false;
    const mat = new THREE.MeshStandardMaterial({ color, transparent: true, opacity, depthWrite: false, roughness: 0.5,
      emissive: color, emissiveIntensity: 0.25 });
    for (const { g, geo } of objGeoms) {
      const mesh = new THREE.Mesh(geo, mat);
      const p = m.geom_pos.subarray(3 * g, 3 * g + 3), q = m.geom_quat.subarray(4 * g, 4 * g + 4);
      mesh.position.set(p[0], p[1], p[2]);
      mesh.quaternion.set(q[1], q[2], q[3], q[0]);
      mesh.userData = { ghost: true };
      root.add(mesh);
    }
    return root;
  }

  setGoalBox(box) {
    if (!box) { this.boxHelper.visible = false; return; }
    this.boxHelper.box.min.set(...box.mins);
    this.boxHelper.box.max.set(...box.maxs);
  }

  static poseMatrix(pos, quat) {
    return new THREE.Matrix4().compose(new THREE.Vector3(pos[0], pos[1], pos[2]), new THREE.Quaternion(quat[1], quat[2], quat[3], quat[0]),
      new THREE.Vector3(1, 1, 1));
  }

  /** Per-frame sync. ov: {goal:{pos,quat}, draft:{pos,quat}|null, point:{p,n}|null, previewPoint:{p,n}|null (point-mode hover),
   * palm:[3]|null, push:{from,to}|null, showGoalBox} */
  draw(ov) {
    const sim = this.sim;
    if (sim) {
      const xp = sim.d.geom_xpos, xm = sim.d.geom_xmat;
      for (const mesh of this.meshes) {
        const g = mesh.userData.geom;
        const o = 9 * g;
        mesh.matrix.set(xm[o], xm[o + 1], xm[o + 2], xp[3 * g], xm[o + 3], xm[o + 4], xm[o + 5], xp[3 * g + 1], xm[o + 6], xm[o + 7], xm[o + 8],
          xp[3 * g + 2], 0, 0, 0, 1);
        mesh.matrixWorldNeedsUpdate = true;
      }
      const [op, oq] = sim.objectPose();
      const Mo = View.poseMatrix(op, oq);
      if (ov.goal) {
        this.goalGhost.matrix.copy(View.poseMatrix(ov.goal.pos, ov.goal.quat));
        this.goalGhost.matrixWorldNeedsUpdate = true;
        this.goalAxes.position.set(...ov.goal.pos);
        this.goalAxes.quaternion.set(ov.goal.quat[1], ov.goal.quat[2], ov.goal.quat[3], ov.goal.quat[0]);
      }
      this.draftGhost.visible = !!ov.draft;
      if (ov.draft) {
        this.draftGhost.matrix.copy(View.poseMatrix(ov.draft.pos, ov.draft.quat));
        this.draftGhost.matrixWorldNeedsUpdate = true;
      }
      this.objAxes.position.set(...op);
      this.objAxes.quaternion.set(oq[1], oq[2], oq[3], oq[0]);
      this.objAxes.visible = this.goalAxes.visible = this.showAxes;
      this.pointDot.visible = this.normalArrow.visible = !!ov.point;
      if (ov.point) {
        const p = new THREE.Vector3(...ov.point.p).applyMatrix4(Mo);
        const n = new THREE.Vector3(...ov.point.n).applyQuaternion(new THREE.Quaternion(oq[1], oq[2], oq[3], oq[0])).normalize();
        this.pointDot.position.copy(p);
        this.normalArrow.position.copy(p);
        this.normalArrow.setDirection(n);
      }
      this.draftDot.visible = this.previewArrow.visible = !!ov.previewPoint;
      if (ov.previewPoint) {
        const p = new THREE.Vector3(...ov.previewPoint.p).applyMatrix4(Mo);
        this.draftDot.position.copy(p);
        this.previewArrow.position.copy(p);
        this.previewArrow.setDirection(new THREE.Vector3(...ov.previewPoint.n).applyQuaternion(new THREE.Quaternion(oq[1], oq[2], oq[3], oq[0])).normalize());
      }
      if (this.follow) { // follow-object camera: glide the orbit target to the object, the camera keeps its offset
        const want = new THREE.Vector3(op[0], op[1], op[2] + 0.05);
        const move = want.sub(this.controls.target).multiplyScalar(0.12);
        this.controls.target.add(move);
        this.camera.position.add(move);
      }
      this.palmDot.visible = !!ov.palm;
      if (ov.palm) this.palmDot.position.set(...ov.palm);
      this.pushLine.visible = this.pushDot.visible = !!ov.push;
      if (ov.push) {
        const a = this.pushLine.geometry.attributes.position;
        a.setXYZ(0, ...ov.push.from); a.setXYZ(1, ...ov.push.to); a.needsUpdate = true;
        this.pushDot.position.set(...ov.push.to);
      }
      this.boxHelper.visible = !!ov.showGoalBox;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  _ndc(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    return new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
  }

  /** Raycast against the object (and the green goal ghost) only, so fingers in front of the object do not block a pick:
   * {target: 'object'|'goal', point world [3]} or null. */
  pick(clientX, clientY, { ghost = true } = {}) {
    this.raycaster.setFromCamera(this._ndc(clientX, clientY), this.camera);
    const cands = [...(this.objMeshes || [])];
    if (ghost && this.goalGhost) cands.push(...this.goalGhost.children);
    const h = this.raycaster.intersectObjects(cands, false)[0];
    if (!h) return null;
    const p = [h.point.x, h.point.y, h.point.z];
    return h.object.userData.ghost ? { target: "goal", point: p } : { target: "object", point: p, body: h.object.userData.body };
  }

  /** Where the mouse ray meets the camera-facing plane through `through`. */
  dragPoint(clientX, clientY, through) {
    this.raycaster.setFromCamera(this._ndc(clientX, clientY), this.camera);
    const n = new THREE.Vector3();
    this.camera.getWorldDirection(n);
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(n, new THREE.Vector3(...through));
    const out = new THREE.Vector3();
    return this.raycaster.ray.intersectPlane(plane, out) ? [out.x, out.y, out.z] : null;
  }

  /** Show / hide the gizmo at pivot (world) in mode 'translate' | 'rotate'. */
  setGizmo(on, pivotPos, mode) {
    if (!this.gizmo) return;
    this.gizmo.enabled = on;
    this.gizmo.getHelper().visible = on;
    if (mode) this.gizmo.setMode(mode);
    if (pivotPos && !this.gizmo.dragging) { this.pivot.position.set(...pivotPos); this.pivot.quaternion.identity(); this.pivot.updateMatrixWorld(); }
  }

  pivotPose() {
    const p = this.pivot.position, q = this.pivot.quaternion;
    return [[p.x, p.y, p.z], [q.w, q.x, q.y, q.z]];
  }

  resetPivotRotation() { this.pivot.quaternion.identity(); this.pivot.updateMatrixWorld(); }

  resetCamera() {
    this.camera.position.copy(this.home.pos);
    this.controls.target.copy(this.home.target);
    this.controls.update();
  }

  frameObject(pos) {
    this.controls.target.set(pos[0], pos[1], pos[2] + 0.08);
    this.controls.update();
  }

  setTheme(dark) {
    this.scene.background = null;
    this.renderer.setClearColor(new THREE.Color(cssVar(dark ? "--scene-bg-dark" : "--scene-bg", dark ? "#1b1e24" : "#eef0f3")), 1);
  }
}
