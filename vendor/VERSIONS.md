# Vendored runtime files

Everything the page loads at runtime is served from this folder (relative to `index.html`): no CDN at runtime, so the page works
offline, on GitHub Pages and inside a claude.ai Artifact (whose CSP allows same-origin files only for wasm / data).
Versions are pinned and were at least two weeks old when vendored (2026-10-31). Files were copied unmodified from the npm tarballs.

| package | version | published | files | source | license |
|---|---|---|---|---|---|
| `@mujoco/mujoco` (Google DeepMind's official MuJoCo WASM bindings) | 3.16.0 | 2026-10-14 | `mujoco/mujoco.js`, `mujoco/mujoco.wasm` (single-threaded build: no SharedArrayBuffer / COOP-COEP needed) | https://registry.npmjs.org/@mujoco/mujoco/-/mujoco-3.16.0.tgz (sha1 348a418bc7fd007054ecd18723dc5fe09d6f86a2) | Apache-2.0 |
| `onnxruntime-web` | 1.30.0 | 2026-09-14 | `ort/ort.wasm.min.mjs` (wasm-only ESM bundle), `ort/ort-wasm-simd-threaded.mjs`, `ort/ort-wasm-simd-threaded.wasm` (runs single-threaded when the page is not cross-origin isolated; `ort.env.wasm.numThreads = 1`) | https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.30.0.tgz (sha1 b88b09117992e6f486321e931a826fd2f7775377) | MIT |
| `three` | 0.186.1 | 2026-09-24 | `three/three.module.js`, `three/three.core.js`, `three/addons/{controls/OrbitControls.js, controls/TransformControls.js, environments/RoomEnvironment.js, utils/BufferGeometryUtils.js}`, `three/LICENSE` | https://registry.npmjs.org/three/-/three-0.186.1.tgz (sha1 6d50f70c2c437f844179bbb56d6f5b774e1ca38a) | MIT |

MuJoCo 3.16.0 matches agent A's scene build (`mjcf/manifest.json` `"mujoco": "3.16.0"`). The bindings' API used here:
`MjModel.from_xml_path(path, MjVFS)`, `MjVFS.addBuffer`, `MjData`, `mj_step`, `mj_forward`, `mj_resetDataKeyframe`, `mj_name2id`,
live typed-array views (`qpos`, `qvel`, `ctrl`, `xfrc_applied`, `geom_xpos`, `geom_xmat`, `mocap_pos`, model arrays incl. `mesh_vert` /
`mesh_face` for rendering and `actuator_*` for the hand-drive switch). A VFS path with `../` (scenes/x.xml + meshdir ../assets) resolves.

## sha256

```
5d5d89ddd0857a6518c628fa502d42f99d6dd67ca240182f1068cd30a45465ac  mujoco/mujoco.js
e05ac96db7f45bd46e76fed6e3469bf9204329e093344d742a18bca2cc0e230e  mujoco/mujoco.wasm
219e6a1fc8a9938268d18efca3c91d310bd2f4a59bbd13744df5b2b7fc6cee3b  ort/ort.wasm.min.mjs
e13f7f94fc51b4ca72b12faeb1ee95f4ace6dfbc8939bc718aabdc0a27c4299b  ort/ort-wasm-simd-threaded.mjs
3398c10d07d229bd91b364548e130e0e51a8e5704b88c7c083ebbeb78842dee2  ort/ort-wasm-simd-threaded.wasm
9052042d676cb0fdc1ddfefe193053f34b7ac0513a616fdac4535d49987812ea  three/three.module.js
9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6  three/three.core.js
3d79d07ecb686b4e5d93232eedab255331c1beef711e13164eaa1f68655a5f2b  three/addons/controls/OrbitControls.js
151befe25bb0d68626f9a6b033625b7b0cf6848e39ba71623b7bed8e3df565d6  three/addons/controls/TransformControls.js
55f466192cc84298755a424c5e040345006b2ee1455589b3b54126c2ea4123f4  three/addons/environments/RoomEnvironment.js
9fb63427ce6641fa14fd0baff9cc4d1b5f9c3d85fd084bf2e90e803c44ec1797  three/addons/utils/BufferGeometryUtils.js
8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc  three/LICENSE
```

Check: `cd posedex/web_demo/app/vendor && sha256sum -c <(sed -n '/^```$/,/^```$/p' VERSIONS.md | grep -v '```')`
