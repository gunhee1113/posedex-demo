# PoseDex sim demo

A trained xArm6 + Allegro hand policy reaching 6D object goals with a commanded palm contact point, running entirely in the
browser: MuJoCo (WASM) for physics, onnxruntime-web for the policy, three.js for rendering. No server-side computation.

Open the page, wait for the policy to lift the object, then drag the goal gizmo (G), press R for a random goal, or press P and
click the object to choose a new contact point. H shows all keys.

- Policy: PoseDex research checkpoint (fp32 ONNX, LSTM actor), trained in simulation; this page is an interactive preview,
  not a benchmark.
- Objects: 15 YCB objects (Calli et al., *The YCB Object and Model Set*, CC BY 4.0).

## Third-party software (vendored, unmodified)

| package | version | license |
|---|---|---|
| @mujoco/mujoco (Google DeepMind) | 3.16.0 | Apache-2.0 |
| onnxruntime-web (Microsoft) | 1.30.0 | MIT |
| three.js | 0.186.1 | MIT (`vendor/three/LICENSE`) |

See `vendor/VERSIONS.md` for sources and checksums.
