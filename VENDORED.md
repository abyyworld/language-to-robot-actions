# Reused code

This project builds on two earlier repositories by the same author rather than
rebuilding what they already do. Code was **copied** (not imported as a
dependency) because neither is published as a package, and because the ICM
package pins MuJoCo `<3.4` while this one runs on current MuJoCo. Every copied
file says where it came from in its docstring.

| File here | Source | Commit | Changes |
|---|---|---|---|
| `src/lang2act/stats.py` | [vla-evals](https://github.com/abyyworld/vla-evals) `src/policy_evals/stats.py` | `346c494` | Unchanged up to `required_episodes`. Added: Wilson interval, template-clustered paired bootstrap, McNemar sample size (`required_pairs`), Holm correction. |
| `tests/test_stats.py` (first half) | vla-evals `tests/test_stats.py` | `346c494` | Import path only. |
| `src/lang2act/compare.py` | vla-evals `src/policy_evals/compare.py` | `346c494` | Rewritten around the same rules: paired episodes only, interval plus exact test, "underpowered" with a sample-size estimate instead of a silent pass. The ship/no-ship gate became better / worse / equivalent / underpowered. |
| `src/lang2act/harness.py` (suite + pairing rule) | vla-evals `benchmark.py`, `conf/benchmarks/*.yaml` | `346c494` | Same idea (versioned suite file, identical seeds for every condition, per-episode records); new implementation for LLM planners with resumable JSONL output. |
| `src/lang2act/sim/ik.py` | [Interactive-correction-for-manipulation](https://github.com/abyyworld/Interactive-correction-for-manipulation) `src/icm/control/ik.py` | `72bfedf` | Unchanged. |
| `src/lang2act/sim/panda.py` | ICM `src/icm/envs/panda.py` | `72bfedf` | Import paths only. |
| `src/lang2act/sim/assets/fetch.py` | ICM `src/icm/envs/assets/fetch.py` | `72bfedf` | Docstring; same pinned MuJoCo Menagerie commit. |
| `src/lang2act/sim/scene.py` | ICM `src/icm/envs/assets/scene.py` | `72bfedf` | Table, cameras, Menagerie include, wrist camera, TCP site, gravity compensation and the MjSpec helpers kept. Object set replaced: any number of coloured blocks plus a fixed bowl, instead of one target, two distractors and a goal pad. |
| `src/lang2act/sim/world.py` | ICM `src/icm/envs/pick_place.py` | `72bfedf` | Kept: integrated-setpoint Cartesian control, contact + geometric grasp detection, `get_state`/`set_state` (now also restoring the solver warm start). Removed: phase tracker, reward, single target, cameras rig. Added: reachability query. |
| `src/lang2act/sim/skills.py` | ICM `src/icm/control/scripted.py` | `72bfedf` | Same P-controller, gains, timeouts and yaw folding, split into `pick` and `place` primitives. Added: aiming the held block rather than the TCP, slowed final descents, partial release opening. |
| `src/lang2act/__init__.py` (GL backend choice) | ICM `src/icm/__init__.py` | `72bfedf` | Unchanged logic. |

Not vendored: the Franka Panda model and meshes. `lang2act assets` fetches
them from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache-2.0) at the commit pinned in `fetch.py`.
