# CLAUDE.md

Orientation for an AI assistant (or a returning human) picking this up cold.

## What this is

An LLM turns an instruction plus a scene description into a plan
(`pick` / `place_on` / `place_at`). A verifier rejects impossible plans and
feeds the reasons back. The plan then runs in MuJoCo with per-step checks. A
harness runs a fixed, seeded suite (30 templates × 10 seeds) under different
conditions and compares runs pairwise. README.md has the full picture.

## Rules that keep the numbers meaningful

- **Never quote a number that is not produced from `runs/` by `lang2act report`.**
  No LLM results exist until someone runs them with an API key.
- **The verifier must never see the instruction or the goal.** It checks
  executability only; otherwise its "improvement" is a leak.
- **Prompt and grader share thresholds.** `predicates.py` defines "left of",
  "between", spacing; `planners/prompts.py` states them; `tests/test_prompts.py`
  fails if they drift. Change both or neither.
- **Suites are never edited in place.** Add `suites/core_v2.yaml` instead; old
  runs stay comparable only to runs on the same suite file.
- **The oracle must stay at 100% on the suite** (CI checks the smoke suite). If a
  change to the sim or skills drops it, LLM failures can no longer be told apart
  from motion failures.
- **No refusal fallbacks, explicit effort, record the served model.** See the
  docstring of `planners/anthropic_planner.py` for why.

## Landmines (inherited and new)

- Import `lang2act` before `mujoco` in scripts: the GL backend is chosen at
  import time (logic from the ICM repo). Headless Linux needs `libosmesa6` to render.
- `MjSpec` body lookup differs across MuJoCo versions: use `find_spec_body`.
- Gravity compensation must be set before compile (see `sim/scene.py`).
- Bowl slots are the four corners, 8.4 cm apart. A centre slot let the second
  block shove the first over the wall.
- Workspace starts at x = 0.36 m: closer to the base the forearm clips tall stacks.
- `skills.place` aims the *held block*, not the TCP; the grasp offset is ~7 mm.

## Commands

`make help`. The common ones: `make test`, `make lint`, `make ceiling`,
`make synthetic`, `make exp-verifier MODEL=...`, `make report`.
