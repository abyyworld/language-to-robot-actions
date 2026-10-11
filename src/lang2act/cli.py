"""Command line: ``lang2act <command>``. Run ``lang2act -h`` for the list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .verifier import VERIFIER_MODES


def _cmd_assets(args) -> int:
    from .sim.assets import fetch

    return fetch.main(["--force"] if args.force else [])


def _cmd_tasks(args) -> int:
    from .tasks import TEMPLATES, make_task

    for tid, (cat, _) in TEMPLATES.items():
        task = make_task(tid, args.seed)
        print(f"{cat:<13} {tid:<20} {task.instruction}")
    return 0


def _cmd_run(args) -> int:
    from .episode import Condition
    from .harness import Suite, run_suite

    suite = Suite.load(args.suite)
    condition = Condition(
        planner=args.planner,
        verifier=args.verifier,
        max_repairs=args.max_repairs,
        prompt=args.prompt,
        effort=None if args.effort == "none" else args.effort,
        structured=not args.no_structured,
        sample=args.sample,
    )
    out = (
        args.out
        or f"runs/{suite.name}/{args.planner.replace(':', '_')}-{args.verifier}-{args.prompt}"
    )
    run = run_suite(
        suite,
        condition,
        out,
        workers=args.workers,
        limit=args.limit,
        label=args.label,
        cache_dir=None if args.no_cache else args.cache_dir,
        retry_errors=args.retry_errors,
    )
    scored = [r for r in run.records if r["stage"] != "api_error"]
    if scored:
        from .stats import wilson_interval

        k = sum(r["success"] for r in scored)
        ci = wilson_interval(k, len(scored))
        print(
            f"{run.name}: {k}/{len(scored)} = {ci.point:.1%} (95% CI {ci.low:.1%}-{ci.high:.1%})  -> {out}"
        )
    return 0


def _cmd_compare(args) -> int:
    from .compare import compare_runs, format_comparison
    from .harness import load_run

    cmp = compare_runs(
        load_run(args.baseline), load_run(args.candidate), by=args.by, margin=args.margin
    )
    print(json.dumps(cmp.to_dict(), indent=2) if args.json else format_comparison(cmp))
    return 0


def _cmd_report(args) -> int:
    from .harness import load_run
    from .report import build_report

    runs = [load_run(p) for p in args.runs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot = out.with_suffix(".png") if args.plot else None
    out.write_text(build_report(runs, margin=args.margin, plot_path=plot))
    print(f"wrote {out}" + (f" and {plot}" if plot and plot.exists() else ""))
    return 0


def _cmd_show(args) -> int:
    from .harness import load_run

    run = load_run(args.run)
    rec = run.by_key().get((args.template, args.seed))
    if rec is None:
        print("no such episode", file=sys.stderr)
        return 1
    print(f"[{rec['category']}] {rec['template']} seed={rec['seed']}  ->  {rec['stage']}")
    print(f"instruction: {rec['instruction']}")
    for o in rec["scene"]["objects"]:
        print(f"  {o['id']:<16} {o['position']}  {o.get('on', '')}")
    for rd in rec["rounds"]:
        print(
            f"round {rd['round']}: {rd.get('plan_text') or rd.get('parse_error') or rd['completion'].get('error')}"
        )
        for v in rd.get("violations", []):
            print(f"    ! {v['message']}")
    if rec["execution"]:
        for s in rec["execution"]["outcomes"]:
            print(f"  exec {s['step']:<28} {'ok' if s['ok'] else 'FAILED: ' + s['error']}")
    if rec["goal_detail"]:
        print(f"goal: {rec['goal_detail']}")
    return 0


def _cmd_demo(args) -> int:
    from . import planners
    from .episode import Condition, run_episode
    from .sim.world import TabletopWorld
    from .tasks import make_task

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("demo needs imageio: pip install 'lang2act[viz]'", file=sys.stderr)
        return 1
    task = make_task(args.template, args.seed)
    world = TabletopWorld(task.scene.to_sim_objects())
    world.reset()
    caption = _captioner(task.instruction, args.width)
    first = world.render(args.camera, args.width, args.height)
    motion: list = []
    counter = {"n": 0}

    def grab(w):
        counter["n"] += 1
        if counter["n"] % args.every == 0:
            motion.append(caption(w.render(args.camera, args.width, args.height), w.status))

    condition = Condition(planner=args.planner, verifier=args.verifier, prompt=args.prompt)
    planner = planners.make_planner(args.planner, prompt=args.prompt)
    rec = run_episode(task, planner, condition, on_step=grab, world=world)

    # The planning rounds happen before anything moves: show each draft and
    # what the verifier said about it over the opening frame.
    hold = int(args.fps * 2.5)
    frames = []
    for rd in rec["rounds"]:
        plan = rd.get("plan_text") or rd.get("parse_error") or "(no plan)"
        verdict = (
            "verifier: " + rd["violations"][0]["message"]
            if rd.get("violations")
            else ("verifier: ok" if args.verifier != "none" else "")
        )
        frames += [caption(first, f"plan {rd['round'] + 1}: {plan}", verdict)] * hold
    frames += motion
    outcome = "success" if rec["success"] else rec["stage"]
    final = caption(world.render(args.camera, args.width, args.height), f"result: {outcome}")
    frames += [final] * hold
    imageio.mimsave(args.out, frames, duration=1000 / args.fps, loop=0)
    print(
        f"{task.instruction}\n  plan: {rec['final_plan_text']}\n  -> {rec['stage']}  "
        f"({len(frames)} frames -> {args.out})"
    )
    return 0


def _captioner(instruction: str, width: int):
    """Draw the instruction and up to two status lines above each frame (needs Pillow)."""
    import textwrap

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=13)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    chars = max(20, width // 7)

    def caption(frame, status: str, detail: str = ""):
        lines = [(f'"{instruction}"', (240, 239, 236))]
        lines += [(s, (195, 194, 183)) for s in textwrap.wrap(status, chars)[:2]]
        lines += [(s, (235, 140, 110)) for s in textwrap.wrap(detail, chars)[:2]]
        header = 8 + 17 * 5
        img = Image.new("RGB", (width, frame.shape[0] + header), (24, 26, 30))
        img.paste(Image.fromarray(frame), (0, header))
        d = ImageDraw.Draw(img)
        for i, (text, colour) in enumerate(lines):
            d.text((8, 5 + 17 * i), text, fill=colour, font=font)
        return np.asarray(img)

    return caption


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="lang2act", description="Language to robot actions, verified."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("assets", help="fetch the pinned Franka Panda meshes (~33 MB)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=_cmd_assets)

    p = sub.add_parser("tasks", help="list task templates with a sample instruction")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_cmd_tasks)

    p = sub.add_parser("run", help="run a suite under one condition")
    p.add_argument("--suite", default="suites/core_v1.yaml")
    p.add_argument(
        "--planner",
        required=True,
        help="e.g. anthropic:claude-haiku-5-5, ollama:qwen2.5:7b, oracle, noisy:0.5",
    )
    p.add_argument("--verifier", choices=VERIFIER_MODES, default="symbolic")
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--prompt", default="basic", help="basic | fewshot")
    p.add_argument(
        "--effort", default="medium", help="Claude effort: low|medium|high|xhigh|max|none"
    )
    p.add_argument(
        "--no-structured", action="store_true", help="do not constrain output to the plan schema"
    )
    p.add_argument(
        "--sample", type=int, default=0, help="sample index (a new index draws fresh responses)"
    )
    p.add_argument("--out", help="run directory (default: runs/<suite>/<condition>)")
    p.add_argument("--label", help="name shown in reports")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="run at most this many new episodes")
    p.add_argument("--cache-dir", default=".cache/llm")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument(
        "--retry-errors", action="store_true", help="re-run episodes that hit API errors"
    )
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("compare", help="paired comparison of two runs")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--by", default="category", choices=["category", "template"])
    p.add_argument("--margin", type=float, default=0.05, help="equivalence margin")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_compare)

    p = sub.add_parser("report", help="markdown report over runs (first run is the baseline)")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="results/report.md")
    p.add_argument(
        "--plot", action="store_true", help="also write a per-category chart (needs matplotlib)"
    )
    p.add_argument("--margin", type=float, default=0.05)
    p.set_defaults(fn=_cmd_report)

    p = sub.add_parser("show", help="print one episode's trace")
    p.add_argument("run")
    p.add_argument("--template", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(fn=_cmd_show)

    p = sub.add_parser("demo", help="render one episode to a GIF (needs OpenGL and imageio)")
    p.add_argument("--template", default="bowl_then_stack")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--planner", default="oracle")
    p.add_argument("--verifier", choices=VERIFIER_MODES, default="symbolic")
    p.add_argument("--prompt", default="basic")
    p.add_argument("--camera", default="scene")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--every", type=int, default=2, help="keep every Nth control step")
    p.add_argument("--fps", type=float, default=20)
    p.add_argument("--out", default="demo.gif")
    p.set_defaults(fn=_cmd_demo)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
