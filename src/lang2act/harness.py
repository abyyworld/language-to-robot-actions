"""Run a suite under one condition and record every episode.

The rule the comparisons depend on, inherited from vla-evals: **every
condition sees the same episodes.** A suite is a list of templates and a seed
range; episode ``(template, seed)`` is the same scene and instruction for every
planner, prompt and verifier setting. That turns any two runs into matched
pairs.

Per-episode records go to ``<run>/episodes.jsonl`` as they finish, so a run
that dies (rate limits, a laptop lid) resumes where it stopped instead of
paying for the same calls twice. ``run.json`` pins the condition; resuming
into a directory recorded under a different condition is refused, because
silently mixing two conditions in one run is how a comparison stops meaning
anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import __version__
from .episode import Condition, run_episode
from .planners import make_planner
from .tasks import CATEGORIES, TEMPLATES, make_task, template_ids


@dataclass
class Suite:
    name: str
    seed_base: int = 0
    seeds_per_template: int = 10
    templates: list[str] = field(default_factory=list)

    @staticmethod
    def load(path: str | Path) -> Suite:
        data = yaml.safe_load(Path(path).read_text())
        templates = data.get("templates", "all")
        if templates == "all":
            templates = template_ids()
        elif isinstance(templates, dict) and "categories" in templates:
            templates = template_ids(templates["categories"])
        unknown = [t for t in templates if t not in TEMPLATES]
        if unknown:
            raise ValueError(f"unknown templates in {path}: {unknown}")
        return Suite(
            name=data["name"],
            seed_base=int(data.get("seed_base", 0)),
            seeds_per_template=int(data.get("seeds_per_template", 10)),
            templates=list(templates),
        )

    def episodes(self) -> list[tuple[str, int]]:
        return [
            (t, self.seed_base + i) for t in self.templates for i in range(self.seeds_per_template)
        ]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed_base": self.seed_base,
            "seeds_per_template": self.seeds_per_template,
            "templates": self.templates,
        }


@dataclass
class Run:
    path: Path
    meta: dict
    records: list[dict]

    @property
    def name(self) -> str:
        return self.meta.get("label") or self.path.name

    def by_key(self) -> dict[tuple[str, int], dict]:
        return {(r["template"], r["seed"]): r for r in self.records}


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def load_run(path: str | Path) -> Run:
    path = Path(path)
    meta = json.loads((path / "run.json").read_text())
    records = []
    episodes = path / "episodes.jsonl"
    if episodes.is_file():
        for line in episodes.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    order = {t: i for i, t in enumerate(TEMPLATES)}
    records.sort(key=lambda r: (order.get(r["template"], 1_000), r["seed"]))
    return Run(path, meta, records)


def run_suite(
    suite: Suite,
    condition: Condition,
    out_dir: str | Path,
    *,
    workers: int = 4,
    limit: int | None = None,
    label: str | None = None,
    cache_dir: str | None = ".cache/llm",
    retry_errors: bool = False,
    progress: bool = True,
) -> Run:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "run.json"
    planner = make_planner(
        condition.planner,
        prompt=condition.prompt,
        effort=condition.effort,
        structured=condition.structured,
        cache_dir=cache_dir,
    )
    meta = {
        "label": label or out.name,
        "suite": suite.to_dict(),
        "condition": condition.to_dict(),
        "planner": planner.describe(),
        "categories": list(CATEGORIES),
        "lang2act_version": __version__,
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if meta_path.is_file():
        previous = json.loads(meta_path.read_text())
        if previous["condition"] != meta["condition"] or previous["suite"] != meta["suite"]:
            raise SystemExit(
                f"{out} already holds a run with a different condition or suite.\n"
                f"  recorded: {previous['condition']}\n  requested: {meta['condition']}\n"
                "Use a new --out directory."
            )
        meta["started_at"] = previous["started_at"]
    meta_path.write_text(json.dumps(meta, indent=2))

    episodes_path = out / "episodes.jsonl"
    done: dict[tuple[str, int], dict] = {}
    if episodes_path.is_file():
        for line in episodes_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["template"], r["seed"])] = r
    if retry_errors:
        kept = {k: r for k, r in done.items() if r["stage"] != "api_error"}
        if len(kept) != len(done):
            episodes_path.write_text("".join(json.dumps(r) + "\n" for r in kept.values()))
        done = kept

    todo = [e for e in suite.episodes() if e not in done]
    if limit is not None:
        todo = todo[:limit]
    lock = threading.Lock()
    started = time.perf_counter()
    counts = {"done": 0, "success": 0, "errors": 0}

    def one(key: tuple[str, int]) -> dict:
        template, seed = key
        return run_episode(make_task(template, seed), planner, condition)

    if progress:
        print(
            f"[{meta['label']}] {len(todo)} episodes to run ({len(done)} already recorded)",
            flush=True,
        )
    with episodes_path.open("a") as fh, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, k): k for k in todo}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:  # a bug, not a planner failure: surface it
                raise RuntimeError(f"episode {key} crashed") from exc
            with lock:
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                counts["done"] += 1
                counts["success"] += record["success"]
                counts["errors"] += record["stage"] == "api_error"
                if progress and (counts["done"] % 10 == 0 or counts["done"] == len(todo)):
                    rate = counts["success"] / counts["done"]
                    el = time.perf_counter() - started
                    print(
                        f"  {counts['done']}/{len(todo)}  success {rate:.1%}  "
                        f"api errors {counts['errors']}  {el:.0f}s",
                        flush=True,
                    )
    if progress and counts["errors"]:
        print(
            f"  {counts['errors']} episodes hit API errors; rerun with --retry-errors to redo them.",
            flush=True,
        )
    return load_run(out)
