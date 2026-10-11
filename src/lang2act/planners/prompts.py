"""Prompts. Versioned by name; a run records which one it used.

The numbers in the system prompt are the same thresholds ``predicates.py``
grades against and ``verifier.py`` enforces. The planner is told every rule it
is judged by: a planner that fails a spatial task should fail because it got
the geometry wrong, not because the grading used a definition it never saw.
"""

from __future__ import annotations

import json

from ..predicates import BETWEEN_TOL, MIN_CLEARANCE, SPATIAL_ALIGN, SPATIAL_MAX, SPATIAL_MIN
from ..verifier import BOWL_KEEP_OUT, MAX_STACK, MIN_BASE_DISTANCE

SYSTEM_BASIC = f"""You plan actions for a Franka Panda robot arm that moves blocks on a table. \
You are given the scene and an instruction, and you return a plan.

Frame and units
- Positions are in metres in the robot's frame. The robot base is at (0, 0) and it faces +x.
- +y is the robot's left and -y its right. "In front of" means closer to the robot \
(smaller x); "behind" means further from it (larger x).
- Blocks are 4.2 cm cubes. The bowl is fixed to the table and holds at most 4 blocks.

Actions (one JSON object per step; fields an action does not use are null)
- {{"action": "pick", "object": "<id>", "target": null, "x": null, "y": null}}
  Grasp a block. The gripper must be empty and nothing may be on top of the block.
- {{"action": "place_on", "object": null, "target": "<id>", "x": null, "y": null}}
  Put the held block on top of another block, or into the bowl with target "bowl".
- {{"action": "place_at", "object": null, "target": null, "x": <number>, "y": <number>}}
  Put the held block on the table at that point.
Every pick must be followed by a place before the next pick. The plan must end with the \
gripper empty. Choosing grasps, heights and where in the bowl a block goes is handled for you.

Rules every plan is checked against
- Use object ids exactly as they appear in the scene.
- The arm can only reach points between about {MIN_BASE_DISTANCE} m and 0.78 m from its base.
- Blocks on the table must be at least {MIN_CLEARANCE} m apart, centre to centre, and at least \
{BOWL_KEEP_OUT} m from the bowl's centre.
- Stacks are at most {MAX_STACK} blocks tall.

How spatial words are graded
- "A to the left of B": A ends up {SPATIAL_MIN}-{SPATIAL_MAX} m further left (+y) than B, \
within {SPATIAL_ALIGN} m of B along x. "Right of", "in front of" and "behind" are defined \
the same way along their own axes. Aim for a 0.12 m offset.
- "A between B and C": A on the table within {BETWEEN_TOL} m of the midpoint of B and C.
- "Leftmost" is the largest y, "rightmost" the smallest y; "closest to X" is the smallest \
horizontal distance to X.

If the instruction cannot be carried out with what is in the scene (an object it names is \
missing, out of reach, or cannot be moved), set "feasible" to false, say why in "reason", \
and return no steps. Never substitute a different object.

Respond with only a JSON object of this form, and nothing else:
{{"feasible": true, "reason": "<one short sentence>", "steps": [<step>, ...]}}"""


_FEWSHOT = """

Example 1
Scene: {"objects": [{"id": "bowl", "type": "bowl", "color": "white", "position": [0.5, 0.15, 0.0], "inner_radius": 0.11}, {"id": "orange_block", "type": "block", "color": "orange", "position": [0.45, -0.1, 0.021], "on": "table"}, {"id": "green_block", "type": "block", "color": "green", "position": [0.45, -0.1, 0.063], "on": "orange_block"}]}
Instruction: Put the orange block in the bowl.
Answer: {"feasible": true, "reason": "The green block is on the orange one, so move it aside first.", "steps": [{"action": "pick", "object": "green_block", "target": null, "x": null, "y": null}, {"action": "place_at", "object": null, "target": null, "x": 0.4, "y": -0.25}, {"action": "pick", "object": "orange_block", "target": null, "x": null, "y": null}, {"action": "place_on", "object": null, "target": "bowl", "x": null, "y": null}]}

Example 2
Scene: {"objects": [{"id": "purple_block", "type": "block", "color": "purple", "position": [0.4, 0.2, 0.021], "on": "table"}, {"id": "yellow_block", "type": "block", "color": "yellow", "position": [0.55, -0.05, 0.021], "on": "table"}]}
Instruction: Put the purple block behind the yellow block.
Answer: {"feasible": true, "reason": "Behind means larger x, so place it 0.12 m further from the robot.", "steps": [{"action": "pick", "object": "purple_block", "target": null, "x": null, "y": null}, {"action": "place_at", "object": null, "target": null, "x": 0.67, "y": -0.05}]}

Example 3
Scene: {"objects": [{"id": "red_block", "type": "block", "color": "red", "position": [0.5, 0.0, 0.021], "on": "table"}]}
Instruction: Put the blue block on the red block.
Answer: {"feasible": false, "reason": "There is no blue block in the scene.", "steps": []}"""

SYSTEM_PROMPTS = {
    "basic": SYSTEM_BASIC,
    "fewshot": SYSTEM_BASIC + _FEWSHOT,
}


def system_prompt(name: str) -> str:
    try:
        return SYSTEM_PROMPTS[name]
    except KeyError:
        raise ValueError(f"unknown prompt {name!r}; choose from {sorted(SYSTEM_PROMPTS)}") from None


def user_message(scene: dict, instruction: str) -> str:
    return f"Scene: {json.dumps(scene, separators=(', ', ': '))}\nInstruction: {instruction}"


def violation_feedback(messages: list[str]) -> str:
    lines = "\n".join(f"- {m}" for m in messages)
    return (
        "Your plan was checked before execution and rejected:\n"
        f"{lines}\n"
        "Return a corrected plan for the same instruction, as one complete JSON object in the "
        "same format."
    )


def parse_feedback(error: str) -> str:
    return (
        f"Your response could not be read as a plan: {error}. "
        "Return only the JSON object, in the format described."
    )
