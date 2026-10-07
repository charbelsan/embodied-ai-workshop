# Your team space

Your mission is to make the workstation tidying reliable. Start by measuring the starter program, change one thing, then measure again. The challenge sheet describes the parts, the rules and the scoring.

## First pass

In the VS Code terminal (menu **Terminal**, then **New Terminal**: it opens directly in `~/vinci-workshop`) or in the desktop Terminal, run:

```bash
./evaluate.sh
```

Then, in `team/config.yaml`, switch `verification.enabled` from `false` to `true`. Save and run the same command again. Compare `Complete successes`, `Average score`, `Average sim time` and the score detail: `identified` goes from 0 to 10, `final_state_verified` stays at 0 (checking the final state after the last action: that one is yours to earn).

A complete success describes a correct final state. The score also counts the steps, verification included. You can therefore gain points without changing the number of successful missions.

## The files

| File | Role |
|---|---|
| `brain.py` | `solve(tools)` decides what to do in each episode |
| `skills.py` | Your manipulation functions; `place()` and `where_is()` are examples |
| `config.yaml` | Providers, verification, retries and evaluation settings |

**Brain** is the decision role; **`brain.py`** is the Python file that fills it here. The starter program follows a simple sequence. You can teach it to observe, adapt its plan and verify.

## The Brain interface

```python
tools.rules()                    # current destinations; "keep" means preserve
tools.inspect()                  # parts, slots, drawer and work order
tools.call("open_drawer", provider="rl")
tools.call("pick_and_place", obj="bracket", target="kitting_tray")
tools.call("close_drawer")
tools.time_left()                # simulated seconds left
tools.decide("I check before closing")  # trace of your decision
```

This example shows the calls, not a complete solution. Read the destinations from `tools.rules()` instead of hard-coding them. The opening providers are `rl`, `analytic` and `visual`. The placement targets are `top_drawer` and `kitting_tray`.

`tools.inspect()` provides privileged state, allowed in this challenge. Some skills use it too. Every skill returns `ok`, `reason`, `duration_s` and `evidence`. The `ok` field does not replace an observation of the result.

For your own movements, `tools.lab` exposes `observe()`, `move_to(pos, quat)` and `gripper(open)`, among others. Available helpers include `move_to`, `go_pick_start`, `clear_of_drawer` and `stop`. Read the provided code for their parameters before calling them.

## The cameras in your code

Use the cameras (front and wrist) in your code, for example to check a step, write a visual skill, or look at a failure before deciding on a retry:

```python
o = tools.lab.observe()
if "rgb_front" in o:                                  # absent when the cameras are not running (see below)
    front, wrist = o["rgb_front"], o["rgb_wrist"]     # numpy uint8 images, height x width x 3
```

During `./evaluate.sh`, the cameras run only if a visual provider is chosen in `config.yaml`
(`open_drawer: visual`): the evaluation then switches to a single process with rendering, which is slower. With `rl` or `analytic`,
it runs without rendering and these keys are absent. To look at the images before writing code: the notebook
(`o = lab.observe(); lab.show(o)`) or your coding agent (`observe(scene="cabinet")`, see `AGENTS.md`).

## A useful work loop

1. State a hypothesis: "This failure comes from…".
2. Find an episode and its traces in `results/`.
3. Change a single behaviour and save.
4. Run the same evaluation again, then compare successes, score and time.
5. Keep the change if the observations justify it; write down what remains uncertain.

The latest summary is `results/latest/report.txt`. The terminal shows where to find the traces. Note the paths of your runs: the `latest` link is replaced at the next test.

```bash
./evaluate.sh                    # 8 DEV episodes; about 2 min
./evaluate.sh --episodes 20      # more runs
./evaluate.sh --videos 2         # extra videos; slower
```

Each episode has a budget of 240 **simulated** seconds. Your retry loop must have a limit on attempts and respect the remaining time. Eight episodes are still a small measurement: a difference of a few points can come from physical variability.

## Models and coding agent: optional

`./train.sh rl` and `./train.sh dagger` start the allowed trainings. To use a trained model, run `./train.sh select` followed by the actual path of its folder in `results/`, as printed at the end of the training. Check the selected model with its sha256 checksum in the outputs. Keep time to evaluate and submit.

You can use your own account with `claude` or `codex`, following `AGENTS.md`. The agent can help edit the code and call the `vinci-robot` MCP tools. This connection is not needed to finish the challenge. The RAM manipulation is a separate experiment; it gives no points in the tidying mission.

## Reset and submit

```bash
./reset.sh          # reset the scene
./reset.sh --team   # back to the starter code; your current code is saved in results/
./reset.sh --all    # start everything over (RESTART WORKSHOP icon): original notebook and team/, work archived in results/
./submit.sh         # freeze and evaluate your submission in the background
./submit.sh --status  # your submissions and their scores
```

`./submit.sh` freezes your code, your configuration and the selected models, then runs **8 DEV episodes, seeds 0 to 7, without video** in the background. Its average score out of 100 is shown by `./submit.sh --status`. **The best valid submitted score counts** for the ranking, ties are allowed. There is no additional hidden evaluation for this ranking.

You can close the terminal. The computation can take several minutes; wait for it to finish before another training. A copy stays in `results/submissions/` and the work queue in `~/.local/state/vinci-submissions/`. "Frozen copy" means saved locally; the status `complete` in `./submit.sh --status` means the score is final.
