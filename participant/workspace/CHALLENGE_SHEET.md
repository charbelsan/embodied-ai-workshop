# Team challenge · put the workstation back in order

You have **55 min, submission included**, then 5 min for results and discussion. Your goal: make the robot reliable, and explain with measurements what your change brings.

## The mission: put the workstation back in order

| Part | Looks like | Starting destination |
|---|---|---|
| `bolt_box` | Red cube, 4.5 cm | Top drawer |
| `sensor` | Red cube, 4 cm | Top drawer |
| `bracket` | Blue bar, 8 × 3 × 3 cm | Kitting tray |
| `gauge` | Yellow cube, 5 cm | Do not touch |

**Final state:** parts in their place, drawer closed (less than 2 cm open), gauge moved less than 2 cm, nothing on the floor. Before dropping a part into the drawer, open it at least 18 cm. **Read the destinations from `tools.rules()` instead of copying this table into your program.**

## Your first improvement · 10 to 15 min

**Measure first.** In VS Code, open a terminal (menu **Terminal**, then **New Terminal**: it opens in `~/vinci-workshop`), or use the **Terminal** desktop icon, and run `./evaluate.sh`. The 8 development episodes take about 2 min. Write down the complete successes, the average score and the simulated time.

**One option per evaluation:** `./evaluate.sh` without video (recommended, about 1 to 2 min), or `./evaluate.sh --videos 2` with two videos (optional, about 9 to 10 min in our test). Both compute the score. With video, the desktop may become less fluid: let the command finish. You can skip videos and submit without generating any. Durations depend on your settings and on the chosen provider.

**Change one thing.** In `team/config.yaml`, under `verification:`, replace `enabled: false` with `enabled: true`. Keep the indentation and save (**Ctrl + S**).

**Measure again** with the same settings, and compare the score detail printed at the end of the evaluation: `identified` goes from 0 to 10, because the Brain now inspects the scene. `final_state_verified` stays at 0: checking the final state after the last action is a lead for what comes next.

**Explain:** did you improve the final state, its verification, or both? A change of score is not automatically a higher success rate.

| Run | Complete successes | Score /100 | Average simulated time |
|---|---|---|---|
| Start | | | |
| Verification enabled | | | |
| Your improvement | | | |

**Example observed in rehearsal:** 80 → 90/100 with 8/8 successes in both runs. It illustrates the difference between score and success; it is not a guaranteed result.

## Then choose a track · 35 to 40 min

**Discovery track:** compare the drawer-opening providers in `config.yaml`, one change at a time: `rl`, a trained RL policy, fast, which receives the exact handle position; `analytic`, a scripted geometric controller; `visual`, a DAgger student acting from the RGB-D cameras and its joints, slower because it switches the cameras on. Explain their results from the traces.

**Programming track:** in `brain.py`, inspect after an action, compare with the expected result, then retry if needed, with a limit on attempts and time. Check the final state.

**To go further:** handle parts and destinations from `tools.rules()`; look for a failure in the traces; test your fix on more episodes. A personal coding agent is optional: see `AGENTS.md`.

# Measure and submit

## Where to work and what to call

Edit `team/brain.py` (decisions), `team/skills.py` (movements) and `team/config.yaml` (settings). You may add files in `team/`. `solve(tools)` is called once per episode.

| Call in your Brain | Purpose |
|---|---|
| `tools.rules()` | Read the destinations and the preservation rule |
| `tools.inspect()` | Observe parts, slots and drawer; privileged state allowed |
| `tools.call("open_drawer", provider="rl")` | Open; providers: `rl` (RL policy), `analytic` (geometric), `visual` (DAgger student, cameras) |
| `tools.call("pick_and_place", obj="bracket", target="kitting_tray")` | Place; targets: `top_drawer`, `kitting_tray` |
| `tools.call("close_drawer")` | Close the drawer |
| `tools.time_left()` / `tools.decide("reason")` | Read the remaining time / explain a choice in the trace |
| `tools.lab.observe()["rgb_front"]`, `["rgb_wrist"]` | Front and wrist camera images, when the cameras run (see `team/README.md`) |

A skill returns `ok`, `reason`, `duration_s`, `evidence`. **`ok` is its opinion: check the resulting state.** Other calls and the `tools.lab` API are described in the team README.

## Measure a change

```bash
./evaluate.sh                  # no video, 8 DEV episodes, 1 to 2 min
./evaluate.sh --episodes 20    # more runs; allow more time
./evaluate.sh --videos 2       # video option: 9 to 10 min in our test
./submit.sh                    # submit the current version
```

Read `results/latest/report.txt`, then the traces in the evaluation folder it prints. Compare with identical settings and keep the path of each run: `latest` changes at every evaluation. A result on 8 episodes is still a small measurement.

| Score component | Maximum |
|---|---:|
| Parts identified / grasped / correctly placed | 10 / 15 / 25 |
| No wrong placement / drawer handled correctly | 10 / 15 |
| Final state verified / task complete | 10 / 15 |
| **Total** | **100** |

**Complete success ≠ 100/100:** success describes the final state; points also count the steps. Budget: **240 simulated seconds per episode**. Ranking on **8 identical DEV episodes (seeds 0 to 7), no video**: best valid average score submitted, out of 100; ties allowed.

## Rules and troubleshooting

You may modify your team code, choose the skills and retrain the allowed models. You may not modify the simulator, the evaluator, the levels or the physics. Access the world only through the allowed interfaces `tools` / `tools.lab.observe()`.

`./reset.sh` resets the scene. `./reset.sh --team` restores the starter code and saves your current version in `results/`. To start everything over (notebook included), double-click the **RESTART WORKSHOP** icon: nothing is deleted, your work is moved to `results/archive_…`. If you do not understand an error, show its message to the facilitator.

**Before leaving:** save, run `./submit.sh`, then check its score with `./submit.sh --status` (a few minutes later). The computation continues in the background, even if you close the terminal. You can submit several times; **the best valid submitted score counts**. Prepare one sentence: "We changed… ; our measurement shows… ; what remains to check is…".
