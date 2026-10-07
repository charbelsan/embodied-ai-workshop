# VINCI Embodied AI · running the workshop in 2 hours

**Audience:** participants with no robotics background, in teams. **Intention:** let them experience the difference between learning a gesture, executing it, checking its result and deciding what comes next.

The notebook is the course material. The sheets help find an action quickly. Your role is to connect the experiments, collect predictions and help teams interpret their measurements.

## Before the group arrives

Check on a participant machine: desktop access, START WORKSHOP up to READY, the notebook opening, the desktop icons, the reference models and videos. Open the challenge sheet PDF and check it is readable on screen. Prepare your demo access; hand out personal access separately from the shared material.

To let a team start from zero without restarting its machine, the **RESTART WORKSHOP** icon stops its computations, restores the original notebook and `team/`, and moves its work to `results/archive_…`.

Identify a colleague who can handle connection problems while you continue with the group. Say where to ask for help. Never let a participant wait silently for several minutes on an error.

## Key notions to give before the lab

In the spoken introduction, present in a few minutes: an observation (what the robot receives), an action (its command), a policy (the function that chooses), a reward (the signal designed by the engineer), an environment (a simulated copy) and an episode (one attempt between resets). Present PPO when RL starts: collect experience, estimate what does better than expected, update the weights with bounded changes. Go into detail during the computation to keep the 45/55 split.

Before each experiment, say **where to look**:

| Step | What the participant should watch |
|---|---|
| RL or BC/DAgger training | Notebook output and curves. The Isaac window may stay still: training simulations run separately. |
| RL videos 0/50/250, RAM comparisons, VLA, world model, agent results | Pre-recorded references in the notebook; do not attribute them to the team's run. |
| Cube pick, RL model trial, student trial, final sequence | Live robot in Isaac Sim. Weights are not updated during these trials. |
| RAM insertion | Live robot through the notebook's images and video, not in another Isaac window. |
| Challenge evaluation | Score in the terminal. Without video: about 1–2 min; two optional videos: about 9–10 min observed, with possible lag. |

Remind them that `[*]` means "computing", that a parameter cell does not start the training, and that trials of the team's models are followed by an automatic return to the reference models. The pauses below are facilitation options, not mandatory stops.

## Timeline

| Workshop clock | Sequence | Your objective |
|---|---|---|
| 00:00–00:03 | Access and START WORKSHOP | Everyone opens the first cell |
| 00:03–00:11 | Observe and act | Distinguish gesture, final state and score |
| 00:11–00:29 | RL then BC/DAgger | Understand what is learned, and from which information |
| 00:29–00:36 | RAM | Accept not being able to confirm from the image |
| 00:36–00:45 | VLA, world model, Brain | Connect each component to its measured usefulness |
| 00:45–01:00 | First challenge test | Each team gets a before/after comparison |
| 01:00–01:35 | One chosen improvement | Hypothesis, change, measurement |
| 01:35–01:40 | Last evaluation | Keep and explain the result |
| 01:40–02:00 | Submission, ranking and discussion | Each team reads `./submit.sh --status`; announce the ranking, collect what was learned |

These slots are a suggestion. The four 1–2 min pauses are **included** in the notebook blocks; place them during the RL and DAgger trainings when possible. Computation times are included: use them to discuss, without starting other concurrent trainings.

## Suggested opening · 30 seconds

"Today you will put a robot to work. It can execute a command perfectly and still fail the mission. We will learn to spot that difference, then change its program so it decides better. You do not need to know robotics or to understand every line of code. Before each experiment: predict, observe, then explain."

## Four pauses to bring out the notions

**1. After the first evaluation: what proves success?**

Ask two teams for a short answer. Expected answer: the result in the world counts; the call that was sent and the `ok` message are not enough. Have them distinguish final success from progress points. Do not present 80/100 as a bug if the final state is correct.

**2. After DAgger: what did we add to the data?**

Have them complete: "After a perturbation, the student meets… ; DAgger brings…". Expected answer: situations different from the initial demonstrations, and the teacher's corrections in those situations. Recall the two thresholds: success beyond 20 cm, skill target at 30 cm. These are not the challenge criteria.

**3. After the RAM insertion, before the verdict: can you conclude?**

Quick vote: succeeded, failed, impossible to confirm — once with the sharp view (for humans only), once with only the 128 px images the Brain receives (the policy reduces its RGB-D inputs to 64 px). Ask for the visual clue, then reveal the verdict. A cautious, justified answer is acceptable. Ask them to propose another viewpoint or a measurement. The exact evaluator used for this demonstration is not a tool allowed to be copied into the Brain.

**4. After the mission: what did verification bring?**

Have them distinguish the visual student's technical return, the measured opening, the RL retry and the final verdict. The retry is a Python rule limited to one attempt, using the simulator's privileged state; it is neither an LLM nor a success of the visual model. If the opening is insufficient, the pick is not started. The arm goes through `go_pick_start` to clear the open drawer before reaching the cube; a failure of this move also stops the sequence. If the placement is not confirmed, the drawer is not closed. The Brain is a role; `brain.py` is its implementation. Ask how to replace the exact measurement with a visual check and how to handle uncertainty.

## Explanations to preserve

- RL: distinguish training reward and measured success; the drawer policy receives privileged state.
- Imitation: the student sees RGB-D images and its joints. The DAgger teacher was prepared before the workshop; it is not the short RL model the participant just trained.
- RAM: a strong expert can be a hard teacher to imitate. Corrections, observations and data volume matter. Overlapping intervals are not enough to conclude there is no statistical difference.
- VLA: the negative results concern the configurations tested in this scene. Do not generalize to a whole family of models.
- World model: accuracy and usefulness for decisions are two different measurements.
- Verification: distinguish a skill's message, an observation, an evaluation verdict and an agent's claim.

## Starting the challenge

"Everyone starts by measuring the starter version, enabling verification, then measuring again. Then choose a single idea. At the end we want an argued improvement, even a modest one, and a confirmed submission."

Suggest three roles: one person operates, one reads the instructions and outputs, one writes down hypotheses and results. Rotate roles after the first run. The discovery track lets people take part without writing a Python function; more experienced teams work on retries and on reading the rules dynamically.

At 01:25, announce 10 minutes of work left. At 01:35, ask them to finish their changes and run a last measurement. At 01:40, everyone submits. The personal coding agent and retraining are optional; they must not prevent a team from delivering a working version.

## If the group falls behind

Do not rerun a busy cell. Distinguish slow computation, a Python error and a normal task failure. If a training fails, report it as an incident and use the reference results collectively, clearly announced as such; do not present them as the participants' results.

To keep time, limit the detailed reading of the RAM checks, the viewing to one VLA success and one failure, and free changes of the comparator. Preserve the RL/DAgger experiments, the RAM vote, the first challenge before/after and the submission. If a computation stays stuck, call technical support rather than improvising a sequence of dependent cells.

## Final discussion · after submissions are confirmed

Each team shares in 30 seconds: "We changed… ; on … episodes, we observed… ; we do not know yet…". Adapt the number of speakers to the group size.

End with three questions:

1. Why does `ok=True` not prove that the mission succeeded?
2. Which problem does DAgger try to address?
3. Which additional measurement would you want before using your solution elsewhere?

Expected answers: check the obtained state; learn corrections in the situations the student actually meets; test more situations and the limits of the verification. Recall that the ranking uses the 8 DEV episodes (seeds 0 to 7): a good score on these episodes does not prove robustness to other situations.

### Operator check: images really updated

An available HTTP service does not guarantee that the cameras follow the physics. The smoke test includes `live camera motion`: a small movement from the initial position must change both images beyond rendering noise, then the scene is reset. Run it outside a participant session:

```bash
/opt/vinci-workshop/nb-venv/bin/python /opt/vinci-workshop/scripts/check_live_cameras.py
```

If this check fails, do not interpret the visual student's results. Check the scene and, once it is free, restart only `vinci-skill-server`, then run the check again. This restart does not stop the machine but resets the Isaac scene. The check tests neither every model nor the challenge success.
