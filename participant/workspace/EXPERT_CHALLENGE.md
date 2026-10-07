# Expert challenge — your models in a mission

Optional, unranked, 30 to 45 minutes. Open **notebooks/02_defi_expert_v2.ipynb**.

**Goal:** reuse the policies you trained in notebook 01, or train a variant, then measure their effect on a complete mission: open the drawer, put the cube away, close the drawer. Everything happens in the same scene. The opening is learned (RL or BC/DAgger visual student); placement and closing are programmed. There is no automatic fallback to another policy.

## Steps

1. Measure the reference: `./expert.sh evaluate --reference --provider visual --episodes 3`.
2. Choose one of your existing models, or train one: `./train.sh dagger` / `./train.sh rl`.
3. Select its actual folder: `./train.sh select results/<run>`.
4. Measure: `./expert.sh evaluate --provider visual --episodes 3` (or `--provider rl` for an RL model).
5. Present your hypothesis, the failure you observed, the checkpoint and the before/after results. Increase the number of runs to confirm.

**Watch Isaac Sim during the measurement.** Count about 70 seconds per mission, i.e. 3 to 4 minutes for three runs. No coding agent or external account is needed. Do not run any other training, reset, notebook or agent on the scene during a measurement. The commands refuse concurrent evaluations that use the shared lock.

Reports are in `results/expert/measure_*/report.json`: model and checksum, episodes, opening success, mission success, errors. Only `status: complete` is a usable result. A technical failure never produces a success. A failed opening counts as a failure, with no hidden retry.

## Freedom to explore

Edit your notebook, train, compare, write your own composition or a visual tool. A coding agent can help. The standard measurement stays separate from your exploration code: it isolates the effect of the opening model. Alternative compositions are shown at the debrief, unranked.

The DAgger teacher is provided: the RL model of notebook 01 does not automatically become the student's teacher. The student learns from RGB-D images and joints; the RL policy uses privileged state. No VLA or RAM model is retrained in this track.

## Honest measurement

The mission result is measured physically: no checkpoints declared by the agent, no success JSON written by the agent. The measurement code and the simulator must not be modified. Results shown at the debrief can be replayed by the facilitator from the checkpoint.

This is an open workshop on a machine you administer, not a competition secured against a malicious user. Training access can read the simulator's ground truth; we do not claim it cannot be misused. No automatic expert ranking is published.

The resets of this scene are not a hidden benchmark or a seed-paired protocol. A 3/3 does not establish general robustness. If both the reference and your variant succeed, study data cost, duration or perception; do not announce an unmeasured improvement.

## Extension: visual RAM verifier

In notebook v2, the optional extension collects labelled images with distinct seeds for training and validation, then trains a simple classifier. The RAM insertion policy stays provided. Easy negatives do not validate the detection of an almost-successful insertion. The classifier's score is not a calibrated confidence.
