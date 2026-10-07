# VINCI Embodied AI Workshop

**Part 1 — guided track (~45 min):** **START WORKSHOP** icon (or `./start.sh`), then follow the notebook that opens (`notebooks/01_guided_lab.ipynb`).

**Part 2 — team challenge (~55 min):** **VS Code** icon to edit `team/`, **Terminal** icon to run the commands.

| Command (in the Terminal) | What it does |
|---|---|
| `./evaluate.sh` | plays training episodes with your Brain and prints your score |
| `./train.sh rl` / `./train.sh dagger` | retrains a model (also used by the notebook) |
| `./reset.sh` | puts the robot and the scene back in their starting state (`./reset.sh --team`: back to the starter Brain) |
| `./submit.sh` | freezes code and models, evaluates 8 DEV episodes without video in the background: the best valid submitted score counts (`./submit.sh --status`) |
| `./start.sh` | restarts whatever is missing (this is what START WORKSHOP does) |

```text
notebooks/01_guided_lab.ipynb   part 1: the guided track
notebooks/02_defi_expert_v2.ipynb  optional expert challenge (EXPERT_CHALLENGE.md)
team/                           part 2: YOUR space (brain.py, skills.py, config.yaml)
results/                        your evaluations, videos, trainings and submissions
WORKSHOP_SHEET.pdf              the workshop sheet, 2 pages: getting in, the desktop, the two hours
CHALLENGE_SHEET.pdf             the challenge sheet, 2 pages (also CHALLENGE_SHEET.md)
CHALLENGE_GUIDE.pdf             the detailed guide: every setting, every call, troubleshooting
```
