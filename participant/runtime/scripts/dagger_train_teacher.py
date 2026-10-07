"""Train the PRIVILEGED teacher for open_drawer with PPO (randomized cabinet pose + drawer dynamics).

    python scripts/dagger_train_teacher.py --headless --num_envs 1024 --max_iterations 400

Thin wrapper around Isaac Lab's rsl_rl train.py. Checkpoints: logs/rsl_rl/vinci_dagger_teacher/<run>/model_<it>.pt
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vinci_lab.dagger  # noqa: F401,E402  (registers Vinci-Dagger-Teacher-v0; cfg modules load lazily)

ISAACLAB = os.environ.get("ISAACLAB_DIR", os.path.expanduser("~/upstream/IsaacLab"))
if "--task" not in sys.argv:
    sys.argv += ["--task", "Vinci-Dagger-Teacher-v0"]
script_dir = os.path.join(ISAACLAB, "scripts/reinforcement_learning/rsl_rl")
sys.path.insert(0, script_dir)
sys.argv[0] = "train.py"
runpy.run_path(os.path.join(script_dir, "train.py"), run_name="__main__")
