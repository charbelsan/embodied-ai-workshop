"""Train the open_drawer primitive with PPO (Isaac Lab rsl_rl), on our task Vinci-Open-Drawer-Franka-v0.

    python scripts/train_open_drawer.py --headless --num_envs 1024 --max_iterations 300 \\
        --video --video_interval 3000 --video_length 240

Thin wrapper: registers our task, then runs Isaac Lab's own rsl_rl train.py unchanged.
Checkpoints: logs/rsl_rl/vinci_open_drawer/<run>/model_<it>.pt ; videos: .../videos/train/
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vinci_lab  # noqa: F401  (registers Vinci-Open-Drawer-Franka-v0)

ISAACLAB = os.environ.get("ISAACLAB_DIR", os.path.expanduser("~/upstream/IsaacLab"))
if "--task" not in sys.argv:
    sys.argv += ["--task", "Vinci-Open-Drawer-Franka-v0"]
script_dir = os.path.join(ISAACLAB, "scripts/reinforcement_learning/rsl_rl")
sys.path.insert(0, script_dir)  # train.py imports its sibling cli_args.py
sys.argv[0] = "train.py"
runpy.run_path(os.path.join(script_dir, "train.py"), run_name="__main__")

# Isaac/Kit may crash during Python interpreter finalization after a successful
# training run and simulation_app.close(). Other standalone workshop jobs use
# the same explicit exit. Reach this point only if upstream train.py returned
# normally: training exceptions must still propagate with a non-zero status.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
