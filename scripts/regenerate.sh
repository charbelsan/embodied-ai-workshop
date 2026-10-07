#!/bin/bash
# Rebuild every model / result of the guided notebook from scratch, instead of downloading them.
# Run on a workshop machine (Isaac python from /etc/vinci/workshop.env). Durations measured on one g6e.2xlarge-4xlarge
# (NVIDIA L40S), partly under concurrent load; alone they are usually shorter.
#
#   scripts/regenerate.sh rl         PPO drawer opening, 1024 envs x 300 it (27.9 min) + checkpoint eval/videos (~5 min)
#   scripts/regenerate.sh dagger     privileged teacher PPO 400 it (~3.4 min at 100 it, ~14 min at 400)
#                                    + demos -> BC -> DAgger r1 -> r2 with perturbations (~9-11 min)
#   scripts/regenerate.sh demos      40 scripted pick-and-place demonstrations, front+wrist 256 px (55.8 min)
#   scripts/regenerate.sh lerobot    convert them to a LeRobot 0.6.1 dataset (4.4 min, needs a lerobot venv: LEROBOT_PY)
#   scripts/regenerate.sh smolvla    fine-tune SmolVLA on it, 3000 steps (25.2 min, lerobot-train in the lerobot venv)
# Outputs go to ./regenerated/<step>/. To use a regenerated model in the workshop, copy it into
# participant/runtime/policies/ and re-run workshop/install_workshop.sh.
set -euo pipefail
source /etc/vinci/workshop.env
RT="${VINCI_RUNTIME:-/opt/vinci-workshop}"; PY="$WS_PY"; OUT="$PWD/regenerated/$1"; mkdir -p "$OUT"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
t0=$(date +%s)
case "$1" in
  rl) (cd "$OUT" && "$PY" "$RT/scripts/train_open_drawer.py" --headless --num_envs 1024 --max_iterations 300 </dev/null)
      run=$(ls -d "$OUT"/logs/rsl_rl/vinci_open_drawer/*/ | tail -1)
      (cd "$OUT" && "$PY" "$RT/scripts/eval_checkpoints.py" --headless --run_dir "$run" --num_envs 32 --every 50 \
        --export_to "$OUT/open_drawer_rl.pt" </dev/null) ;;
  dagger) (cd "$OUT" && "$PY" "$RT/scripts/dagger_train_teacher.py" --headless --num_envs 1024 --max_iterations 400 </dev/null)
      ckpt=$(ls -t "$OUT"/logs/rsl_rl/vinci_dagger_teacher/*/model_*.pt | head -1)
      echo "teacher checkpoint: $ckpt (export it to TorchScript like policies/dagger_teacher.pt, or use the shipped teacher)"
      W="$OUT/work"; mkdir -p "$W/scripts" "$W/policies"; cp "$RT/scripts/dagger_pipeline.py" "$W/scripts/"
      ln -sfn "$RT/vinci_lab" "$W/vinci_lab"; cp "$RT/policies/dagger_teacher.pt" "$W/policies/"
      (cd "$OUT" && "$PY" "$W/scripts/dagger_pipeline.py" --headless --num_envs 128 --perturb --out "$OUT/pipeline" </dev/null)
      cp "$W/policies/open_drawer_visual.pt" "$OUT/" ;;
  demos) (cd "$OUT" && "$PY" "$RT/scripts/gen_demos.py" --headless --episodes 40 --out "$OUT/raw" </dev/null) ;;
  lerobot) : "${LEROBOT_PY:?set LEROBOT_PY to the python of a lerobot 0.6.1 venv (pip install 'lerobot[dataset]==0.6.1')}"
      "$LEROBOT_PY" "$RT/scripts/to_lerobot.py" --raw "$PWD/regenerated/demos/raw" --root "$OUT/vinci_pick_place_drawer" --repo_id vinci/pick_place_drawer ;;
  smolvla) : "${LEROBOT_PY:?set LEROBOT_PY to the python of a lerobot 0.6.1 venv}"
      "$(dirname "$LEROBOT_PY")/lerobot-train" --policy.path=lerobot/smolvla_base --dataset.repo_id=vinci/pick_place_drawer \
        --dataset.root="$PWD/regenerated/lerobot/vinci_pick_place_drawer" --steps=3000 --batch_size=32 \
        --output_dir="$OUT/smolvla_d40" --policy.push_to_hub=false ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
echo "$1 done in $(( ($(date +%s) - t0) / 60 )) min -> $OUT"
