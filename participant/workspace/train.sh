#!/bin/bash
# ./train.sh rl [extra args]         train the drawer-opening RL policy (PPO, a few minutes), then export the best checkpoint
# ./train.sh dagger [extra args]     demonstrations -> BC -> perturbation -> DAgger, then export the last student
# ./train.sh select results/<run>    make the robot use THIS run's model (writes path + sha256 into team/config.yaml)
# ./train.sh select reference [name] go back to the workshop's reference model(s)
# Everything a training run produces (logs, curves, videos, model + sha256) goes to results/train_*.
source "${VINCI_RUNTIME:-/opt/vinci-workshop}/lib/env.sh"
cd "$WS"; what="${1:-}"; shift || true

export_model() {  # name file stage teacher_file -> results/<run>/policy/<name>.pt + MODEL.json
  local name=$1 src=$2 stage=$3 teacher=${4:-}
  mkdir -p "$OUT/policy"; cp "$src" "$OUT/policy/$name.pt"; chmod a-w "$OUT/policy/$name.pt"
  "$PY" - "$OUT" "$name" "$stage" "$teacher" <<'PY'
import hashlib, json, sys, time
out, name, stage, teacher = sys.argv[1:]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = {"name": name, "path": f"{out}/policy/{name}.pt", "sha256": h(f"{out}/policy/{name}.pt"), "stage": stage,
     "produced_by": "./train.sh", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
if teacher:
    m["labels_from_teacher"] = {"path": teacher, "sha256": h(teacher)}
json.dump(m, open(f"{out}/policy/MODEL.json", "w"), indent=1)
print(f"model: {m['path']}\nsha256: {m['sha256']}")
PY
  echo "to make the robot use it:  ./train.sh select results/$(basename "$OUT")"
}

case "$what" in
  rl) OUT="$WS/results/train_rl_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"; tl train_start what=rl
      (set -o pipefail; cd "$OUT" && timeout -k 20 1800 "$PY" "$RT/scripts/train_open_drawer.py" --headless --num_envs 256 --max_iterations 50 "$@" \
        </dev/null 2>&1 | tee "$OUT/train.log") || { echo "RL training failed: see $OUT/train.log" >&2; exit 1; }
      run=$(ls -d "$OUT"/logs/rsl_rl/vinci_open_drawer/*/ 2>/dev/null | tail -1)
      if [ -n "$run" ] && ls "$run"/model_*.pt >/dev/null 2>&1; then
        echo "evaluating the checkpoints and exporting the best one (about 1-2 min)..."
        (cd "$OUT" && timeout -k 20 900 "$PY" "$RT/scripts/eval_checkpoints.py" --headless --run_dir "$run" --num_envs 32 \
          --export_to "$OUT/export/open_drawer_rl.pt" </dev/null > "$OUT/eval_checkpoints.log" 2>&1)
        grep -E "^EVAL|^EXPORTED" "$OUT/eval_checkpoints.log" | tail -6
        [ -f "$OUT/export/open_drawer_rl.pt" ] && export_model open_drawer_rl "$OUT/export/open_drawer_rl.pt" \
          "$(grep -o 'best it=[0-9]*' "$OUT/eval_checkpoints.log" | tail -1)" \
          || echo "no model exported: see results/$(basename "$OUT")/eval_checkpoints.log"
      else echo "no checkpoint written: see results/$(basename "$OUT")/train.log"; fi
      tl train_end what=rl ;;
  dagger) OUT="$WS/results/train_dagger_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"; tl train_start what=dagger
      # the pipeline saves its student next to its own code: run it from a private copy inside results/
      W="$OUT/work"; mkdir -p "$W/scripts" "$W/policies"
      cp "$RT/scripts/dagger_pipeline.py" "$W/scripts/"; ln -s "$RT/vinci_lab" "$W/vinci_lab"
      cp "$RT/policies/dagger_teacher.pt" "$W/policies/"
      (set -o pipefail; cd "$OUT" && timeout -k 20 2400 "$PY" "$W/scripts/dagger_pipeline.py" --headless --out "$OUT" \
        --timings "$OUT/timings.jsonl" "$@" </dev/null 2>&1 | tee "$OUT/train.log") || { echo "DAgger training failed: see $OUT/train.log" >&2; exit 1; }
      last=""; for st in r2 r1 bc; do [ -f "$OUT/student_$st.pt" ] && { last=$st; break; }; done
      [ -n "$last" ] && export_model open_drawer_student "$OUT/student_$last.pt" "$last" "$W/policies/dagger_teacher.pt" \
        || echo "no student trained: see results/$(basename "$OUT")/train.log"
      tl train_end what=dagger ;;
  select) "$PY" - "$WS" "${1:?usage: ./train.sh select results/<run> | reference [name]}" "${2:-}" <<'PY'
import json, re, sys
from pathlib import Path
import yaml
ws, arg, only = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cfg = ws / "team" / "config.yaml"
text = cfg.read_text()
chosen = dict((yaml.safe_load(text) or {}).get("checkpoints") or {})
text = re.sub(r"\n?# >>> models selected by ./train.sh select.*?# <<<\n", "\n", text, flags=re.S).rstrip("\n") + "\n"
if arg == "reference":
    for k in ([only] if only else list(chosen)):
        chosen.pop(k, None)
    msg = f"back to the workshop's reference model: {only or 'all models'}"
else:
    run = (ws / arg) if not Path(arg).is_absolute() else Path(arg)
    meta = json.load(open(run / "policy" / "MODEL.json"))
    chosen[meta["name"]] = {"path": str(Path(meta["path"]).relative_to(ws)), "sha256": meta["sha256"]}
    msg = f"the robot now uses {meta['name']} = {chosen[meta['name']]['path']} (sha256 {meta['sha256'][:12]}...)"
if chosen:
    text += "\n# >>> models selected by ./train.sh select (change them with ./train.sh select ...)\ncheckpoints:\n"
    for k, v in chosen.items():
        text += f"  {k}:\n    path: {v['path']}\n    sha256: {v['sha256']}\n"
    text += "# <<<\n"
cfg.write_text(text)
print(msg)
PY
      tl select what="${1:-}" ;;
  *) sed -n '2,6p' "$0"; exit 1 ;;
esac
