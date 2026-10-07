"""Privileged teacher -> BC -> DAgger round 1 -> DAgger round 2 (+ failure mining), in ONE Isaac process.

    python scripts/dagger_pipeline.py --headless --teacher policies/dagger_teacher.pt --num_envs 128

Stages (each one saved under --out, so a later run can resume with --stages):
  bench         camera-env throughput only
  teacher_eval  the privileged teacher on the fixed eval set (reference + videos)
  bc            teacher drives, student imitates (behaviour cloning) -> eval
  r1            STUDENT drives, teacher labels the states the student reached, aggregate, retrain -> eval
  r2            same + failure mining (over-sample failed episodes and high-disagreement states) -> eval
  bc_more       control: plain BC with as many TEACHER samples as r2 has in total ("is it just more data?")
  report        distribution-shift metrics, comparison video, summary.json
The student never gets a reward and never runs PPO: it only learns supervised from teacher labels.
Every eval replays the SAME forced episodes (cabinet pose, drawer damping/friction, start pose).
"""
import argparse
import copy
import datetime as dt
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--teacher", default=os.path.join(os.path.dirname(__file__), "..", "policies", "dagger_teacher.pt"))
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--out", default="dagger_run")
p.add_argument("--stages", default="teacher_eval,bc,r1,r2,bc_more,report")
p.add_argument("--bc_batches", type=int, default=2, help="x num_envs teacher episodes for BC (+1 held-out batch)")
p.add_argument("--round_batches", type=int, default=2, help="x num_envs student episodes per DAgger round")
p.add_argument("--bc_steps", type=int, default=6000)
p.add_argument("--round_steps", type=int, default=4000)
p.add_argument("--eval_seed", type=int, default=2026)
p.add_argument("--n_video", type=int, default=8)
p.add_argument("--reeval", type=int, default=3, help="stage reeval: how many repeated evals per policy")
p.add_argument("--video_ids", default="", help="comma-separated eval episode ids to film (default: 0..n_video-1)")
p.add_argument("--mine_factor", type=float, default=4.0)
p.add_argument("--perturb", action="store_true", help="deployment disturbances (push + handle slip) in evals and student rollouts")
p.add_argument("--perturb_demos", action="store_true", help="also disturb the TEACHER's BC demos (off: clean demos)")
p.add_argument("--episodes_per_round", type=int, default=0,
               help="demo budget K: BC keeps K teacher episodes, each DAgger round adds K student episodes (0 = all collected)")
p.add_argument("--push_rad", type=float, default=0.35)
p.add_argument("--slip_prob", type=float, default=0.7)
p.add_argument("--timings", default="dagger_timings.jsonl")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import vinci_lab.dagger  # noqa: F401,E402
from vinci_lab.dagger.env_cfg import PARAM_NAMES, SUCCESS_OPENING, VinciDaggerStudentEnvCfg, eval_params  # noqa: E402
from vinci_lab.dagger.student import Dataset, Student, to_student_image, train_student  # noqa: E402

DEV = "cuda:0"
OUT = args.out
os.makedirs(os.path.join(OUT, "videos"), exist_ok=True)
STAGES = args.stages.split(",")
PID = os.getpid()


def log(msg):
    print(f"[dagger {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_status():
    """(our process VRAM GB, description of other GPU processes)."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL).stdout
        mine, others = 0.0, []
        for line in q.strip().splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if int(pid) == PID:
                mine = int(mem) / 1024
            elif int(mem) > 1500:
                others.append(f"pid{pid}:{int(mem) / 1024:.1f}GB")
        return round(mine, 1), ", ".join(others)
    except Exception as e:  # noqa: BLE001
        return None, f"nvidia-smi failed: {e}"


def timing(step, t0, **kw):
    vram, others = gpu_status()
    rec = {"phase": "DAGGER", "step": step,
           "start": dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(timespec="seconds"),
           "end": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "wall_minutes": round((time.time() - t0) / 60, 2), "gpu": "L40S 48GB", "max_vram_gb": vram,
           "ram_gb": round(int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) / 1e6, 1),
           "cpu_load_1m": round(os.getloadavg()[0], 1), "concurrent_gpu_load": others or None,
           "num_envs": args.num_envs, "human_interventions": 0}
    rec.update(kw)
    with open(args.timings, "a") as f:
        f.write(json.dumps(rec) + "\n")
    with open(os.path.join(OUT, "timings.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"TIMING {step}: {rec['wall_minutes']} min {json.dumps(kw)[:300]}")


# ---------------------------------------------------------------- env + teacher
N = args.num_envs
VID = (torch.tensor([int(x) for x in args.video_ids.split(",")], device=DEV) if args.video_ids
       else torch.arange(min(args.n_video, N), device=DEV))
if STAGES != ["report"]:  # the report only reads saved results: no simulation needed
    t_boot = time.time()
    cfg = VinciDaggerStudentEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = DEV
    env = ManagerBasedRLEnv(cfg=cfg)
    sc = env.scene
    robot, cab, ee, hf = sc["robot"], sc["cabinet"], sc["ee_frame"], sc["cabinet_frame"]
    front, wrist = sc["front_cam"], sc["wrist_cam"]
    JID = cab.find_joints("drawer_top_joint")[0][0]
    T = int(env.max_episode_length) - 1
    teacher = torch.jit.load(args.teacher, map_location=DEV).eval() if "bench" not in STAGES else None
    timing("boot_student_env", t_boot, episode_steps=T, control_hz=round(1 / env.step_dt), cameras="front+wrist RGB-D 128px -> 64px")


def student_images():
    fo, wo = front.data.output, wrist.data.output
    return (to_student_image(fo["rgb"], fo["distance_to_image_plane"]),
            to_student_image(wo["rgb"], wo["distance_to_image_plane"]))


def sample_perturbations(gen: torch.Generator | None):
    """Deployment disturbances, one draw per episode: an arm PUSH during the approach and a handle SLIP mid-pull."""
    g = gen if gen is not None else torch.default_generator
    return {"t_push": torch.randint(8, 36, (N,), generator=g).to(DEV),
            "delta": ((torch.rand(N, 7, generator=g) * 2 - 1) * args.push_rad).to(DEV),
            "do_slip": (torch.rand(N, generator=g) < args.slip_prob).to(DEV),
            "slip_start": torch.full((N,), -1, device=DEV, dtype=torch.long)}


@torch.no_grad()
def rollout(driver: str, student=None, forced=None, store=True, video=False, ep_offset=0, perturb=False, pgen=None):
    """One episode in every env. driver='teacher' (BC data) or 'student' (DAgger data / eval). The teacher ALWAYS labels.

    perturb=True adds deployment disturbances on the EXECUTED action (never on the stored teacher label):
    a push on the arm joints during the approach, and a slip (gripper forced open) once the drawer passes 6 cm.
    """
    env.dagger_forced = forced
    obs, _ = env.reset()
    env.dagger_forced = None
    pert = sample_perturbations(pgen) if perturb else None
    buf = {k: [] for k in ("front", "wrist", "prop", "a_t", "a_s")}
    max_open = torch.zeros(N, device=DEV)
    min_dist = torch.full((N,), 9.0, device=DEV)
    states, frames, sview = [], [], []
    t_teacher = 0.0
    t0 = time.time()
    for t in range(T):
        rel = hf.data.target_pos_w[:, 0] - ee.data.target_pos_w[:, 0]
        opening = cab.data.joint_pos[:, JID]
        f, w = student_images()
        prop = obs["student"]
        tt = time.time()
        a_t = teacher(obs["policy"])
        t_teacher += time.time() - tt
        a_s = student(f, w, prop) if student is not None else None
        a = a_t if driver == "teacher" else a_s
        if pert is not None:
            a = a.clone()
            push = (t >= pert["t_push"]) & (t < pert["t_push"] + 10)
            a[push, :7] += pert["delta"][push]
            newly = pert["do_slip"] & (pert["slip_start"] < 0) & (opening > 0.06)
            pert["slip_start"][newly] = t
            slip = (pert["slip_start"] >= 0) & (t < pert["slip_start"] + 15)
            a[slip, 7] = 1.0
        if store:
            buf["front"].append(f.cpu())
            buf["wrist"].append(w.cpu())
            buf["prop"].append(prop.cpu())
            buf["a_t"].append(a_t.cpu())
        if a_s is not None:
            buf["a_s"].append(a_s)
        max_open = torch.maximum(max_open, opening)
        min_dist = torch.minimum(min_dist, rel.norm(dim=-1))
        if t % 2 == 0:  # state descriptor for distribution-shift analysis: handle-gripper vector, drawer, arm joints
            states.append(torch.cat([rel / 0.05, opening.unsqueeze(-1) / 0.05, robot.data.joint_pos[:, :7] / 0.1], -1).cpu())
        if video:
            frames.append(np.ascontiguousarray(front.data.output["rgb"][VID, ..., :3].cpu().numpy()))
            sview.append(torch.cat([f[0], w[0]], dim=-1).cpu().numpy())  # student's own 64x64 RGB-D view, env 0
        obs, _, _, _, _ = env.step(a)
    torch.cuda.synchronize()
    max_open = torch.maximum(max_open, cab.data.joint_pos[:, JID])
    success = max_open > SUCCESS_OPENING
    cat = torch.full((N,), 0, dtype=torch.long, device=DEV)  # 0 success
    cat[~success & (min_dist > 0.035)] = 1  # missed the handle
    cat[~success & (min_dist <= 0.035) & (max_open < 0.02)] = 2  # reached, never pulled / no grasp
    cat[~success & (max_open >= 0.02)] = 3  # pulled partially / slipped
    res = {"success": success.cpu(), "max_open": max_open.cpu(), "min_dist": min_dist.cpu(), "category": cat.cpu(),
           "params": env.dagger_params.clone().cpu(), "states": torch.stack(states, 1).cpu(),  # (N, T/2, 11)
           "rollout_s": time.time() - t0, "teacher_ms_per_step": 1000 * t_teacher / T, "frames": frames, "sview": sview,
           "slipped": (pert["slip_start"] >= 0).cpu() if pert is not None else None}
    if buf["a_s"]:
        a_s = torch.stack(buf["a_s"], 1)  # (N,T,8) on GPU
        res["a_s"] = a_s.cpu()
    if store:
        ep = (torch.arange(N) + ep_offset).repeat_interleave(T)
        stack = lambda k: torch.stack(buf[k], 1).flatten(0, 1)  # noqa: E731  env-major: (N*T, ...)
        res["data"] = dict(front=stack("front"), wrist=stack("wrist"), prop=stack("prop"), a_t=stack("a_t"), episode=ep,
                           failed=(~success).cpu().repeat_interleave(T))
    return res


CATS = ["success", "missed_handle", "reached_no_pull", "partial_or_slipped"]


def summarize(res):
    c = res["category"]
    s = {"success_rate": round(res["success"].float().mean().item(), 3), "n": int(res["success"].numel()),
         "mean_max_opening_m": round(res["max_open"].mean().item(), 3),
         "categories": {CATS[i]: int((c == i).sum()) for i in range(4)}}
    if res.get("slipped") is not None:
        sl = res["slipped"]
        s["slips"] = int(sl.sum())
        s["recovered_after_slip"] = round(res["success"][sl].float().mean().item(), 3) if sl.any() else None
        s["success_without_slip"] = round(res["success"][~sl].float().mean().item(), 3) if (~sl).any() else None
    return s


def action_error(model, res):
    """Mean normalized |student - teacher| action error on the states of a rollout that stored teacher labels."""
    d = res["data"]
    errs = []
    with torch.no_grad():
        for i in range(0, d["a_t"].shape[0], 4096):
            s = slice(i, i + 4096)
            a_s = model(d["front"][s].to(DEV), d["wrist"][s].to(DEV), d["prop"][s].to(DEV))
            errs.append(((a_s - d["a_t"][s].to(DEV)) / model.act_std).abs().mean(-1).cpu())
    return torch.cat(errs)


def save_grid_video(name, res, outcome_labels=True):
    frames = res["frames"]
    if not frames:
        return None
    k = min(8, frames[0].shape[0])
    cols = 4
    rows = (k + cols - 1) // cols
    tile = 256
    out = []
    for fr in frames:
        tiles = []
        for i in range(rows * cols):
            if i < k:
                im = cv2.resize(fr[i], (tile, tile), interpolation=cv2.INTER_LINEAR)
                if outcome_labels:
                    e = int(VID[i])
                    ok = bool(res["success"][e])
                    color = (40, 200, 40) if ok else (220, 40, 40)
                    im = cv2.copyMakeBorder(im[4:-4, 4:-4], 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
                    txt = f"ep{e} {CATS[int(res['category'][e])]} {res['max_open'][e]:.2f}m"
                    cv2.putText(im, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                im = np.zeros((tile, tile, 3), np.uint8)
            tiles.append(im)
        grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(rows)], 0)
        cv2.putText(grid, name, (8, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        out.append(grid)
    path = os.path.join(OUT, "videos", f"{name}.mp4")
    imageio.mimsave(path, out, fps=30, macro_block_size=1)
    np.savez_compressed(os.path.join(OUT, f"frames_{name}.npz"), frames=np.stack(frames), success=res["success"].numpy(),
                        video_ids=VID.cpu().numpy(),
                        category=res["category"].numpy(), max_open=res["max_open"].numpy())
    return path


def save_student_view(name, res):
    if not res["sview"]:
        return
    out = []
    for v in res["sview"]:  # (4, 64, 128): RGB-D of front|wrist side by side
        rgb = np.transpose(v[:3], (1, 2, 0)).astype(np.uint8)
        d = cv2.applyColorMap(v[3].astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1]
        im = np.concatenate([rgb, d], 0)
        out.append(cv2.resize(im, (im.shape[1] * 4, im.shape[0] * 4), interpolation=cv2.INTER_NEAREST))
    imageio.mimsave(os.path.join(OUT, "videos", f"student_view_{name}.mp4"), out, fps=30, macro_block_size=1)


FORCED = eval_params(N, args.eval_seed, DEV)


def evaluate(name, model=None):
    t0 = time.time()
    pgen = torch.Generator(device="cpu").manual_seed(args.eval_seed + 7)  # the SAME disturbances for every method
    res = rollout("teacher" if model is None else "student", model, forced=FORCED, store=True, video=True,
                  perturb=args.perturb, pgen=pgen)
    s = summarize(res)
    extra = {}
    if model is not None:  # error of the student on ITS OWN states (labels from the teacher, computed during the rollout)
        err_own = ((res["a_s"].flatten(0, 1).to(DEV) - res["data"]["a_t"].to(DEV)) / model.act_std).abs().mean(-1)
        extra["action_err_on_own_states"] = round(err_own.mean().item(), 4)
    s.update(extra)
    vid = save_grid_video(name, res)
    if name in ("bc", "r2"):
        save_student_view(name, res)
    torch.save({k: res[k] for k in ("success", "max_open", "min_dist", "category", "params", "states", "slipped")}, os.path.join(OUT, f"eval_{name}.pt"))
    s["video"] = vid
    s["eval_rollout_s"] = round(res["rollout_s"], 1)
    s["teacher_ms_per_step"] = round(res["teacher_ms_per_step"], 2)
    s["perturbed"] = bool(args.perturb)
    json.dump(s, open(os.path.join(OUT, f"eval_{name}.json"), "w"), indent=1)
    slip_kw = {k: s[k] for k in ("slips", "recovered_after_slip", "success_without_slip") if k in s}
    timing(f"eval_{name}", t0, success_rate=s["success_rate"], mean_max_opening_m=s["mean_max_opening_m"],
           categories=s["categories"], perturbed=bool(args.perturb), **slip_kw, **extra)
    log(f"EVAL {name}: {json.dumps(s)}")
    return s, res


def collect(driver, model, batches, rnd, ep_offset, keep_episodes=0):
    """keep_episodes>0: keep only the first K episodes in total (demo budget), the rest of the batch is discarded."""
    t0 = time.time()
    ds = Dataset()
    fails = []
    kept = 0
    for b in range(batches):
        if keep_episodes and kept >= keep_episodes:
            break
        disturb = args.perturb and (driver == "student" or args.perturb_demos)
        res = rollout(driver, model, store=True, ep_offset=ep_offset + b * N, perturb=disturb)
        d = res["data"]
        k = min(N, keep_episodes - kept) if keep_episodes else N
        kept += k
        if k < N:  # env-major layout: the first k episodes are the first k*T samples
            d = {key: v[: k * T] for key, v in d.items()}
            res = dict(res, success=res["success"][:k], category=res["category"][:k], max_open=res["max_open"][:k],
                       slipped=None if res.get("slipped") is None else res["slipped"][:k],
                       a_s=res["a_s"][:k] if "a_s" in res else None)
        w = torch.ones(d["a_t"].shape[0])
        dis = None
        if model is not None:
            dis = ((res["a_s"].flatten(0, 1).to(DEV) - d["a_t"].to(DEV)) / model.act_std).abs().mean(-1).cpu()
        ds.add(d["front"], d["wrist"], d["prop"], d["a_t"], d["episode"], rnd, d["failed"], w)
        if dis is not None:
            ds.parts.setdefault("disagree", []).append(dis)
        fails.append(summarize(res))
    sr = float(np.mean([f["success_rate"] for f in fails]))
    timing(f"collect_{rnd}_{driver}", t0, episodes=kept, samples=len(ds), driver_success_rate=round(sr, 3),
           rollout_s_per_batch=round((time.time() - t0) / batches, 1),
           perturbed=bool(args.perturb and (driver == "student" or args.perturb_demos)))
    return ds, sr


def new_student(stats_ds):
    m = Student().to(DEV)
    m.set_stats(stats_ds.cat("act").to(DEV), stats_ds.cat("prop").to(DEV))
    return m


def train(model, ds, steps, name, use_weights=False):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    losses = train_student(model, ds, steps, use_weights=use_weights, log=log)
    torch.save(model.state_dict(), os.path.join(OUT, f"student_{name}.pt"))
    timing(f"train_{name}", t0, steps=steps, samples=len(ds), final_loss=round(float(np.mean(losses[-20:])), 4),
           torch_peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    return model


def load_student(name):
    m = Student().to(DEV)
    m.load_state_dict(torch.load(os.path.join(OUT, f"student_{name}.pt"), map_location=DEV))
    return m.eval()


def weight_for_mining(ds: Dataset):
    """Failure mining: samples from FAILED student episodes x mine_factor; top-20% teacher/student disagreement x2."""
    w = torch.ones(len(ds))
    failed = ds.cat("failed")
    rnd = ds.cat("round")
    w[failed & (rnd > 0)] *= args.mine_factor
    if ds.parts.get("disagree"):
        dis = torch.cat(ds.parts["disagree"])  # only student-driven rounds have it, in order
        idx = torch.nonzero(rnd > 0).squeeze(-1)
        thr = torch.quantile(dis[:200000], 0.8)
        w[idx[dis > thr]] *= 2.0
    ds.parts["weight"] = [w]
    return w


# ---------------------------------------------------------------- stages
if "bench" in STAGES:
    for n_steps in (50, 150):
        env.reset()
        t0 = time.time()
        a = torch.zeros(N, 8, device=DEV)
        for _ in range(n_steps):
            env.step(a)
            student_images()
        torch.cuda.synchronize()
        dt_ = time.time() - t0
        timing("bench_camera_env", t0, steps=n_steps, env_steps_per_s=round(n_steps * N / dt_, 1), steps_per_s=round(n_steps / dt_, 1))
    os._exit(0)

summary = {}
if "teacher_eval" in STAGES:
    summary["teacher"], _ = evaluate("teacher")

if "bc" in STAGES:
    bc_ds, sr = collect("teacher", None, args.bc_batches, rnd=0, ep_offset=0, keep_episodes=args.episodes_per_round)
    val_res = rollout("teacher", None, store=True, ep_offset=10_000)  # held-out teacher episodes
    torch.save(val_res["data"], os.path.join(OUT, "val_teacher.pt"))
    torch.save(torch.cat([val_res["states"].flatten(0, 1)]), os.path.join(OUT, "teacher_val_states.pt"))
    bc_ds.save(os.path.join(OUT, "dataset_bc.pt"))
    model = new_student(bc_ds)
    train(model, bc_ds, args.bc_steps, "bc")
    summary["bc"], _ = evaluate("bc", model)
    summary["bc"]["action_err_on_teacher_states"] = round(action_error(model, val_res).mean().item(), 4)
    json.dump(summary["bc"], open(os.path.join(OUT, "eval_bc.json"), "w"), indent=1)

if "r1" in STAGES:
    bc_ds = Dataset.load(os.path.join(OUT, "dataset_bc.pt"))
    model = load_student("bc")
    r1_ds, sr = collect("student", model, args.round_batches, rnd=1, ep_offset=100_000, keep_episodes=args.episodes_per_round)
    agg = Dataset()
    agg.extend(bc_ds)
    agg.extend(r1_ds)
    agg.parts["disagree"] = r1_ds.parts.get("disagree", [])
    agg.save(os.path.join(OUT, "dataset_r1.pt"))
    torch.save(r1_ds.parts.get("disagree", []), os.path.join(OUT, "disagree_r1.pt"))
    model = copy.deepcopy(model)
    train(model, agg, args.round_steps, "r1")
    summary["r1"], _ = evaluate("r1", model)
    val = torch.load(os.path.join(OUT, "val_teacher.pt"))
    summary["r1"]["action_err_on_teacher_states"] = round(action_error(model, {"data": val}).mean().item(), 4)
    json.dump(summary["r1"], open(os.path.join(OUT, "eval_r1.json"), "w"), indent=1)

if "r2" in STAGES:
    agg = Dataset.load(os.path.join(OUT, "dataset_r1.pt"))
    agg.parts["disagree"] = torch.load(os.path.join(OUT, "disagree_r1.pt"))
    model = load_student("r1")
    r2_ds, sr = collect("student", model, args.round_batches, rnd=2, ep_offset=200_000, keep_episodes=args.episodes_per_round)
    agg.extend(r2_ds)
    agg.parts["disagree"] = agg.parts["disagree"] + r2_ds.parts.get("disagree", [])
    t0 = time.time()
    w = weight_for_mining(agg)
    n_fail = int((agg.cat("failed") & (agg.cat("round") > 0)).sum())
    timing("failure_mining_r2", t0, mined_failed_samples=n_fail, weight_mean=round(w.mean().item(), 2),
           share_of_updates_on_failures=round((w * (agg.cat("failed") & (agg.cat("round") > 0)).float()).sum().item() / w.sum().item(), 3))
    model = copy.deepcopy(model)
    train(model, agg, args.round_steps, "r2", use_weights=True)
    summary["r2"], _ = evaluate("r2", model)
    val = torch.load(os.path.join(OUT, "val_teacher.pt"))
    summary["r2"]["action_err_on_teacher_states"] = round(action_error(model, {"data": val}).mean().item(), 4)
    json.dump(summary["r2"], open(os.path.join(OUT, "eval_r2.json"), "w"), indent=1)
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "..", "policies", "open_drawer_visual.pt"))

if "bc_more" in STAGES:
    bc_ds = Dataset.load(os.path.join(OUT, "dataset_bc.pt"))
    more, _ = collect("teacher", None, 2 * args.round_batches, rnd=0, ep_offset=300_000, keep_episodes=2 * args.episodes_per_round)
    bc_ds.extend(more)
    model = new_student(bc_ds)
    train(model, bc_ds, args.bc_steps + 2 * args.round_steps, "bc_more")
    summary["bc_more"], _ = evaluate("bc_more", model)
    json.dump(summary["bc_more"], open(os.path.join(OUT, "eval_bc_more.json"), "w"), indent=1)

if "reeval" in STAGES:
    # evaluation noise: the SAME policy on the SAME forced episodes + disturbances, several times
    t0 = time.time()
    rep = {}
    for n in ("teacher", "bc", "bc_more", "r1", "r2"):
        if n != "teacher" and not os.path.exists(os.path.join(OUT, f"student_{n}.pt")):
            continue
        m = None if n == "teacher" else load_student(n)
        runs = []
        for r in range(args.reeval):
            pgen = torch.Generator(device="cpu").manual_seed(args.eval_seed + 7)
            res = rollout("teacher" if m is None else "student", m, forced=FORCED, store=False, perturb=args.perturb, pgen=pgen)
            mo = res["max_open"]
            runs.append({"success": round(res["success"].float().mean().item(), 3), "reached_0.30m": round((mo >= 0.30).float().mean().item(), 3)})
        sr = np.array([x["success"] for x in runs]); tg = np.array([x["reached_0.30m"] for x in runs])
        rep[n] = {"runs": runs, "success_mean": round(float(sr.mean()), 3), "success_std": round(float(sr.std()), 3),
                  "reached_0.30m_mean": round(float(tg.mean()), 3), "reached_0.30m_std": round(float(tg.std()), 3)}
        log(f"REEVAL {n}: {json.dumps(rep[n])}")
    json.dump(rep, open(os.path.join(OUT, "reeval.json"), "w"), indent=1)
    timing("reeval", t0, repeats=args.reeval, results={n: (v["reached_0.30m_mean"], v["reached_0.30m_std"]) for n, v in rep.items()})

if "replay" in STAGES:
    # re-run saved policies on chosen eval episodes (same forced params + same disturbances) and film them side by side
    t0 = time.time()
    rows = [("teacher", None)] + [(n, load_student(n)) for n in ("bc", "bc_more", "r1", "r2")
                                  if os.path.exists(os.path.join(OUT, f"student_{n}.pt"))]
    labels = {"teacher": "TEACHER (privileged)", "bc": "BC", "bc_more": "BC x3 demos", "r1": "DAgger r1", "r2": "DAgger r2 + mining"}
    k = VID.numel()
    out_rows, outcome = [], {}
    for n, m in rows:
        pgen = torch.Generator(device="cpu").manual_seed(args.eval_seed + 7)
        res = rollout("teacher" if m is None else "student", m, forced=FORCED, store=False, video=True, perturb=args.perturb, pgen=pgen)
        ids = VID.tolist()
        outcome[n] = {int(e): {"success": bool(res["success"][e]), "category": CATS[int(res["category"][e])],
                               "max_open_m": round(float(res["max_open"][e]), 3),
                               "slipped": None if res.get("slipped") is None else bool(res["slipped"][e])} for e in ids}
        seq = []
        for fr in res["frames"]:
            tiles = []
            for i, e in enumerate(ids):
                im = cv2.resize(fr[i], (224, 224))
                ok = bool(res["success"][e])
                im = cv2.copyMakeBorder(im[4:-4, 4:-4], 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(40, 200, 40) if ok else (220, 40, 40))
                cv2.putText(im, f"ep{e} {res['max_open'][e]:.2f}m", (8, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(im)
            row = np.concatenate(tiles, 1)
            cv2.putText(row, labels[n], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            seq.append(row)
        out_rows.append(seq)
    tlen = min(len(s_) for s_ in out_rows)
    video = [np.concatenate([r[t] for r in out_rows], 0) for t in range(tlen)]
    path = os.path.join(OUT, "videos", "replay_" + "_".join(str(e) for e in VID.tolist()) + ".mp4")
    imageio.mimsave(path, video, fps=30, macro_block_size=1)
    json.dump(outcome, open(path.replace(".mp4", ".json"), "w"), indent=1)
    timing("replay_video", t0, video=path, episodes=VID.tolist(), outcomes={n: [o["success"] for o in v.values()] for n, v in outcome.items()})

if "report" in STAGES:
    t0 = time.time()
    names = [n for n in ("teacher", "bc", "bc_more", "r1", "r2") if os.path.exists(os.path.join(OUT, f"eval_{n}.json"))]
    table = {n: json.load(open(os.path.join(OUT, f"eval_{n}.json"))) for n in names}
    # distribution shift: distance of the states each policy visits to the states the TEACHER visited in the BC data
    ref = torch.load(os.path.join(OUT, "teacher_val_states.pt")).to(DEV)
    g =torch.Generator(device="cpu").manual_seed(0)
    ref = ref[torch.randperm(ref.shape[0], generator=g)[:20000].to(DEV)]

    def nn_dist(states):
        s = states.flatten(0, 1).to(DEV)
        s = s[torch.randperm(s.shape[0], generator=g)[:4000].to(DEV)]
        return torch.cat([torch.cdist(s[i:i + 1000], ref).min(1).values for i in range(0, s.shape[0], 1000)])

    base = nn_dist(torch.load(os.path.join(OUT, "eval_teacher.pt"))["states"])
    thr = torch.quantile(base, 0.95).item()
    for n in names:
        ev_n = torch.load(os.path.join(OUT, f"eval_{n}.pt"))
        d = nn_dist(ev_n["states"])
        table[n]["shift_median_nn_dist"] = round(d.median().item(), 3)
        table[n]["shift_frac_states_off_teacher_support"] = round((d > thr).float().mean().item(), 3)
        # stricter outcome: the target opening of the lab's open_drawer skill (0.30 m), and its failure given a slip
        mo = ev_n["max_open"]
        table[n]["reached_skill_target_0.30m"] = round((mo >= 0.30).float().mean().item(), 3)
        sl = ev_n.get("slipped")
        if sl is not None:
            table[n]["stall_below_0.30m_if_slip"] = round(((mo < 0.30) & sl).sum().item() / max(1, sl.sum().item()), 3)
            table[n]["stall_below_0.30m_if_no_slip"] = round(((mo < 0.30) & ~sl).sum().item() / max(1, (~sl).sum().item()), 3)
    # failure vs randomization: which parameters make BC fail?
    ev = torch.load(os.path.join(OUT, "eval_bc.pt")) if "bc" in names else None
    if ev is not None:
        prm = ev["params"]
        fail = ~ev["success"]
        table["bc_failure_vs_params"] = {PARAM_NAMES[j]: {"fail_mean": round(prm[fail, j].mean().item(), 3) if fail.any() else None,
                                                          "success_mean": round(prm[~fail, j].mean().item(), 3) if (~fail).any() else None}
                                         for j in range(5)}
    # comparison video: rows = methods, cols = the same 4 eval episodes
    rows = []
    for n in [x for x in ("teacher", "bc", "r1", "r2") if x in names]:
        with np.load(os.path.join(OUT, f"frames_{n}.npz")) as zf:  # decompress ONCE (NpzFile re-reads on every access)
            rows.append((n, {k: zf[k] for k in zf.files}))
    if rows:
        # columns: the filmed episodes where BC fell shortest of the 0.30 m skill target (failures first), same for all rows
        zb = dict(rows).get("bc", rows[0][1])
        vids = [int(v) for v in zb.get("video_ids", range(zb["frames"].shape[1]))]
        cols = sorted(range(len(vids)), key=lambda i: float(zb["max_open"][vids[i]]))[:4]
        tlen = min(z["frames"].shape[0] for _, z in rows)
        out = []
        for t in range(tlen):
            rimgs = []
            for n, z in rows:
                tiles = []
                for i in cols:
                    e = vids[i]
                    mo = float(z["max_open"][e])
                    im = cv2.resize(z["frames"][t, i], (224, 224))
                    color = (40, 200, 40) if mo >= 0.30 else ((240, 160, 30) if mo > SUCCESS_OPENING else (220, 40, 40))
                    im = cv2.copyMakeBorder(im[4:-4, 4:-4], 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
                    cv2.putText(im, f"ep{e} max {mo:.2f}m", (8, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(im)
                row = np.concatenate(tiles, 1)
                label = {"teacher": "TEACHER (privileged)", "bc": "BC", "r1": "DAgger r1", "r2": "DAgger r2 + mining"}[n]
                tgt = table[n].get("reached_skill_target_0.30m")
                txt = f"{label}: drawer >=0.30m in {tgt * 100:.0f}% of 128 eps" if tgt is not None else label
                cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
                rimgs.append(row)
            out.append(np.concatenate(rimgs, 0))
        imageio.mimsave(os.path.join(OUT, "videos", "compare_teacher_bc_r1_r2.mp4"), out, fps=30, macro_block_size=1)
        table["compare_video_episodes"] = [vids[i] for i in cols]
    json.dump(table, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    timing("report", t0, summary={n: table[n].get("success_rate") for n in names})
    log("SUMMARY " + json.dumps(table))

log("done")
os._exit(0)
