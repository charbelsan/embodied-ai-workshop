#!/opt/vinci-workshop/nb-venv/bin/python
"""Starter of a LEARNED VERIFIER for the expert challenge: does the scene look like a success?

    ./expert_verifier.py collect --scene pc
    ./expert_verifier.py fit --scene pc
    ./expert_verifier.py predict --scene pc

Input: ONLY the images the agent is allowed to see (/api/observe_allowed, the same as the MCP tool observe).
Model: logistic regression on two 32 x 32 grey thumbnails, trained by the optional extension
in notebooks/02_defi_expert_v2.ipynb and saved in results/verifier/<scene>.npz.
Train and validation episodes use disjoint seeds. You may explore a CNN or a different threshold.
The output is an uncalibrated model score, not a physical success verdict.
"""
import argparse
import fcntl
import secrets
import time
import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

URL = {"cabinet": os.environ.get("VINCI_SERVICE_URL", "http://127.0.0.1:8765"),
       "pc": os.environ.get("VINCI_PC_URL", "http://127.0.0.1:8766")}
WS = Path(os.environ.get("VINCI_WORKSPACE", Path(__file__).resolve().parent))
SIZE = 32


def thumb(img) -> np.ndarray:
    """(H, W, 3) uint8 image -> SIZE x SIZE grey thumbnail in [0, 1], flattened."""
    g = Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(g, dtype=np.float32).ravel() / 255.0


def features(front, wrist) -> np.ndarray:
    return np.concatenate([thumb(front), thumb(wrist), [1.0]]).astype(np.float32)


def allowed_images(scene: str):
    req = urllib.request.Request(URL[scene] + "/api/observe_allowed", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        o = json.loads(r.read())
    dec = lambda k: np.array(Image.open(io.BytesIO(base64.b64decode(o[k]))).convert("RGB"))
    return dec("rgb_front"), dec("rgb_wrist")


def predict(scene: str, front, wrist) -> float:
    path = WS / "results" / "verifier" / f"{scene}.npz"
    if not path.exists():
        sys.exit(f"no trained verifier for {scene}: run notebooks/02_defi_expert_v2.ipynb, optional extension")
    m = np.load(path)
    x = (features(front, wrist) - m["mean"]) / m["std"]
    return float(1.0 / (1.0 + np.exp(-np.clip(x @ m["w"], -50, 50))))


def collect(scene, n_train, n_val):
    if scene != "pc":
        raise ValueError("This seeded collector supports pc. Cabinet prediction from older models remains available.")
    if min(n_train, n_val) < 4:
        raise ValueError("Use at least four examples in each split")
    sys.path.insert(0, os.environ.get("VINCI_RUNTIME", "/opt/vinci-workshop"))
    from vinci_lab.client import Lab
    pc = Lab(URL[scene], timeout=300)
    dest = WS / "results/verifier"; dest.mkdir(parents=True, exist_ok=True)
    # Separate split namespaces, unique episode seeds across every saved collection.
    used = set()
    for file in dest.glob(f"{scene}_split_*.npz"):
        with np.load(file) as data: used.update(map(int, data["seeds"]))
    lock = open(WS / ".evaluation.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        for split, n in (("train", n_train), ("validation", n_val)):
            xs, ys, seeds = [], [], []
            for i in range(n):
                while True:
                    seed = secrets.randbelow(900_000_000) + (1_000_000_000 if split == "validation" else 0)
                    if seed not in used: break
                used.add(seed)
                pc._post("/eval/reset", {"seed": seed})
                if i % 3:
                    r = pc.skill("insert_ram_student")
                    if not r.get("ok"): raise RuntimeError(f"RAM skill error: {r}")
                with urllib.request.urlopen(URL[scene] + "/eval/grade", timeout=120) as r:
                    truth = json.load(r)
                front, wrist = allowed_images(scene)
                xs.append(features(front, wrist)); ys.append(float(truth["ram_seated"])); seeds.append(seed)
                print(f"{split} {i+1}/{n}: label={int(ys[-1])}, seed={seed}", flush=True)
            path = dest / f"{scene}_split_{split}_{time.time_ns()}.npz"
            np.savez(path, X=np.array(xs), y=np.array(ys), seeds=np.array(seeds), split=split)
            print("SAVED", path, flush=True)
    finally:
        lock.close()


def datasets(dest, scene):
    groups = {}
    for split in ("train", "validation"):
        files = sorted(dest.glob(f"{scene}_split_{split}_*.npz"))
        if not files: raise ValueError(f"No {split} examples; collect both splits first")
        with_data = [np.load(f) for f in files]
        try:
            groups[split] = {k: np.concatenate([d[k] for d in with_data]) for k in ("X", "y", "seeds")}
        finally:
            for d in with_data: d.close()
    tr, va = groups["train"], groups["validation"]
    if set(tr["seeds"]) & set(va["seeds"]): raise ValueError("Episode leakage between training and validation")
    for split, data in groups.items():
        if len(set(data["seeds"])) != len(data["seeds"]): raise ValueError(f"Duplicate episode in {split}")
        if len(np.unique(data["y"])) < 2: raise ValueError(f"Need both outcomes in {split}; collect more examples")
    return tr, va


def fit(scene):
    dest = WS / "results/verifier"
    tr, va = datasets(dest, scene)
    mean, std = tr["X"].mean(0), tr["X"].std(0) + 1e-6
    mean[-1], std[-1] = 0, 1
    z = (tr["X"]-mean)/std; w = np.zeros(z.shape[1])
    sigmoid = lambda x: 1/(1+np.exp(-np.clip(x, -50, 50)))
    for _ in range(400):
        pr = sigmoid(z@w); w -= 0.1*(z.T@(pr-tr["y"])/len(pr)+0.01*w)
    metrics = {}
    for split, data in (("train",tr),("validation",va)):
        predicted = sigmoid(((data["X"]-mean)/std)@w) > .5
        y = data["y"].astype(bool)
        metrics[split] = dict(n=len(y), accuracy=float((predicted==y).mean()),
                             false_successes=int((predicted & ~y).sum()), missed_successes=int((~predicted & y).sum()))
    np.savez(dest/f"{scene}.npz",w=w,mean=mean,std=std)
    (dest/f"{scene}_metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
    print("Separate seeded episodes, but easy negatives. This is not evidence of near-failure detection or calibrated confidence.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["predict", "collect", "fit"], nargs="?", default="predict")
    p.add_argument("--scene", choices=list(URL), default="cabinet")
    p.add_argument("--train", type=int, default=12)
    p.add_argument("--validation", type=int, default=8)
    a = p.parse_args()
    if a.command == "collect": collect(a.scene, a.train, a.validation)
    elif a.command == "fit": fit(a.scene)
    else:
        prob = predict(a.scene, *allowed_images(a.scene))
        what = "drawer open > 20 cm" if a.scene == "cabinet" else "RAM stick seated"
        print(json.dumps({"scene":a.scene,"model_score":round(prob,3),"meaning":what,"calibrated":False}))
