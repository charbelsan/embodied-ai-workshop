"""Notebook-side client of the skill server (scripts/skill_server.py). Pure stdlib + numpy + PIL: no Isaac import.

    from vinci_lab.client import Lab
    lab = Lab()                 # http://127.0.0.1:8765
    o = lab.observe(); lab.show(o)
    lab.move_to([0.30, 0.30, 0.90]); lab.gripper(open=False)
    lab.skill("open_drawer", provider="rl")
"""
from __future__ import annotations

import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image


class Lab:
    def __init__(self, url: str = "http://127.0.0.1:8765", timeout: float = 900):
        self.url, self.timeout = url.rstrip("/"), timeout

    def _post(self, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    # ---- low-level participant API
    def observe(self) -> dict:
        o = self._post("/api/observe")
        for k in ("rgb_front", "rgb_wrist"):
            o[k] = np.array(Image.open(io.BytesIO(base64.b64decode(o[k]))))
        return o

    def move_to(self, pos, quat=(0.0, 1.0, 0.0, 0.0)) -> dict:
        return self._post("/api/move_to", {"pos": list(pos), "quat": list(quat)})

    def move_joints(self, q=None) -> dict:
        return self._post("/api/move_joints", {"q": q})

    def gripper(self, open: bool = True) -> dict:
        return self._post("/api/gripper", {"open": open})

    def reset(self) -> dict:
        return self._post("/api/reset")

    def success(self) -> dict:
        return self._post("/api/success")

    # ---- skills + Brain trace
    def skill(self, name: str, **args) -> dict:
        return self._post("/skill", {"name": name, "args": args})

    def attach_vla(self, port: int = 6200, checkpoint: str = "", max_actions: int = 750) -> dict:
        """Connect pick_and_place(provider="vla") to a running vla_pipeline.policy_server."""
        return self._post("/attach_vla", {"port": port, "checkpoint": checkpoint, "max_actions": max_actions})

    def start_run(self, run_id: str, goal: str, reset: bool = True, brain: str = "participant") -> dict:
        return self._post("/run", {"run_id": run_id, "goal": goal, "reset": reset, "brain": brain})

    def decision(self, text: str) -> dict:
        return self._post("/decision", {"text": text})

    def end_run(self, notes: str = "") -> dict:
        return self._post("/end", {"notes": notes})

    def showcase(self) -> np.ndarray:
        """PC scene: the 1280x720 view of the slot, for humans (the student and the Brain only get observe())."""
        with urllib.request.urlopen(self.url + "/showcase", timeout=self.timeout) as r:
            return np.array(Image.open(io.BytesIO(base64.b64decode(json.loads(r.read())["rgb_showcase"]))))

    def show_scene(self, title: str = "", student_label: str = "what the student sees"):
        """PC scene: the sharp view, and next to it the two 128 px images the student actually gets."""
        import matplotlib.pyplot as plt
        hd, o = self.showcase(), self.observe()
        fig = plt.figure(figsize=(13, 5.4))
        g = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.62], wspace=0.04, hspace=0.12)
        a = fig.add_subplot(g[:, :2]); a.imshow(hd); a.axis("off")
        for i, k in enumerate(("rgb_front", "rgb_wrist")):
            b = fig.add_subplot(g[i, 2]); b.imshow(o[k]); b.axis("off")
            b.set_title(f"{student_label} · {k[4:]} · {o[k].shape[1]} px", fontsize=9)
        fig.suptitle(title, fontsize=12)
        plt.show()
        return o

    @staticmethod
    def show(o: dict, title: str = ""):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for a, k in zip(ax, ("rgb_front", "rgb_wrist")):
            a.imshow(o[k]); a.set_title(k); a.axis("off")
        fig.suptitle(title or f"cube {o['cube_pos']}  |  tiroir {o['drawer_opening_m']} m  |  pince {o['gripper_width']:.3f} m")
        plt.show()
