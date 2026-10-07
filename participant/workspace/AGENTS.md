# Working with your coding agent (Claude Code, Codex, ...)

Bring your own agent and your own account: open the **Terminal** icon, then

```bash
claude            # Claude Code: first run asks you to log in (browser link) with YOUR account
codex             # Codex CLI: `codex login` with YOUR account
```

If the login page never answers, the machine cannot reach the service from this network: tell an organizer.

## The robot, as tools (MCP server `vinci-robot`, already configured for both agents)

| tool | what it does |
|---|---|
| `observe(scene)` | front + wrist camera images, and the robot's own joint state. No object pose, no measurement of the drawer. |
| `open_drawer()` | **learned skill**: the vision student (RGB-D + joints) opens the top drawer. It reports that it ran and which model (sha256) ran, **not** whether the drawer is open: look at the images. |
| `close_drawer()` | **declared controller** (scripted; it uses the handle pose): pushes the drawer closed. |
| `insert_ram()` | **learned skill** of the PC scene: the vision student inserts the RAM stick already held by the gripper into its slot (the case moves from one episode to the next). It reports that it ran and which model ran, **not** whether the stick is seated: check with `observe(scene='pc')`. At the end the stick is released and the hand climbs so the wrist camera looks down on the slot; there is no re-grasp: `./reset.sh` starts a new PC episode. Measured limit: from these views, a reference agent never claimed a false success but could not confirm any real one (the last millimetres do not show at 128 px). |

In the team code (`team/brain.py`), the same two images come from `tools.lab.observe()["rgb_front"]` / `["rgb_wrist"]`:
see `team/README.md` (during `./evaluate.sh` they exist only when a visual provider is selected).

Your agent is the **Brain**: it decides what to call, checks the result from the images, and tries again if needed.
Your models come from `./train.sh` (then `./train.sh select results/<run>`): the tools run exactly the selected sha256.
You may code anything in this folder with any tool. The simulator runtime (`/opt/vinci-workshop`) is not meant to be changed.
