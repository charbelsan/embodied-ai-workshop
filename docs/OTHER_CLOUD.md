# Running the workshop on another cloud provider

`deploy/` automates **AWS only** (machine image, one machine per team, network access, cleanup). For **Microsoft Azure**, follow [AZURE.md](AZURE.md). Everything else is plain Ubuntu: the same installers run on any provider that rents **Ubuntu 22.04 x86_64 virtual machines with an RTX-class NVIDIA GPU**.

## 1. Choose the GPU

Isaac Sim renders with RT cores. Check the GPU before renting:

| Works | Does not work |
|---|---|
| L40S, L40, L4, A10, A10G, A40, RTX 6000 Ada, RTX A6000, RTX 4090/4080, RTX 3090 | **A100, H100, H200, V100, P100** (no RT cores) |

Per team machine: 24 GB+ of GPU memory (48 GB was used for the event), 16 vCPU, 64-128 GB of RAM, a 200 GB disk. Smaller machines (8 vCPU, 32 GB RAM, L4 24 GB) run the guided notebook and the challenge more slowly. Once a machine is up, `./install/check_machine.sh` gives a verdict in a few seconds. Use NVIDIA driver **580**: provider images often ship a newer driver (595 crashed Isaac Sim 5.1 in our tests); `workshop/bootstrap_machine.sh` installs 580.

## 2. Choose how participants see the desktop

Teams use a desktop with Isaac Sim, a browser (Jupyter) and VS Code.

- **Amazon DCV** (used for the event, browser access, very good for 3D). It is free on AWS EC2 only; on other providers it needs a license (a demo license works for 30 days). `workshop/bootstrap_machine.sh` installs it.
- **Any other remote desktop** that streams an Xorg session on the GPU: your provider's own solution, NoMachine, a VNC server with a web client… Install the desktop part of `bootstrap_machine.sh` (sections 1 and 2) and replace section 3.
- **No remote desktop**: a single person with SSH access can run everything headless (`./start.sh` detects that there is no display) and open Jupyter through an SSH tunnel: `ssh -L 8888:127.0.0.1:8888 user@machine`, then http://127.0.0.1:8888 locally. The 3D window of the simulator is not visible in that mode.

## 3. Install one machine

As a user with sudo, on a fresh Ubuntu 22.04 GPU machine:

```bash
git clone https://github.com/charbelsan/embodied-ai-workshop.git && cd embodied-ai-workshop
scripts/fetch_assets.sh
# a) full team machine (driver 580 + XFCE desktop + Amazon DCV + Isaac Sim + machine services), ~40-60 min:
sudo DESKTOP_USER=$USER workshop/bootstrap_machine.sh      # run it again after the reboot it may ask for
sudo workshop/install_workshop.sh --user $USER --isaac-python /opt/isaac-venv/bin/python
scripts/smoke_test.sh                                     # exit code 0 expected
# b) or, if the provider's image already has NVIDIA driver 580 and a desktop:
./install/install.sh
```

## 4. Three things to adapt

1. **Creating the machines.** `deploy/team.sh` uses the AWS API. Elsewhere, create the machines with your provider's console or tooling: either install each one as above, or install one, save it as an image/snapshot, and create the other machines from it (that is what `deploy/build_image.sh` does on AWS). Then give each machine its team: `sudo vinci-team-init t03 vinci-t03 "$(openssl passwd -6 'password')"` (team id, desktop account, DCV access, warm-up).
2. **Accounts and access.** On AWS, `deploy/team.sh` creates one desktop user and password per team (`vinci-t01`…) and opens port 8443 only to the venue network. Do the same with your provider's firewall: DCV (8443) open to the venue only, SSH only to the organizers.
3. **Saving team outputs.** `vinci-persist` copies team outputs to S3 before a machine is destroyed; leave `PERSIST_BUCKET` empty in `/etc/vinci/workshop.env` to skip it, and copy `~/vinci-workshop/results` and `~/vinci-workshop/team` yourself if you want to keep them.

## Budget

Compare per-hour prices for the GPU above. For reference, the event used 10 machines with an NVIDIA L40S for ~8 h (setup, rehearsal, 4 h of workshop), i.e. ~350 $ on AWS on-demand prices. Always delete the machines and their disks after the event.
