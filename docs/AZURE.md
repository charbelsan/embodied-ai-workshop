# Running the workshop on Microsoft Azure

`deploy/` automates AWS. On Azure, the same machine installer is used; only the cloud steps differ.

**What is validated and what is not.** The machine installer (`workshop/bootstrap_machine.sh` + `install_workshop.sh` on Ubuntu 22.04 with NVIDIA driver 580.178.04) and the end-to-end smoke test were validated on real GPUs (NVIDIA L40S and L4). The Azure-specific steps below follow Microsoft's documentation but **have not been run by us on Azure yet**: run the single-machine part first (sections 1-5) and check `scripts/smoke_test.sh` before creating ten machines.

## 1. The virtual machine

| Item | Value |
|---|---|
| Size | **`Standard_NV36ads_A10_v5`**: one full NVIDIA A10 (24 GB), 36 vCPU, 440 GiB of RAM |
| Not suitable | `NV6/12/18ads_A10_v5` (1/6 to 1/2 of a GPU, 4-12 GB); `NC…A100`, `NC…H100` (no RT cores: Isaac Sim cannot render); `NCasT4_v3` (T4 16 GB, below Isaac Sim's minimum) |
| Quota | "Standard NVADSA10v5 Family vCPUs": 36 per machine, i.e. **432 vCPU for 10 teams + 2 spares**. Request it days in advance. |
| Image | Ubuntu Server 22.04 LTS, Gen2 |
| Security type | **Standard** (Secure Boot and vTPM must be off for the NVIDIA GRID driver) |
| Disk | 200 GB Premium SSD |
| Price | per-hour price of the size in your region: Azure pricing calculator. GRID licensing is included in the VM price. |

Name the admin user **`ubuntu`**: the installer and the smoke test use it as the desktop user, exactly as on AWS.

```bash
az group create -n vinci-workshop -l <region>
az network nsg create -g vinci-workshop -n vinci-nsg
az network nsg rule create -g vinci-workshop --nsg-name vinci-nsg -n ssh-organizers --priority 100 \
  --protocol Tcp --destination-port-ranges 22 --source-address-prefixes <your-ip>/32 --access Allow
az network nsg rule create -g vinci-workshop --nsg-name vinci-nsg -n dcv-venue --priority 110 \
  --protocol Tcp --destination-port-ranges 8443 --source-address-prefixes <venue-network-cidr> --access Allow
az vm create -g vinci-workshop -n vinci-builder --size Standard_NV36ads_A10_v5 \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest --security-type Standard \
  --admin-username ubuntu --ssh-key-values ~/.ssh/id_ed25519.pub --nsg vinci-nsg --os-disk-size-gb 200
```

## 2. The NVIDIA driver: Azure's GRID driver, version 580

A10 VMs need the GRID driver redistributed by Microsoft (the public data-center driver is not supported there). Microsoft lists two for NVadsA10_v5: **vGPU 19.6 LTS = 580.178.04** and vGPU 20.x = 595.x. **Use 580.178.04**: the workshop is validated with 580.178.04, and Isaac Sim 5.1 crashed at start-up with driver 595 in our tests. Take the current link from Microsoft's page "Azure N-series GPU driver setup for Linux" (GRID drivers table), then on the VM:

```bash
sudo apt update && sudo apt install -y build-essential
wget -O NVIDIA-Linux-x86_64-580.178.04-grid-azure.run "<link of vGPU19.6 (LTS) for NVadsA10_v5>"
sudo sh NVIDIA-Linux-x86_64-580.178.04-grid-azure.run
nvidia-smi                       # NVIDIA A10, driver 580.178.04
```

## 3. Install the workshop machine

`bootstrap_machine.sh` sees that driver 580 is already installed and skips its own driver step.

```bash
git clone https://github.com/charbelsan/embodied-ai-workshop.git && cd embodied-ai-workshop
scripts/fetch_assets.sh
sudo workshop/bootstrap_machine.sh        # desktop, Amazon DCV, Isaac Sim 5.1 + Isaac Lab 2.3.2 (~40-60 min)
sudo reboot
sudo workshop/install_workshop.sh --isaac-python /opt/isaac-venv/bin/python
scripts/smoke_test.sh                     # 16 checks, exit code 0 expected
```

**Amazon DCV outside AWS:** DCV is free on AWS only. Elsewhere it starts with an automatic **30-day evaluation license from its installation**: build the machine image less than 30 days before the event, or buy a license from an Amazon DCV reseller. Any other GPU-accelerated remote desktop also works (see `docs/OTHER_CLOUD.md`).

## 4. One machine per team

Capture the machine as a **specialized** image (it keeps the installed workshop and the `ubuntu` user; a generalized image would remove them), then create one VM per team from it:

```bash
az vm deallocate -g vinci-workshop -n vinci-builder
az sig create -g vinci-workshop --gallery-name vinciGallery
az sig image-definition create -g vinci-workshop --gallery-name vinciGallery --gallery-image-definition workshop \
  --publisher vinci --offer workshop --sku v1 --os-type Linux --os-state specialized --hyper-v-generation V2
az sig image-version create -g vinci-workshop --gallery-name vinciGallery --gallery-image-definition workshop \
  --gallery-image-version 1.0.0 --virtual-machine $(az vm show -g vinci-workshop -n vinci-builder --query id -o tsv)
IMAGE=$(az sig image-version show -g vinci-workshop --gallery-name vinciGallery --gallery-image-definition workshop \
  --gallery-image-version 1.0.0 --query id -o tsv)
for t in t01 t02 t03; do   # ... up to t10
  az vm create -g vinci-workshop -n vinci-$t --image "$IMAGE" --specialized --size Standard_NV36ads_A10_v5 \
    --security-type Standard --nsg vinci-nsg
done
```

Then give each machine its team account (the password is passed as a hash, never in clear):

```bash
PASS=$(openssl rand -base64 12)            # keep each team's password in a private file
az vm run-command invoke -g vinci-workshop -n vinci-t01 --command-id RunShellScript \
  --scripts "vinci-team-init t01 vinci-t01 '$(openssl passwd -6 "$PASS")'"
```

Each team then opens `https://<public-ip-of-its-VM>:8443` in a browser and logs in with its user (`vinci-t01`…) and password. The certificate warning is normal: see `participant/PARTICIPANT_ACCESS.md`.

## 5. During and after the event

- Check a machine: `ssh ubuntu@<ip> 'bash -s' < scripts/smoke_test.sh`.
- Collect the scores: each team runs `./submit.sh --status` (best valid score on the 8 DEV episodes).
- Pause: `az vm deallocate -g vinci-workshop -n vinci-t01` stops the compute billing (disks are still billed).
- **Delete everything after the event:** `az group delete -n vinci-workshop` (machines, disks, image, network).
