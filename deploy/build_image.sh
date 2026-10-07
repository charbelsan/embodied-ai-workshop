#!/bin/bash
# Build the workshop machine image from a bare Ubuntu 22.04 (~1h30, one g6e instance, then terminated).
#   ./build_image.sh [--keep-builder]
# Steps: launch a builder -> copy workshop/, participant/, scripts/ -> bootstrap_machine.sh (driver, desktop, DCV, Isaac)
#        -> reboot -> install_workshop.sh -> smoke_test.sh (must pass) -> bake_image.sh -> stop -> create image.
# The image id is written to state/image.env (WORKSHOP_AMI) and used by team.sh.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
: "${SG_ID:?run setup_account.sh first}"
REPO="$(cd "$D/.." && pwd)"
[ -f "$REPO/participant/runtime/policies/open_drawer_rl.pt" ] || { echo "models missing: run scripts/fetch_assets.sh first"; exit 1; }
BASE_AMI="${BASE_AMI:-$(aws_ ssm get-parameter --name /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id --query Parameter.Value --output text)}"
echo "$(now) base image $BASE_AMI"
ID=""
for t in $INSTANCE_TYPES; do
  for s in $(subnets); do
    ID=$(aws_ ec2 run-instances --image-id "$BASE_AMI" --instance-type "$t" --subnet-id "${s#*:}" --key-name "$KEY_NAME" \
      --security-group-ids "$SG_ID" --iam-instance-profile "Name=$PROFILE" --metadata-options HttpTokens=required,HttpEndpoint=enabled \
      --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$ROOT_GB,VolumeType=gp3,Iops=3000,Throughput=250,DeleteOnTermination=true}" \
      --tag-specifications "ResourceType=instance,Tags=[$(tags "$PREFIX-builder")]" --query 'Instances[0].InstanceId' --output text 2>&1) && [[ "$ID" == i-* ]] && break 2
    echo "  $t ${s%%:*}: $(echo "$ID" | grep -oE 'InsufficientInstanceCapacity|Unsupported[A-Za-z]*|[A-Za-z]*LimitExceeded|Invalid[A-Za-z.]*|UnauthorizedOperation' | head -1)"; ID=""
  done
done
[[ "$ID" == i-* ]] || { echo "no capacity for a builder ($INSTANCE_TYPES)"; exit 1; }
echo "$(now) builder $ID"
aws_ ec2 wait instance-running --instance-ids "$ID"; IP=$(public_ip "$ID"); wait_ssh "$IP"
rsh() { ssh "${SSHO[@]}" ubuntu@"$IP" "$@"; }
reboot_wait() { rsh 'sudo reboot' || true; sleep 30; wait_ssh "$IP"; }
tar -C "$REPO" -czf - workshop participant scripts | rsh 'rm -rf ~/src && mkdir ~/src && tar -xzf - -C ~/src'
echo "$(now) bootstrap (driver, desktop, DCV, Isaac Sim/Lab)"
rc=0; rsh 'cd ~/src/workshop && sudo ./bootstrap_machine.sh' || rc=$?
if [ "$rc" = 3 ]; then reboot_wait; rsh 'cd ~/src/workshop && sudo ./bootstrap_machine.sh'; elif [ "$rc" != 0 ]; then exit $rc; fi
reboot_wait
echo "$(now) install the workshop"
rsh 'cd ~/src/workshop && sudo ./install_workshop.sh --isaac-python /opt/isaac-venv/bin/python --reset-workspace'
# the boot warm-up ran before the workshop existed: run it now, as on every team machine
# (Isaac UI cold + warm = GUI shader caches, then the workshop service), otherwise the smoke test meets a cold GUI
echo "$(now) boot warm-up (Isaac UI cold+warm, workshop service)"
rsh 'sudo systemctl restart vinci-warmup.service; grep -E "env_loaded|env_failed|smoke_|workspace_" /var/log/vinci-timings.log | tail -6'
echo "$(now) smoke test"
rsh 'bash ~/src/scripts/smoke_test.sh'
rsh 'rm -rf ~/src; sudo bash -s' < "$REPO/workshop/bake_image.sh"
echo "$(now) stop + image"
aws_ ec2 stop-instances --instance-ids "$ID" >/dev/null
until aws_ ec2 wait instance-stopped --instance-ids "$ID" 2>/dev/null; do :; done   # the AWS waiter gives up after 10 min
AMI=$(aws_ ec2 create-image --instance-id "$ID" --name "$PREFIX-$(date -u +%Y%m%d-%H%M)" \
  --tag-specifications "ResourceType=image,Tags=[$(tags "$PREFIX-image")]" "ResourceType=snapshot,Tags=[$(tags "$PREFIX-image")]" --query ImageId --output text)
until aws_ ec2 wait image-available --image-ids "$AMI" 2>/dev/null; do :; done   # a 200 GB image can take more than the waiter's 10 min
echo "WORKSHOP_AMI=$AMI" > "$STATE_DIR/image.env"
[ "${1:-}" = --keep-builder ] || aws_ ec2 terminate-instances --instance-ids "$ID" >/dev/null
echo "$(now) image $AMI ready (state/image.env). Next: ./team.sh create t01"
