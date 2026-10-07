#!/bin/bash
# Who may reach the team machines. Applied live (a few seconds) to every machine of the security group.
#   ./allow_cidr.sh list
#   ./allow_cidr.sh add CIDR [dcv|ssh]       (default dcv = TCP+UDP 8443, the browser access for participants)
#   ./allow_cidr.sh remove CIDR [dcv|ssh]
#   ./allow_cidr.sh open | close             DCV from anywhere (fallback when the venue network is unknown); the per-team
#                                            DCV password still protects each machine
set -euo pipefail
source "$(dirname "$0")/lib.sh"
: "${SG_ID:?run setup_account.sh first}"
rule() {  # verb cidr kind
  local v=$1 c=$2 k=${3:-dcv}
  [ "$k" = ssh ] && [ "$c" = 0.0.0.0/0 ] && { echo "refused: SSH from everywhere"; exit 1; }
  if [ "$k" = ssh ]; then aws_ ec2 "$v-security-group-ingress" --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$c" >/dev/null
  else for p in tcp udp; do aws_ ec2 "$v-security-group-ingress" --group-id "$SG_ID" --protocol $p --port 8443 --cidr "$c" >/dev/null 2>&1 || true; done; fi
  echo "$v $k $c"
}
case "${1:-list}" in
  list) aws_ ec2 describe-security-groups --group-ids "$SG_ID" --query 'SecurityGroups[0].IpPermissions[].[IpProtocol,FromPort,IpRanges[].CidrIp]' --output text ;;
  add) rule authorize "$2" "${3:-dcv}" ;;
  remove) rule revoke "$2" "${3:-dcv}" ;;
  open) rule authorize 0.0.0.0/0 dcv ;;
  close) rule revoke 0.0.0.0/0 dcv ;;
  *) sed -n '2,8p' "$0" ;;
esac
