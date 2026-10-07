# Shared by the deploy scripts (sourced).
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$D/config.env" ] || { echo "create $D/config.env from config.env.example first"; exit 1; }
set -a; source "$D/config.env"; set +a
STATE_DIR="${STATE_DIR:-$D/state}"; mkdir -p "$STATE_DIR"; chmod 700 "$STATE_DIR"
[ -f "$STATE_DIR/account.env" ] && source "$STATE_DIR/account.env"
[ -f "$STATE_DIR/image.env" ] && source "$STATE_DIR/image.env"
aws_() { aws --region "$REGION" "$@"; }
now() { date -u +%FT%TZ; }
KEY_FILE="${KEY_FILE:-$STATE_DIR/$KEY_NAME.pem}"
SSHO=(-i "$KEY_FILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10)
CREDS="$STATE_DIR/credentials_${EVENT}.csv"
tags() { local t="{Key=Name,Value=$1},{Key=Project,Value=$PREFIX},{Key=event,Value=$EVENT}"; [ -n "${2:-}" ] && t="$t,{Key=team,Value=$2}"; echo "$t"; }
subnets() {  # "az:subnet" of the default VPC, filtered by $AZS
  local vpc; vpc=$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  aws_ ec2 describe-subnets --filters Name=vpc-id,Values="$vpc" --query 'Subnets[].[AvailabilityZone,SubnetId]' --output text |
    while read -r az sn; do [ -z "$AZS" ] || [[ " $AZS " == *" $az "* ]] && echo "$az:$sn"; done | sort
}
wait_ssh() { for _ in $(seq 90); do ssh "${SSHO[@]}" ubuntu@"$1" true 2>/dev/null && return 0; sleep 10; done; return 1; }
public_ip() { aws_ ec2 describe-instances --instance-ids "$1" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text; }
