#!/bin/bash
# Team machines.
#   ./team.sh create t01 [t02 ...]        launch (instance type / AZ fallback) with a new DCV user + 16-char password each
#   ./team.sh create-all N                t01 .. tNN
#   ./team.sh status [TEAM|all]           READY / NOT READY [failed checks] + URL + user
#   ./team.sh credentials                 the table to hand out (team, URL, user, password)
#   ./team.sh reset TEAM                  restore the workspace + restart the desktop and the simulator (~1 min)
#   ./team.sh stop|start TEAM|all         (the public IP, hence the URL, changes after a start)
#   ./team.sh destroy TEAM|all [--no-save]  copy the team outputs to OUTPUT_BUCKET (verified), then terminate
#   ./team.sh ssh TEAM [command]
set -uo pipefail
source "$(dirname "$0")/lib.sh"
: "${SG_ID:?run setup_account.sh first}"
ids_for() {
  local f=(Name=tag:Project,Values="$PREFIX" Name=tag:event,Values="$EVENT" Name=instance-state-name,Values=pending,running,stopping,stopped)
  [ "$1" != all ] && f+=(Name=tag:team,Values="$1")
  aws_ ec2 describe-instances --filters "${f[@]}" --query 'Reservations[].Instances[].InstanceId' --output text
}
creds_set() {  # team id url user password
  ( flock 9; { grep -v "^$1," "$CREDS" 2>/dev/null; echo "$1,$2,$3,$4,$5"; } > "$CREDS.tmp"; mv "$CREDS.tmp" "$CREDS"; chmod 600 "$CREDS" ) 9>"$STATE_DIR/.lock"
}
launch() {
  local team=$1 duser="vinci-$1" pw hash ud id out
  : "${WORKSHOP_AMI:?no workshop image: run build_image.sh (or set WORKSHOP_AMI in config.env)}"
  pw=$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 16); hash=$(openssl passwd -6 "$pw")
  ud=$(mktemp); chmod 600 "$ud"; "$D/userdata.sh" "$team" "$duser" "$hash" > "$ud"
  for t in $INSTANCE_TYPES; do
    for s in $(subnets); do
      out=$(aws_ ec2 run-instances --image-id "$WORKSHOP_AMI" --instance-type "$t" --subnet-id "${s#*:}" --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" --iam-instance-profile "Name=$PROFILE" --user-data "file://$ud" \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled --instance-initiated-shutdown-behavior stop \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$ROOT_GB,VolumeType=gp3,Iops=3000,Throughput=250,DeleteOnTermination=true}" \
        --tag-specifications "ResourceType=instance,Tags=[$(tags "$PREFIX-$team" "$team")]" "ResourceType=volume,Tags=[$(tags "$PREFIX-$team" "$team")]" \
        --query 'Instances[0].InstanceId' --output text 2>&1)
      if [[ "$out" == i-* ]]; then
        id=$out; creds_set "$team" "$id" PENDING "$duser" "$pw"; rm -f "$ud"
        echo "$(now) $team launched $id ($t, ${s%%:*})"; return 0
      fi
      echo "$(now) $team $t ${s%%:*}: $(echo "$out" | grep -oE 'InsufficientInstanceCapacity|InstanceLimitExceeded|VcpuLimitExceeded|Unsupported[A-Za-z]*|Invalid[A-Za-z.]*|UnauthorizedOperation' | head -1)"
    done
  done
  rm -f "$ud"; echo "$(now) $team FAILED: no capacity for $INSTANCE_TYPES (see README: capacity)"; return 1
}
refresh_urls() {
  [ -f "$CREDS" ] || return 0
  while IFS=, read -r team id url user pw; do
    ip=$(public_ip "$id" 2>/dev/null); [ -n "$ip" ] && [ "$ip" != None ] && url="https://$ip:8443"
    echo "$team,$id,$url,$user,$pw"
  done < "$CREDS" > "$CREDS.tmp"; mv "$CREDS.tmp" "$CREDS"; chmod 600 "$CREDS"
}
status_one() {
  local id=$1 team st typ ip h fails="" user
  read -r team st typ ip <<< "$(aws_ ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].[Tags[?Key==`team`]|[0].Value,State.Name,InstanceType,PublicIpAddress]' --output text)"
  [ "$st" = running ] || fails="ec2_$st"
  if [ "$st" = running ]; then
    h=$(timeout 25 ssh "${SSHO[@]}" ubuntu@"$ip" 'sudo /usr/local/bin/vinci-health' 2>/dev/null | tail -1)
    [ -z "$h" ] && h='{}'
    fails=$(echo "$h" | python3 -c 'import json,sys
try: c=json.load(sys.stdin).get("checks",{})
except Exception: c={}
print(" ".join(k for k,v in c.items() if not v) if c else "health_unreachable")')
  fi
  user=$(grep "^$team," "$CREDS" 2>/dev/null | cut -d, -f4)
  if [ -z "$fails" ]; then echo "$team READY — https://$ip:8443 (user $user) — $typ — $id"
  else echo "$team NOT READY [$fails] — https://$ip:8443 (user $user) — $typ $st — $id"; fi
}
cmd=${1:-help}; shift || true
case "$cmd" in
  create) for t in "$@"; do launch "$t" & done; wait
          ids=$(for t in "$@"; do ids_for "$t"; done); [ -n "$ids" ] && aws_ ec2 wait instance-running --instance-ids $ids
          refresh_urls; echo "ready in ~5-15 min: ./team.sh status   (credentials: $CREDS, chmod 600)" ;;
  create-all) n=${1:?N}; exec "$0" create $(for i in $(seq 1 "$n"); do printf 't%02d ' "$i"; done) ;;
  status) refresh_urls; for id in $(ids_for "${1:-all}"); do status_one "$id" & done; wait ;;
  credentials) refresh_urls; column -s, -t < "$CREDS" | awk '{print $1, $3, $4, $5}' ;;
  reset) ip=$(public_ip "$(ids_for "$1")"); timeout 900 ssh "${SSHO[@]}" ubuntu@"$ip" 'sudo /usr/local/bin/vinci-reset' ;;
  ssh) team=$1; shift; ip=$(public_ip "$(ids_for "$team")"); ssh "${SSHO[@]}" ubuntu@"$ip" "$@" ;;
  stop) ids=$(ids_for "${1:?team or all}"); [ -n "$ids" ] && aws_ ec2 stop-instances --instance-ids $ids --output text >/dev/null && echo "stopping $ids" ;;
  start) ids=$(ids_for "${1:?team or all}"); [ -n "$ids" ] && aws_ ec2 start-instances --instance-ids $ids --output text >/dev/null \
           && aws_ ec2 wait instance-running --instance-ids $ids && refresh_urls && echo "started (new URLs: ./team.sh credentials)" ;;
  destroy)
    for id in $(ids_for "${1:?team or all}"); do
      read -r team st ip <<< "$(aws_ ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].[Tags[?Key==`team`]|[0].Value,State.Name,PublicIpAddress]' --output text)"
      if [ -n "$OUTPUT_BUCKET" ] && [ "${2:-}" != --no-save ]; then
        [ "$st" = running ] || { echo "REFUSED $team: $st, outputs cannot be saved (start it, or --no-save)"; continue; }
        r=$(timeout 1800 ssh "${SSHO[@]}" ubuntu@"$ip" 'sudo /usr/local/bin/vinci-persist' 2>&1 | tail -1)
        [[ "$r" == PERSIST_OK* ]] || { echo "REFUSED $team: outputs not verified ($r); use --no-save to skip"; continue; }
        echo "$team: $r"
      fi
      aws_ ec2 terminate-instances --instance-ids "$id" --output text >/dev/null && echo "$team: terminated $id"
      sed -i "/^$team,/d" "$CREDS" 2>/dev/null
    done ;;
  *) sed -n '2,11p' "$0" ;;
esac
