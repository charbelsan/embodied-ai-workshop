#!/bin/bash
# After the event: terminate every team machine (outputs saved first if OUTPUT_BUCKET is set), then optionally
# delete the images, the security group and the instance role.   ./cleanup_account.sh [--all]
set -uo pipefail
source "$(dirname "$0")/lib.sh"
"$D/team.sh" destroy all
left=$(aws_ ec2 describe-instances --filters Name=tag:Project,Values="$PREFIX" Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)
[ -z "$left" ] || { echo "machines still present: $left (see messages above)"; exit 1; }
if [ "${1:-}" = --all ]; then
  for ami in $(aws_ ec2 describe-images --owners self --filters Name=tag:Project,Values="$PREFIX" --query 'Images[].ImageId' --output text); do
    snaps=$(aws_ ec2 describe-images --image-ids "$ami" --query 'Images[0].BlockDeviceMappings[].Ebs.SnapshotId' --output text)
    aws_ ec2 deregister-image --image-id "$ami"; for s in $snaps; do aws_ ec2 delete-snapshot --snapshot-id "$s"; done; echo "deleted image $ami"
  done
  # a terminated GPU machine keeps its network interface until it is fully shut down (several minutes with Isaac)
  gone=$(aws_ ec2 describe-instances --filters Name=tag:Project,Values="$PREFIX" Name=instance-state-name,Values=shutting-down --query 'Reservations[].Instances[].InstanceId' --output text)
  [ -z "$gone" ] || until aws_ ec2 wait instance-terminated --instance-ids $gone 2>/dev/null; do :; done
  if [ -n "${SG_ID:-}" ]; then
    for _ in $(seq 10); do aws_ ec2 delete-security-group --group-id "$SG_ID" 2>/dev/null && { echo "deleted $SG_ID"; break; }; sleep 30; done
  fi
  aws iam remove-role-from-instance-profile --instance-profile-name "$PREFIX-profile" --role-name "$PREFIX-role" 2>/dev/null
  aws iam delete-instance-profile --instance-profile-name "$PREFIX-profile" 2>/dev/null
  aws iam delete-role-policy --role-name "$PREFIX-role" --policy-name "$PREFIX-machine" 2>/dev/null
  aws iam delete-role --role-name "$PREFIX-role" 2>/dev/null && echo "deleted role and profile"
  aws_ ec2 delete-key-pair --key-name "$KEY_NAME" 2>/dev/null && rm -f "$STATE_DIR/$KEY_NAME.pem" && echo "deleted key pair $KEY_NAME"
  [ -n "${OUTPUT_BUCKET:-}" ] && echo "kept bucket $OUTPUT_BUCKET (team outputs): delete it yourself once copied (aws s3 rb s3://$OUTPUT_BUCKET --force)"
  rm -f "$STATE_DIR/image.env" "$STATE_DIR/account.env"
fi
echo "clean: no workshop machine left in $REGION"
