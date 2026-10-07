#!/bin/bash
# One-time account setup: security group, instance role/profile, SSH key pair, optional output bucket.
#   ./setup_account.sh          (idempotent: re-running only adds what is missing)
set -euo pipefail
source "$(dirname "$0")/lib.sh"
ADMIN_CIDR="${ADMIN_CIDR:-$(curl -s https://checkip.amazonaws.com)/32}"
VPC=$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
[ "$VPC" != "None" ] || { echo "no default VPC in $REGION"; exit 1; }

SG_ID=$(aws_ ec2 describe-security-groups --filters Name=group-name,Values="$PREFIX-sg" Name=vpc-id,Values="$VPC" --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG_ID" = "None" ]; then
  SG_ID=$(aws_ ec2 create-security-group --group-name "$PREFIX-sg" --description "$PREFIX team machines (DCV 8443, SSH admin)" \
    --vpc-id "$VPC" --tag-specifications "ResourceType=security-group,Tags=[$(tags "$PREFIX-sg")]" --query GroupId --output text)
  echo "security group $SG_ID"
fi
aws_ ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$ADMIN_CIDR" >/dev/null 2>&1 && echo "SSH from $ADMIN_CIDR" || true

ROLE="$PREFIX-role"; PROFILE="$PREFIX-profile"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  echo "role $ROLE"
fi
STMTS="{\"Sid\":\"DcvLicense\",\"Effect\":\"Allow\",\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::dcv-license.$REGION/*\"}"
if [ -n "$OUTPUT_BUCKET" ]; then
  aws_ s3api head-bucket --bucket "$OUTPUT_BUCKET" 2>/dev/null || {
    if [ "$REGION" = us-east-1 ]; then aws_ s3api create-bucket --bucket "$OUTPUT_BUCKET" >/dev/null
    else aws_ s3api create-bucket --bucket "$OUTPUT_BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
    aws_ s3api put-public-access-block --bucket "$OUTPUT_BUCKET" --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    echo "bucket $OUTPUT_BUCKET"; }
  STMTS="$STMTS,{\"Sid\":\"TeamOutputs\",\"Effect\":\"Allow\",\"Action\":[\"s3:PutObject\",\"s3:GetObject\"],\"Resource\":\"arn:aws:s3:::$OUTPUT_BUCKET/team-outputs/*\"}"
  STMTS="$STMTS,{\"Sid\":\"ListTeamOutputs\",\"Effect\":\"Allow\",\"Action\":\"s3:ListBucket\",\"Resource\":\"arn:aws:s3:::$OUTPUT_BUCKET\",\"Condition\":{\"StringLike\":{\"s3:prefix\":[\"team-outputs/*\"]}}}"
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name "$PREFIX-machine" --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[$STMTS]}"
if ! aws iam get-instance-profile --instance-profile-name "$PROFILE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$PROFILE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" --role-name "$ROLE"
  echo "instance profile $PROFILE (IAM needs ~15 s to propagate)"; sleep 15
fi

if ! aws_ ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
  aws_ ec2 create-key-pair --key-name "$KEY_NAME" --query KeyMaterial --output text > "$STATE_DIR/$KEY_NAME.pem"
  chmod 600 "$STATE_DIR/$KEY_NAME.pem"; echo "key pair $KEY_NAME -> $STATE_DIR/$KEY_NAME.pem (keep it safe)"
fi
printf 'SG_ID=%s\nPROFILE=%s\nADMIN_CIDR=%s\n' "$SG_ID" "$PROFILE" "$ADMIN_CIDR" > "$STATE_DIR/account.env"
# the venue network needs account.env (SG_ID), written just above
if [ -n "$PARTICIPANT_CIDR" ]; then "$D/allow_cidr.sh" add "$PARTICIPANT_CIDR" dcv; fi
echo "account ready ($REGION): $STATE_DIR/account.env"
