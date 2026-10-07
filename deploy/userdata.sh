#!/bin/bash
# Print the cloud-init user-data of one team machine.
#   userdata.sh TEAM DCV_USER PASSWORD_HASH
# Only the SHA-512 hash of the team password goes into the user-data, never the password itself.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
TEAM=$1; DUSER=$2; HASH=$3
cat <<UD
#!/bin/bash
M=/var/log/vinci-timings.log
mark() { echo "\$1 \$(date +%s.%N) \$(date -u +%FT%TZ)" >> \$M; }
mark userdata_start
echo "$TEAM" > /etc/vinci-team
echo "$EVENT" > /etc/vinci-event
id -u $DUSER >/dev/null 2>&1 || useradd -m -s /bin/bash $DUSER
echo '$DUSER:$HASH' | chpasswd -e
passwd -l ubuntu >/dev/null 2>&1
printf '[permissions]\n%%owner%% allow builtin\n$DUSER allow builtin\n' > /etc/dcv/default.perm
sed -i "s#^PERSIST_BUCKET=.*#PERSIST_BUCKET=$OUTPUT_BUCKET#; s#^PERSIST_REGION=.*#PERSIST_REGION=$REGION#" /etc/vinci/workshop.env
/usr/local/bin/vinci-xorg-fix --restart && mark xorg_fixed
systemctl is-active --quiet vinci-warmup.service || systemctl start --no-block vinci-warmup.service
mark userdata_end
UD
