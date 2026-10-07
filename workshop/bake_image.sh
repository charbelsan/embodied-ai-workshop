#!/bin/bash
# Run as root on a builder machine (bootstrap_machine.sh + install_workshop.sh done, smoke test passed), right before
# creating the machine image: refresh the pristine workspace, remove secrets / personal / per-instance state, audit.
#   sudo ./bake_image.sh
# Then stop the instance and create the image (deploy/build_image.sh does all of this for you).
set -u
R=/var/log/workshop-bake-report.txt; : > $R
say() { echo "$*" | tee -a $R; }
say "== bake $(date -u +%FT%TZ)"
source /etc/vinci/workshop.env

# 1. services: the warm-up starts the simulator at every boot; the simulator service itself is started by the warm-up
systemctl daemon-reload; systemctl enable vinci-xorg-fix.service vinci-warmup.service; systemctl disable vinci-skill-server.service vinci-pc-scene.service 2>/dev/null

# 2. pristine workspace = the state restored by deploy/team.sh reset
rm -rf "$WS_DIR/results"/* "$WS_DIR/results"/.[!.]* 2>/dev/null
find "$WS_DIR" -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
rm -rf /opt/vinci/pristine; mkdir -p /opt/vinci/pristine; rsync -a "$WS_DIR/" /opt/vinci/pristine/workspace/
say "pristine workspace: $(du -sh /opt/vinci/pristine/workspace | cut -f1) from $WS_DIR"

# 3. caches kept in the image (they save minutes at the first launch)
for d in /home/*/.cache/ov /home/*/.nv/ComputeCache /home/*/.cache/nvidia /home/*/.cache/warp /home/*/.local/share/ov; do
  [ -e "$d" ] && say "cache $(du -sh "$d" 2>/dev/null)"
done

# 4. secrets, personal data, per-instance state
for u in /home/* /root; do
  rm -f  $u/.ssh/authorized_keys $u/.ssh/id_* $u/.ssh/known_hosts* $u/.git-credentials $u/.netrc $u/.gitconfig \
         $u/.bash_history $u/.python_history $u/.lesshst $u/.viminfo $u/.wget-hsts $u/.zsh_history \
         $u/.cache/huggingface/token $u/.huggingface/token
  rm -rf $u/.aws $u/.cache/huggingface/stored_tokens $u/.config/wandb $u/.ipython/profile_default/history.sqlite \
         $u/.local/share/jupyter/runtime $u/.config/gh $u/.docker/config.json
done
for tu in $(getent passwd | cut -d: -f1 | grep '^vinci-'); do userdel -r "$tu" 2>/dev/null && say "removed team user $tu"; done
passwd -l ubuntu >/dev/null 2>&1
printf '[permissions]\n%%owner%% allow builtin\n' > /etc/dcv/default.perm
rm -f /etc/vinci-team /etc/vinci-event /etc/vinci-deadline /etc/cron.d/vinci-deadline /var/log/vinci-timings.log
rm -rf /var/log/vinci-warmup/* /var/log/dcv/* /tmp/* /var/tmp/*
journalctl --rotate >/dev/null 2>&1; journalctl --vacuum-time=1s >/dev/null 2>&1
find /var/log -type f \( -name '*.gz' -o -name '*.[0-9]' \) -delete
cloud-init clean --logs >/dev/null 2>&1
truncate -s 0 /etc/machine-id; rm -f /var/lib/dbus/machine-id; ln -sf /etc/machine-id /var/lib/dbus/machine-id
say "per-instance state cleaned (cloud-init, machine-id, logs, team users, DCV permissions, SSH keys)"

# 5. audit: paths only, never values (library sources are skipped: they contain documentation fixtures)
PAT='AKIA[0-9A-Z]{16}|aws_secret_access_key|hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github''_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9-]{40,}|xox[bp]-[A-Za-z0-9-]{20,}|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY'
hits=$(grep -rIlE "$PAT" /home /root /etc /opt --exclude-dir=site-packages --exclude-dir=dist-packages \
        --exclude-dir=.venv --exclude-dir='*venv*' --exclude-dir=node_modules --exclude-dir=.cache \
        --exclude-dir=extscache --exclude-dir=.vscode --exclude-dir=pip_prebundle --exclude-dir=exts --exclude-dir=ssl --exclude-dir=ssh --exclude-dir=dcv --exclude-dir=embodiedswe 2>/dev/null)
say "secret audit: ${hits:-none}"
say "== bake done; next: stop the instance and create the image"
[ -z "$hits" ]
