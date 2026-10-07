"""Allow only active Nginx Let's Encrypt keys with root-only POSIX access.

Called by the root leak scanner. A failure keeps the file in the leak report.
No key contents are printed; no blanket exclusion of certificate directories.
"""
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


def protected_active_key(filename, nginx_config, archive_root=Path('/etc/letsencrypt/archive')):
    path = Path(filename)
    if path.is_symlink() or path.parent.parent != archive_root:
        return False
    if not re.fullmatch(r'privkey[0-9]+\.pem', path.name):
        return False
    try:
        st = path.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_uid != 0 or stat.S_IMODE(st.st_mode) != 0o600:
            return False
        for parent in [path.parent, archive_root]:
            st = parent.stat()
            if st.st_uid != 0 or stat.S_IMODE(st.st_mode) & 0o022:
                return False
        keys = re.findall(r'(?:^|;)\s*ssl_certificate_key\s+["\']?([^;"\'\s]+)["\']?\s*;', nginx_config, re.M)
        return any(Path(k).resolve(strict=True) == path.resolve(strict=True) for k in keys)
    except OSError:
        return False


if __name__ == '__main__':
    try:
        if os.geteuid() != 0:
            sys.exit(1)
        r = subprocess.run(['nginx', '-T'], capture_output=True, text=True, timeout=10)
        sys.exit(0 if r.returncode == 0 and protected_active_key(sys.argv[1], r.stdout) else 1)
    except (OSError, subprocess.TimeoutExpired, IndexError):
        sys.exit(1)
