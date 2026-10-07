"""Verify pinned model/video archives before writing any file into the runtime."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile


def metadata(path):
    m = json.loads(Path(path).read_text())
    a = m['assets']
    if m['schema'] != 1 or m['version'] != a['tag']:
        raise ValueError('Inconsistent release manifest')
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*)?', a['tag']):
        raise ValueError('Invalid release tag')
    if a['name'] != f"workshop-assets-{a['tag']}.tar.gz":
        raise ValueError('Invalid asset name')
    return a


def safe_relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name or not p.parts:
        raise ValueError(f'Invalid archive path: {name}')
    return p


def install(manifest, archive, destination):
    a = metadata(manifest)
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != a['sha256']:
        raise ValueError('Archive does not match this release; runtime unchanged')
    wanted = a['files']
    for name in wanted:
        safe_relative(name)
    dest = Path(destination).resolve()
    with tempfile.TemporaryDirectory(prefix='workshop-assets-') as tmp:
        staged = Path(tmp)
        with tarfile.open(archive, 'r:gz') as tf:
            members = tf.getmembers()
            names = [m.name for m in members]
            if len(set(names)) != len(names) or set(names) != set(wanted) | {'ASSETS.sha256'}:
                raise ValueError('Archive members do not match the pinned manifest')
            for member in members:
                safe_relative(member.name)
                if not member.isfile():
                    raise ValueError('Only regular asset files are permitted')
                data = tf.extractfile(member).read()
                if member.name in wanted and hashlib.sha256(data).hexdigest() != wanted[member.name]:
                    raise ValueError(f'Asset checksum mismatch: {member.name}')
                target = staged / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        # The archive's own manifest cannot override the checksums committed with the code.
        sums = ''.join(f'{digest}  {name}\n' for name, digest in sorted(wanted.items()))
        (staged / 'ASSETS.sha256').write_text(sums)
        for name in names:
            target = dest / name
            if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != dest and dest in p.parents):
                raise ValueError(f'Symlink in destination path: {name}')
            if dest not in target.resolve().parents:
                raise ValueError(f'Asset would escape runtime: {name}')
        for name in names:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staged / name, target)
    print(f"assets OK: {a['tag']}, {len(wanted)} verified files")


if __name__ == '__main__':
    try:
        if sys.argv[1] == 'metadata':
            a = metadata(sys.argv[2]); print(a['tag']); print(a['name'])
        elif sys.argv[1] == 'install':
            install(*sys.argv[2:5])
        else:
            raise ValueError('Expected metadata or install')
    except (ValueError, KeyError, OSError, tarfile.TarError) as e:
        print(f'Asset verification failed: {e}', file=sys.stderr)
        sys.exit(1)
