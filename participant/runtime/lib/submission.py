"""Frozen, local challenge submissions with a durable outbox. No portal dependency."""
import argparse
import fcntl
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import yaml

PROTOCOL = 'vinci-dev8-seeds0-7-v1'
RT = Path(os.environ.get('VINCI_RUNTIME', '/opt/vinci-workshop'))
WS = Path(os.environ.get('VINCI_WORKSPACE', Path.home() / 'vinci-workshop')).resolve()
QUEUE = Path(os.environ.get('VINCI_SUBMISSION_QUEUE', Path.home() / '.local/state/vinci-submissions'))
CONFIG = Path(os.environ.get('VINCI_PORTAL_CONFIG', '/etc/vinci/portal.json'))
LIMIT = 120 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def freeze(dry=False):
    """Publish a queue entry only after snapshot + archive are complete."""
    QUEUE.mkdir(parents=True, exist_ok=True, mode=0o700)
    sid = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:12]
    pending = QUEUE / ('.building-' + sid)
    dest = QUEUE / sid
    snap = pending / 'workspace'
    snap.mkdir(parents=True)
    try:
        source = WS / 'team'
        total = 0
        for file in source.rglob('*'):
            if file.is_symlink():
                raise ValueError('team/ contains a symbolic link: copy the file into team/ before submitting')
            if file.is_file():
                total += file.stat().st_size
        if total > LIMIT:
            raise ValueError('team/ exceeds 120 MB; remove the datasets')
        shutil.copytree(source, snap / 'team', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
        for name in ('brain.py', 'skills.py'):
            py_compile.compile(str(snap / 'team' / name), doraise=True)
        shutil.rmtree(snap / 'team' / '__pycache__', ignore_errors=True)
        cfg_path = snap / 'team/config.yaml'
        cfg = yaml.safe_load(cfg_path.read_text())
        if not isinstance(cfg, dict) or cfg.get('task', 'kitting_cabinet') != 'kitting_cabinet':
            raise ValueError('Cette soumission attend la tâche kitting_cabinet')
        (snap / 'original-config.yaml').write_text(cfg_path.read_text())
        selections = cfg.get('checkpoints') or {}
        if not isinstance(selections, dict):
            raise ValueError('checkpoints must be a dictionary')
        references = {'open_drawer_rl':'open_drawer_rl.pt', 'open_drawer_student':'open_drawer_visual.pt', 'dagger_teacher':'dagger_teacher.pt'}
        if set(selections) - set(references):
            raise ValueError('Nom de checkpoint inconnu')
        models = {}
        (snap / 'models').mkdir()
        for name, ref in references.items():
            sel = selections.get(name)
            path = Path(sel['path']) if sel else RT / 'policies' / ref
            if not path.is_absolute():
                path = WS / path
            total += path.stat().st_size
            if total > LIMIT:
                raise ValueError('Code and models exceed 120 MB')
            frozen = snap / 'models' / (name + '.pt')
            shutil.copy2(path, frozen)
            digest = sha(frozen)
            if sel and sel.get('sha256') and sel['sha256'] != digest:
                raise ValueError(f'Checkpoint {name} differs from its declared checksum')
            models[name] = {'path':'models/' + frozen.name, 'sha256':digest}
        cfg['checkpoints'] = models
        cfg['evaluate'] = {'episodes':8, 'procs':4, 'seed0':0, 'videos':0}
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest = {str(p.relative_to(snap)):sha(p) for p in sorted(snap.rglob('*')) if p.is_file()}
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        meta = {'id':sid, 'snapshot':digest, 'protocol':PROTOCOL, 'state':'queued', 'files':manifest,
                'created':time.time(), 'runtime_version':(RT / 'VERSION').read_text().strip() if (RT / 'VERSION').exists() else 'unrecorded',
                'runtime_evaluator_sha256':sha(RT / 'lib/evaluate.py')}
        write(pending / 'manifest.json', meta)
        with tarfile.open(pending / 'snapshot.tar.gz', 'w:gz') as tar:
            tar.add(snap, arcname='workspace')
            tar.add(pending / 'manifest.json', arcname='manifest.json')
        if (pending / 'snapshot.tar.gz').stat().st_size > LIMIT:
            raise ValueError('Archive too large')
        write(pending / 'state.json', dict(meta, state='dry-run' if dry else 'queued'))
        os.rename(pending, dest)
        local = WS / 'results/submissions'
        local.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest / 'snapshot.tar.gz', local / (sid + '.tar.gz'))
        print(f'FROZEN COPY: {sid}\nArchive: {local / (sid + ".tar.gz")}', flush=True)
        if dry:
            print('DRY RUN OK: nothing computed or sent.')
        else:
            print('Evaluation in the background: 8 DEV episodes without video. It can take several minutes.\n'
                  'You can close the terminal. Your score, when it is ready: ./submit.sh --status\n'
                  f'Local status: {dest / "state.json"}\nLog: {dest / "evaluate.log"}', flush=True)
            if CONFIG.exists():
                print('The best valid submitted score is sent to the organizers. A portal outage does not delete your copy; '
                      'the upload will retry.', flush=True)
        return dest
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def request(cfg, route, payload=None, archive=None):
    headers = {'X-Team':cfg['team'], 'Authorization':'Bearer ' + cfg['token']}
    data = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
    headers['Content-Type'] = 'application/json'
    if archive:
        data = archive.read_bytes()
        headers.update({'Content-Type':'application/gzip', 'X-Archive-SHA256':sha(archive)})
    req = urllib.request.Request(cfg['url'].rstrip('/') + '/api/worker/' + route, data=data, headers=headers, method='PUT' if archive else 'POST')
    with urllib.request.urlopen(req, timeout=30 if archive else 5) as r:
        return json.load(r)


def publish(job):
    state = json.loads((job / 'state.json').read_text())
    if state['state'] == 'dry-run' or (job / 'published.json').exists():
        return
    if not CONFIG.exists():
        return
    cfg = json.loads(CONFIG.read_text())
    if not cfg['url'].startswith('https://') and not os.environ.get('VINCI_PORTAL_ALLOW_HTTP_TEST'):
        raise ValueError('Le portail doit utiliser HTTPS')
    # Retry the same immutable ID. A late queued event cannot regress an existing result.
    base = {k:state[k] for k in ('id', 'snapshot', 'protocol')}
    request(cfg, 'event', dict(base, state='queued'))
    if state['state'] == 'queued':
        return
    request(cfg, 'event', dict(base, state='running'))
    if state['state'] not in ('complete', 'error'):
        return
    request(cfg, 'archive/' + state['id'], archive=job / 'snapshot.tar.gz')
    payload = {**base, **{k:v for k,v in state.items() if k in ('state', 'seeds', 'episode_scores', 'error', 'runtime_version', 'runtime_evaluator_sha256')}}
    request(cfg, 'event', payload)
    write(job / 'published.json', {'at':time.time()})


def publisher(stop):
    while not stop.is_set():
        for job in sorted(QUEUE.glob('*')):
            if not (job / 'state.json').exists():
                continue
            try:
                publish(job)
                (job / 'sync-error.txt').unlink(missing_ok=True)
            except Exception as exc:
                # No token or request headers in logs.
                (job / 'sync-error.txt').write_text(f'{time.strftime("%H:%M:%S")} {type(exc).__name__}: {exc}\n')
        stop.wait(10)


def evaluate(job):
    state = json.loads((job / 'state.json').read_text())
    for rel, digest in state['files'].items():
        if sha(job / 'workspace' / rel) != digest:
            raise ValueError('Frozen copy modified before evaluation: ' + rel)
    if sha(RT / 'lib/evaluate.py') != state['runtime_evaluator_sha256']:
        raise ValueError('Evaluator version changed since the submission')
    state['state'] = 'running'
    write(job / 'state.json', state)
    env = dict(os.environ, VINCI_WORKSPACE=str(job / 'workspace'), VINCI_TEAM_DIR=str(job / 'workspace/team'),
               VINCI_EVAL_LOCK=str(WS / '.evaluation.lock'), MPLBACKEND='Agg')
    with (job / 'evaluate.log').open('w') as log:
        # evaluate.py enforces its own process-group timeout on all Isaac descendants.
        proc = subprocess.run([sys.executable, str(RT / 'lib/evaluate.py'), '--episodes', '8', '--seed0', '0',
                               '--videos', '0', '--level', 'dev', '--procs', '4'],
                              env=env, cwd=job / 'workspace', stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError('Evaluation interrupted or incomplete; see evaluate.log')
    report = json.loads((job / 'workspace/results/latest/report.json').read_text())
    if not report['complete'] or report['seeds'] != list(range(8)):
        raise RuntimeError('The eight expected episodes are not available')
    # Persist the evaluation evidence off the GPU along with the immutable input snapshot.
    shutil.copy2(job / 'snapshot.tar.gz', job / 'input.tar.gz')
    with tarfile.open(job / 'snapshot.tar.gz', 'w:gz') as tar:
        tar.add(job / 'input.tar.gz', arcname='input.tar.gz')
        tar.add(job / 'manifest.json', arcname='manifest.json')
        actual = (job / 'workspace/results/latest').resolve()
        tar.add(actual, arcname='evaluation-details')
    state.update(state='complete', seeds=report['seeds'], episode_scores=report['episode_scores'])
    write(job / 'state.json', state)


def loop(once=False):
    QUEUE.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (QUEUE / '.worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        # A prior running job was interrupted by a reboot/service restart: never report success.
        for job in QUEUE.glob('*'):
            if (job / 'state.json').exists():
                s = json.loads((job / 'state.json').read_text())
                if s['state'] == 'running':
                    s.update(state='error', error='Worker interrupted during the evaluation. Submit again.')
                    write(job / 'state.json', s)
        stop = threading.Event()
        thread = threading.Thread(target=publisher, args=(stop,), daemon=True)
        thread.start()
        try:
            while True:
                for job in sorted(QUEUE.glob('*')):
                    if not (job / 'state.json').exists():
                        continue
                    state = json.loads((job / 'state.json').read_text())
                    if state['state'] != 'queued':
                        continue
                    try:
                        evaluate(job)
                    except Exception as exc:
                        state.update(state='error', error=str(exc))
                        write(job / 'state.json', state)
                if once:
                    break
                time.sleep(3)
        finally:
            stop.set()
            thread.join(timeout=35)
            if once:
                for job in QUEUE.glob('*'):
                    if (job / 'state.json').exists():
                        try:
                            publish(job)
                        except Exception:
                            pass


def status():
    """Every submission of this machine, oldest first, with its average DEV score once evaluated."""
    rows = []
    for job in sorted(QUEUE.glob('*')):
        f = job / 'state.json'
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        scores = s.get('episode_scores') or []
        avg = sum(scores) / len(scores) if s['state'] == 'complete' and scores else None
        rows.append((s['id'], s['state'], avg, s.get('error', '')))
        print(f"{s['id']}  {s['state']:<9} {'%.1f/100' % avg if avg is not None else '—':>9}  {s.get('error', '')}".rstrip())
    if not rows:
        print('No submission yet: run ./submit.sh')
    else:
        best = max((r[2] for r in rows if r[2] is not None), default=None)
        print(f"Best valid score: {'%.1f/100' % best if best is not None else '— (no complete evaluation yet)'}")
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--once', action='store_true')
    p.add_argument('--status', action='store_true', help='list your submissions and their scores')
    a = p.parse_args()
    os.umask(0o077)
    if a.status:
        status()
    elif a.worker:
        loop(a.once)
    else:
        freeze(a.dry_run)
        if not a.dry_run:
            with (QUEUE / 'worker.log').open('a') as log:
                subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
