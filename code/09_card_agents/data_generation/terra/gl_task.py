#!/usr/bin/env python3
"""Great Lakes array task for the Terra cohort (MAIN_PLAN.md, Great Lakes
amendment). Runs cases[task::tasks] of pending.json with main.run_case on one
ChatGPT account. Per-case work stays in node-local storage; the task writes its
own started/results JSONL and one archive tar to out/ in the run folder. Touch
<run folder>/STOP to stop new claims.

  python3 code/gl_task.py --task 0 --tasks 4 --account second=<run>/auth/second:12
  python3 code/gl_task.py --smoke --account second=<run>/auth/second:1
"""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import threading
import time

RUN = Path(__file__).resolve().parents[1]
G = Path('<cluster-project>')
WORK = G / 'terra-work-gl'  # per-case run folders; Slurm's private /tmp breaks bubblewrap
ALLOWED = {'codex-cli-0.165.0', 'terra-rlib-4.5.1', WORK.name}
PROXY = '<cluster-proxy>'
NO_PROXY = 'localhost,127.0.0.1,::1'
R_STACK = '<cluster-software>/arc/stacks/gcc/13.2.0'


def configure():
    """Great Lakes settings for code/pilot.py, set before it is imported."""
    parent = WORK / f"{os.environ.get('SLURM_JOB_ID', 'x')}-{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    parent.mkdir(parents=True, exist_ok=True)
    proxies = {k: PROXY for k in ('HTTPS_PROXY', 'HTTP_PROXY', 'https_proxy', 'http_proxy')}
    proxies.update(NO_PROXY=NO_PROXY, no_proxy=NO_PROXY)
    deny = ['<cluster-home>', '<cluster-other-project>', '<cluster-storage>',
            f'{WORK}/*/ws-*/codex-home', f'{WORK}/*/rec-*', f'{WORK}/*/cases']
    deny += [str(p) for p in sorted(G.iterdir()) if p.name not in ALLOWED]
    agent = {'PATH': f'<cluster-software>/arc/python/3.12.1/bin:{R_STACK}/R/4.5.1/bin:<cluster-software>/arc/gcc/13.2.0/bin:/usr/local/bin:/usr/bin:/bin',
             'LD_LIBRARY_PATH': f'<cluster-software>/arc/python/3.12.1/lib:{R_STACK}/R/4.5.1/lib64/R/lib:<cluster-software>/coe/o/image-libraries/220318/lib:<cluster-software>/arc/gcc/13.2.0/lib64',
             'LIBRARY_PATH': '<cluster-software>/arc/gcc/13.2.0/lib64', 'PKG_CONFIG_PATH': f'{R_STACK}/R/4.5.1/lib64/pkgconfig',
             'R_LIBS': f'{{root}}/home/Rlib:{G}/terra-rlib-4.5.1', 'R_LIBS_SITE': f'{R_STACK}/Rtidyverse/4.5.1/2025-09-30',
             **proxies}
    slash_tmp = parent / 'slash-tmp'
    slash_tmp.mkdir(mode=0o700, exist_ok=True)
    os.environ['REPLAY_WRAP'] = json.dumps(['/usr/bin/bwrap', '--dev-bind', '/', '/', '--bind', str(slash_tmp), '/tmp', '--'])
    os.environ.update(REPLAY_CODEX=str(G / 'codex-cli-0.165.0/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex'),
                      REPLAY_RUN_PARENT=str(parent), REPLAY_DENY_READ=json.dumps(deny),
                      REPLAY_AGENT_ENV=json.dumps(agent), REPLAY_CODEX_ENV=json.dumps(proxies))


def account(text):
    name, rest = text.split('=', 1)
    home, workers = rest.rsplit(':', 1)
    return name, Path(home), int(workers)


def smoke(home):
    import pilot
    root = pilot.new_case(pilot.INPUTS / 'data.csv', auth_file=home / 'auth.json', isolate=True)
    out = pilot.RUN_PARENT / 'smoke'
    try:
        r = pilot.invoke(root, 'Run `Rscript -e "library(sandwich); library(lmtest); cat(nrow(read.csv(\'data.csv\')))"`, '
                               '`ls <cluster-home>` and `curl -sS -o /dev/null -w %{http_code} https://cran.r-project.org/` '
                               'in the shell, then reply with the three outputs only.', out)
        print(r['answer'])
        print(json.dumps(r['usage']))
    except Exception as error:
        print('ERR', error)
        print((out / 'stderr.txt').read_text()[-2000:])
        print((out / 'events.jsonl').read_text()[-2000:])
    finally:
        pilot.remove_auth(root)
        shutil.rmtree(root, ignore_errors=True)


def run(task, ntasks, name, home, workers):
    import main
    import pilot
    # Arrays after the first (65033654, which wrote to out/ and watched STOP)
    # write to out/<array job id>/ and watch STOP-ALL.
    out = RUN / 'out' / os.environ.get('SLURM_ARRAY_JOB_ID', 'x')
    out.mkdir(parents=True, exist_ok=True)
    main.STARTED, main.RESULTS, main.STOP = out / f'started-{task}.jsonl', out / f'results-{task}.jsonl', RUN / 'STOP-ALL'
    main.CASES = pilot.RUN_PARENT / 'cases'
    main.CASES.mkdir(exist_ok=True)
    archive, archive_lock = out / f'cases-{task}.tar', threading.Lock()

    def pack(case_id, root, record):
        local = main.CASES / f'{case_id}-{int(time.time() * 1000)}.tar.gz'
        with tarfile.open(local, 'x:gz') as tar:
            tar.add(record, arcname='record')
            tar.add(Path(root) / 'codex-home/sessions', arcname='sessions')
            tar.add(Path(root) / 'workspace', arcname='workspace', filter=lambda t: None if t.name == 'workspace/data.csv' else t)
        with tarfile.open(local) as tar:
            tar.getmembers()
        with archive_lock, tarfile.open(archive, 'a') as tar:
            tar.add(local, arcname=local.name)
        local.unlink()
        shutil.rmtree(root)
        shutil.rmtree(record)
        return f'{archive.relative_to(RUN)}:{local.name}'

    main.pack = pack
    main.state[name] = {'paused': False, 'used_percent': None}
    cases = json.loads((RUN / os.environ.get('PENDING', 'pending.json')).read_text())['cases'][task::ntasks]
    done = {r['case_id']: r['status'] for r in main.read_jsonl(main.RESULTS)}
    for case_id in {r['case_id'] for r in main.read_jsonl(main.STARTED)} - set(done):
        main.append(main.RESULTS, {'case_id': case_id, 'status': 'unavailable', 'error': 'runner ended before the case finished'})
        done[case_id] = 'unavailable'
    queue = deque(c for c in cases if c['case_id'] not in done or done[c['case_id']] in main.RETRY_ELIGIBLE)
    target = json.loads((RUN / 'runs/original/result.json').read_text())['commitment']

    def worker():
        while not (main.STOP.exists() or main.state[name]['paused']):
            with main.lock:
                case = queue.popleft() if queue else None
            if case is None:
                return
            try:
                result = main.run_case(case, target, (name, home))
            except Exception as error:
                main.append(main.RESULTS, {'case_id': case['case_id'], 'position': case['position'],
                                           'status': 'unavailable', 'error': f'runner: {error}'[-1500:]})
                continue
            if result['status'] in main.RETRY_ELIGIBLE:
                with main.lock:
                    queue.append(case)
                time.sleep(60)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in [pool.submit(worker) for _ in range(workers)]:
            f.result()
    print(json.dumps({'task': task, 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'account': main.state[name],
                      'left_in_queue': len(queue), 'stop_file': main.STOP.exists()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    parser.add_argument('--tasks', type=int, default=1)
    parser.add_argument('--account', type=account, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    configure()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    name, home, workers = args.account
    if args.smoke:
        smoke(home)
    else:
        run(args.task, args.tasks, name, home, workers)
