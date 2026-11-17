# Run on Great Lakes, read-only: stream one Card3000 replay pack and print, for each
# replay, the tool calls and user messages in its final provider request, which
# carries the whole conversation. Usage: python3 - <pack path> < gl_extract_calls.py
import json, os, sqlite3, sys, tarfile, tempfile

tf = tarfile.open(sys.argv[1], 'r|')
reqs, jobs, last = {}, {}, None
for m in tf:
    n = m.name
    if '/results/replays/' in n and n.endswith('/started.json'):
        s = json.load(tf.extractfile(m))
        jobs[s['prestart_admission']['case_id']] = s['job_id']
    elif '/results/replays/' in n and n.endswith('/child-ledger-checkpoint.sqlite'):
        fd, p = tempfile.mkstemp(dir=os.environ.get('TMPDIR', '/tmp'))
        with os.fdopen(fd, 'wb') as t:
            t.write(tf.extractfile(m).read())
        c = sqlite3.connect(p)
        for rid, case, ts in c.execute('select r.id, rc.case_id, r.created_at from requests r '
                                       'join request_cases rc on r.id = rc.request_id'):
            reqs[rid] = (case, ts)
        c.close(); os.unlink(p)
    elif '/provider-receipts/' in n and n.endswith('/request.json'):
        if last is None:
            best = {}
            for rid, (case, ts) in reqs.items():
                if case not in best or ts > best[case][1]:
                    best[case] = (rid, ts)
            last = {rid: case for case, (rid, ts) in best.items()}
            sys.stderr.write('replays %d requests %d\n' % (len(best), len(reqs)))
        rid = n.split('/')[-2]
        if rid in last:
            r = json.load(tf.extractfile(m))
            items = []
            for i in r.get('input', []):
                t = i.get('type')
                if t in ('custom_tool_call', 'function_call'):
                    items.append({'t': t, 'name': i.get('name'), 'input': i.get('input') or i.get('arguments')})
                elif t == 'message' and i.get('role') == 'user':
                    items.append({'t': 'user', 'text': json.dumps(i.get('content'))[:300]})
                elif t == 'compaction' or (t and 'compact' in t):
                    items.append({'t': 'compaction'})
            print(json.dumps({'case_id': last[rid], 'job_id': jobs.get(last[rid]), 'request_id': rid, 'items': items}))
