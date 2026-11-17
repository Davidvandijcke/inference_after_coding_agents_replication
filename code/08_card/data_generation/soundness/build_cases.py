"""Rebuilds the prompts of the full run (one per replay with a recovered script, in the order
sent to Great Lakes) and checks them against the hash of the file the run used.
Usage: python3 code/build_cases.py [out.jsonl]"""
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit

audit.PROMPT = (audit.HERE / 'inputs/audit_prompt.txt').read_text()
ids = sorted(k for k in audit.REVIEW if audit.PACKETS[k]['code'])
random.Random(audit.SEED).shuffle(ids)
text = ''.join(json.dumps({'replay_id': k, 'prompt': audit.prompt_for(k)}) + '\n' for k in ids)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(text)
digest = hashlib.sha256(text.encode()).hexdigest()
saved = (audit.HERE / 'inputs/cases.sha256').read_text().split()[0]
print(len(ids), 'cases;', 'prompts match the run' if digest == saved else f'PROMPTS DIFFER: {digest}')
sys.exit(digest != saved)
