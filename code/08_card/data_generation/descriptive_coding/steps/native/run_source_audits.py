"""Full-source, prediction-visible audits of sealed A annotations. No source execution."""
from pathlib import Path
import argparse, concurrent.futures, fcntl, importlib.util, json, tempfile, time, subprocess
import run_reviews as native
import copy
import signal, threading

BASE = native.BASE
spec = importlib.util.spec_from_file_location('quote_refs', BASE/'pipeline/quote_refs.py')
refs = importlib.util.module_from_spec(spec); spec.loader.exec_module(refs)
REF_RULES = '''All evidence quotes must be selected using an exact quote_ref from the supplied reference view. Select a reference belonging to the same item. Set ev to include that reference's actual event line. For a link select a reference whose line is in from or to. The host expands the reference to the exact source fragment; it never selects evidence for you. Field aliases are exact duplicates only; all unique content is supplied. The displayed span boundaries are presentation wraps, not source edits. Do not concatenate references or invent reference IDs. Reconstruct source code across adjacent spans before interpreting it. Every command must be judged using its actual body, executed paths and output.\n'''
REF_RULES += '''Before returning, reconcile every explanatory note with structured fields: if a mislabeled Headcount calculation actually used full-time columns, remove Headcount from that command's families. Rebuild each command's families only from its executed code/output. Group means, missingness and printed counts have exec none; row-permutation/input-contract checks are verification_only even when the function computes a result. Read the entire sentence and nearby context around evidence; preserve qualifiers such as before, will, if, and negation. A valid substring may reverse source meaning when clipped. Planned commitment is not completed adoption. An unresolved implementation identity cannot support a confirmed specification revision elsewhere. Every required item-map key must contain that item's annotation, never a copied row from another key.\n'''

def keyed_schema(schema,packet,alias_map):
    """Require each source item exactly once by JSON object key; semantics unchanged."""
    out=copy.deepcopy(schema)
    group_types={'commands':'command_execution','messages':'agent_message','files':'file_change'}
    for parent in ('annotation','item_audit'):
        props=out['properties'][parent]['properties']
        for group,kind in group_types.items():
            ids=[x['item_id'] for x in packet['items'] if x['type']==kind]
            row=props[group]['items']; children={}
            for iid in ids:
                child=copy.deepcopy(row); child['properties']['id']={'type':'string','enum':[iid]}
                own=[q for q,r in alias_map['aliases'].items() if alias_map['source_map']['spans'][r]['item_id']==iid and alias_map['source_map']['spans'][r]['citeable']]
                if not own: raise ValueError('No citable own source for '+iid)
                if 'quote_ref' in child['properties']:
                    child['properties']['quote_ref']={'type':'string','enum':own}
                children[iid]=child
            props[group]={'type':'object','properties':children,'required':ids,'additionalProperties':False}
    return out

def unkey(response):
    out=copy.deepcopy(response)
    for parent in ('annotation','item_audit'):
        for group in ('commands','messages','files'):
            mapping=out[parent][group]
            if not isinstance(mapping,dict): raise ValueError('Expected exact source-item map')
            for iid,row in mapping.items():
                if row['id']!=iid: raise ValueError('Source-item key/id mismatch')
            out[parent][group]=list(mapping.values())
    return out

def strict_schema_check(value):
    if isinstance(value,dict):
        if value.get('type')=='object':
            if set(value.get('required',[]))!=set(value.get('properties',{})):
                raise ValueError('Strict schema required/property mismatch before dispatch')
        for child in value.values(): strict_schema_check(child)
    elif isinstance(value,list):
        for child in value: strict_schema_check(child)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--selection',required=True,type=Path)
    p.add_argument('--a-dir',required=True,type=Path); p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--workers',type=int,default=16); p.add_argument('--credit-ceiling',type=float,default=1500)
    args=p.parse_args(); args.run_dir=args.run_dir.resolve(); args.a_dir=args.a_dir.resolve()
    args.run_dir.mkdir(parents=True,exist_ok=True)
    lock=(args.run_dir/'controller.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ids=json.loads(args.selection.read_text()); assert len(ids)==len(set(ids))
    store=native.api.PacketStore(production=True).preload_ids(ids)
    contract=(BASE/'design/ANNOTATION_CONTRACT.md').read_text()
    workspace=Path(tempfile.mkdtemp(prefix='card-audit-isolated-'))
    (workspace/'AGENTS.md').write_text('Review supplied archived source only. No tools or execution. Return requested structured annotation.\n')
    meter=native.Meter(args.credit_ceiling)
    receipts=list(args.run_dir.glob('*/B/RECEIPT.json'))
    meter.spent=sum(json.loads(x.read_text())['estimated_credit_upper'] for x in receipts); meter.calls=len(receipts)
    freeze={'source_epoch':store.epoch,'selected':ids,'workers':args.workers,'credit_ceiling':args.credit_ceiling,
            'audit_type':'full-source prediction-visible audit','model':'gpt-6-luna','effort':'high',
            'script_sha256':native.sha(Path(__file__).read_bytes()),
            'bridge_sha256':native.sha((BASE/'pipeline/quote_refs.py').read_bytes()),
            'contract_sha256':native.sha(contract.encode()),'rules_sha256':native.sha((native.AUDIT_RULES+native.RULES+REF_RULES).encode())}
    freeze_path=args.run_dir/'RUN_FREEZE.json'
    if freeze_path.exists(): freeze_path=args.run_dir/('RUN_RESUME_'+str(int(time.time()))+'.json')
    native.save(freeze_path,freeze)
    stop_new=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_: stop_new.set())
    signal.signal(signal.SIGINT,lambda *_: stop_new.set())

    def work(identifier):
        if stop_new.is_set():
            return {'id':identifier,'B_complete':False,'audit_valid':False,'not_started_due_to_drain':True}
        packet,_=store.get(identifier); cid=packet.get('coding_id') or packet['case_id']; directory=args.run_dir/cid
        result={'id':cid,'B_complete':False,'audit_valid':False,'semantic_accepted':False}
        if (directory/'RESULT.json').exists(): return json.loads((directory/'RESULT.json').read_text())
        try:
            apath=args.a_dir/cid/'A/response.json'; arpath=args.a_dir/cid/'A/RECEIPT.json'
            # A receipt is written after the child process terminates. Never read a live response.
            while not arpath.exists():
                if native.os.environ.get('CARD_REVIEW_A_TERMINAL')=='1':
                    raise RuntimeError('A phase terminal without receipt; preserve unavailable attempt for repair')
                if stop_new.is_set():
                    return {'id':cid,'B_complete':False,'audit_valid':False,'not_started_due_to_drain':True}
                time.sleep(2)
            ar=json.loads(arpath.read_text())
            if ar['status']!=0 or any(x!='error' for x in ar['tool_items']): raise RuntimeError('A failed or used a tool; retain for bounded repair')
            prediction=json.loads(apath.read_text())
            if prediction['id']!=cid or prediction['hash']!=packet['eventstream_sha256']:
                raise RuntimeError('A source identity/hash mismatch')
            view,smap=refs.compact_view(cid,store)
            directory.mkdir(parents=True,exist_ok=True)
            native.save(directory/'SOURCE_MAP.json',smap)
            native.save(directory/'A_BINDING.json',{'path':str(apath),'sha256':native.sha(apath.read_bytes()),'receipt_sha256':native.sha(arpath.read_bytes())})
            errors=native.api.collect_record_errors(prediction,store)
            prompt=native.AUDIT_RULES+native.RULES+REF_RULES+'\nCONTRACT:\n'+contract
            prompt+='\nFALLIBLE DRAFT:\n'+json.dumps(prediction,ensure_ascii=False)+'\nMECHANICAL ERRORS:\n'+json.dumps(errors,ensure_ascii=False)
            prompt+='\nFULL SOURCE REFERENCE VIEW:\n'+view
            schema=keyed_schema(refs.schema_with_refs(native.api.audit_output_schema(prediction,store)),packet,smap)
            strict_schema_check(schema)
            if stop_new.is_set():
                return {'id':cid,'B_complete':False,'audit_valid':False,'not_started_due_to_drain':True}
            raw,receipt=native.call(prompt,schema,directory/'B','high',meter,workspace)
            result['B_complete']=True
            expanded,lookup=refs.expand_compact_refs(unkey(raw),smap)
            native.save(directory/'audit-expanded.json',expanded); native.save(directory/'REFERENCE_LOOKUP.json',lookup)
            validation=native.api.collect_audit_errors(prediction,expanded,store,deduplicate_sets=True)
            native.save(directory/'VALIDATION.json',validation)
            result['validation']=validation
            canonical,reconciliation=native.api.canonicalize_record(expanded['annotation'],store,deduplicate_sets=True)
            native.save(directory/'audited-derived.json',canonical); native.save(directory/'DERIVED_RECONCILIATION.json',reconciliation)
            result['audit_valid']=validation['valid']
            result['unresolved']=expanded['unresolved_adjudication']; result['needs_semantic_acceptance']=True
        except Exception as e: result.update(error=str(e),error_type=type(e).__name__)
        native.save(directory/'RESULT.json',result); return result

    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,i) for i in ids]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            status={'requested':len(ids),'finished':len(results),'B_complete':sum(x['B_complete'] for x in results),
                    'audit_valid':sum(x['audit_valid'] for x in results),'estimated_credit_upper':meter.spent,
                    'inflight_credit_reservation':meter.reserved,'model_calls':meter.calls,
                    'not_started_due_to_drain':sum(x.get('not_started_due_to_drain',False) for x in results)}
            native.save(args.run_dir/'STATUS.json',status); print(json.dumps(status),flush=True)
    native.save(args.run_dir/'RESULTS.json',results)

if __name__=='__main__': main()
