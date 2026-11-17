#!/usr/bin/env python3
"""Offline metadata inventory. Archived commands are opaque data and never executed."""
import collections, gzip, hashlib, io, json, pathlib, statistics, tarfile, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
JOIN = ROOT / 'output/card-trace-coding-20261018-v1/extraction/PRIVATE_join.jsonl'
EVIDENCE = ROOT / 'output/card-trace-reporting-brainstorm-20261018/EVIDENCE.json'
def sha(b): return hashlib.sha256(b).hexdigest()
def dump(p, x): p.write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n')
def js(x): return json.dumps(x, ensure_ascii=False, separators=(',', ':'))
def counts(xs): return dict(collections.Counter(xs))

def inspect(data, metadata):
    events = [json.loads(l) for l in data.splitlines() if l.strip()]
    types = collections.Counter(); itemtypes = collections.Counter(); commands = {}; messages = {}; changes = {}; other = {}
    thread = ''; timestamps = []; reason = []; summaries = []; usage = collections.Counter(); boundaries=[]
    def scan(x, path, line):
        if isinstance(x,dict):
            for k,v in x.items():
                p = path+'.'+k; low=k.lower()
                if low in {'timestamp','time','created_at','started_at','completed_at','finished_at','ended_at','duration','duration_ms','duration_seconds'}:
                    timestamps.append({'line':line,'field':p,'value':v})
                if ('reasoning' in low or low in {'analysis','thought','thoughts'}) and not path.startswith('event.usage'):
                    reason.append({'line':line,'field':p,'nonempty':bool(v),'value_type':type(v).__name__})
                if 'summary' in low:
                    summaries.append({'line':line,'field':p,'nonempty':bool(v),'value_type':type(v).__name__})
                scan(v,p,line)
        elif isinstance(x,list):
            for i,v in enumerate(x): scan(v,path+f'[{i}]',line)
    for line,e in enumerate(events,1):
        typ=e.get('type','<absent>'); types[typ]+=1
        if typ=='thread.started':thread=e.get('thread_id','')
        if typ in {'thread.started','turn.started','turn.completed','turn.failed','error'}: boundaries.append({'line':line,'type':typ})
        scan(e,'event',line)
        for k,v in e.get('usage',{}).items():
            if isinstance(v,(int,float)):usage[k]+=v
        if not isinstance(e.get('item'),dict):continue
        it=e['item']; ityp=it.get('type','<absent>'); itemtypes[ityp]+=1
        key=(thread,it.get('id',f'missing-id-line-{line}'))
        if ityp=='command_execution':
            d=commands.setdefault(key,{'item_id':key[1],'thread_id':thread,'started_lines':[],'completed_lines':[]})
            if typ=='item.started':d['started_lines'].append(line)
            if typ=='item.completed':d['completed_lines'].append(line)
            if typ=='item.completed' or 'command_sha256' not in d:
                d.update({'status':it.get('status'),'exit_code':it.get('exit_code'),'command_sha256':sha(it.get('command','').encode()),'command_chars':len(it.get('command','')),'output_chars':len(it.get('aggregated_output',''))})
        elif ityp=='agent_message':
            if typ=='item.completed':messages[key]={'item_id':key[1],'line':line,'chars':len(it.get('text','')),'text_sha256':sha(it.get('text','').encode())}
        elif ityp=='file_change':
            if typ=='item.completed':changes[key]={'item_id':key[1],'line':line,'status':it.get('status'),'changed_path_records':len(it.get('changes',[]))}
        else:other[key]={'item_id':key[1],'type':ityp,'line':line,'event_type':typ}
    cs=list(commands.values()); n=len(cs)
    for i,c in enumerate(cs):
        c.update({'command_sequence_index':i+1,'command_sequence_fraction':(i+.5)/n,'sequence_third':['early','middle','late'][min(2,3*i//n)]})
    completed=[c for c in cs if c['completed_lines']]
    failed=[c for c in completed if c['status'] in {'failed','error'} or c['exit_code'] not in (None,0)]
    reasoning_value=usage.pop('reasoning_output_tokens',None)
    reasoning_usage={'present':reasoning_value is not None,'state':'absent' if reasoning_value is None else 'zero' if reasoning_value==0 else 'nonzero'}
    return dict(metadata,reasoning_usage=reasoning_usage,eventstream_sha256=sha(data),eventstream_bytes=len(data),event_lines=len(events),event_type_counts=dict(types),item_event_type_counts=dict(itemtypes),unique_command_count=n,unique_completed_commands=len(completed),started_unfinished_commands=sum(bool(c['started_lines']) and not c['completed_lines'] for c in cs),failed_completed_commands=len(failed),duplicate_completed_command_events=sum(max(0,len(c['completed_lines'])-1) for c in cs),commands=cs,visible_message_count=len(messages),visible_message_chars=sum(m['chars'] for m in messages.values()),visible_messages=list(messages.values()),completed_file_change_items=len(changes),changed_path_records=sum(c['changed_path_records'] for c in changes.values()),file_changes=list(changes.values()),other_items=list(other.values()),reasoning_items=sum(v['type'] in {'reasoning','analysis','thinking'} or 'reasoning' in v['type'] for v in other.values()),reasoning_fields=reason,summary_fields=summaries,timestamp_fields=timestamps,usage=dict(usage),phase_boundaries={'scope':'analysis/events.jsonl only; all stream lines precede commitment phase by saved phase separation','events':boundaries,'commitment_comparison_events_consumed':False,'sequence_position_is_elapsed_time':False})

def main():
    started=time.time(); rows=[json.loads(l) for l in JOIN.read_text().splitlines()]; evidence=json.loads(EVIDENCE.read_text())
    pilot=sorted(rows,key=lambda r:sha(('card-tool-trace-pilot-v1:'+r['case_id']).encode()))[:12]
    dump(OUT/'PILOT_SELECTION.json',{'rule':'12 smallest SHA256(UTF8(card-tool-trace-pilot-v1: + case_id)); uses only IDs, before stream inspection; no labels, outcomes, or content','cases':[{'case_id':r['case_id'],'coding_id':r['coding_id'],'rank_hash':sha(('card-tool-trace-pilot-v1:'+r['case_id']).encode())} for r in pilot]})
    packmeta={}; handles={}; inventory=[]; errors=[]
    for r in rows:
        s=r['source_pack']; p=pathlib.Path(s['path'])
        if str(p) in packmeta:continue
        receipt=pathlib.Path(s['export_receipt']['path']); index=pathlib.Path(s['index']['path']); rb=receipt.read_bytes(); ib=index.read_bytes(); rr=json.loads(rb)
        assert sha(rb)==s['export_receipt']['sha256'],str(receipt)
        assert sha(ib)==s['index']['sha256']==rr['index_sha256'],str(index)
        entries=[json.loads(l) for l in ib.splitlines() if l.strip()]
        lookup={(q['case_id'],q['offset'],q['length'],q['payload_sha256'],q['sha256']) for q in entries if q.get('kind')=='case'}
        st=p.stat(); packmeta[str(p)]={'path':str(p),'file_size':st.st_size,'mtime_ns':st.st_mtime_ns,'export_receipt_sha256':sha(rb),'index_sha256':sha(ib),'committed_pack_bytes':rr['committed_pack_bytes'],'whole_pack_sha256':None,'whole_pack_hash_note':'No digest supplied by receipt; only authorized closed byte ranges consumed, not whole pack. Payload SHA256 is recorded for each consumed range.','lookup':lookup}
        handles[str(p)]=p.open('rb')
    pilot_ids={r['case_id'] for r in pilot} | {'original-001'}
    with (OUT/'pilot-analysis-events.jsonl.gzpack').open('wb') as archive, (OUT/'pilot-analysis-events.index.jsonl').open('w') as idx, (OUT/'inventory.jsonl').open('w') as inv:
        def save(data,meta):
            rec=inspect(data,meta)
            if 'source_pack' in meta:rec['eventstream_member']='case/analysis/events.jsonl'
            if meta['case_id'] in pilot_ids:
                compressed=gzip.compress(data,compresslevel=6,mtime=0); off=archive.tell(); archive.write(compressed)
                a={'case_id':meta['case_id'],'offset':off,'length':len(compressed),'uncompressed_bytes':len(data),'eventstream_sha256':rec['eventstream_sha256'],'gzip_member_sha256':sha(compressed)};idx.write(js(a)+'\n');rec['derived_archive']=a
            inventory.append(rec);inv.write(js(rec)+'\n')
        for i,r in enumerate(rows):
            try:
                s=r['source_pack']; q=s['record']; pm=packmeta[s['path']]
                assert q['case_id'].endswith('-closed') and q['kind']=='case'
                assert (q['case_id'],q['offset'],q['length'],q['payload_sha256'],q['index_record_sha256']) in pm['lookup']
                assert q['offset']+q['length']<=pm['committed_pack_bytes']<=pm['file_size']
                f=handles[s['path']]; f.seek(q['offset']);payload=f.read(q['length']); assert len(payload)==q['length']; assert sha(payload)==q['payload_sha256']
                with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as t:
                    m=t.getmember('case/analysis/events.jsonl'); data=t.extractfile(m).read(); phase=json.loads(t.extractfile('case/analysis/phase.json').read()); assert sha(data)==phase['events_sha256']
                    names=set(t.getnames())
                save(data,{'case_id':r['case_id'],'coding_id':r['coding_id'],'session_id':r['raw_identifiers']['session_id'],'stratum':r['stratum'],'source_terminal_status':r['terminal_status'],'source_pack':{k:v for k,v in s.items()},'source_pack_sha256':None,'source_pack_digest_available':False,'payload_sha256_verified':True,'source_eventstream_sha256_verified':True,'analysis_phase_returncode':phase.get('returncode'),'analysis_phase_timed_out':phase.get('timed_out'),'separate_commitment_stream_present':'case/commitment/events.jsonl'in names,'separate_comparison_stream_present':'case/comparison/events.jsonl'in names})
            except Exception as e: errors.append({'case_id':r['case_id'],'exception':type(e).__name__,'detail':str(e)})
            if (i+1)%500==0:print(js({'processed':i+1,'verified':len(inventory),'errors':len(errors),'elapsed_seconds':round(time.time()-started,1)}),flush=True)
        original=pathlib.Path(evidence['original_evidence']['events_path']); data=original.read_bytes();assert sha(data)==evidence['original_evidence']['sha256'];save(data,{'case_id':'original-001','coding_id':None,'source_path':str(original),'expected_eventstream_sha256':evidence['original_evidence']['sha256'],'source_eventstream_sha256_verified':True,'source_pack_sha256':None,'payload_sha256_verified':None})
    for p,m in packmeta.items():
        st=pathlib.Path(p).stat(); assert (st.st_size,st.st_mtime_ns)==(m['file_size'],m['mtime_ns']);m.pop('lookup');handles[p].close()
    dump(OUT/'SOURCE_PACKS.json',list(packmeta.values()));dump(OUT/'ERRORS.json',errors)
    replays=[r for r in inventory if r['case_id']!='original-001']; orig=inventory[-1]
    def distribution(key):
        a=[r[key] for r in replays]; a.sort();return {'total':sum(a),'min':min(a),'p25':a[(len(a)-1)//4],'median':statistics.median(a),'p75':a[3*(len(a)-1)//4],'max':max(a)}
    summary={'schema':'card-precommitment-tool-trace-inventory-v1','script_sha256':sha(pathlib.Path(__file__).read_bytes()),'inputs':{'private_join':{'path':str(JOIN),'sha256':sha(JOIN.read_bytes())},'brainstorm_evidence':{'path':str(EVIDENCE),'sha256':sha(EVIDENCE.read_bytes())}},'coverage':{'expected_replays':len(rows),'verified_replays':len(replays),'originals':1,'errors':len(errors),'source_packs':len(packmeta),'selected_closed_payload_bytes':sum(r['source_pack']['record']['length'] for r in rows),'eventstream_bytes':sum(r['eventstream_bytes'] for r in inventory),'gzip_archive_bytes':(OUT/'pilot-analysis-events.jsonl.gzpack').stat().st_size},'replay_event_type_counts':dict(sum((collections.Counter(r['event_type_counts']) for r in replays),collections.Counter())),'replay_item_event_type_counts':dict(sum((collections.Counter(r['item_event_type_counts']) for r in replays),collections.Counter())),'replay_metrics':{k:distribution(k) for k in ['unique_command_count','unique_completed_commands','started_unfinished_commands','failed_completed_commands','visible_message_count','visible_message_chars','completed_file_change_items','changed_path_records','event_lines']},'reasoning_items_replays':sum(r['reasoning_items'] for r in replays),'cases_with_reasoning_fields':sum(bool(r['reasoning_fields']) for r in replays),'cases_with_summary_fields':sum(bool(r['summary_fields']) for r in replays),'cases_with_structured_timestamps':sum(bool(r['timestamp_fields']) for r in replays),'usage_reasoning_output_tokens_cases':sum(r['reasoning_usage']['present'] for r in replays),'original_summary':{k:v for k,v in orig.items() if k not in ['commands','visible_messages','file_changes','source_path']},'scope':'Only analysis/events.jsonl inspected for process; commitment/comparison contents, SQLite, wire records, archived commands and model fits not executed or analyzed. Original retained separately. No semantic coding of cohort.','elapsed_seconds':round(time.time()-started,1)}
    summary['reasoning_usage_field_states']=dict(collections.Counter(r['reasoning_usage']['state'] for r in replays))
    summary['usage_policy']='Reasoning usage field retained only as present/absent and zero/nonzero; neither text nor a faithful reasoning/effort measure.'
    dump(OUT/'SUMMARY.json',summary);print(js(summary),flush=True)

if __name__=='__main__':main()
