#!/usr/bin/env python3
"""Local declaration-only extraction. Never reads packs, databases, ledgers or provider wire."""
import collections, hashlib, json, pathlib, re, shutil
BASE=pathlib.Path.cwd()
OUT=BASE/'output/card-trace-coding-20261018-v1/extraction'
EVIDENCE=BASE/'output/card-trace-reporting-brainstorm-20261018/EVIDENCE.json'
FIELDS=('model','null','outcome','population','sample_restrictions')
def sha(b): return hashlib.sha256(b).hexdigest()
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def jl(p,rows): p.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in rows))
def bind(p): return {'path':str(p),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
def check(ref):
 p=pathlib.Path(ref['path']); assert sha(p.read_bytes())==ref['sha256'],str(p);return p
E=json.loads(EVIDENCE.read_text()); MAP=pathlib.Path(E['source_private_map']); LAB=pathlib.Path(E['source_reviewed_labels']); ACC=pathlib.Path(E['source_acceptance'])
m=json.loads(MAP.read_text()); labels=json.loads(LAB.read_text()); accepted=json.loads(ACC.read_text())
assert len(m['rows'])==len(labels['rows'])==3000
lab={r['job_id']:r for r in labels['rows']}; assert len(lab)==3000
# Follow the exact committed private-map chain, verifying each prior link.
chain=[]; batches=[]; cur=MAP
while True:
 d=json.loads(cur.read_text()); chain.append(bind(cur));batches.extend(d.get('public_batches',[]))
 if not d.get('previous_private'):break
 cur=check(d['previous_private'])
public={};public_sources=[]
for ref in batches:
 p=check(ref);public_sources.append(bind(p));x=json.loads(p.read_text())
 for i,pair in enumerate(x['pairs']):
  bid=pair['blinded_id'];assert bid not in public
  assert sha(canonical(pair))==x['pair_hashes'][bid],(str(p),bid)
  public[bid]=(pair,{'path':str(p),'sha256':ref['sha256'],'json_pointer':f'/pairs/{i}'},x['pair_hashes'][bid])
assert len(public)==3000,len(public)
# Context and target are hash-verified, target stays private.
contexts=[]
for ref in m['contexts']:
 p=check(ref);dst=OUT/'neutral'/'context'/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
 contexts.append({'name':p.name,'sha256':ref['sha256'],'relative_path':'context/'+p.name,'source':bind(p)})
original_path=check(m['original']); original=json.loads(original_path.read_text())
pack_paths={r['source_pack']['path'] for r in m['rows']};assert len(pack_paths)==52
assert all(pathlib.Path(p).is_file() for p in pack_paths)
neutral=[];private=[];groups=collections.defaultdict(list);field_groups={f:collections.defaultdict(list) for f in FIELDS};counts=collections.Counter();all_source=[]
for i,r in enumerate(m['rows']):
 l=lab[r['job_id']]; pair,source,pairhash=public[r['blinded_id']]
 assert pairhash==r['public_pair_sha256']==l['public_pair_sha256']
 d=pair['candidate_declaration'];assert d==r['raw_result']['declared_analysis'];assert set(d)==set(FIELDS)
 assert sha(json.dumps(d,sort_keys=True,separators=(',',':')).encode())==r['raw_result']['commitment_hash']
 assert pair['target_declaration']==original['commitment']
 for f in FIELDS:assert isinstance(d[f],str) and d[f]
 for key in ('result_sha256','row_hash','canonical_world_hash','stratum','position','order_position'):assert r[key]==l[key],(r['job_id'],key)
 for c in pair['public_context']:
  assert sha(c['text'].encode())==c['sha256']
  assert c['sha256'] in [v['sha256'] for v in contexts]
 coding='ct-'+sha(('card-trace-descriptive-coding-v1\0'+r['job_id']).encode())[:24]
 # Whitespace-only normalization preserves every word, number, formula and punctuation.
 norm={f:re.sub(r'\s+',' ',d[f]).strip() for f in FIELDS};sig=sha(canonical(norm)); fs={f:sha(norm[f].encode()) for f in FIELDS}
 groups[sig].append(coding)
 for f in FIELDS:field_groups[f][fs[f]].append(coding)
 nr={'coding_id':coding,'declaration':d,'declaration_sha256':sha(canonical(d)),'evidence_signature':sig,'field_signatures':fs,'source':source,'field_json_pointers':{f:source['json_pointer']+'/candidate_declaration/'+f for f in FIELDS},'context_version':pair['context_version'],'evidence_scope':'frozen selected-analysis declaration; not a chronological analysis trace'}
 neutral.append(nr)
 pr={'coding_id':coding,'case_id':r['job_id'],'private_map_row_index':i,'private_map_json_pointer':f'/rows/{i}','independent_label_row_index':labels['rows'].index(l),'stratum':r['stratum'],'reviewed_status':l['reviewed_status'],'reviewed_match':l.get('reviewed_match'),'terminal_status':l['terminal_status'],'source_status':r['source_status'],'declaration_status':r['declaration_status'],'declaration_sha256':nr['declaration_sha256'],'public_source':source,'public_pair_sha256':pairhash,'raw_identifiers':{k:r[k] for k in ('blinded_id','world_id','session_id','shard','order_position','position','shared_world_cluster','canonical_world_hash','target_id')},'source_hashes':{k:r[k] for k in ('row_hash','result_sha256','closed_record_sha256')},'commitment_hash':r['raw_result']['commitment_hash'],'source_pack':r['source_pack'],'review_provenance':l['review_provenance'],'accepted_label_source':l.get('label_source'),'raw_status':r['raw_result']['status'],'raw_self_verdict':r.get('raw_self_verdict')}
 private.append(pr); counts[(r['stratum'],l['reviewed_status'],l.get('reviewed_match'))]+=1
assert len({r['coding_id'] for r in neutral})==3000
assert len({r['case_id'] for r in private})==3000
assert {r['case_id'] for r in private}==set(m['claimed_ids'])
for stratum,expected in {'tail':(1142,1135,48,7),'body':(1858,1854,50,4)}.items():
 rows=[r for r in private if r['stratum']==stratum]; observed=(len(rows),sum(r['reviewed_status']=='classified' for r in rows),sum(r['reviewed_match'] is True and r['reviewed_status']=='classified' for r in rows),sum(r['reviewed_status']!='classified' for r in rows));assert observed==expected,(stratum,observed)
 assert accepted['counts'][stratum]==[expected[2],expected[1],expected[3]]
neutral.sort(key=lambda x:x['coding_id'])
jl(OUT/'neutral'/'packets.jsonl',neutral)
# Transparent fixed sample selection uses raw case identifiers privately, exposes only blind IDs.
pilot=sorted(private,key=lambda r:sha(('card-trace-schema-pilot-v1\0'+r['case_id']).encode()))[:48]; pilot_ids={r['coding_id'] for r in pilot}
jl(OUT/'neutral'/'pilot48.jsonl',[r for r in neutral if r['coding_id'] in pilot_ids])
dump(OUT/'neutral'/'pilot48_ids.json',sorted(pilot_ids))
for i in range(0,3000,250):jl(OUT/'neutral'/f'packets-{i//250+1:02}.jsonl',neutral[i:i+250])
jl(OUT/'PRIVATE_join.jsonl',private)
dump(OUT/'PRIVATE_original.json',{'source':bind(original_path),'original_result':original})
# No scientifically meaningful normalization or category extraction is performed.
jl(OUT/'neutral'/'evidence_signature_groups.jsonl',[{'evidence_signature':sig,'coding_ids':ids,'count':len(ids)} for sig,ids in sorted(groups.items())])
jl(OUT/'neutral'/'field_signature_groups.jsonl',[{'field':f,'field_signature':sig,'coding_ids':ids,'count':len(ids)} for f,g in field_groups.items() for sig,ids in sorted(g.items())])
dump(OUT/'neutral'/'context_manifest.json',{'contexts':contexts,'scope':'Only design/codebook; original target declaration intentionally omitted'})
summary={'schema':'card-trace-declaration-extraction-v1','cases':3000,'unique_case_ids':3000,'unique_coding_ids':3000,'public_pairs_verified':3000,'public_batch_file_hashes_verified':len(public_sources),'private_map_chain_hashes_verified':len(chain)-1,'all_five_fields_present':3000,'counts':{s:{'attempted':n,'classified':c,'matches':a,'unresolved':u} for s,(n,c,a,u) in {'tail':(1142,1135,48,7),'body':(1858,1854,50,4)}.items()},'exact_full_declaration_groups':len({sha(canonical(r['declaration'])) for r in neutral}),'whitespace_only_full_declaration_groups':len(groups),'whitespace_only_field_groups':{f:len(g) for f,g in field_groups.items()},'closed_pack_pointer_paths_exist':len(pack_paths),'public_pair_sha256_recomputed':3000,'frozen_commitment_sha256_recomputed':3000,'public_targets_equal_frozen_original':3000,'pack_payloads_consumed':0,'pack_payload_hashes_verified':0,'evidence_sufficiency':'All outcome/sample/model/null/population fields directly available for every case. No need to read analysis events for endpoint coding. Declarations cannot establish temporal exploration.','blinding':'Neutral packets contain only new opaque coding IDs, selected declarations, field signatures and declaration/context provenance. No raw case IDs, strata, reviewed labels, original target narrative or realized numerical results supplied. Structural numeric constants are retained.','source_files':{'private_map':bind(MAP),'independent_labels':bind(LAB),'acceptance':bind(ACC),'original':bind(original_path)},'public_batch_sources':public_sources,'private_map_chain':chain}
dump(OUT/'CHECKS.json',summary)
dump(OUT/'SCHEMA.json',{'neutral':'coding_id; declaration{model,null,outcome,population,sample_restrictions}; declaration_sha256 (canonical JSON UTF-8); evidence_signature (all fields whitespace-only normalized); field_signatures; source{public batch path,sha256,json_pointer}; field_json_pointers; context_version; evidence_scope','private':'PRIVATE_join.jsonl carries every raw ID/disposition/stratum/accepted independent label and provenance, plus pack pointer. PRIVATE_original.json preserves original result and target declaration.','pilot':'48 smallest SHA256(card-trace-schema-pilot-v1\\0 + raw case_id); no content, stratum or label used','normalization':'Only collapse Unicode whitespace and trim. No text removal, lowercasing, number removal, synonym mapping or equivalence inference.'})
print(json.dumps({k:v for k,v in summary.items() if k not in ('source_files','public_batch_sources','private_map_chain')},indent=2))
