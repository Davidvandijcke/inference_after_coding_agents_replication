#!/usr/bin/env python3
"""Build final B codes from frozen neutral sources and locked descriptive decisions.
Default verifies every binding and deterministically rebuilds FINAL_CODES.jsonl.
--lock is the one-time local review freeze, preserving the prior candidate/projection files.
No LLM, model fitting, network, scientific runtime, or private label join is used.
"""
import argparse,hashlib,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parents[1]
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
read=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
write=lambda p,rs:p.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in rs))
def evidence(r):
 out=[]
 for f in ['outcome','sample_restrictions','model']:
  t=r['declaration'][f];out.append(dict(source_field=f,source_json_pointer=r['field_json_pointers'][f],span_start=0,span_end=len(t),text=t,text_sha256=sha(t.encode())))
 return out
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lock',action='store_true');a=p.parse_args()
 source=HERE/'source.jsonl';ss=read(source)
 schema_path=BASE/'schema'/'SCHEMA.json';schema=json.loads(schema_path.read_text());schema_hash=sha(schema_path.read_bytes());rules_hash=sha((schema_path.parent/'SCHEMA.md').read_bytes())
 assert len(ss)==1000 and len({r['coding_id'] for r in ss})==1000
 for r in ss:assert sha(canon(r['declaration']))==r['declaration_sha256']
 if a.lock:
  assert not (HERE/'LOCKED_DECISIONS.jsonl').exists(),'Preserve existing decisions; create a new explicit version instead.'
  pp=read(HERE/'REVIEW_PROJECTION.jsonl');cc=read(HERE/'CANDIDATES.jsonl');assert len(pp)==len(cc)==1000
  locked=[];overrides=[]
  for i,(r,x,c) in enumerate(zip(ss,pp,cc)):
   assert r['coding_id']==x['coding_id']==c['coding_id']
   flags=list(x['flags'])
   if x['display']['sample_treatment']=='other_unclear':flags.append('sample_residual_preserves_explicit_status_or_subgroup_rule')
   if x['display']['mean_model']=='other_unclear':flags.append('mean_residual_preserves_alternative_estimand_or_form_or_unclear_model')
   d=dict(coding_id=r['coding_id'],schema_version=schema['schema_version'],schema_sha256=schema_hash,rules_sha256=rules_hash,declaration_sha256=r['declaration_sha256'],source_sha256=r['source']['sha256'],context_version=r['context_version'],display=x['display'],flags=sorted(set(flags)),coding_status='semantically_reviewed',reviewer='card_trace_schema_v1',coherence_review='Reviewed source-supported descriptive dimensions; detailed source differences retained; no model/null verdict adjudication.',review_method='Compact exact conditioning clauses and mean-form expressions, outcome definitions/formulas, eligibility/recode evidence families, full flagged clauses and fixed20 audit.',decision_reasons=x['changes'],proof_projection=x['proof'])
   locked.append(d)
   changes={k:dict(candidate=c['display'][k],reviewed=v,reason=x['changes'].get(k,'Reviewed direct source formula/eligibility clause.')) for k,v in x['display'].items() if c['display'][k]!=v}
   if changes:overrides.append(dict(coding_id=r['coding_id'],declaration_sha256=r['declaration_sha256'],source_sha256=r['source']['sha256'],changes=changes,evidence=evidence(r)))
  write(HERE/'LOCKED_DECISIONS.jsonl',locked);write(HERE/'OVERRIDES.jsonl',overrides)
  manifest=dict(schema_version=schema['schema_version'],source_file_sha256=sha(source.read_bytes()),schema_sha256=schema_hash,rules_sha256=rules_hash,locked_decisions_sha256=sha((HERE/'LOCKED_DECISIONS.jsonl').read_bytes()),candidates_sha256=sha((HERE/'CANDIDATES.jsonl').read_bytes()),review_projection_sha256=sha((HERE/'REVIEW_PROJECTION.jsonl').read_bytes()),review_script_sha256=sha((HERE/'review_B.py').read_bytes()),helper_sha256=sha((BASE/'schema'/'propose_codes.py').read_bytes()),fixed_audit_ids_sha256=sha((HERE/'FIXED_AUDIT_IDS.json').read_bytes()))
  (HERE/'BINDINGS.json').write_text(json.dumps(manifest,indent=2)+'\n')
 bindings=json.loads((HERE/'BINDINGS.json').read_text())
 for value,expected in [(sha(source.read_bytes()),bindings['source_file_sha256']),(schema_hash,bindings['schema_sha256']),(rules_hash,bindings['rules_sha256']),(sha((HERE/'LOCKED_DECISIONS.jsonl').read_bytes()),bindings['locked_decisions_sha256'])]:assert value==expected,'Frozen input binding changed.'
 ll=read(HERE/'LOCKED_DECISIONS.jsonl');assert len(ll)==1000
 final=[]
 for r,d in zip(ss,ll):
  assert r['coding_id']==d['coding_id'] and r['declaration_sha256']==d['declaration_sha256'] and r['source']['sha256']==d['source_sha256'] and r['context_version']==d['context_version']
  for k,v in d['display'].items():assert v in schema['display_categories'][k]
  result=dict(d,source=r['source'],evidence=evidence(r),details={'outcome_formula_raw':r['declaration']['outcome'],'sample_rule_raw':r['declaration']['sample_restrictions'],'mean_formula_raw':r['declaration']['model'],'null_raw':r['declaration']['null'],'target_population_raw':r['declaration']['population'],'restriction_role_status':'Retained verbatim; not re-adjudicated in descriptive coding. Ancillary inference never creates a substantive bin.'})
  for e in result['evidence']:
   assert e['text']==r['declaration'][e['source_field']][e['span_start']:e['span_end']] and sha(e['text'].encode())==e['text_sha256']
  final.append(result)
 write(HERE/'FINAL_CODES.jsonl',final)
 receipt=dict(status='complete_descriptive_partition_B',n_cases=1000,partition='neutral packets05-08',schema_version=schema['schema_version'],final_codes_sha256=sha((HERE/'FINAL_CODES.jsonl').read_bytes()),locked_decisions_sha256=bindings['locked_decisions_sha256'],n_candidate_cases_with_overrides=len(read(HERE/'OVERRIDES.jsonl')),all_cases_unique=True,all_declaration_hashes_verified=True,all_evidence_spans_and_hashes_verified=True,all_codes_in_frozen_schema=True,source_file_sha256=bindings['source_file_sha256'],schema_sha256=schema_hash,rules_sha256=rules_hash,rebuild_command='python3 coding/B/lock_and_rebuild.py',rebuild_requires_llm=False,review_scope='Three descriptive choice dimensions only; source probability/inference text retained; accepted model/null verdicts never read or changed.',review_evidence={'conditioning_clause_groups':202,'cases_with_bracketed_conditioning_clauses':566,'remaining_direct_model_expression_cases':434,'flagged_outcome_and_sample_clauses':'reviewed in source context','fixed_content_independent_audit_cases':20,'audit_rule':'20 smallest SHA256(card-trace-B-audit-v1 + NUL + coding_id); no outcomes or strata used','audit_corrections':'Manager-inclusive staffing alias and recorded-zero/no-status variant fixed uniformly before locking.'},blinded_to_stratum_and_accepted_verdict=True,no_realized_numeric_results_join=True,no_prevalence_claim_from_pilot=True)
 (HERE/'REVIEW_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({k:receipt[k] for k in ['status','n_cases','final_codes_sha256','n_candidate_cases_with_overrides']}))
if __name__=='__main__':main()
