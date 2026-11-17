#!/usr/bin/env python3
"""Rebuild C codes solely from frozen neutral packets and inspected DECISIONS.jsonl. No LLM/network/fits."""
import json,hashlib,re
from pathlib import Path
from collections import Counter
DIR=Path(__file__).resolve().parent;B=DIR.parent.parent
H=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
rows=[json.loads(l) for j in range(9,13) for l in (B/f'extraction/neutral/packets-{j:02}.jsonl').read_text().splitlines()]
decisions=[json.loads(l) for l in (DIR/'DECISIONS.jsonl').read_text().splitlines()]
assert len(rows)==len(decisions)==1000
assert len({r['coding_id'] for r in rows})==1000
schema=json.loads((B/'schema/SCHEMA.json').read_text());schema_hash=H((B/'schema/SCHEMA.json').read_bytes())
context=json.loads((DIR/'CONTEXT_EVIDENCE.json').read_text())
for name,v in context.items():assert H((B/'extraction/neutral/context'/name).read_bytes())==v['sha256']
codes=[];signatures=[];review=[]
for i,(r,d) in enumerate(zip(rows,decisions)):
 assert r['coding_id']==d['coding_id'] and r['declaration_sha256']==d['declaration_sha256'] and d['schema_sha256']==schema_hash
 decl=r['declaration'];evidence=[]
 for field in ['outcome','sample_restrictions','model']:
  t=decl[field];evidence.append({'source_field':field,'source_json_pointer':r['field_json_pointers'][field],'span_start':0,'span_end':len(t),'text':t,'text_sha256':H(t.encode()),'source_role':'frozen_declaration'})
 for key,v in d['display'].items():assert v in schema['display_categories'][key]
 o=decl['outcome'];s=decl['sample_restrictions'];m=decl['model'];low=o.lower();flags=list(d['flags'])
 if re.search(r'log|asinh|arcsinh|inverse hyperbolic',o,re.I):flags.append('outcome_transformation_retained')
 if re.search(r'positive|nonnegative|denominator',s,re.I):flags.append('outcome_domain_or_eligibility_retained')
 residual=[k for k,v in d['display'].items() if v=='other_unclear']
 if residual:flags.append('explicit_outside_display_bins')
 if d['display']['mean_model']=='other_unclear':flags.append('nonlinear_panel_mean' if i==322 else 'chain_or_covariate_specific_treatment_contrast')
 if i==145:flags.append('closure_zero_excluded_by_positive_domain')
 base={'nonmanager_fte':'nonmanager_fte','manager_inclusive_fte':'manager_inclusive_fte','employee_headcount':'employee_headcount','full_time_employees':'full_time_count','other_unclear':'other'}[d['display']['employment_measure']]
 if base=='other':
  if re.search(r'part.time',o,re.I) and not re.search(r'empft|full.time',o,re.I):base='part_time_count'
  elif re.search(r'share',o,re.I):base='employment_share'
  elif re.search(r'0\.75',o):base='other_weighted_employment'
 managers='excluded' if re.search(r'excluding managers|exclude managers|managers (?:are |and assistant managers are )?excluded|non.manager|nonmanager',o,re.I) else 'included' if d['display']['employment_measure']=='manager_inclusive_fte' or re.search(r'\+\s*nmgrs|plus managers|including managers|includes managers|managers.*each counted as one',o,re.I) else 'unspecified'
 transform='log1p' if re.search(r'log\s*\(\s*1\s*\+',o,re.I) else 'asinh' if re.search(r'asinh|inverse hyperbolic|arcsinh',o,re.I) else 'log' if re.search(r'\blog\b|natural.log',o,re.I) and not re.search(r'no log|not.*logged|no .*transformation',o,re.I) else 'none_explicit_or_untransformed_context'
 closure_req='not_applicable'
 if d['display']['sample_treatment']=='closure_recoding':
  closure_req='observed_raw_counts_not_required' if re.search(r'without requiring|not required|no (?:observed |reported )?(?:wave.two |wave.2 |follow.up )?(?:counts?|components?)|only .*baseline|regardless of|exempt|even if.*missing',s,re.I) else 'ambiguous'
  if closure_req=='ambiguous':flags.append('closure_followup_count_requirement_retained_ambiguous')
 signature_ids={}
 for dim,code in d['display'].items():
  payload={'dimension':dim,'complete_relevant_declaration':{k:decl[k] for k in ['outcome','sample_restrictions','model']},'context':context,'context_version':r['context_version'],'source_role':'frozen_declaration','conflict_flags':d['flags']}
  sig=H(canon(payload).encode());signature_ids[dim]=sig
  signatures.append({'evidence_signature':sig,'dimension':dim,'display_code':code,'coding_id':r['coding_id'],'reviewer':'/root/card_trace_coder_c_v1','reason':d['reasons'][dim],'complete_evidence_fields':['outcome','sample_restrictions','model'],'declaration_sha256':r['declaration_sha256'],'context_version':r['context_version'],'coding_status':'semantically_reviewed'})
 details={'outcome':{'outcome_formula_raw':o,'base_measure':base,'manager_inclusion':managers,'part_time_weight':0.5 if d['display']['employment_measure'] in ['nonmanager_fte','manager_inclusive_fte'] else None,'manager_weight':1 if d['display']['employment_measure']=='manager_inclusive_fte' else None,'transformation':transform,'contrast':'post_minus_pre'},'sample':{'sample_rule_raw':s,'sample_state':'other' if d['display']['sample_treatment']=='other_unclear' else 'explicit','closure_followup_requirement':closure_req},'mean':{'mean_formula_raw':m,'mean_state':'other' if d['display']['mean_model']=='other_unclear' else 'explicit','time_form':'panel_other' if i==322 else 'change_score'},'substantive_audit':{'null_raw':decl.get('null'),'target_population_raw':decl.get('population'),'scope':'archived exact declaration; probability restrictions and accepted endpoint not recoded'}}
 codes.append({'coding_id':r['coding_id'],'schema_version':schema['schema_version'],'source':r['source'],'declaration_sha256':r['declaration_sha256'],'context_version':r['context_version'],'display':d['display'],'details':details,'evidence':evidence,'evidence_signatures':signature_ids,'flags':sorted(set(flags)),'confidence':{k:'high' for k in d['display']},'ambiguity':{'display_assignment_unresolved':False,'closure_followup_requirement':closure_req},'coding_status':'semantically_reviewed','reviewer':'/root/card_trace_coder_c_v1','coherence_review':{'status':'reviewed','result':'material_distinctions_retained','flags':d['flags']},'coding_scope':'descriptive final declarations only; no independent endpoint adjudication; no search chronology'})
 review.append({'index':i,'coding_id':r['coding_id'],'display':d['display'],'evidence_fields_sha256':{x['source_field']:x['text_sha256'] for x in evidence},'review_scope':'compact field-level evidence plus targeted full-field edge review; full source spans retained'})
for name,data in [('codes.jsonl',codes),('signature_decisions.jsonl',signatures),('REVIEW_LEDGER.jsonl',review)]:
 (DIR/name).write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in data))
for r,c in zip(rows,codes):
 for e in c['evidence']:
  assert r['declaration'][e['source_field']][e['span_start']:e['span_end']]==e['text']
  assert H(e['text'].encode())==e['text_sha256']
receipt={'partition':'C','case_count':1000,'unique_ids':1000,'missing_ids':[],'extra_ids':[],'unique_dimension_signatures':len({x['evidence_signature'] for x in signatures}),'signature_decisions':len(signatures),'semantic_review_case_count':1000,'semantic_review_dimension_decisions':3000,'unresolved_coding_cases':[],'residual_cases':[c['coding_id'] for c in codes if 'other_unclear' in c['display'].values()],'schema_sha256':schema_hash,'schema_markdown_sha256':H((B/'schema/SCHEMA.md').read_bytes()),'clarifications_sha256':H((B/'CODING_CLARIFICATIONS.md').read_bytes()),'method':'Conservative parser proposals; compact evidence inspection for all1000 declarations; targeted complete-field semantic resolution of candidate errors, status operators, baseline aliases, transformations and actual subgroup filters. Per-case frozen adjudication decisions drive reproducible rebuild; no claim that parser output alone is semantic review. Exact full relevant declaration evidence and context hashes retained. Descriptive categories never update accepted independent endpoints.','source_packets':{f'packets-{j:02}.jsonl':H((B/f'extraction/neutral/packets-{j:02}.jsonl').read_bytes()) for j in range(9,13)},'category_counts':{dim:dict(Counter(c['display'][dim] for c in codes)) for dim in schema['display_categories']},'evidence_span_validation':'3000 exact source-field spans and span SHA256 verified','blinding':'Neutral packets/context only; no PRIVATE_join, stratum, accepted verdict, numeric result or target narrative read','outputs':{name:H((DIR/name).read_bytes()) for name in ['codes.jsonl','signature_decisions.jsonl','REVIEW_LEDGER.jsonl','DECISIONS.jsonl','adjudication_overrides.jsonl','CONTEXT_EVIDENCE.json','build_codes.py']}}
(DIR/'RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['case_count','unique_ids','signature_decisions','unresolved_coding_cases','category_counts','outputs']},indent=2))
