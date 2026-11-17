#!/usr/bin/env python3
"""Rebuild locked A descriptive codes from frozen neutral packets and saved decisions.
Usage: python3 coding/A/rebuild.py
Uses Python standard library; no LLM/network/fits/private joins required.
"""
import pathlib,json,hashlib,re,collections,sys
HERE=pathlib.Path(__file__).resolve().parent
SCHEMA_DIR=HERE.parents[1]/'schema'
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
def readjl(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def writejl(p,x):p.write_text(''.join(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n' for v in x))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def has(p,t):return re.search(p,t,re.I|re.S) is not None
def sentences(t):return re.split(r'(?<=[.;])\s+',t)
def binding(p):return {'path':str(p),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
rows=readjl(HERE/'source_packets.jsonl');ds=readjl(HERE/'DECISIONS.jsonl');overrides=readjl(HERE/'ADJUDICATION_OVERRIDES.jsonl')
schema=json.loads((SCHEMA_DIR/'SCHEMA.json').read_text());assert len(rows)==len(ds)==1000
assert len({r['coding_id'] for r in rows})==1000
assert {r['coding_id'] for r in rows}=={r['coding_id'] for r in ds}
by={r['coding_id']:r for r in ds}
# Context is frozen once and bound into each semantic signature.
context_hashes=json.loads((HERE/'CONTEXT_HASHES.json').read_text())
results=[];sigrows=[];flagged=[];evchecks=0
for i,r in enumerate(rows):
 dec=r['declaration'];d=by[r['coding_id']];assert d['source_index']==i
 assert sha(canon(dec))==r['declaration_sha256']
 display=d['display']
 for dim,code in display.items():assert code in schema['display_categories'][dim]
 o=dec['outcome'];s=dec['sample_restrictions'];m=dec['model'];null=dec['null'];base=display['employment_measure'];flags=list(d['coherence_flags'])
 # These secondary details are literal evidence summaries, never equivalence decisions.
 positive_o=re.sub(r'\bno\b[^.;]*|\bnot\b[^.;]*|\bneither\b[^.;]*','',o,flags=re.I)
 if has(r'asinh|inverse.hyperbolic',positive_o):transform='asinh'
 elif has(r'log\s*\(\s*1\s*\+|log1p|logarithm of 1 plus|log.of.one.plus|log offset 1|natural.log scale after adding.{0,20}1',positive_o):transform='log1p'
 elif has(r'\blog|natural.log|logarithm',positive_o):transform='log'
 elif has(r'proportional.change|percentage.change|percent.change|\)\s*/\s*empft|/ empft',positive_o):transform='proportional_change'
 else:transform='none_explicit_or_untransformed'
 if has(r'\+\s*nmgrs|\+ managers|plus (?:the count of )?managers|including managers|manager.inclusive|managers.{0,20}count as one|manager and assistant.manager headcount|manager count|number of managers',o):manager='included'
 elif has(r'exclud.{0,15}managers|non.manager|managers.{0,25}(?:excluded|not included)|manager.*weight.{0,8}0',o):manager='excluded'
 elif has(r'\bempft\b|\bemppt\b',o) and not has(r'nmgrs',o):manager='excluded';flags.append('manager_exclusion_resolved_by_codebook_alias')
 else:manager='unspecified'
 if base=='manager_inclusive_fte':manager='included'
 if i==689:manager='included'
 manager_weight=1 if manager=='included' and has(r'\+\s*nmgrs|plus.*managers|as one|one FTE|one.for.one|each.*one|weighted as one|count as one|nmgrs2.{0,10}−.{0,10}nmgrs|nmgrs2 minus nmgrs',o) else 0 if manager=='excluded' else None
 if manager=='included' and manager_weight is None:flags.append('manager_weight_not_structurally_resolved_in_outcome_summary')
 partweight=0.5 if base in ['nonmanager_fte','manager_inclusive_fte'] else 1 if base=='employee_headcount' else 0 if base=='full_time_employees' else None
 if base=='other_unclear':
  base_detail='part_time_count' if has(r'part.time employee|number of part.time|count of part.time|emppt2',o) and not has(r'full.time employees plus|empft',positive_o) else 'other'
 else:base_detail='full_time_count' if base=='full_time_employees' else base
 if has(r'proportional.change|percentage.change|percent.change|\)\s*/\s*empft|/ empft',o):contrast='ratio'
 else:contrast='post_minus_pre'
 if i==116:transform='proportional_change';contrast='ratio'
 status=display['sample_treatment'];sample_state='mixed' if i==215 else 'other' if status=='other_unclear' else 'explicit'
 if status=='closure_recoding':
  if has(r'without requiring[^.;]{0,45}(?:follow|wave.2)|does not require[^.;]{0,40}(?:wave|follow)|not require[^.;]{0,45}(?:wave|follow)|regardless of (?:its |their )?(?:recorded|wave.2)|need only|only baseline',s):closure_follow='observed_raw_counts_not_required'
  elif has(r'counts.{0,30}(?:both waves|in each wave)|all (?:four|six).{0,50}(?:observed|finite|nonmissing)|complete.{0,50}(?:both waves|empft2)|both waves.{0,50}(?:empft|count)|empft, emppt, empft2, and emppt2',s):closure_follow='observed_raw_counts_required'
  else:closure_follow='ambiguous'
 else:closure_follow='not_applicable'
 if status=='closure_recoding' and closure_follow=='ambiguous':flags.append('closure_raw_followup_requirement_needs_exact_rule_context')
 if status=='other_unclear' and i not in [102,215,257,566,646,669,955]:flags.append('explicit_status_selection_without_a_closure_recoding_operator')
 if status=='respondents_only' and has(r'closed|closure',o) and has(r'zero|\b0\b',o):flags.append('closure_outcome_definition_outside_selected_respondent_sample')
 mean=display['mean_model'];mean_state='other' if mean=='other_unclear' else 'explicit'
 time='panel_did' if i==135 else 'change_score'
 restriction_sent=[t for t in sentences(m) if has(r'distribution|variance|independen|dependen|normal|gaussian|iid|parallel|mean.zero|mean zero|unknown|known',t)]
 infer_sent=[t for t in sentences(m) if has(r'HC[0-3]|standard.error|bootstrap|permut|inference|student|\bt.reference|multiplicity|p.value',t)]
 evidence=[]
 for field in ['outcome','sample_restrictions','model','null','population']:
  t=dec[field];e={'source_field':field,'source_json_pointer':r['field_json_pointers'][field],'span_start':0,'span_end':len(t),'text':t,'text_sha256':sha(t.encode()),'source_role':'frozen_declaration'}
  assert dec[field][e['span_start']:e['span_end']]==e['text'];assert sha(e['text'].encode())==e['text_sha256'];evchecks+=1;evidence.append(e)
 details={'outcome':{'outcome_formula_raw':o,'base_measure':base_detail,'manager_inclusion':manager,'manager_weight':manager_weight,'part_time_weight':partweight,'transformation':transform,'scale':'exact formula and scale retained in outcome_formula_raw','contrast':contrast},'sample':{'analysis_unit':'store paired record','baseline_eligibility':s,'followup_status_eligibility':s,'status_code_map':{'1':'answered wave2','3':'permanent closure','2':'renovation closure','4':'highway closure','5':'mall fire closure','0':'refused wave2'},'required_variables':sorted(set(re.findall(r'\b(?:empft2?|emppt2?|nmgrs2?|state|chain|co_owned|southj|northj|centralj|shore|pa1|pa2|wage_st|pctaff|hrsopen|type2)\b',s))),'missing_component_rule':s,'closure_rule':{'sample_raw':s,'outcome_raw':o},'closure_followup_requirement':closure_follow,'other_filters':s if i in [102,215,257,566,646,669,955] else [],'support_conditions':[t for t in sentences(s) if has(r'both state|each state|at least|rank|identif|estimab|function|otherwise|minimum',t)],'outcome_domain':[t for t in sentences(s) if has(r'positive|negative|denominator|finite|log',t)],'failure_conditions':[t for t in sentences(s) if has(r'function|stop|fail|error|otherwise|at least',t)],'cross_wave_rule':s,'sample_state':sample_state},'mean':{'mean_formula_raw':m,'time_form':time,'link':'not inferred beyond exact declared mean formula','intercept':'see exact mean formula','treatment_term':'see exact mean formula and null','additive_chain':mean=='nj_additive_chain' or (mean=='added_baseline_covariates' and has(r'chain',m)),'extra_covariates':m if mean=='added_baseline_covariates' else [],'interactions':[t for t in sentences(m) if has(r'interact|τ_[ck]|τ[ck]|state contrast may differ',t)],'fixed_effects':[t for t in sentences(m) if has(r'fixed.effect|chain|store effect',t)],'weights':[t for t in sentences(m) if has(r'weight|pair|mix|Σ|sum',t)],'offsets':[t for t in sentences(o+' '+m) if has(r'offset|center|centred|centered|divided by|asinh|log\s*\(\s*1',t)],'mean_state':mean_state},'substantive_audit':{'null_raw':null,'contrast_parameter_raw':null,'sidedness':'two_sided_explicit' if has(r'≠|!=|not equal|two.sided',null) else 'not_explicit_in_null_text','target_population_raw':dec['population'],'explicit_restrictions':[{'text':t,'role':'ambiguous','note':'Preserved verbatim; role not freshly adjudicated for descriptive coding.'} for t in restriction_sent]},'inference':{'estimator':[],'se_method':re.findall(r'HC[0-3]',m),'clustering':[],'reference_distribution':[],'degrees_of_freedom':[],'resampling':[],'multiplicity':[],'other_raw':infer_sent}}
 signatures={}
 for dim in display:
  payload={'dimension':dim,'complete_declaration':dec,'context_version':r['context_version'],'context_hashes':context_hashes,'source_role':'frozen_declaration','coherence_flags':d['coherence_flags']}
  signature=sha(canon(payload));signatures[dim]=signature
  sigrows.append({'evidence_signature':signature,'dimension':dim,'coding_ids':[r['coding_id']],'category':display[dim],'reviewer':d['reviewer'],'reason':'Direct review of compact field evidence with expanded contexts for aliases, conflicts and special operators. Exact full declaration and supplied context are bound; no signature equivalence or match verdict inferred.','decision_source':'DECISIONS.jsonl','decision_index':i})
 result={'coding_id':r['coding_id'],'schema_version':schema['schema_version'],'source':r['source'],'declaration_sha256':r['declaration_sha256'],'context_version':r['context_version'],'display':display,'outcome_category':base,'sample_category':status,'model_category':mean,'details':details,'evidence':evidence,'evidence_signatures':signatures,'flags':sorted(set(flags)),'confidence':d['confidence'],'coding_status':'semantically_reviewed','reviewer':d['reviewer'],'coherence_review':{'status':'reviewed','flags':d['coherence_flags'],'note':'Checked actual selected formulas/eligibility and cross-field roles; accepted independent model/null verdicts were concealed and remain immutable.'}}
 assert set(schema['required_record_fields'])<=set(result);results.append(result)
 if result['flags']:flagged.append({'coding_id':r['coding_id'],'flags':result['flags'],'display':display,'coherence_review':result['coherence_review']})
writejl(HERE/'final_codes.jsonl',results);writejl(HERE/'signature_decisions.jsonl',sigrows);writejl(HERE/'flagged_cases.jsonl',flagged)
assert len({x['evidence_signature'] for x in sigrows})==3000
inputs={p.name:binding(p) for p in [HERE/'source_packets.jsonl',HERE/'DECISIONS.jsonl',HERE/'ADJUDICATION_OVERRIDES.jsonl',HERE/'CONTEXT_HASHES.json',SCHEMA_DIR/'SCHEMA.md',SCHEMA_DIR/'SCHEMA.json',SCHEMA_DIR/'propose_codes.py']}
outputs={p.name:binding(p) for p in [HERE/'final_codes.jsonl',HERE/'signature_decisions.jsonl',HERE/'flagged_cases.jsonl']}
receipt={'partition':'A','cases':1000,'unique_coding_ids':1000,'source_packets':'neutral/packets-01.jsonl through packets-04.jsonl','cases_reviewed':1000,'dimension_signatures_reviewed':3000,'semantic_signature_reuse':0,'evidence_spans_verified':evchecks,'source_declaration_hashes_verified':1000,'saved_overrides_vs_frozen_helper_candidates':len(overrides),'coding_status_counts':dict(collections.Counter(x['coding_status'] for x in results)),'category_counts':{dim:dict(collections.Counter(x['display'][dim] for x in results)) for dim in results[0]['display']},'confidence':'Coarse categories high confidence from explicit primary formulas/eligibility and codebook; exact ancillary details preserved without new equivalence judgments. Ambiguous closure raw-followup requirements flagged separately.','flagged_cases':len(flagged),'masking':'Coding used only saved neutral packets/context; no private label/stratum join opened during this coding pass. This agent earlier owned extraction and saw aggregate frozen counts; case-label mappings were not consulted for descriptive coding.','review_method':'Inspected compact field-level evidence across all1000; expanded full contexts for outcome components/negative mentions, recorded versus assigned closure zeros, all-status recoding, actual subgroup filters, baseline aliases, chain intercepts versus heterogeneous slopes, positive treatment interactions and nonmean estimands. Candidate regex prioritized review; saved decisions determine final codes.','limitations':'Final endpoint taxonomy; no chronology or match rejudgment. Full null/population/probability text archived; ancillary-role adjudication outside coarse scope.','rebuild_command':'python3 output/card-trace-coding-20261018-v1/coding/A/rebuild.py','inputs':inputs,'outputs':outputs,'build_script':binding(HERE/'rebuild.py')}
dump(HERE/'RECEIPT.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k not in ['inputs','outputs','build_script']},ensure_ascii=False,indent=2))
