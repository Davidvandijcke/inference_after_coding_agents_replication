#!/usr/bin/env python3
"""Conservative descriptive candidates and exact evidence spans. All outputs require semantic review.
Usage: python3 propose_codes.py INPUT.jsonl OUTPUT.jsonl
No inference, match labels, numerical result extraction, fits, network, or paid calls.
"""
import argparse,hashlib,json,pathlib,re
VERSION='card-declaration-choices-v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
NAMES=['employment_measure','sample_treatment','mean_model']
def norm(t):
 return t.lower().replace('–','-').replace('—','-').replace('−','-').replace('‐','-').replace('‑','-').replace('₁','1').replace('₂','2')
def has(p,t): return re.search(p,t,re.I|re.S) is not None
def ev(r,field,start=0,end=None):
 t=r['declaration'].get(field) or ''; end=len(t) if end is None else end
 return dict(source_field=field,source_json_pointer=r.get('field_json_pointers',{}).get(field),span_start=start,span_end=end,text=t[start:end],text_sha256=sha(t[start:end].encode()))
def compact(r,field):
 t=r['declaration'].get(field) or ''
 # Keep source text verbatim; split only at sentence-like boundaries, never decimal points.
 spans=[]
 for m in re.finditer(r'.+?(?:[.!?](?=\s+[A-Z])|$)',t,re.S):
  if field!='model' or has(r'mean|model|covariat|control|chain|baseline|interact|estimand|weight|log|share|regress|Δ|τ|[+|=]',m.group()):
   spans.append(ev(r,field,m.start(),m.end()))
 return spans or [ev(r,field)]
def propose(r):
 dec=r.get('declaration') or {}; o=norm(dec.get('outcome') or ''); s=norm(dec.get('sample_restrictions') or ''); m=norm(dec.get('model') or '')
 flags=[]; detail_flags=[]; outcome='other_unclear'; sample='other_unclear'; mean='other_unclear'
 # Base employment measure: numerical definitions dominate vague labels.
 weighted=has(r'0[.]5|one[ -]half|one half|half (?:of )?(?:the )?part',o)
 manager=has(r'\+\s*nmgrs|manager[ -]inclusive|including managers|managers?[^.;]{0,55}(?:counted|included)|plus[^.;]{0,25}managers',o)
 share=has(r'(?:full[ -]time|employment|workforce)[ -]share|empft\s*/\s*\(',o)
 if share: detail_flags.append('employment_share_not_count')
 elif has(r'headcount',o): outcome='employee_headcount'
 elif weighted and has(r'full[ -]time|empft',o) and has(r'part[ -]time|emppt',o): outcome='manager_inclusive_fte' if manager else 'nonmanager_fte'
 elif has(r'(?:count|number) (?:of )?full[ -]time employees|full[ -]time (?:employee |employment )?count',o) and not weighted: outcome='full_time_employees'
 elif has(r'part[ -]time employee count',o): detail_flags.append('parttime_only_not_FTE')
 else: flags.append('outcome_definition_requires_review')
 if manager: detail_flags.append('manager_inclusion_explicit')
 if has(r'log\s*\(\s*1\s*\+|log offset 1',o+' '+m): detail_flags.append('log1p_transform')
 elif has(r'log',o) and not has(r'no log|not log|no other.*transformation',o): detail_flags.append('possible_transform_check_negation')
 if has(r'winsori|trim|standardiz|percent|ratio',o): detail_flags.append('outcome_transform_or_scale')
 # Sample: detect actual status eligibility and preserve mixed/state-specific restrictions.
 status1=has(r'status(?:2)?\s*(?:==|=|equal(?:s| to)?)\s*1|status[ -]1\b|status (?:codes? |values? )?1\b|statuses? exactly 1|status2 values 1|status2 must equal 1',s)
 status3=has(r'status(?:2)?\s*(?:==|=)\s*3|status[ -]3\b|status (?:codes? |values? )?3\b|(?:or|and)\s*3\s*\(permanently closed',s)
 pair13=has(r'status(?:2)?(?:\s+(?:codes?|values?))?\s*(?:(?:==|=|equal(?:s| to)?)\s*)?(?:exactly\s*)?1[^.;]{0,65}(?:or|and)\s*(?:status(?:2)?\s*(?:=|==)\s*)?3',s)
 zero=has(r'(?:set|assign|defin|replac|recod)[^.;]{0,90}zero|zero[^.;]{0,55}(?:assigned|set|defined|replaced|recoded)',s+' '+o)
 included_closure=(pair13 or (status3 and status1)) and zero
 mixed=has(r'(?:new jersey|\bnj\b) permanent closures|closures[^.;]{0,45}(?:new jersey|\bnj\b) only|state[ -]specific[^.;]{0,30}(?:eligib|sample|closure)',s)
 actual_subgroup=has(r'(?:excluding|exclude) chain 4|retain chain codes 1[-–]3|(?:retain|restrict[^.;]{0,20}to|include only)[^.;]{0,35}(?:burger king only|single chain|chains 1[-–]3)',s)
 no_status=has(r'no[^.;]{0,90}(?:status[ -]based|follow[ -]up[ -]status|status[ -]filter|wave[ -]2 status|wave[ -]two status|interview status)|no additional restriction by[^.;]{0,40}(?:status|closure)',s)
 complete=has(r'complete|nonmissing|non[ -]missing|finite|observed|required|counts in both waves',s)
 if mixed or actual_subgroup:
  flags.append('mixed_state_or_subgroup_sample'); sample='other_unclear'
 elif included_closure or (no_status and zero and has(r'permanent.{0,15}clos',s+' '+o)): sample='closure_recoding'
 elif status1 and complete: sample='respondents_only'
 elif complete and (no_status or not has(r'status',s)): sample='complete_case_no_status_filter'
 else: flags.append('sample_eligibility_requires_review')
 if sample=='closure_recoding': detail_flags.append('check_closure_status_set_baseline_and_followup_requirements')
 if has(r'both states|both state|each state|at least|identifi|estimab|unique store',s): detail_flags.append('support_or_failure_condition')
 if has(r'positive|nonnegative|negative|denominator|stops|otherwise.*stop',s): detail_flags.append('outcome_domain_or_stop_rule')
 if has(r'complete[^.;]{0,60}(?:empft|full[ -]time)',s) and (has(r'part[ -]time employee count',o) or outcome=='full_time_employees'): detail_flags.append('check_required_variables_beyond_outcome')
 # Mean candidates: strip explicit negative control clauses rather than treating keyword presence as a positive choice.
 core=m
 core=re.split(r'no parametric|no observation distribution',core)[0]
 core=re.sub(r'\bno\s+(?:additional|other|chain|baseline|ownership|covariate|treatment[ -]by)[^.;]*','',core)
 hetero=has(r'chain[ -]specific[^.;]{0,65}(?:contrast|τ|tau)|τ_[ck]|τ[ck]\b|tau_[ck]|δ[₂₃23][^.;]{0,10}state|state_i[·*][^.;]{0,20}i\(chain',core)
 panel=has(r'panel[ -]level|state[ -]by[ -]post|nj[ -]by[ -]post|nj\s*[*×]\s*post',core)
 nonlinear=has(r'poisson|logistic|logit|probit|nonlinear link',core)
 # Broader explicit interactions are flags, never silently reduced to additive models.
 positive_interaction=has(r'(?:includes?|with|and)\s+(?:a |the )?(?:state[ -]by|nj[ -]by|treatment[ -]by)[^.;]{0,40}interaction',core)
 conditioning=' '.join(re.findall(r'\|([^]\n]+)\]',core))
 extra=has(r'co_owned|company_owned|company ownership|wage_st',core) or has(r'baseline|fte_pre|ft_i,pre|empft|emppt|y_i[01]|s_pre|b_i',conditioning) or has(r'(?:β|γ|ρ|λ|δ|beta|gamma|rho|lambda)\s*[·*]?\s*(?:baseline|fte[^.;]{0,12}pre|ft_i,pre|y_i[01]|s_pre|empft|emppt|b_i)',core)
 chain=has(r'chain',core)
 clearmean=has(r'mean|conditional[ -]mean|conditional mean|regression|e\s*[\[(]|model',core)
 change=has(r'change|Δ|difference|post.minus.pre|post[ -]minus[ -]pre',m+' '+o)
 if hetero or positive_interaction or panel or nonlinear:
  flags.append('mean_interaction_panel_or_nonlinear'); mean='other_unclear'
 elif extra and clearmean: mean='added_baseline_covariates'
 elif chain and clearmean and change: mean='nj_additive_chain'
 elif clearmean and change: mean='nj_only'
 else: flags.append('mean_structure_requires_review')
 if hetero: detail_flags.append('chain_specific_treatment_contrasts')
 if has(r'weight|Σ|sum',m): detail_flags.append('weights_or_fixed_definition_constants')
 if has(r'interact',m): detail_flags.append('interaction_word_present_read_negation_and_role')
 # Inspect source text for ambiguity/contradiction rather than claiming automatic certification.
 if has(r'unclear|ambiguous|either|alternativ|multiple primary|not specified',o+' '+s): detail_flags.append('explicit_ambiguity_word')
 signature=sha(json.dumps({'dimension_context':dec,'context_version':r.get('context_version'),'source_role':'frozen_declaration'},ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
 return dict(coding_id=r['coding_id'],schema_version=VERSION,source=r.get('source'),declaration_sha256=r.get('declaration_sha256'),context_version=r.get('context_version'),evidence_signature=signature,display=dict(zip(NAMES,[outcome,sample,mean])),candidate_display=dict(zip(NAMES,[outcome,sample,mean])),flags=sorted(set(flags)),detail_flags=sorted(set(detail_flags)),coding_status='candidate_unreviewed',reviewer=None,coherence_review='required',evidence=[e for f in ['outcome','sample_restrictions','model'] for e in compact(r,f)],details={'outcome_formula_raw':dec.get('outcome'),'sample_rule_raw':dec.get('sample_restrictions'),'mean_formula_raw':dec.get('model'),'null_raw':dec.get('null'),'target_population_raw':dec.get('population')},source_scope='declaration only; no match adjudication; candidate lexical extraction requires semantic review')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);a=p.parse_args()
 rows=[json.loads(l) for l in a.input.read_text().splitlines() if l.strip()]
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(''.join(json.dumps(propose(r),ensure_ascii=False)+'\n' for r in rows))
 print(json.dumps({'cases':len(rows),'output':str(a.output),'status':'candidate_unreviewed'}))
if __name__=='__main__':main()
