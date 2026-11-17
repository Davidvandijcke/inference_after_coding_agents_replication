"""Validate frozen descriptive codes, apply recorded adjudications, and package displays."""
from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import json
import shutil

TASK=Path(__file__).resolve().parents[1]
PROJECT=Path('<manuscript checkout>')
DATA=PROJECT/'results/data'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loadlines(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def savegz(path,rows):
    raw=''.join(json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n' for d in rows).encode()
    with Path(path).open('wb') as file:
        with gzip.GzipFile(filename='',fileobj=file,mode='wb',mtime=0) as zipped:
            zipped.write(raw)


def evidence_check(e,decl):
    text=decl[e['source_field']][e['span_start']:e['span_end']]
    assert text==e['text']
    assert hashlib.sha256(text.encode()).hexdigest()==e['text_sha256']


def main():
    packets=loadlines(TASK/'extraction/neutral/packets.jsonl')
    neutral={r['coding_id']:r for r in packets}
    assert len(neutral)==3000
    allowed=json.loads((TASK/'schema/SCHEMA.json').read_text())['display_categories']
    source_files={'A':TASK/'coding/A/final_codes.jsonl',
                  'B':TASK/'coding/B/FINAL_CODES.jsonl','C':TASK/'coding/C/codes.jsonl'}
    original={};parts={};spans=0
    for part,path in source_files.items():
        rows=loadlines(path);assert len(rows)==1000
        for r in rows:
            cid=r['coding_id'];assert cid not in original
            assert r['declaration_sha256']==neutral[cid]['declaration_sha256']
            for dim,cat in r['display'].items():assert cat in allowed[dim]
            for e in r['evidence']:
                evidence_check(e,neutral[cid]['declaration']);spans+=1
            original[cid]=r;parts[cid]=part
    assert set(original)==set(neutral)
    final={cid:{'coding_id':cid,'declaration_sha256':r['declaration_sha256'],
                'display':dict(r['display'])} for cid,r in original.items()}
    changes=json.loads((TASK/'integration/ADJUDICATIONS.json').read_text())
    seen=set()
    for d in changes:
        cid,dim=d['coding_id'],d['dimension'];assert (cid,dim) not in seen
        seen.add((cid,dim));assert final[cid]['display'][dim]==d['old']
        assert d['new'] in allowed[dim]
        for e in d['evidence']:evidence_check(e,neutral[cid]['declaration'])
        final[cid]['display'][dim]=d['new']
    audit=loadlines(TASK/'audit/review/blind_audit_codes.jsonl')
    mismatches=[]
    for d in audit:
        for dim,cat in d['display'].items():
            if final[d['coding_id']]['display'][dim]!=cat:
                mismatches.append({'coding_id':d['coding_id'],'dimension':dim,
                    'final':final[d['coding_id']]['display'][dim],'audit':cat})
    assert not mismatches,mismatches
    # Join only after descriptive decisions and adjudications have been locked.
    joins=loadlines(TASK/'extraction/PRIVATE_join.jsonl')
    cases={r['coding_id']:r for r in joins};assert set(cases)==set(final)
    records=[];counts={};nodes={}
    for cid in sorted(final):
        d=final[cid];j=cases[cid];classified=j['reviewed_status']=='classified'
        assert (type(j['reviewed_match']) is bool) if classified else j['reviewed_match'] is None
        row={'case_id':cid,'stratum':j['stratum'],'classifiable':classified,
             'reviewed_match':j['reviewed_match'] if classified else None,
             'outcome_category':d['display']['employment_measure'],
             'sample_category':d['display']['sample_treatment'],
             'model_category':d['display']['mean_model']}
        records.append(row)
    for stratum,expected in [('tail',(1142,1135,48,7)),('body',(1858,1854,50,4))]:
        rows=[r for r in records if r['stratum']==stratum]
        classified=[r for r in rows if r['classifiable']]
        actual=(len(rows),len(classified),sum(r['reviewed_match'] for r in classified),len(rows)-len(classified))
        assert actual==expected,(stratum,actual)
        counts[stratum]=dict(zip(['attempted','classified','matches','excluded'],actual))
        nodes[stratum]={dim:dict(Counter(r[dim] for r in classified))
            for dim in ['outcome_category','sample_category','model_category']}
    (TASK/'integration/FINAL_CODES.jsonl').write_text(''.join(json.dumps(final[cid],ensure_ascii=False,sort_keys=True)+'\n' for cid in sorted(final)))
    (TASK/'integration/FLOW_CASES.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    savegz(DATA/'card-reviewer-codes.jsonl.gz',[original[cid] for cid in sorted(original)])
    savegz(DATA/'card-coding-decisions.jsonl.gz',[final[cid] for cid in sorted(final)])
    savegz(DATA/'card-declarations.jsonl.gz',[neutral[cid] for cid in sorted(neutral)])
    savegz(DATA/'card-flow-cases.jsonl.gz',records)
    with (DATA/'card-case-map.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['coding_id','stratum','reviewed_status','source_classification'])
        writer.writeheader()
        for cid in sorted(cases):
            j=cases[cid]
            label='match' if j['reviewed_match'] is True else 'nonmatch' if j['reviewed_match'] is False else 'unresolved'
            writer.writerow({'coding_id':cid,'stratum':j['stratum'],'reviewed_status':label,
                             'source_classification':j['reviewed_status']})
    shutil.copy2(TASK/'integration/ADJUDICATIONS.json',DATA/'card-coding-overrides.json')
    shutil.copy2(TASK/'integration/SCHEMA_v1_1.json',DATA/'card-coding-schema.json')
    shutil.copy2(TASK/'integration/SCHEMA_v1_1.md',DATA/'card-coding-rules.md')
    shutil.copy2(TASK/'CODING_CLARIFICATIONS.md',DATA/'card-coding-clarifications.md')
    shutil.copy2(TASK/'extraction/neutral/context/CODEBOOK.md',DATA/'card-codebook.md')
    shutil.copy2(TASK/'extraction/neutral/context/DESIGN.md',DATA/'card-design.md')
    savegz(DATA/'card-blind-audit.jsonl.gz',audit)
    audit_summary={'cases':72,'dimensions':216,'initial_agreement':209,'final_agreement':216,
        'adjudicated_dimensions_full_cohort':len(changes),'adjudicated_cases_full_cohort':len({d['coding_id'] for d in changes}),
        'scope':'Independent masked descriptive coding; all reported existing match labels and correction estimates preserved.',
        'selection':json.loads((TASK/'audit/SELECTION.json').read_text())['selection'],
        'projection_clarification':'The display groups regression specifications/projections by explicit regressors/form, without inferring conditional-mean restrictions. Actual model/null verdicts remain independent.'}
    (DATA/'card-audit-summary.json').write_text(json.dumps(audit_summary,indent=2)+'\n')
    renderer=TASK/'rendering/render_alluvial.py'
    shutil.copy2(renderer,PROJECT/'results/scripts/render_alluvial.py')
    spec=json.loads(json.dumps(__import__('runpy').run_path(str(renderer))['DEFAULT_SCHEMA']))
    spec[2]['title']='Specification'
    (DATA/'card-render-schema.json').write_text(json.dumps(spec,indent=2)+'\n')
    report={'cohort_cases':3000,'verified_original_evidence_spans':spans,
        'dimension_codes':9000,'blind_audit':audit_summary,'counts':counts,'node_counts':nodes,
        'frozen_source_files':{part:{'path':str(p),'sha256':sha(p)} for part,p in source_files.items()},
        'frozen_declared_packet_sha256':sha(TASK/'extraction/neutral/packets.jsonl'),
        'independent_label_sha256':json.loads((TASK/'extraction/CHECKS.json').read_text())['source_files']['independent_labels']['sha256']}
    (TASK/'integration/VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    (DATA/'card-flow-summary.json').write_text(json.dumps({'counts':counts,'node_counts':nodes},indent=2)+'\n')
    print(json.dumps({'counts':counts,'node_counts':nodes,'adjudications':len(changes),'audit_agreement':'216/216 after source-based adjudication'},indent=2))


if __name__=='__main__':main()
