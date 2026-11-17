#!/usr/bin/env python3
"""In-text numbers of the Card-Krueger application, each with its quote from paper/main.tex.

Inputs:  data/derived/08_card/original.csv, bank.csv, correction.csv, threshold.csv, choice-summary.json,
         replay-summary.json, activity.json; data/agent_runs/card_luna/packet/DESIGN.md
Output:  results/numbers/08_card.json
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / 'data' / 'derived' / '08_card'
sys.path.insert(0, str(ROOT / 'code' / 'lib'))
from paper_numbers import write_numbers  # noqa: E402


def named(name):
    return {r['name']: float(r['value']) for r in csv.DictReader(open(DERIVED / name, encoding='utf-8'))}


def main():
    o, b, c, t = named('original.csv'), named('bank.csv'), named('correction.csv'), named('threshold.csv')
    choice = json.load(open(DERIVED / 'choice-summary.json', encoding='utf-8'))
    replay = json.load(open(DERIVED / 'replay-summary.json', encoding='utf-8'))
    activity = json.load(open(DERIVED / 'activity.json', encoding='utf-8'))
    before, after = map(float, re.search(r'from \$(\d+\.\d+) to \$(\d+\.\d+)',
                                         (ROOT / 'data/agent_runs/card_luna/packet/DESIGN.md').read_text(encoding='utf-8')).groups())
    ratios = [x['chosen_when_most_significant'] / x['chosen_otherwise'] for k in ('measure', 'specification') for x in choice[k]]
    nonmanager = next(x for x in choice['measure'] if x['option'] == 'nonmanager_fte')
    review = replay['review']['all']
    timing = activity['timing']
    entries = [
        ('card_stores', 'collected survey data from 410 fast-food restaurants', o['stores']),
        ('card_min_wage_before', 'an increase in the minimum wage from \\$4.25', before),
        ('card_min_wage_after', 'to \\$5.05 there', after),
        ('card_replays', 'replay it on 3,000 random subsamples', b['replays']),
        ('card_subset_size', 'random subsamples of 150 restaurants', b['subset_size']),
        ('card_tail_replays', '1,142 in the tail', b['tail_replays']),
        ('card_body_replays', '1,858 in the body', b['body_replays']),
        ('card_fact_classified', 'The match review classifies 2,985 of these replays', c['b1'] + c['b0']),
        ('card_fact_unresolved', '(15 remain unresolved)', b['replays'] - c['b1'] - c['b0']),
        ('card_display_replays', "the displays of the replays' programs use the 2,983", replay['replays']),
        ('card_activity_replays', 'the 2,944 with a purpose code for every step',
         activity['replays']['tail'] + activity['replays']['body']),
        ('card_original_estimate', 'The estimate it reported for the New-Jersey DiD coefficient was $2.97$', o['estimate']),
        ('card_original_estimate_employees', 'raised employment by about 2.97 employees per restaurant', o['estimate']),
        ('card_original_p', 'was $2.97$ ($p=0.025$)', o['u']),
        ('card_ck_estimate', 'and find an increase of 2.30 employees', o['ck_estimate']),
        ('card_ck_p', '2.30 employees ($p=0.055$)', o['ck_p']),
        ('card_ck_stores', 'restrict their main analysis to the 357 restaurants with data on employment and starting wages',
         o['ck_stores']),
        ('card_tail_reselection', 'reselecting it in 12.6\\% of tail replays', c['x1'] / c['b1']),
        ('card_body_reselection', 'but only in 7.2\\% of body replays', c['x0'] / c['b0']),
        ('card_lambda', 'This corresponds to a selection multiplier of 1.72', c['multiplier']),
        ('card_lambda_lower', 'multiplier of 1.72 with a one-sided 95\\% lower bound of 1.43', c['multiplier_lower_one95']),
        ('card_lambda_005', 'the multiplier increases further to $1.90$', t['multiplier']),
        ('card_lambda_005_lower', '$1.90$, with a one-sided 95\\% lower bound of 1.55', t['multiplier_lower_one95']),
        ('card_u', 'the original $p$-value of $0.0252$', c['u']),
        ('card_p_corrected', 'gets corrected upward to $0.0433$', c['p_corrected']),
        ('card_p_upper', 'cannot reject the null anymore at $0.052$', c['upper_one95']),
        ('card_sound_reviewer', 'judges 98.5\\% of GPT-6-Luna\'s analyses to be methodologically sound',
         review['pass'] / review['replays']),
        ('card_sound_after_audit', 'and 99.5\\% after a second model (Claude Opus 5.5, at high effort) audits its verdicts',
         review['pass_audited'] / review['replays']),
        ('card_tail_matches', 'Matching replays: 143 of 1,135 in the tail', c['x1']),
        ('card_tail_classified', '143 of 1,135 in the tail', c['b1']),
        ('card_body_matches', 'and 133 of 1,850 in the body', c['x0']),
        ('card_body_classified', '133 of 1,850 in the body', c['b0']),
        ('card_employment_options', 'For each of the 4 employment options', len(choice['measure'])),
        ('card_choice_ratio_min', 'choosing it at least 2$\\times$', min(ratios)),
        ('card_choice_ratio_max', 'and up to 12.6$\\times$ as frequently', max(ratios)),
        ('card_nonmanager_default', 'being chosen 39\\% of the time even when it is not the most significant',
         nonmanager['chosen_otherwise']),
        ('card_sound_up_to', 'up to 99.5\\% of replay agents\' analyses were methodologically sound',
         review['pass_audited'] / review['replays']),
        ('card_activity_categories', 'We classify agent actions into 8 categories', len(activity['steps'])),
        ('card_first_bin_inspect_steps', '85\\% of all steps in the first out of ten bins involve data inspection',
         activity['composition'][1][0]),
        ('card_first_bin_inspect_replays', 'with 94\\% of all replay agents inspecting the data at least once in that first bin',
         timing['inspect_in_first_part']),
        ('card_first_bin_setup_only', 'the remaining 6\\% of agents spending the entire first bin setting up the computing environment',
         timing['only_setup_in_first_part']),
        ('card_estimate_before_script', '97.9\\% of replay agents ran an analysis before writing any external scripts',
         timing['estimate_before_script']),
        ('card_edit_after_report', 'about 23.5\\% of agents edit the \\texttt{analysis.R} script',
         timing['analysis_R_edit_after_report']),
        ('card_activity_parts', 'divided by chronological position into ten equal parts', len(activity['composition'][0])),
        ('card_activity_tail', 'Coded from 1,113 tail', activity['replays']['tail']),
        ('card_activity_body', 'and 1,831 body replays', activity['replays']['body']),
        ('card_stores_table', 'for 410 restaurants total surveyed', o['stores']),
        ('card_min_wage_before_table', 'minimum wage rose from \\$4.25', before),
        ('card_min_wage_after_table', 'to \\$5.05 on April 1', after),
        ('card_replay_significance_n', 'Based on 2,983 classified replays', replay['replays']),
    ]
    # Counts read from CSV as floats are written as integers.
    write_numbers('08_card', [(i, quote, int(value) if float(value).is_integer() else value) for i, quote, value in entries])


if __name__ == '__main__':
    main()
