#!/usr/bin/env python3
"""Collects the (id, quote, value) entries of scripts 01, 03 and 04 and writes the module's in-text numbers.

Inputs:  data/derived/10_self_correction/numbers_stance.json, numbers_card.json, numbers_power.json
Output:  results/numbers/10_self_correction.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'lib'))
from paper_numbers import write_numbers  # noqa: E402

DERIVED = ROOT / 'data' / 'derived' / '10_self_correction'
write_numbers('10_self_correction', [e for part in ('stance', 'card', 'power')
                                     for e in json.load(open(DERIVED / f'numbers_{part}.json', encoding='utf-8'))])
