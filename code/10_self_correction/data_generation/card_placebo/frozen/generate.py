"""Known-null Card randomization placebo; never mounted in agent workspace.

Globally permute the coherent state/geography bundle across intact store
histories, preserving the observed total NJ count. No status/chain strata.
Original store/sheet identifiers are replaced to remove historical state.
"""
import csv
import random
from pathlib import Path

LABELS = ('state','southj','centralj','northj','pa1','pa2','shore')
ORIGINAL_SEED = 117_020_000
CALIBRATION_SEED = 118_020_000

def read_source(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames), rows

def generate_rows(source_rows, seed):
    order = list(range(len(source_rows)))
    random.Random(int(seed)).shuffle(order)
    result = []
    for i, donor in enumerate(order):
        row = dict(source_rows[i])
        row.update({k: source_rows[donor][k] for k in LABELS})
        row['sheet'] = str(i + 1)
        row['store_id'] = 'store_' + str(i + 1).zfill(4)
        result.append(row)
    return result

def generate_csv(source, destination, seed):
    fields, rows = read_source(source)
    generated = generate_rows(rows, seed)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with open(destination, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(generated)
    return generated

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');p.add_argument('seed',type=int)
    a=p.parse_args();generate_csv(a.source,a.destination,a.seed)
