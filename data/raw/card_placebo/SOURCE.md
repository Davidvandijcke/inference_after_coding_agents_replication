# Card-Krueger placebo datasets (derived from public data)

Placebo versions of the Card and Krueger (1994) fast-food survey analyzed by the sessions in
`data/agent_runs/card_placebo_sol/` (Figure `fig:self_correction_family`, note). Derived from the public Card and
Krueger data shipped by module 08 (`data/raw/card_krueger/`, with its citation and terms); the authors' derivation
is CC BY 4.0.

- Files: `placebo-NNN.csv.gz`, NNN = 000 to 199, 410 stores each (3 MB compressed).
- Construction: the frozen generator
  `code/10_self_correction/data_generation/card_placebo/frozen/generate.py` takes the Card analysis file the
  Card-Krueger agents receive (`data/agent_runs/card_luna/packet/data.csv`, SHA-256
  f84acec0e243bc13e2070cee5017a4199b44524872dfec1c603fb9baae8c917a; the run code calls its copy
  `frozen/source_card.csv`), permutes the joint bundle of state and
  region labels across the 410 store histories with seed 117,020,000 + NNN, keeps the 331 New Jersey and 79
  Pennsylvania labels, and replaces store identifiers. Employment, wages, prices, status and missingness of every
  store are unchanged, so every state effect is zero by construction.
- Regeneration: `generate_csv(source, destination, seed)` of that generator, applied to the packet's `data.csv`
  with seed 117,020,000 + NNN, rewrites each file from the package alone.
  `code/10_self_correction/data_generation/extract_records.py` does this for all 200 and checks each against the
  SHA-256 its session recorded when the dataset was served.
