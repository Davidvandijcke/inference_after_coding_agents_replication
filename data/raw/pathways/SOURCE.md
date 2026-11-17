# Pathways to Work synthetic evaluation data

Author-generated synthetic data, CC BY 4.0.

| File | Contents |
|---|---|
| `main-00.csv` ... `main-99.csv` | The 100 datasets of the open-ended agent design, one per original agent: 2,000 participants, 29 integer columns, every treatment effect zero |
| `CODEBOOK.md` | The codebook every agent received with its data |

`main-SS.csv` is `generate(8_000_000 + SS, 2000)` from `code/02_open_ended/data_generation/generate.py`;
`cluster/make_originals.py OUT_DIR OUT_DIR --n 100 --seed-base 8000000 --prefix main` in that folder, run beside
`generate.py`, writes all 100 files to `OUT_DIR/data/originals/` byte for byte with numpy 2.4.4. A replay agent received
a subsample: subsample `j` of dataset `SS` holds the 200 rows listed by
`random.Random(f"{8_000_000 + SS}-bank-{j}").sample(range(2000), 200)`, in that order, under the same header.
