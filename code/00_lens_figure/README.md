# 00 Introduction figure

Figure 1 (`paper/figs/lens_figure.pdf`): a stylized illustration of the correction, drawn from simulated values
(seed 7). It uses no data from the experiments.

`01_lens_figure.py` draws it in a few seconds. `run_all.sh` runs this module in its own Python environment
(`.venv-lens`, `environment/requirements-lens.txt`), because the figure's text layout was set with matplotlib
3.11.2 and the other modules use 3.10.8. The typeface is Inter 4.000, shipped in `fonts/` under the SIL Open
Font License 1.1 (`fonts/OFL.txt`).
