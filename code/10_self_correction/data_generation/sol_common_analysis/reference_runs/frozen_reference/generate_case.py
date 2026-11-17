"""Host-only generation of a fresh full-size known-null reference dataset."""
from pathlib import Path
import importlib.util

def generate_case(case, destination):
    source=Path(__file__).with_name('generate.py')
    spec=importlib.util.spec_from_file_location('sol_reference_generator',source)
    gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    data=gen.generate(int(case['seed']),2000,tau=0.0)
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    gen.write_csv(data,Path(destination))
