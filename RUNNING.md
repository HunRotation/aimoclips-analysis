Use Python 3.10.12.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py
```

Numerical outputs are written to `results/`. The included CLAP and FAD scores are the manuscript's saved statistical inputs; audio metric recomputation and figure rendering are outside this package.
