# articles-GridGuard

Stability screening of a decentralised smart-grid control model with class-wise conformal prediction and referral
to an analyst. Data: UCI Electrical Grid Stability Simulated Data (**a simulation of a 4-node star grid, not field measurements**).
TRL 4 (public simulation-based dataset, laboratory validation only).

```
pip install -e ".[dev]" && python -m pytest
# data: https://archive.ics.uci.edu/dataset/471 (not committed)
python research/run_study.py /path/to/Data_for_UCI_named.csv
```
