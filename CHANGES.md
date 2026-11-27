# Changes

## Fixed 24-month outcome window and out-of-time test (Nov 2026)

**Why.** The earlier version kept only finished loans (fully paid or charged off). In recent vintages most loans were still running at the data snapshot, so each vintage's default rate depended on which loans happened to have finished, not only on risk (20% overall; 23% among finished 2016 and 2017 loans). Loans that default early finish early, so recent vintages over-represent them. The model was also tested on a random split, which mixes years between training and test.

**What changed.**
- `lending_club.py` (new, shared with the other credit projects): loads the raw file, dates each default at last payment plus four months, and records the snapshot (March 2019).
- `build_dataset.py` now keeps **all** 2.26M loans and adds outcome columns for fixed 12- and 24-month windows, with a flag for whether each loan's window was fully observable by the snapshot.
- `analysis.py` models default within 24 months on the 1.22M loans whose window is observable, trains on loans issued up to 2014 and tests on 2015-2016. The random split is kept only for comparison. Chart titles that stated conclusions now follow the results.
- `analysis.py` is written in notebook cell format; `make_notebook.py` rebuilds and executes `affordability_default_risk.ipynb` from it, so the two cannot drift apart.
- The derived parquet is no longer committed (it is now 53 MB); `build_dataset.py` rebuilds it from the Kaggle file in about 30 seconds.
- `results.json` holds every number quoted in the README.

**Numbers.**

| | Before (finished loans, random split) | After (24-month window, out of time) |
|---|---|---|
| Loans modelled | 1.35M, 20.0% default | 1.22M, 10.8% default (test: 752k, 11.6%) |
| Credit + bureau ROC-AUC / PR-AUC | 0.617 / 0.271 | 0.620 / 0.166 |
| Affordability ROC-AUC / PR-AUC | 0.699 / 0.361 | 0.683 / 0.216 |
| Both ROC-AUC / PR-AUC | 0.706 / 0.371 | 0.683 / 0.218 |
| KS (combined) | 0.30 | 0.26 |
| Default rate, payment-to-income under 5% vs over 20% | 14% vs 34% | 7.7% vs 20.4% |

PR-AUC is not comparable across the two columns because the default rate differs (no-skill PR-AUC is 0.20 before and 0.116 after). The main finding holds: affordability alone out-predicts the credit score and bureau record. What did not survive is the gain from combining the two, which disappears out of time.
