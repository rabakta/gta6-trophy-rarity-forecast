# GTA VI trophy rarity forecast — make.fun entry

Forecasts each Grand Theft Auto VI PS5 trophy's earn rate on Exophase at
2026-11-26 00:00 UTC, as five quantiles per trophy, from the house's recorded
reads up to the deadline. `main.py` (Python 3.13, stdlib only, no network)
reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one
`*.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Model

A trophy's rate is earners / players_tracked, and the two move for different
reasons, so they are forecast separately.

- **Players N(t)** follow a power law in time since release,
  N ~ (1 + t)^alpha. Alpha is fitted by least squares on log N vs log(1 + t)
  over the case's distinct reads when the history shows growth; when the
  history is flat (launch night: every read 638 players) the prior alpha = 1
  is used, anchored at the first read — a flat page is a stale Exophase
  refresh, not a frozen player count.
- **Earners** come in two cohorts. Players already tracked keep playing and
  convert toward the trophy's ceiling with a time constant that shrinks for
  easy trophies (high current rate). Players who join by the target are young:
  a fresh player earns nothing for the first few hours, then converts toward
  the same ceiling. The forecast rate is (old + new earners) / N(target).
- **Spread** is on the logit scale, combining uncertainty in player growth
  and in progression, calibrated on the documented launch-night refresh and
  on synthetic week-long finals. A trophy at ~100% is forecast near 100%
  with a one-sided spread; a trophy nobody holds yet stays near the floor.

## Local trials

Practice cases (answers reconstructed from the documented refresh:
3,868 players, 4,276 trophies earned; five reconstruction variants for how
the 103 new non-prologue earnings split):

| case | this entry | house baseline |
| --- | --- | --- |
| p00 | loss 0.070–0.081 (plausible variants) | 1.39–1.43 |
| p01 | loss 0.082–0.089 | 1.39–1.43 |

On three synthetic week-long finals built from an independent cohort model
(steady / dilution-dominated / fast-progression): loss 0.17–0.21 vs
baseline 0.54–0.59, coverage 0.91–0.96. Runtime ~0.2–0.4 s per case,
deterministic (two runs byte-identical).

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT (see LICENSE).
