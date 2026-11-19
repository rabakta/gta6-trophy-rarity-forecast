"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Model: a trophy's rate is earners / players_tracked, and the two move for
different reasons, so forecast them separately.

Players N(t): Exophase only refreshes a game's counts every few hours, so the
page a read shows is stale, then jumps. N(t) is modelled as a power law in
time since release, N ~ (1 + t)^alpha: alpha is fitted by least squares on
log N vs log(1 + t) over the case's distinct reads when the history shows
growth; when the history is flat (launch night: every read 638 players) the
prior alpha = 1 is used, anchored at the first read, because the flat page is
a stale refresh, not a frozen player count.

Earners E_i(t): two cohorts.
  * The players already tracked at the checkpoint keep playing: their
    non-earners convert toward the trophy's ceiling S_i with time constant
    tau_i, E_old = E + (N - E) * (1 - exp(-H / tau_i)).
  * The players who join by the target arrive through the horizon and are
    young: a fresh player earns nothing for the first D_i hours, then
    converts toward the same ceiling, E_new = N_new * S_i *
    (1 - exp(-(A_new - D_i) / tau_i)) for A_new > D_i, else 0, where A_new is
    the new cohort's mean age at the target (about half the refresh
    interval before it, since the answering read catches the last refresh).
  * tau_i and D_i shrink for easy trophies (high current rate), S_i grows
    with the current rate: a trophy everybody already holds is early and
    nearly universal; a trophy almost nobody holds is late and rare.
The forecast rate is (E_old + E_new) / N_target, evaluated per trophy.

Spread: quantiles on the logit scale, sigma combining uncertainty in the
player growth (wider the more N must grow) and in progression (wider the
longer the horizon). A trophy at ~100% is forecast near 100% with a
one-sided spread, since its logit punishes any miss hard either way.

Stdlib only, deterministic, no network. Reads every *.in in the input dir,
writes one *.out per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tunables (calibrated on the documented launch-night refresh) ---
ALPHA_PRIOR = 1.0        # power-law exponent for N(t) when history is flat
ALPHA_MIN, ALPHA_MAX = 0.15, 1.2
GROWTH_CAP = 60.0        # N_target / N_last is never forecast above this
# TAU0 = 800: the documented refresh added only ~103 new earnings across
# 54 trophies in 4.5h, which bounds the average conversion time constant
# near 600-800h; 500 overshoots the practice totals, 1000+ undershoots
# week-long progression in synthetic finals. TAU0 = 800 splits both.
TAU0 = 800.0             # hours: conversion time constant at a 1% trophy
D0 = 2.5                 # hours: delay before a fresh player can earn a 1% trophy
S_A, S_B = 2.0, 0.5      # ceiling S_i = min(0.95, S_A * p_i ** S_B)
REFRESH_PRIOR_H = 5.0    # hours between Exophase refreshes when unobserved
# Spread, in logit units. Calibrated, not guessed: on the documented
# launch-night refresh the median's logit error has std ~0.05, and on
# synthetic week-long finals built from an independent cohort model it is
# ~0.5 with heavy tails; a half-width of the naive random-walk spread
# scores best on both (practice loss 0.08 vs 0.15 at full width).
SIG_N = 0.22             # logit sigma per sqrt(log growth) of N
SIG_P0, SIG_P1 = 0.13, 0.08  # progression sigma: SIG_P0 + SIG_P1 * sqrt(days)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since(dt, origin):
    return max(0.0, (dt - origin).total_seconds() / 3600.0)


def distinct_reads(history):
    """Consecutive reads with identical N and rates carry no signal; keep one
    of each run, oldest first."""
    out = []
    for read in history:
        if out and out[-1]["players_tracked"] == read["players_tracked"] \
                and out[-1]["rates"] == read["rates"]:
            continue
        out.append(read)
    return out


def fit_players(history, release, target_dt):
    """Return (N_target, growth_factor, refresh_interval_hours).

    Power law N(t) = N_ref * ((1 + t) / (1 + t_ref)) ** alpha, fitted on
    distinct reads when possible, else the prior alpha anchored at the
    first read (the flat page is a stale refresh).
    """
    reads = distinct_reads(history)
    first = reads[0]
    last = reads[-1]
    n_last = last["players_tracked"]
    t_first = hours_since(parse_ts(first["at"]), release)
    t_last = hours_since(parse_ts(last["at"]), release)
    t_target = hours_since(target_dt, release)

    points = []
    seen_n = set()
    for read in reads:
        n = read["players_tracked"]
        if n in seen_n or n <= 0:
            continue
        seen_n.add(n)
        points.append((hours_since(parse_ts(read["at"]), release), n))

    alpha = ALPHA_PRIOR
    anchor_t, anchor_n = t_first, first["players_tracked"]
    if len(points) >= 3:
        xs = [math.log1p(t) for t, _ in points]
        ys = [math.log(n) for _, n in points]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        if sxx > 1e-9:
            b = sxy / sxx
            a = my - b * mx
            # Trust the fit only if it roughly reproduces the last read.
            pred_last = math.exp(a + b * math.log1p(t_last))
            if 0.75 * n_last <= pred_last <= 1.33 * n_last:
                alpha = min(ALPHA_MAX, max(ALPHA_MIN, b))
                anchor_t, anchor_n = t_last, n_last
    growth = ((1.0 + t_target) / (1.0 + anchor_t)) ** alpha
    growth = min(GROWTH_CAP, max(1.0, growth))
    n_target = anchor_n * growth
    if n_target < n_last:
        n_target = float(n_last)
    # What the spread prices is growth from the last read, not from the anchor.
    growth_vs_last = n_target / n_last if n_last > 0 else 1.0

    # Refresh cadence: median gap between distinct reads, else the prior.
    gaps = []
    for prev, cur in zip(reads, reads[1:]):
        if cur["players_tracked"] != prev["players_tracked"]:
            gaps.append((parse_ts(cur["at"]) - parse_ts(prev["at"])).total_seconds() / 3600.0)
    if gaps:
        gaps.sort()
        interval = gaps[len(gaps) // 2]
    else:
        interval = REFRESH_PRIOR_H
    return n_target, growth_vs_last, interval


def trophy_params(rate_pct):
    """Ceiling, delay and time constant for one trophy from its last rate."""
    p = max(rate_pct, 0.01) / 100.0
    ceiling = min(0.95, S_A * p ** S_B)
    if p >= 0.5:
        ceiling = 1.0
    ceiling = max(ceiling, p)
    scale = (0.01 / p) ** 0.5
    tau = TAU0 * scale
    delay = D0 * scale
    return ceiling, delay, tau


def forecast_trophy(rate, n_last, n_target, horizon_h, interval_h):
    """Median forecast rate (percent) for one trophy."""
    if rate is None:
        rate = 0.0
    if rate >= 99.99:
        return 100.0
    earners = rate / 100.0 * n_last
    ceiling, delay, tau = trophy_params(rate)
    # Players already tracked: convert toward the ceiling over the horizon.
    old = earners + (n_last - earners) * (1.0 - math.exp(-horizon_h / tau))
    # Players who join by the target: mean age ~ half a refresh interval
    # before the answering read, and nothing earned before the delay.
    n_new = max(0.0, n_target - n_last)
    age_new = max(0.0, interval_h / 2.0)
    if age_new > delay:
        new = n_new * ceiling * (1.0 - math.exp(-(age_new - delay) / tau))
    else:
        new = 0.0
    total = old + new
    return 100.0 * total / n_target if n_target > 0 else rate


def sigma_for(growth, horizon_h):
    return math.sqrt((SIG_N * math.sqrt(max(0.0, math.log(growth)))) ** 2
                     + (SIG_P0 + SIG_P1 * math.sqrt(horizon_h / 24.0)) ** 2)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    release_s = case["game"].get("release_date", "2026-11-19")
    release = datetime.fromisoformat(release_s + "T00:00:00+00:00")
    target_dt = parse_ts(case["target"])
    horizon_h = case["horizon_minutes"] / 60.0
    n_target, growth, interval = fit_players(case["history"], release, target_dt)
    n_last = case["history"][-1]["players_tracked"]
    sigma = sigma_for(growth, horizon_h)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = None
        for read in reversed(case["history"]):
            if read["rates"][i] is not None:
                rate = read["rates"][i]
                break
        if rate is None:
            rate = 0.0
        med = forecast_trophy(rate, n_last, n_target, horizon_h, interval)
        if rate >= 99.99 or med >= 99.99:
            qs = [99.8, 99.97, 100.0, 100.0, 100.0]
        elif rate < 0.005 and rate * n_last / 100.0 < 0.5:
            # Nobody holds it yet: a trophy with zero earners in days of
            # tracking has a near-zero hazard, so stay near the floor.
            qs = [0.01, 0.01, 0.02, 0.05, 0.15]
        else:
            lg = logit(med)
            qs = [round4(min(100.0, max(0.01, inv_logit(lg + z * sigma))))
                  for z in Z]
            qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
