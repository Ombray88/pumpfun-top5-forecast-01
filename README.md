# pumpfun-top5-forecast-01

Momentum + growth-extrapolation forecaster for the make.fun pump.fun top-5
forecast competition: predict which 5 pump.fun coins launched since T0 will be
the largest by market cap 24h out, and how large each place's market cap will be.

## Method

Pump.fun's top 5 turns over completely every few minutes, so the persistence
baseline (last snapshot holds) is weak. This entry extrapolates every coin's
log10 market cap to the middle of the answer window from a recency-weighted
linear fit of its own trajectory, damps the growth (coins saturate), and turns
the extrapolation into honest probabilities with a Monte Carlo over the
candidates (20,000 draws, fixed seed, deterministic). Coins born after the cut
cannot be named, so a share of every place is left unnamed for them (more the
longer the horizon). Each place's size distribution is a normal in log10(cap)
blended halfway between the Monte Carlo mean and the last mark's rank-k cap.

## Run

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22+, standard library only. No network, no secrets.

## Backtest

`PUMPFUN_BACKTEST=1` predicts each next mark from the earlier marks on the
kit's practice history and scores this entry vs the persistence baseline with
the ranked.ts log-score rule. Mean over 3 cuts: ours -35.47 vs baseline -35.76
(delta +0.29) on 10-minute horizons; the real final horizon is ~10h, where
persistence is far weaker and momentum should win by more.

## How to run

To execute this program:
- **Runtime:** Requires Node.js 22 or newer. Uses exclusively the Node.js standard library with zero external dependencies, no network access, and no secrets.
- **Inputs and Outputs:** Reads input data from `/input/task.json` and `/input/history.ndjson` (or via `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR` environment variables to override) and writes the forecast to `/output/forecast.json`.
- **Backtest Mode:** Enable backtesting by setting the environment variable `PUMPFUN_BACKTEST=1`.
