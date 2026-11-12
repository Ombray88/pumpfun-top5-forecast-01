// pumpfun-top5-forecast-01: momentum + growth-extrapolation forecaster (Node 22+, stdlib only).
//
// Idea: pump.fun's top 5 turns over completely every few minutes (in the practice
// history the #1 changes three times in 40 minutes). The persistence baseline (last
// snapshot holds) is weak here. This entry instead extrapolates every coin's
// log10 market cap to the middle of the answer window, from a recency-weighted fit
// of its own trajectory, damps the growth (coins saturate), and turns the
// extrapolation into honest probabilities with a Monte Carlo over the candidates.
// Coins born after the cut cannot be named, so a share of every place is left
// unnamed for them (more the longer the horizon).
//
// Input: /input/task.json (places, size-bin edges, output name, answer reads) and
// /input/history.ndjson (marks, each a chart of eligible coins by market cap).
// Output: /output/forecast.json { probabilities, top5, sizes }.
// Outside the grader: MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input / /output.
// PUMPFUN_BACKTEST=1 runs a local backtest instead (see backtest()): it predicts
// the last mark from the earlier marks and scores this entry vs the persistence
// baseline with the ranked.ts log-score rule. It never runs in the grader.

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MICRO = 1_000_000;
/** Monte Carlo draws for the rank probabilities. Fixed seed: the run is deterministic. */
const MC_DRAWS = 20000;
/** Candidates kept for the Monte Carlo (top by extrapolated mean). */
const MC_CANDIDATES = 400;
/** Growth half-life (hours) for the damping factor 1 / (1 + horizon / 2). */
const DAMP_TAU_H = 2;
/** Shrinkage applied to the fitted slope: pump growth is noisy and decelerates. */
const SLOPE_SHRINK = 0.7;
/** Floor / cap on a coin's predictive sd in log10(cap). */
const SIGMA_FLOOR = 0.25;
const SIGMA_CAP = 1.4;
/** Sigma for a coin seen only once (no slope): it can move a lot. */
const SIGMA_SINGLE = 1.0;
/** Sigma floor for a coin's size distribution at a place. */
const SIZE_SD_FLOOR = 0.3;
const SIZE_SD_CAP = 1.2;

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  created_at?: string;
  mayhem?: boolean;
}

interface Mark {
  at: string;
  ok: boolean;
  rows: Row[] | null;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  size: { edges: number[] };
  reads?: string[];
  window?: { answer_at?: string };
};

function readMarks(): Mark[] {
  const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
    .split('\n')
    .filter((l) => l !== '');
  return lines
    .map((l) => JSON.parse(l) as Mark)
    .filter((m) => m.ok === true && Array.isArray(m.rows));
}

/** log10 of a cap in SOL, as the scorer takes it: log10(lamports) - 9. */
const sizeOf = (lamports: string | number): number =>
  Math.log10(Number(BigInt(lamports))) - 9;

/** erfc(x), Numerical Recipes' erfcc (fractional error under 1.2e-7 everywhere). */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  // Horner form of the polynomial, evaluated stepwise.
  let p = 0.17087277;
  p = -0.82215223 + t * p;
  p = 1.48851587 + t * p;
  p = -1.13520398 + t * p;
  p = 0.27886807 + t * p;
  p = -0.18628806 + t * p;
  p = 0.09678418 + t * p;
  p = 0.37409196 + t * p;
  p = 1.00002368 + t * p;
  const r = t * Math.exp(-z * z - 1.26551223 + t * p);
  return x >= 0 ? r : 2 - r;
}
/** P(X >= x) for X normal with mean `mean` and sd `sd`. */
const above = (x: number, mean: number, sd: number): number =>
  x === -Infinity ? 1 : x === Infinity ? 0 : 0.5 * erfc((x - mean) / (sd * Math.SQRT2));

/** Deterministic PRNG (mulberry32). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** One standard-normal draw (Box-Muller, uses two uniforms). */
function randn(rng: () => number): number {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

interface CoinFit {
  mint: string;
  /** log10 cap at the last mark. */
  last: number;
  /** Extrapolated log10 cap at the target time. */
  mu: number;
  /** Predictive sd of the log10 cap at the target time. */
  sigma: number;
  /** Number of marks the coin was seen in. */
  n: number;
}

/**
 * Fit every coin present in the last mark: recency-weighted linear fit of its
 * log10 cap against time, extrapolated to the target time with damped growth.
 * Returns the fits sorted by extrapolated mean, descending.
 */
function fitCoins(marks: Mark[], targetMs: number): CoinFit[] {
  const lastAt = Date.parse(marks[marks.length - 1]!.at);
  const firstAt = Date.parse(marks[0]!.at);
  const spanH = Math.max((lastAt - firstAt) / 3_600_000, 0.25);
  const horizonH = Math.max((targetMs - lastAt) / 3_600_000, 0);
  const damp = 1 / (1 + horizonH / DAMP_TAU_H);
  // Recency weight half-life: one quarter of the history span.
  const halfLifeMs = Math.max(spanH / 4, 0.1) * 3_600_000;

  const series = new Map<string, Array<{ t: number; y: number }>>();
  for (const m of marks) {
    const t = Date.parse(m.at);
    for (const r of m.rows!) {
      const y = sizeOf(r.market_cap_lamports);
      if (!Number.isFinite(y)) continue;
      let s = series.get(r.mint);
      if (!s) {
        s = [];
        series.set(r.mint, s);
      }
      s.push({ t, y });
    }
  }

  const lastRows = marks[marks.length - 1]!.rows!;
  const fits: CoinFit[] = [];
  for (const r of lastRows) {
    const s = series.get(r.mint);
    if (!s || s.length === 0) continue;
    const last = s[s.length - 1]!.y;
    if (s.length === 1) {
      fits.push({ mint: r.mint, last, mu: last, sigma: SIGMA_SINGLE, n: 1 });
      continue;
    }
    // Weighted linear fit y ~ a + b * (t - lastAt), weights decay with age.
    let sw = 0;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (const p of s) {
      const w = Math.pow(0.5, (lastAt - p.t) / halfLifeMs);
      const x = (p.t - lastAt) / 3_600_000; // hours before the last mark (<= 0)
      sw += w;
      sx += w * x;
      sy += w * p.y;
      sxx += w * x * x;
      sxy += w * x * p.y;
    }
    const denom = sw * sxx - sx * sx;
    let slope = 0; // log10 cap per hour
    let intercept = sy / sw;
    if (denom > 1e-12) {
      slope = (sw * sxy - sx * sy) / denom;
      intercept = (sy - slope * sx) / sw;
    }
    // Residual sd around the fit (weighted).
    let sse = 0;
    for (const p of s) {
      const w = Math.pow(0.5, (lastAt - p.t) / halfLifeMs);
      const x = (p.t - lastAt) / 3_600_000;
      const e = p.y - (intercept + slope * x);
      sse += w * e * e;
    }
    const residSd = Math.sqrt(sse / Math.max(sw, 1e-9));
    // Slope uncertainty grows with the horizon; residual sd is the floor.
    const slopeVar = Math.max(0.02, residSd * residSd) / Math.max(s.length - 1, 1);
    const sigma = Math.min(
      SIGMA_CAP,
      Math.max(SIGMA_FLOOR, Math.sqrt(residSd * residSd + slopeVar * horizonH * horizonH + 0.02)),
    );
    // Damped extrapolation, anchored on the last observed value (the fitted
    // intercept lags a fast mover). Clamp the slope: a coin cannot sustain >10x/hour.
    const slopeClamped = Math.max(-1.5, Math.min(1.5, slope)) * SLOPE_SHRINK;
    const mu = last + slopeClamped * horizonH * damp;
    fits.push({ mint: r.mint, last, mu, sigma, n: s.length });
  }
  fits.sort((a, b) => b.mu - a.mu);
  return fits;
}

interface Forecast {
  probabilities: Array<{ mint: string; place: number; p: number }>;
  top5: string[];
  sizes: Array<{ place: number; bin: number; p: number }>;
}

/** Mean of the answer reads in task.json (fallback: window.answer_at, else last mark + 1h). */
function targetTimeMs(marks: Mark[]): number {
  if (task.reads && task.reads.length > 0) {
    const sum = task.reads.reduce((a, r) => a + Date.parse(r), 0);
    return sum / task.reads.length;
  }
  if (task.window?.answer_at) return Date.parse(task.window.answer_at);
  return Date.parse(marks[marks.length - 1]!.at) + 3_600_000;
}

function buildForecast(marks: Mark[]): Forecast {
  const targetMs = targetTimeMs(marks);
  const lastAt = Date.parse(marks[marks.length - 1]!.at);
  const horizonH = Math.max((targetMs - lastAt) / 3_600_000, 0);
  const fits = fitCoins(marks, targetMs);
  const candidates = fits.slice(0, MC_CANDIDATES);
  const places = task.places;

  // Monte Carlo: draw each candidate's log-cap at the target, rank, count.
  const rng = mulberry32(0x9e3779b9);
  const counts: number[][] = Array.from({ length: places }, () =>
    new Array<number>(candidates.length).fill(0),
  );
  const sizeSamples: number[][] = Array.from({ length: places }, () => []);
  const vals = new Array<number>(candidates.length);
  for (let d = 0; d < MC_DRAWS; d++) {
    for (let i = 0; i < candidates.length; i++) {
      vals[i] = candidates[i]!.mu + candidates[i]!.sigma * randn(rng);
    }
    // Top `places` by value (partial selection, no full sort).
    const topIdx: number[] = [];
    for (let i = 0; i < candidates.length; i++) {
      if (topIdx.length < places) {
        topIdx.push(i);
        topIdx.sort((a, b) => vals[b]! - vals[a]!);
      } else if (vals[i]! > vals[topIdx[places - 1]!]!) {
        topIdx[places - 1] = i;
        topIdx.sort((a, b) => vals[b]! - vals[a]!);
      }
    }
    for (let k = 0; k < places; k++) {
      const idx = topIdx[k]!;
      counts[k]![idx]! += 1;
      sizeSamples[k]!.push(vals[idx]!);
    }
  }

  // Unnamed reserve: coins born after the cut can still reach the top 5.
  // It grows with the horizon (more time for a new coin to explode).
  const reserve = Math.min(0.4, Math.max(0.2, 0.15 + 0.02 * horizonH));
  const namedCap = 1 - reserve;

  const probabilities: Forecast['probabilities'] = [];
  for (let k = 0; k < places; k++) {
    const place = k + 1;
    const rows: Array<{ mint: string; p: number }> = [];
    for (let i = 0; i < candidates.length; i++) {
      const p = (counts[k]![i]! / MC_DRAWS) * namedCap;
      if (p >= 1e-4) rows.push({ mint: candidates[i]!.mint, p });
    }
    rows.sort((a, b) => b.p - a.p);
    // Keep at most 40 named rows per place; renormalise if the cap is exceeded.
    const kept = rows.slice(0, 40);
    const sum = kept.reduce((a, r) => a + r.p, 0);
    const scale = sum > namedCap ? namedCap / sum : 1;
    for (const r of kept) {
      const p = Math.floor((r.p * scale) * MICRO) / MICRO;
      if (p > 0) probabilities.push({ mint: r.mint, place, p });
    }
  }

  // top5: each place's most probable coin, distinct.
  const top5: string[] = [];
  const used = new Set<string>();
  for (let k = 0; k < places; k++) {
    let best: string | null = null;
    let bestP = -1;
    for (let i = 0; i < candidates.length; i++) {
      const p = counts[k]![i]! / MC_DRAWS;
      if (p > bestP && !used.has(candidates[i]!.mint)) {
        bestP = p;
        best = candidates[i]!.mint;
      }
    }
    if (best === null) best = candidates[k]!.mint;
    used.add(best);
    top5.push(best);
  }

  // Sizes: per place, the MC distribution of the rank-k coin's log-cap.
  // The sd grows with the horizon (a 10h forecast is far less certain than a
  // 10-minute one), with the baseline's 0.35 as the short-horizon floor.
  // The mean is blended halfway back toward the last mark's rank-k cap:
  // extrapolated growth overshoots (coins saturate), while pure persistence
  // undershoots a rising top; the blend is the honest centre.
  const sizeSdFloor = Math.min(1.0, 0.35 + 0.03 * horizonH);
  const lastChart = [...marks[marks.length - 1]!.rows!].sort((a, b) => a.rank - b.rank);
  const sizes: Forecast['sizes'] = [];
  const edges = task.size.edges;
  for (let k = 0; k < places; k++) {
    const xs = sizeSamples[k]!;
    const mcMean = xs.reduce((a, x) => a + x, 0) / xs.length;
    const lastMean = sizeOf(lastChart[k]!.market_cap_lamports);
    const mean = 0.5 * mcMean + 0.5 * lastMean;
    const variance = xs.reduce((a, x) => a + (x - mean) * (x - mean), 0) / xs.length;
    const sd = Math.min(SIZE_SD_CAP, Math.max(sizeSdFloor, Math.sqrt(variance)));
    for (let bin = 0; bin <= edges.length; bin++) {
      const lo = bin === 0 ? -Infinity : edges[bin - 1]!;
      const hi = bin === edges.length ? Infinity : edges[bin]!;
      const u = Math.floor((above(lo, mean, sd) - above(hi, mean, sd)) * MICRO);
      if (u > 0) sizes.push({ place: k + 1, bin, p: u / MICRO });
    }
  }

  return { probabilities, top5, sizes };
}

function main(): void {
  const marks = readMarks();
  if (marks.length === 0) throw new Error('the history holds no ok mark');
  const forecast = buildForecast(marks);
  writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify(forecast)}\n`);
}

/** Local backtest: for each cut point, predict the next mark from the earlier
 * marks; score this entry vs the persistence baseline with the ranked.ts
 * log-score rule (floor 1e-4; unnamed mass shared equally over the chart's
 * unnamed coins). */
function backtest(): void {
  const marks = readMarks();
  if (marks.length < 3) throw new Error('need at least 3 marks for a backtest');
  const places = task.places;
  const edges = task.size.edges;
  const FLOOR = 1e-4;

  const score = (
    probs: Array<{ mint: string; place: number; p: number }>,
    sizes: Array<{ place: number; bin: number; p: number }>,
    chart: Row[],
  ) => {
    let order = 0;
    let size = 0;
    for (let k = 0; k < places; k++) {
      const place = k + 1;
      const rows = probs.filter((r) => r.place === place);
      const named = new Map(rows.map((r) => [r.mint, r.p]));
      const namedSum = rows.reduce((a, r) => a + r.p, 0);
      const unnamed = chart.filter((c) => !named.has(c.mint));
      const share = unnamed.length > 0 ? Math.max(0, 1 - namedSum) / unnamed.length : 0;
      const actual = chart[k]!;
      const p = named.get(actual.mint) ?? share;
      order += Math.log(Math.max(p, FLOOR));
      const actualSize = sizeOf(actual.market_cap_lamports);
      let bin = 0;
      for (let i = 0; i < edges.length; i++) if (actualSize >= edges[i]!) bin = i + 1;
      const srows = sizes.filter((s) => s.place === place);
      const smap = new Map(srows.map((s) => [s.bin, s.p]));
      const ssum = srows.reduce((a, s) => a + s.p, 0);
      const sbins = edges.length + 1;
      const unnamedBins = sbins - smap.size;
      const sshare = unnamedBins > 0 ? Math.max(0, 1 - ssum) / unnamedBins : 0;
      const sp = smap.get(bin) ?? sshare;
      size += Math.log(Math.max(sp, FLOOR));
    }
    return { order, size, total: order + size };
  };

  let totO = 0;
  let totB = 0;
  let n = 0;
  for (let cut = 3; cut < marks.length; cut++) {
    const train = marks.slice(0, cut);
    const answer = marks[cut]!;
    const chart = [...answer.rows!].sort((a, b) => a.rank - b.rank).slice(0, 100);

    const savedReads = task.reads;
    (task as { reads?: string[] }).reads = [answer.at];
    const ours = buildForecast(train);
    (task as { reads?: string[] }).reads = savedReads;

    const lastChart = [...train[train.length - 1]!.rows!].sort((a, b) => a.rank - b.rank);
    const baseProbs: Array<{ mint: string; place: number; p: number }> = [];
    for (let place = 1; place <= places; place++) {
      baseProbs.push({ mint: lastChart[place - 1]!.mint, place, p: 0.3 });
      if (place > 1) baseProbs.push({ mint: lastChart[place - 2]!.mint, place, p: 0.1 });
      if (lastChart[place] !== undefined)
        baseProbs.push({ mint: lastChart[place]!.mint, place, p: 0.1 });
    }
    const baseSizes: Array<{ place: number; bin: number; p: number }> = [];
    for (let place = 1; place <= places; place++) {
      const mean = sizeOf(lastChart[place - 1]!.market_cap_lamports);
      for (let bin = 0; bin <= edges.length; bin++) {
        const lo = bin === 0 ? -Infinity : edges[bin - 1]!;
        const hi = bin === edges.length ? Infinity : edges[bin]!;
        const u = Math.floor((above(lo, mean, 0.35) - above(hi, mean, 0.35)) * MICRO);
        if (u > 0) baseSizes.push({ place, bin, p: u / MICRO });
      }
    }

    const o = score(ours.probabilities, ours.sizes, chart);
    const b = score(baseProbs, baseSizes, chart);
    const actualTop = chart.slice(0, places).map((r) => r.mint);
    const hits = (top: string[]) => top.filter((m) => actualTop.includes(m)).length;
    console.log(
      `cut ${cut}: answer ${answer.at.slice(11, 19)} ours total=${o.total.toFixed(2)} (order ${o.order.toFixed(2)}, size ${o.size.toFixed(2)}, hits ${hits(ours.top5)}) | base total=${b.total.toFixed(2)} (order ${b.order.toFixed(2)}, size ${b.size.toFixed(2)}, hits ${hits(lastChart.slice(0, places).map((r) => r.mint))}) | delta ${(o.total - b.total).toFixed(2)}`,
    );
    totO += o.total;
    totB += b.total;
    n++;
  }
  console.log(`MEAN over ${n} cuts: ours=${(totO / n).toFixed(2)} base=${(totB / n).toFixed(2)} delta=${((totO - totB) / n).toFixed(2)}`);
}

if (process.env.PUMPFUN_BACKTEST === '1') backtest();
else main();
