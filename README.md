# Kalshi Edge Bot

Automated trading bot for [Kalshi](https://kalshi.com) prediction markets. It
continuously scans a configurable set of market series, looks for **structurally
risk‑free arbitrage** (and, optionally, model‑driven mispricings), sizes
positions with fractional Kelly, and manages exits.

The design goal is *highest return for the least risk*, so the default
configuration trades only the one strategy that is mathematically risk‑free
(`ARBS`) plus an optional model‑based "longshot" engine. The speculative
heuristics (`SPREAD`, `DRIFT`) ship disabled.

> ⚠️ **This is educational software that can move real money. Read the
> [Safety](#safety) section before running it live.**

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Kalshi API credentials

Kalshi authenticates API requests with an **RSA‑PSS signature**. In the Kalshi
web UI (Settings → API Keys) create a key pair, then:

- Save the **private key** to `kalshi_private.pem`.
- Put the **Key ID** in `credentials.json`:

```json
{
  "key_id": "your-key-id-uuid",
  "private_key_path": "kalshi_private.pem"
}
```

Both files are in `.gitignore` — never commit them.

### 3. Run the bot

```bash
# Paper trade on Kalshi's demo environment (safest)
python bot.py --demo

# Scan only — print signals, place nothing (forces dry-run)
python bot.py --scan-only

# Dry run — simulate sizing/positions locally, submit no orders
python bot.py --dry-run --demo

# Live trading (real money — use --demo first!)
python bot.py

# Verbose debug logging
python bot.py --demo -v
```

CLI flags override `config.json`. `--scan-only` implies `--dry-run`.

---

## How It Works

Each cycle the bot:

1. Refreshes balance and open positions from Kalshi.
2. Monitors existing positions for exits.
3. Scans the market universe (by default the curated `target_series` list; set
   `scan_all_markets: true` to paginate every open market), skips markets with
   no order book, generates signals, ranks them, and attempts the **top 5**
   that pass risk checks.

### Primary Engine — Historical Value (`valuation.py`)

The bot's main strategy buys markets that are **priced below their calibrated
historical fair value** and rides the reversion. A Kalshi price *is* the market's
probability estimate; we compute a better one and buy the side that's cheap
relative to it. Two independent edges are combined:

1. **Systematic calibration bias (favorite–longshot).** A robust, well‑documented
   effect: favorites win *more* often than their price implies and longshots
   *less*. We correct the implied probability by sharpening it in logit space:
   `fair = sigmoid(value_sharpen · logit(price))`. With `value_sharpen = 1.10`,
   a 90¢ favorite is nudged to ~92¢ fair and a 10¢ longshot down to ~8¢.
   Set `value_sharpen = 1.0` to disable.

2. **Per‑market mean reversion.** The bot persists a rolling, down‑sampled price
   history per ticker (`price_history.json`) and takes an EMA of the historical
   mid as a stable **anchor**. When the live ask falls at least
   `value_min_gap_cents` below the calibrated anchor, the market has likely
   overshot — so it buys the cheap side (YES *or* NO, whichever is underpriced).

A `VALUE` signal fires only when the calibrated anchor and the live price
disagree by ≥ `value_min_gap_cents` **and** the fee‑adjusted edge ≥
`value_min_edge`, for prices within `[value_min_price, value_max_price]`.
Confidence scales with history depth and deviation size (capped at 0.65). The
exit target is partway back to fair value.

> **Warm‑up:** `VALUE` needs `value_min_obs` history points per market before it
> can trade, and history is down‑sampled to one point per
> `history_min_interval_secs` (default 5 min). So expect **no `VALUE` trades for
> the first ~20–30 minutes** of a fresh run while history accumulates — the scan
> log reports how many markets are "warming up". History persists across
> restarts, so this is a one‑time cost.

### Analytic Strategies (`strategy.py`)

#### `ARBS` — Two‑leg arbitrage *(risk‑free, on by default alongside VALUE)*

A Kalshi market always pays exactly **100¢** to either the YES or the NO holder
at settlement. If you can buy **one YES** and **one NO** for a combined cost
(including both entry fees) **below 100¢**, the pair pays 100¢ no matter the
outcome — a locked, market‑neutral profit.

- Fires when `yes_ask + no_ask + fees < 100¢` with at least
  `MIN_ARB_NET_CENTS` (1¢) of guaranteed profit per pair.
- The bot buys **both legs** and **holds them to settlement** — arb positions
  are never exited early.
- Execution safety: if the first (YES) leg fills but the second (NO) leg order
  fails, the bot immediately tries to **unwind** the YES leg so it is never left
  with a naked directional position. A failed unwind is logged as `CRITICAL`.
- Confidence: 0.99.

#### `EVENT_ARB` — Multi‑market arbitrage *(detector, risk‑free)*

Within a **mutually‑exclusive** event (e.g. CPI ranges, Fed‑decision buckets,
"who wins the division"), at most one outcome resolves YES. Buying **NO on N of
those outcomes** therefore wins on at least `N‑1` of them. When the combined NO
cost (plus fees) is below `100×(N‑1)`, the position pays out no matter which
outcome wins — risk‑free, and requires only mutual exclusivity (not an
exhaustive set, so even a subset of outcomes works).

- The bot groups every scanned market by event, runs `check_event_arb`, and for
  any over‑round it **verifies the event is mutually exclusive** via
  `GET /events/{ticker}` before trusting it.
- **Detector only — it does not auto‑trade.** Placing N legs at once risks a
  partial fill (a half‑filled multi‑leg arb is a naked position), so the bot
  *alerts* instead: it logs the opportunity and appends a record to
  `arb_alerts.jsonl` for you to act on. Controlled by `event_arb_enabled` /
  `event_arb_min_cents`.

#### `FAVORITE` — Favorite–longshot bias *(speculative, off by default)*

A well‑documented prediction‑market bias: bettors overpay for longshots and
underpay for favorites, so heavy favorites (priced 80–95¢) tend to be modestly
underpriced. The bot assumes true probability is ~2pp above the quote and buys
YES. The honest edge is small (~1¢), **below the default `min_edge` of 0.02** —
to actually trade it you must lower `min_edge` (e.g. to `0.005`). Confidence: 0.45.

#### `MOMENTUM` — Trade continuation *(speculative, off by default)*

When the last trade moved at least 4¢ from the previous price on real volume
(≥20 contracts/24h), buys in the direction of the move (YES on up‑moves, NO on
down‑moves), projecting the move to continue. No guaranteed edge. Confidence: 0.32.

#### `SPREAD` — Spread mispricing *(speculative, off by default)*

When a market has a wide bid/ask spread, treats `last_price` as a fair‑value
anchor and buys the side the last trade sits closer to. No guaranteed edge.
Confidence: 0.30.

#### `DRIFT` — Price drift *(speculative, off by default)*

When `last_price` diverges from the current book mid (and recent trend agrees),
buys in the direction of the drift. No guaranteed edge. Confidence: 0.30.

> `SPREAD` and `DRIFT` derive "fair value" from heuristics, not a real model.
> Enable them only if you understand they can lose money to fees and spread.

### Longshot Engine (`longshot.py`) *(enabled in `config.json`)*

Finds cheap markets where an external data model implies the true probability is
materially higher than the price. All data sources are free and need no API key:

| Sub‑strategy | Ticker prefix | Model | Data source |
|---|---|---|---|
| `LONGSHOT_NBA` | `KXNBAPTS/REB/AST/BLK/STL` | Poisson over season average | ESPN stats |
| `LONGSHOT_WEATHER` | `HIGH<CITY>` | Normal(forecast, σ=8°F) | api.weather.gov (NWS) |
| `LONGSHOT_ECON` | `KXCPI/UNEMP/GDP/MORT` | Normal around latest reading | FRED CSV |

A longshot fires only when, for a market priced within
`[longshot_min_price, longshot_max_price]`,
`fair_prob > market_prob × (1 + longshot_min_edge)` **and** the fee‑adjusted EV
is positive. External data is cached for 30 minutes. Confidence: 0.50–0.60.

### Fees (modeled exactly)

Kalshi's trading fee is charged **on entry only** (settlement is free):

```
fee = ceil(0.07 × contracts × P × (1 − P))      # P = price in dollars (0–1)
```

This peaks at ~1.75¢/contract near 50¢ and shrinks toward the extremes. The bot
applies this exact formula in every EV, arbitrage, and Kelly calculation
(`kalshi_fee()` in `strategy.py`).

### Risk Management (`risk.py`)

Position sizing per signal is the **minimum** of:

- **Fractional Kelly** — fee‑adjusted net odds × `kelly_scale`. If Kelly ≤ 0
  (no real edge), the bot bets **nothing** (no forced minimum). `ARBS` skips
  Kelly (it is risk‑free) and sizes purely by the caps below.
- **Per‑position cap** — `max_position_pct` of balance.
- **Exposure headroom** — keeps total deployed capital under
  `max_exposure_pct` of balance.
- **Per‑trade cap** — `max_contracts_per_trade` (fixed at 50 in code).

Other guards:

| Guard | Behavior |
|---|---|
| `min_edge` | Skip signals with edge below this |
| `min_confidence` | Skip signals below this confidence |
| `max_daily_loss_pct` | Halt *all* new trades for the day if balance drops this far from the day's opening balance |
| Duplicate guard | Never open a second position in a ticker already held |
| Liquidity filter | Skip markets with no `yes_ask` or `volume_24h < 1` |
| Close‑time filter | Skip markets closing in under 5 minutes |

### Exit Logic (non‑arb positions)

Exits are **risk‑managed to cut losers fast and let winners run** (the opposite
of the old fixed tiny‑target / 60%‑stop logic, which lost far more on a single
loser than it made on many winners). Checked every `monitor_interval_secs`, a
non‑arb position is closed when **any** trigger hits:

1. **Hard stop** — best bid ≤ `entry − exit_max_loss_cents`. Caps the loss per
   contract (default 8¢) instead of riding it down 60%.
2. **Trailing stop** — once the bid has risen at least `exit_trail_activate_cents`
   above entry, the position exits if the bid falls `exit_trail_give_back_cents`
   from its peak. This lets a winner keep climbing rather than being dumped at a
   few cents of profit.
3. **Take‑profit** — best bid ≥ 97¢ (don't hold a near‑resolved winner for a
   last penny of risk).
4. **Time limit** — held longer than `max_hold_hours`.

With the defaults a 45¢ entry caps its loss at ~37¢ (−8¢) but a winner that runs
to 60¢ trails out near 57¢ (+12¢) — so wins can exceed losses. The exit order is
placed **before** the position is removed from tracking, so a failed exit leaves
the position in place to retry rather than orphaning it. `ARBS` positions are
exempt — both legs are held to settlement.

### Persistence & Reconciliation

- Open positions are written to `open_positions.json` after every change and
  reloaded on startup.
- Each cycle the bot **reconciles** its in‑memory positions against Kalshi's
  actual portfolio: positions closed on Kalshi are dropped; unknown positions on
  Kalshi are **adopted** (strategy `ORPHAN`) so they still get monitored.

---

## Configuration Reference (`config.json`)

| Key | Default | Description |
|---|---|---|
| `demo` | `true`¹ | Use Kalshi's demo (paper) environment |
| `dry_run` | `false` | Simulate locally; submit no orders |
| `scan_only` | `false` | Print signals only (implies `dry_run`) |
| `scan_interval_secs` | `30` | Seconds between full market scans |
| `monitor_interval_secs` | `15` | Seconds between position‑exit checks |
| `scan_all_markets` | `false` | `false` = scan only `target_series` (fast); `true` = paginate **every** open market (~600k, slow) |
| `max_markets_per_scan` | `0` | Cap on markets analyzed per cycle (`0` = no cap) |
| `max_position_pct` | `0.10` | Max fraction of balance per position |
| `max_exposure_pct` | `0.60` | Max fraction of balance deployed at once |
| `kelly_scale` | `0.50` | Kelly fraction for non‑arb sizing |
| `min_edge` | `0.02` | Minimum edge to trade |
| `min_confidence` | `0.30` | Minimum confidence to trade |
| `max_daily_loss_pct` | `0.15` | Daily‑loss halt threshold |
| `value_enabled` | `true` | Run the primary historical‑value engine |
| `value_min_obs` | `4` | History points per market before `VALUE` can trade |
| `value_ema_alpha` | `0.35` | EMA smoothing for the fair‑value anchor |
| `value_sharpen` | `1.10` | Favorite–longshot correction (`1.0` = off) |
| `value_min_gap_cents` | `3` | Live ask must be this far below fair value |
| `value_min_edge` | `0.03` | Min fee‑adjusted edge for a `VALUE` trade |
| `value_min_price` / `value_max_price` | `5` / `95` | Price band `VALUE` trades within (¢) |
| `value_confidence` | `0.50` | Base confidence for `VALUE` signals |
| `history_path` | `price_history.json` | Where price history is persisted |
| `history_max_obs` | `288` | Max history points kept per ticker |
| `history_min_interval_secs` | `300` | Down‑sample interval for history points |
| `strategies` | `["ARBS"]` | Analytic strategies: `ARBS`, `FAVORITE`, `MOMENTUM`, `SPREAD`, `DRIFT` |
| `event_arb_enabled` | `true` | Run the `EVENT_ARB` detector (alerts only) |
| `event_arb_min_cents` | `2` | Minimum guaranteed profit (¢) to alert on |
| `target_series` | see file | Series scanned by default (when `scan_all_markets` is false) |
| `exit_max_loss_cents` | `8` | Hard stop: max loss per contract (¢) |
| `exit_trail_activate_cents` | `3` | Start trailing once up this many ¢ |
| `exit_trail_give_back_cents` | `3` | Exit if bid falls this far from its peak |
| `max_hold_hours` | `24` | Force‑exit non‑arb positions after this |
| `longshot_enabled` | `true` | Run the longshot engine |
| `longshot_min_edge` | `0.15` | Required edge over market price |
| `longshot_max_price` | `15` | Only consider markets priced ≤ this (¢) |
| `longshot_min_price` | `1` | Skip markets priced below this (¢) |
| `longshot_series` | see file | Extra series scanned for longshots |

¹ `config.json` in this repo ships with `demo: false`. The in‑code default
(used if a key is absent) is `demo: true`. CLI flags always win.

> Note: `profit_target_mult`, `max_single_loss_pct`, and `min_orderbook_volume`
> exist in the config/code but are not currently wired into the trading loop.
> Exits are governed by the hard/trailing stop above, not `target_exit_cents`.

---

## Files

| File | Purpose |
|---|---|
| `bot.py` | Main loop, orchestration, order execution, CLI |
| `kalshi_client.py` | REST client with RSA‑PSS auth |
| `valuation.py` | **Primary** historical‑value engine + price‑history tracker |
| `strategy.py` | `ARBS`, `EVENT_ARB`, `FAVORITE`, `MOMENTUM`, `SPREAD`, `DRIFT` + fee/EV/Kelly math |
| `longshot.py` | NBA / weather / economic model signals |
| `risk.py` | Position sizing, exposure limits, drawdown halt |
| `config.json` | Bot settings |
| `credentials.json` | `key_id` + private‑key path *(gitignored)* |
| `kalshi_private.pem` | RSA private key *(gitignored)* |
| `requirements.txt` | Python dependencies |

### Output files (gitignored)

| File | Contents |
|---|---|
| `kalshi_bot.log` | Full structured log of all activity |
| `trade_log.jsonl` | One JSON record per OPEN/CLOSE |
| `arb_alerts.jsonl` | One record per detected `EVENT_ARB` opportunity |
| `price_history.json` | Rolling per‑market price history for the `VALUE` engine |
| `open_positions.json` | Live snapshot of tracked positions |

### Analyzing trades

```python
import json, pandas as pd

trades = [json.loads(l) for l in open("trade_log.jsonl")]
df = pd.DataFrame(trades)
closes = df[df.action == "CLOSE"]
print(f"Total realized P&L: ${closes.pnl_cents.sum()/100:.2f}")
print(closes.groupby("strategy")["pnl_cents"].sum() / 100)
```

---

## Safety

- **Start on `--demo`.** Paper trade until you understand the bot's behavior.
- **Keep `kalshi_private.pem` out of git.** It's gitignored; the real private
  key lives only on your machine. (The `key_id` in `credentials.json` is just a
  public identifier and is harmless on its own — it can't sign requests without
  the private key.)
- **Arbitrage carries execution risk.** The bot places two legs and unwinds on a
  partial fill, but in fast markets a leg can still slip. Keep `max_position_pct`
  modest and watch the logs for `CRITICAL` unwind failures.
- **Edges decay.** Real arbs are rare and fleeting; the longshot models are
  approximations. Monitor `trade_log.jsonl` and disable anything that bleeds.
- **Rate limits.** The bot sleeps 200ms between series fetches. If you hit
  limits, raise `scan_interval_secs`.

---

## Architecture

```
bot.py            ← main loop, order execution, reconciliation, CLI
kalshi_client.py  ← API client (RSA-PSS auth, REST calls)
valuation.py      ← PRIMARY: historical-value / mean-reversion engine + history
strategy.py       ← ARBS / EVENT_ARB / FAVORITE / MOMENTUM / SPREAD / DRIFT + math
longshot.py       ← model-based signals (NBA, weather, economic)
risk.py           ← position sizing, exposure limits, drawdown halt
config.json       ← user configuration
```
