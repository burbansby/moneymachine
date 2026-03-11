# Kalshi Edge Bot

Automated trading bot for Kalshi prediction markets. Continuously scans open markets for statistically mispriced contracts, enters positions using fractional Kelly criterion sizing, and exits when the price reverts toward fair value.

---

## Quick Start

### 1. Install dependencies

```bash
pip install requests cryptography python-dotenv tabulate colorama
```

### 2. Generate your RSA key pair

Kalshi uses RSA-PSS authentication. Generate a key and register the public half in your Kalshi account settings.

```bash
# Generate private key (keep this secret)
openssl genrsa -out kalshi_private.pem 2048

# Extract public key to upload to Kalshi
openssl rsa -in kalshi_private.pem -pubout -out kalshi_public.pem

# Upload kalshi_public.pem at:
# https://kalshi.com/account/api-keys  (or demo.kalshi.com for demo)
```

### 3. Configure credentials

Copy your API Key ID from the Kalshi dashboard and paste it into `config.json`:

```json
{
  "key_id": "YOUR_KEY_ID_HERE",
  "private_key_path": "kalshi_private.pem",
  "demo": true
}
```

Or use environment variables:
```bash
export KALSHI_KEY_ID="your_key_id"
export KALSHI_PRIVATE_KEY_PATH="kalshi_private.pem"
```

### 4. Run the bot

```bash
# Safe start: paper trade on Kalshi's demo environment (default)
python bot.py --demo

# Scan only — print signals, never trade
python bot.py --scan-only

# Dry run — simulate everything except order submission
python bot.py --dry-run --demo

# Live trading (real money — start with demo first!)
python bot.py
```

---

## How It Works

### Strategy Engine (`strategy.py`)

Three independent alpha signals:

#### 1. `ARBS` — Intra-market Arbitrage (highest confidence)
Kalshi contracts pay $1 if the event resolves in your favor. YES + NO must always sum to exactly $1 at settlement. If `YES_ask + NO_ask < $0.97` (after fees), you can buy both sides for guaranteed profit regardless of outcome.

- **Edge**: Risk-free by construction
- **Confidence**: 95%
- **Filter**: Requires ≥3¢ net profit after Kalshi's 7% fee on winnings

#### 2. `MMEAN` — Order Book Skew / Mean Reversion
When significantly more contracts are queued on the bid side than the ask side, institutional demand is unmet. The price tends to drift toward the skewed side. Buy before the move.

- **Edge**: Statistically positive but not guaranteed
- **Confidence**: 45%
- **Filter**: Requires ≥8% imbalance and ≥4% expected return

#### 3. `STALE` — Stale Quote Detection
When recent trades are systematically above or below the current order book mid, the book hasn't caught up to new information. Enter before the book reprices.

- **Edge**: Moderate — relies on quote update lag
- **Confidence**: 40%
- **Filter**: Requires ≥6¢ gap between trade VWAP and book mid

### Risk Management (`risk.py`)

| Parameter | Default | Description |
|---|---|---|
| `max_position_pct` | 5% | Max % of balance per single position |
| `max_exposure_pct` | 40% | Max total % of balance deployed |
| `kelly_scale` | 0.25 | Quarter-Kelly for conservative sizing |
| `min_edge` | 4% | Skip signals below this edge threshold |
| `min_confidence` | 35% | Skip low-confidence signals |
| `max_daily_loss_pct` | 10% | Halt all trading if daily loss hits this |
| `max_hold_hours` | 48 | Force-exit any position held longer than this |

### Exit Logic

Positions are closed when **any** of these trigger:
1. **Profit target**: Current best bid ≥ `target_price` (50% of edge captured)
2. **Stop loss**: Current best bid ≤ `entry_price × 0.4` (60% loss)
3. **Time limit**: Position held longer than `max_hold_hours`

---

## Configuration Reference

| Key | Default | Description |
|---|---|---|
| `key_id` | — | Kalshi API Key ID |
| `private_key_path` | `kalshi_private.pem` | Path to RSA private key |
| `demo` | `true` | Use demo environment |
| `dry_run` | `false` | Simulate without placing orders |
| `scan_interval_secs` | `30` | Seconds between full market scans |
| `monitor_interval_secs` | `10` | Seconds between position checks |
| `max_markets_per_scan` | `150` | Max markets to analyze per cycle |
| `strategies` | `["ARBS","MMEAN","STALE"]` | Which strategies to run |

---

## Output Files

| File | Contents |
|---|---|
| `kalshi_bot.log` | Full structured log of all activity |
| `trade_log.jsonl` | One JSON record per trade (open/close) |

### Analyzing trades

```python
import json, pandas as pd

trades = [json.loads(l) for l in open("trade_log.jsonl")]
df = pd.DataFrame(trades)
closes = df[df.action == "CLOSE"]
print(f"Total P&L: ${closes.pnl_cents.sum()/100:.2f}")
print(closes.groupby("strategy")["pnl_cents"].sum() / 100)
```

---

## Important Notes

- **Always start on demo.** Paper trade until you understand the bot's behavior.
- **Kalshi charges ~7% of profits** as fees. The bot accounts for this in all EV calculations.
- **Prediction markets are illiquid.** Thin order books can cause slippage. The bot skips markets with <5 contracts available.
- **Past edge ≠ future edge.** Market inefficiencies get arbitraged away. Monitor signal quality over time.
- **Rate limits**: Kalshi enforces API rate limits. The bot sleeps 100ms between market fetches. If you hit limits, increase `scan_interval_secs`.
- This bot is educational software. Trade at your own risk.

---

## Architecture

```
bot.py           ← Main loop, orchestration, CLI
kalshi_client.py ← API client (auth, REST calls)
strategy.py      ← Signal generation (ARBS, MMEAN, STALE)
risk.py          ← Position sizing, exposure limits, drawdown halt
config.json      ← User configuration
```
