"""
Longshot Strategy — finds Kalshi markets where the true probability
is significantly higher than the market price implies.

Data sources (all free, no API key required):
  NBA props  → stats.nba.com  (season averages + Poisson model)
  Weather    → api.weather.gov (NWS point forecast)
  Economic   → api.stlouisfed.org FRED (CPI, unemployment latest values)

Ticker parsing:
  KXNBAPTS-26MAR11CHASAC-SACDDEROZAN10-20  → SAC player DeRozan, 20+ pts
  KXNBAREB-26MAR11DETBKN-DETCCUNNINGHAM2-8 → DET player Cunningham, 8+ reb
  KXNBAAST-26MAR11DETBKN-DETCCUNNINGHAM2-5 → DET player Cunningham, 5+ ast
  KXCPI-26MAR-T2.5                          → CPI above 2.5%
  HIGHNY-25MAR10-B72                        → NY high temp above 72°F

Signal fires when:
  fair_prob > market_price_as_prob * (1 + min_longshot_edge)
  AND market price < longshot_max_price (only look at underdog prices)
"""

import logging
import math
import re
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

import requests

from strategy import Signal, fee_adjusted_ev

log = logging.getLogger("kalshi.longshot")

# ── Config (overridden by bot config) ────────────────────────────────────────
MIN_LONGSHOT_EDGE = 0.15      # require 15% edge over market price
MAX_LONGSHOT_PRICE = 45       # only look at markets priced 45¢ or below
MIN_LONGSHOT_PRICE = 3        # skip markets priced below 3¢ (too thin)
LONGSHOT_CONFIDENCE = 0.55    # base confidence for stat-backed signals

# ── HTTP helper ───────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; KalshiBot/1.0)",
    "Accept": "application/json",
})

_cache: Dict[str, Tuple[float, object]] = {}   # key → (timestamp, data)
CACHE_TTL = 1800   # 30 min cache for external data


def _fetch(url: str, headers: dict = None, ttl: int = CACHE_TTL) -> Optional[dict]:
    now = time.time()
    if url in _cache:
        ts, data = _cache[url]
        if now - ts < ttl:
            return data
    try:
        h = dict(SESSION.headers)
        if headers:
            h.update(headers)
        r = SESSION.get(url, headers=h, timeout=10)
        r.raise_for_status()
        data = r.json()
        _cache[url] = (now, data)
        return data
    except Exception as e:
        log.debug(f"Fetch failed {url}: {e} — status: {getattr(e, 'response', None) and e.response.status_code}")
        return None


# ── Poisson probability model ─────────────────────────────────────────────────

def poisson_over_prob(mean: float, threshold: float) -> float:
    """P(X >= threshold) where X ~ Poisson(mean). Used for pts/reb/ast."""
    if mean <= 0:
        return 0.0
    # P(X >= k) = 1 - P(X <= k-1)
    k = int(threshold)
    cumulative = 0.0
    for i in range(k):
        cumulative += (math.exp(-mean) * mean**i) / math.factorial(i)
    return max(0.0, min(1.0, 1.0 - cumulative))


def normal_over_prob(mean: float, std: float, threshold: float) -> float:
    """P(X >= threshold) where X ~ Normal(mean, std). Used for temps."""
    if std <= 0:
        return 0.5
    z = (threshold - mean) / std
    # Approximation of 1 - CDF(z) using error function
    return max(0.0, min(1.0, 0.5 * (1 - _erf(z / math.sqrt(2)))))


def _erf(x: float) -> float:
    """Abramowitz & Stegun approximation of erf(x)."""
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t)
                  + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * math.exp(-x*x)
    return y if x >= 0 else -y


# ── NBA data ──────────────────────────────────────────────────────────────────

# ── NBA data ──────────────────────────────────────────────────────────────────

# ── NBA data ──────────────────────────────────────────────────────────────────

NBA_STATS_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"
    "/statistics/byathlete?region=us&lang=en&contentorigin=espn"
    "&isqualified=true&page=1&limit=500&category=offensive&season=2025&seasontype=2"
)

_nba_players: Dict[str, dict] = {}   # last_name_upper → stats row
_nba_loaded = False


def _load_nba_stats():
    global _nba_players, _nba_loaded
    if _nba_loaded:
        return
    data = _fetch(NBA_STATS_URL, ttl=3600)
    if not data:
        log.warning("Could not load NBA stats from ESPN")
        return
    try:
        athletes = data.get("athletes", [])
        for entry in athletes:
            athlete = entry.get("athlete", {})
            name = athlete.get("displayName", "")
            if not name:
                continue
            last = name.split()[-1].upper()
            cats = {c["name"]: c.get("value", 0)
                    for c in entry.get("categories", [])}

            gp = cats.get("gamesPlayed", 0) or cats.get("games", 0) or 1

            # ESPN may return totals or averages depending on endpoint
            # Prefer avg fields; fall back to dividing totals by gp
            pts = cats.get("avgPoints") or (cats.get("points", 0) / gp)
            reb = cats.get("avgRebounds") or (cats.get("rebounds", 0) / gp)
            ast = cats.get("avgAssists") or (cats.get("assists", 0) / gp)
            blk = cats.get("avgBlocks") or (cats.get("blocks", 0) / gp)
            stl = cats.get("avgSteals") or (cats.get("steals", 0) / gp)

            # Sanity check — skip if values look like totals not averages
            if pts > 60 or reb > 30 or ast > 20:
                pts, reb, ast, blk, stl = pts/gp, reb/gp, ast/gp, blk/gp, stl/gp

            _nba_players[last] = {
                "name": name,
                "pts": round(pts, 1),
                "reb": round(reb, 1),
                "ast": round(ast, 1),
                "blk": round(blk, 1),
                "stl": round(stl, 1),
                "gp":  gp,
            }
        _nba_loaded = True
        log.info(f"NBA stats loaded from ESPN: {len(_nba_players)} players")
    except Exception as e:
        log.warning(f"NBA stats parse error: {e}")
        # Fallback: try parsing alternate ESPN response shape
        try:
            _load_nba_stats_fallback(data)
        except Exception:
            pass


def _load_nba_stats_fallback(data: dict):
    """Handle alternate ESPN response shape with leaders/categories structure."""
    global _nba_players, _nba_loaded
    rows = []
    for group in data.get("categories", []):
        for leader in group.get("leaders", []):
            athlete = leader.get("athlete", {})
            name = athlete.get("displayName", "")
            if not name:
                continue
            last = name.split()[-1].upper()
            if last not in _nba_players:
                _nba_players[last] = {
                    "name": name, "pts": 0, "reb": 0,
                    "ast": 0, "blk": 0, "stl": 0, "gp": 10
                }
            stat_name = group.get("name", "")
            val = leader.get("value", 0)
            gp = max(_nba_players[last]["gp"], 1)
            # If value looks like a season total, divide by GP
            if "point" in stat_name.lower():
                _nba_players[last]["pts"] = round(val / gp if val > 60 else val, 1)
            elif "rebound" in stat_name.lower():
                _nba_players[last]["reb"] = round(val / gp if val > 30 else val, 1)
            elif "assist" in stat_name.lower():
                _nba_players[last]["ast"] = round(val / gp if val > 20 else val, 1)
            elif "block" in stat_name.lower():
                _nba_players[last]["blk"] = round(val / gp if val > 10 else val, 1)
            elif "steal" in stat_name.lower():
                _nba_players[last]["stl"] = round(val / gp if val > 10 else val, 1)
    if _nba_players:
        _nba_loaded = True
        log.info(f"NBA stats loaded via fallback: {len(_nba_players)} players")


# Ticker pattern: KXNBAPTS-26MAR11CHASAC-SACDDEROZAN10-20
# Group 1: stat type suffix (PTS/REB/AST/BLK/STL)
# Group 2: player token e.g. SACDDEROZAN10
# Group 3: threshold e.g. 20
_NBA_RE = re.compile(
    r"KXNBA(PTS|REB|AST|BLK|STL)-\d+\w+-\w{3}[A-Z](\w+?\d+)-(\d+(?:\.\d+)?)$",
    re.IGNORECASE
)


def _parse_player_name(token: str) -> str:
    """
    Extract a name fragment from tokens like 'DDEROZAN10', 'CCUNNINGHAM2'.
    Strip leading single uppercase letter (team abbrev remnant) and trailing digits.
    """
    token = re.sub(r"^\d+", "", token)           # remove leading digits
    token = re.sub(r"\d+$", "", token)            # remove trailing digits
    # Remove one leading uppercase letter if it looks like an initial
    token = re.sub(r"^[A-Z](?=[A-Z])", "", token)
    return token.upper()


def check_nba_longshot(market: dict, cfg: dict) -> Optional[Signal]:
    ticker = market.get("ticker", "")
    yes_ask = market.get("yes_ask", 0)
    volume = market.get("volume_24h", 0)

    max_price = cfg.get("longshot_max_price", MAX_LONGSHOT_PRICE)
    min_price = cfg.get("longshot_min_price", MIN_LONGSHOT_PRICE)
    min_edge  = cfg.get("longshot_min_edge", MIN_LONGSHOT_EDGE)

    if not (min_price <= yes_ask <= max_price) or volume < 1:
        return None

    m = _NBA_RE.search(ticker)
    if not m:
        return None

    stat_type = m.group(1).upper()   # PTS / REB / AST / BLK / STL
    player_token = m.group(2)
    threshold = float(m.group(3))

    name_frag = _parse_player_name(player_token)

    _load_nba_stats()
    if not _nba_players:
        return None

    # Fuzzy match: find player whose last name contains name_frag
    stat_row = None
    for last, row in _nba_players.items():
        if name_frag in last or last in name_frag:
            stat_row = row
            break

    if not stat_row or stat_row["gp"] < 5:
        return None

    stat_map = {"PTS": "pts", "REB": "reb", "AST": "ast",
                "BLK": "blk", "STL": "stl"}
    stat_key = stat_map.get(stat_type)
    if not stat_key:
        return None

    mean = stat_row[stat_key]
    if mean <= 0:
        return None

    # Poisson model — works well for discrete counting stats
    fair_prob = poisson_over_prob(mean, threshold)

    market_prob = yes_ask / 100.0
    if fair_prob <= market_prob * (1 + min_edge):
        return None   # not enough edge

    ev = fee_adjusted_ev(fair_prob, yes_ask)
    edge = ev / yes_ask

    if edge <= 0:
        return None

    log.info(f"  NBA LONGSHOT: {stat_row['name']} {stat_type}>={threshold} "
             f"mean={mean:.1f} fair={fair_prob*100:.1f}¢ market={yes_ask}¢ edge={edge*100:.1f}%")

    return Signal(
        ticker=ticker,
        side="yes",
        strategy="LONGSHOT_NBA",
        fair_value=fair_prob,
        market_price=float(yes_ask),
        edge=edge,
        confidence=LONGSHOT_CONFIDENCE,
        target_exit_cents=min(yes_ask + (fair_prob * 100 - yes_ask) * 0.6, 95.0),
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"player": stat_row["name"], "stat": stat_type,
               "mean": mean, "threshold": threshold, "fair_prob": fair_prob},
    )


# ── Weather data ──────────────────────────────────────────────────────────────

# NWS grid points for major cities (office, gridX, gridY)
CITY_GRIDS = {
    "NY":  ("OKX", 33, 37),
    "LA":  ("LOX", 149, 48),
    "CHI": ("LOT", 74, 74),
    "MIA": ("MFL", 110, 43),
    "DAL": ("FWD", 87, 58),
    "ATL": ("FFC", 51, 81),
    "BOS": ("BOX", 71, 90),
    "SEA": ("SEW", 124, 68),
    "DEN": ("BOU", 52, 62),
    "PHX": ("PSR", 158, 53),
}

# Ticker pattern: HIGHNY-25MAR10-B72  or  HIGHCHI-26MAR11-B55
_WEATHER_RE = re.compile(
    r"HIGH([A-Z]{2,3})-\d+\w+-[BT](\d+(?:\.\d+)?)$", re.IGNORECASE
)


def _get_nws_forecast(office: str, gx: int, gy: int) -> Optional[dict]:
    url = f"https://api.weather.gov/gridpoints/{office}/{gx},{gy}/forecast"
    return _fetch(url, ttl=1800)


def check_weather_longshot(market: dict, cfg: dict) -> Optional[Signal]:
    ticker = market.get("ticker", "")
    yes_ask = market.get("yes_ask", 0)

    max_price = cfg.get("longshot_max_price", MAX_LONGSHOT_PRICE)
    min_price = cfg.get("longshot_min_price", MIN_LONGSHOT_PRICE)
    min_edge  = cfg.get("longshot_min_edge", MIN_LONGSHOT_EDGE)

    if not (min_price <= yes_ask <= max_price):
        return None

    m = _WEATHER_RE.search(ticker)
    if not m:
        return None

    city_code = m.group(1).upper()
    threshold_f = float(m.group(2))

    grid = CITY_GRIDS.get(city_code)
    if not grid:
        return None

    forecast = _get_nws_forecast(*grid)
    if not forecast:
        return None

    try:
        periods = forecast["properties"]["periods"]
        # Get daytime high for the next relevant period
        highs = [p["temperature"] for p in periods[:4]
                 if p.get("isDaytime") and p.get("temperature")]
        if not highs:
            return None
        forecast_high = highs[0]
    except Exception:
        return None

    # Use a ±8°F standard deviation (typical forecast uncertainty)
    fair_prob = normal_over_prob(forecast_high, 8.0, threshold_f)
    market_prob = yes_ask / 100.0

    if fair_prob <= market_prob * (1 + min_edge):
        return None

    ev = fee_adjusted_ev(fair_prob, yes_ask)
    edge = ev / yes_ask

    if edge <= 0:
        return None

    log.info(f"  WEATHER LONGSHOT: {city_code} high>={threshold_f}°F "
             f"NWS={forecast_high}°F fair={fair_prob*100:.1f}¢ market={yes_ask}¢")

    return Signal(
        ticker=ticker,
        side="yes",
        strategy="LONGSHOT_WEATHER",
        fair_value=fair_prob,
        market_price=float(yes_ask),
        edge=edge,
        confidence=0.60,   # weather forecasts are reliable short-term
        target_exit_cents=min(yes_ask + (fair_prob * 100 - yes_ask) * 0.7, 95.0),
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"city": city_code, "threshold": threshold_f,
               "nws_forecast": forecast_high, "fair_prob": fair_prob},
    )


# ── Economic data (FRED) ──────────────────────────────────────────────────────

# FRED series for economic indicators (no API key for recent releases)
FRED_SERIES = {
    "CPI":   "CPIAUCSL",    # CPI All Urban
    "UNEMP": "UNRATE",      # Unemployment rate
    "GDP":   "A191RL1Q225SBEA",  # Real GDP growth
    "MORT":  "MORTGAGE30US",  # 30-yr mortgage rate
}

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

_econ_cache: Dict[str, float] = {}


def _get_fred_latest(series_id: str) -> Optional[float]:
    if series_id in _econ_cache:
        return _econ_cache[series_id]
    try:
        url = FRED_BASE + series_id
        r = SESSION.get(url, timeout=10)
        r.raise_for_status()
        lines = [l for l in r.text.strip().split("\n") if l and not l.startswith("DATE")]
        if not lines:
            return None

        # For CPI we need YoY % change, so fetch last 13 months
        if series_id == "CPIAUCSL":
            if len(lines) >= 13:
                current = float(lines[-1].split(",")[1])
                year_ago = float(lines[-13].split(",")[1])
                val = round((current - year_ago) / year_ago * 100, 2)
            else:
                return None
        else:
            val = float(lines[-1].split(",")[1])

        _econ_cache[series_id] = val
        log.debug(f"FRED {series_id} = {val}")
        return val
    except Exception as e:
        log.debug(f"FRED fetch failed {series_id}: {e}")
        return None


# Ticker patterns for economic markets
# KXCPI-26MAR-T2.5  → CPI threshold 2.5
# KXUNEMP-26MAR-T4.2 → unemployment threshold 4.2
_ECON_RE = re.compile(
    r"KX(CPI|UNEMP|GDP|MORT)-\d+\w+-[TBt](\d+(?:\.\d+)?)$", re.IGNORECASE
)


def check_econ_longshot(market: dict, cfg: dict) -> Optional[Signal]:
    ticker = market.get("ticker", "")
    yes_ask = market.get("yes_ask", 0)

    max_price = cfg.get("longshot_max_price", MAX_LONGSHOT_PRICE)
    min_price = cfg.get("longshot_min_price", MIN_LONGSHOT_PRICE)
    min_edge  = cfg.get("longshot_min_edge", MIN_LONGSHOT_EDGE)

    if not (min_price <= yes_ask <= max_price):
        return None

    m = _ECON_RE.search(ticker)
    if not m:
        return None

    indicator = m.group(1).upper()
    threshold = float(m.group(2))

    series_id = FRED_SERIES.get(indicator)
    if not series_id:
        return None

    latest = _get_fred_latest(series_id)
    if latest is None:
        return None

    # Only look for longshots where the threshold is above current value
    # If threshold < latest it's near-certain, not a longshot
    if threshold <= latest:
        return None

    # Use a normal distribution around latest value
    # Std devs based on historical month-to-month volatility
    std_map = {"CPI": 0.3, "UNEMP": 0.2, "GDP": 0.8, "MORT": 0.15}
    std = std_map.get(indicator, 0.3)

    fair_prob = normal_over_prob(latest, std, threshold)
    market_prob = yes_ask / 100.0

    if fair_prob <= market_prob * (1 + min_edge):
        return None

    ev = fee_adjusted_ev(fair_prob, yes_ask)
    edge = ev / yes_ask

    if edge <= 0:
        return None

    log.info(f"  ECON LONGSHOT: {indicator}>={threshold} "
             f"FRED={latest} fair={fair_prob*100:.1f}¢ market={yes_ask}¢")

    return Signal(
        ticker=ticker,
        side="yes",
        strategy="LONGSHOT_ECON",
        fair_value=fair_prob,
        market_price=float(yes_ask),
        edge=edge,
        confidence=0.50,
        target_exit_cents=min(yes_ask + (fair_prob * 100 - yes_ask) * 0.6, 95.0),
        market_title=market.get("title", ""),
        event_ticker=market.get("event_ticker", ""),
        extra={"indicator": indicator, "threshold": threshold,
               "fred_latest": latest, "fair_prob": fair_prob},
    )


def check_longshot(market: dict, cfg: dict) -> Optional[Signal]:
    """
    Run all longshot sub-strategies on a market.
    Returns the best signal or None.
    """
    ticker = market.get("ticker", "")

    signals = []

    # NBA player props
    if any(s in ticker for s in ("KXNBAPTS", "KXNBAREB", "KXNBAAST",
                                  "KXNBABLK", "KXNBASTL")):
        s = check_nba_longshot(market, cfg)
        if s:
            signals.append(s)

    # Weather markets
    if ticker.upper().startswith("HIGH"):
        s = check_weather_longshot(market, cfg)
        if s:
            signals.append(s)

    # Economic markets
    if any(s in ticker for s in ("KXCPI", "KXUNEMP", "KXGDP", "KXMORT")):
        s = check_econ_longshot(market, cfg)
        if s:
            signals.append(s)

    if not signals:
        return None
    return max(signals, key=lambda s: s.edge * s.confidence)