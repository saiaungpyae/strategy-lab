"""
bt_strategies.py — Classic community trading strategies for backtesting.py.

Each strategy is long-only (buy / flat), which is how most retail crypto
community strategies are actually traded. Indicators are computed from past
data only; backtesting.py fills orders at the NEXT bar's open, so there is no
lookahead bias. A Buy & Hold benchmark is included for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover


# ----------------------------------------------------------------------------
# Indicator helpers (operate on array-like, return pandas Series)
# ----------------------------------------------------------------------------
def SMA(arr, n):
    return pd.Series(arr).rolling(int(n)).mean()


def EMA(arr, n):
    return pd.Series(arr).ewm(span=int(n), adjust=False).mean()


def RSI(arr, n=14):
    s = pd.Series(arr)
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder's smoothing
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - 100 / (1 + rs)


def ATR(high, low, close, n=14):
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bollinger_lower(arr, n, k):
    s = pd.Series(arr)
    mid = s.rolling(int(n)).mean()
    std = s.rolling(int(n)).std()
    return mid - k * std


def supertrend_dir(high, low, close, period=10, mult=3.0):
    """Return the Supertrend trend direction: +1 uptrend, -1 downtrend."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    atr = ATR(high, low, close, period).values
    hl2 = ((high + low) / 2).values
    close_v = close.values
    n = len(close_v)

    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr
    final_upper = np.copy(basic_upper)
    final_lower = np.copy(basic_lower)
    direction = np.ones(n)

    for i in range(1, n):
        if basic_upper[i] < final_upper[i - 1] or close_v[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower[i] > final_lower[i - 1] or close_v[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if close_v[i] > final_upper[i - 1]:
            direction[i] = 1
        elif close_v[i] < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    return pd.Series(direction)


# ----------------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------------
class BuyHold(Strategy):
    """Benchmark: buy on the first bar and never sell."""
    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy()


class SmaCross(Strategy):
    """Golden cross: buy when SMA50 crosses above SMA200, exit on death cross."""
    n1, n2 = 50, 200

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()


class EmaCross(Strategy):
    """Faster trend follow: EMA12 crossing EMA26."""
    n1, n2 = 12, 26

    def init(self):
        self.ema1 = self.I(EMA, self.data.Close, self.n1)
        self.ema2 = self.I(EMA, self.data.Close, self.n2)

    def next(self):
        if crossover(self.ema1, self.ema2):
            self.buy()
        elif crossover(self.ema2, self.ema1):
            self.position.close()


class RsiReversion(Strategy):
    """Mean reversion: buy oversold (RSI<30), exit when RSI recovers past 55."""
    n, lower, exit_level = 14, 30, 55

    def init(self):
        self.rsi = self.I(RSI, self.data.Close, self.n)

    def next(self):
        if not self.position and self.rsi[-1] < self.lower:
            self.buy()
        elif self.position and self.rsi[-1] > self.exit_level:
            self.position.close()


class MacdCross(Strategy):
    """MACD line crossing its signal line."""
    fast, slow, signal = 12, 26, 9

    def init(self):
        macd = lambda a: EMA(a, self.fast) - EMA(a, self.slow)
        self.macd = self.I(macd, self.data.Close)
        self.signal_line = self.I(
            lambda a: (EMA(a, self.fast) - EMA(a, self.slow)).ewm(span=self.signal, adjust=False).mean(),
            self.data.Close,
        )

    def next(self):
        if crossover(self.macd, self.signal_line):
            self.buy()
        elif crossover(self.signal_line, self.macd):
            self.position.close()


class BollingerReversion(Strategy):
    """Buy when price closes below the lower Bollinger band, exit at the midline."""
    n, k = 20, 2.0

    def init(self):
        self.mid = self.I(SMA, self.data.Close, self.n)
        self.lower_band = self.I(bollinger_lower, self.data.Close, self.n, self.k)

    def next(self):
        price = self.data.Close[-1]
        if not self.position and price < self.lower_band[-1]:
            self.buy()
        elif self.position and price > self.mid[-1]:
            self.position.close()


class SupertrendStrat(Strategy):
    """Popular crypto trend filter: long while Supertrend is green."""
    period, mult = 10, 3.0

    def init(self):
        self.dir = self.I(
            supertrend_dir, self.data.High, self.data.Low, self.data.Close,
            self.period, self.mult,
        )

    def next(self):
        if self.dir[-1] == 1 and not self.position:
            self.buy()
        elif self.dir[-1] == -1 and self.position:
            self.position.close()


class DonchianBreakout(Strategy):
    """Turtle-style: buy an N-bar high breakout, exit on an N-bar low breakdown."""
    n = 20

    def init(self):
        self.upper = self.I(lambda h, n: pd.Series(h).rolling(int(n)).max(), self.data.High, self.n)
        self.lower_ch = self.I(lambda l, n: pd.Series(l).rolling(int(n)).min(), self.data.Low, self.n)

    def next(self):
        price = self.data.Close[-1]
        # Compare against the channel formed by the PRIOR bars ([-2]) to avoid
        # using the current bar's own extreme in its breakout test.
        if not self.position and price >= self.upper[-2]:
            self.buy()
        elif self.position and price <= self.lower_ch[-2]:
            self.position.close()


# Registry: name -> class. "buyhold" is the benchmark.
STRATEGIES = {
    "buyhold": BuyHold,
    "sma_cross": SmaCross,
    "ema_cross": EmaCross,
    "rsi_reversion": RsiReversion,
    "macd_cross": MacdCross,
    "bollinger": BollingerReversion,
    "supertrend": SupertrendStrat,
    "donchian": DonchianBreakout,
}
