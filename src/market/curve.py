"""Yield Curve implementation with log-linear discount factor interpolation."""

import bisect
import math
from typing import List, Optional, Sequence


class YieldCurve:
    """Yield Curve representation using Log-Linear interpolation on Discount Factors.

    Log-linear discount factor interpolation corresponds to piecewise constant
    instantaneous forward rates, ensuring positive forward rates and arbitrage-free
    pricing.
    """

    def __init__(
        self,
        pillar_times: Sequence[float],
        discount_factors: Sequence[float],
        curve_name: str = "YieldCurve",
    ) -> None:
        if len(pillar_times) != len(discount_factors):
            raise ValueError(
                f"Length mismatch: {len(pillar_times)} pillars vs {len(discount_factors)} discount factors"
            )
        if len(pillar_times) == 0:
            raise ValueError("YieldCurve must have at least one pillar")

        # Sort pillars and ensure t=0 has DF=1.0
        combined = sorted(zip(pillar_times, discount_factors), key=lambda x: x[0])
        self.times: List[float] = []
        self.dfs: List[float] = []

        if combined[0][0] > 1e-12:
            self.times.append(0.0)
            self.dfs.append(1.0)

        for t, df in combined:
            if df <= 0:
                raise ValueError(f"Discount factor must be positive, got DF({t}) = {df}")
            if self.times and math.isclose(t, self.times[-1], abs_tol=1e-12):
                # Replace duplicate
                self.dfs[-1] = df
            else:
                self.times.append(float(t))
                self.dfs.append(float(df))

        self.log_dfs: List[float] = [math.log(df) for df in self.dfs]
        self.curve_name = curve_name

    def discount_factor(self, t: float) -> float:
        """Returns the discount factor P(0, t) for maturity t >= 0.

        Applies log-linear interpolation between pillars.
        """
        if t <= 0.0:
            return 1.0
        if math.isclose(t, 0.0, abs_tol=1e-12):
            return 1.0

        # Exact match or beyond ends
        if t <= self.times[0]:
            # Extrapolate flat short rate
            r0 = -self.log_dfs[1] / self.times[1] if len(self.times) > 1 else 0.0
            return math.exp(-r0 * t)

        if t >= self.times[-1]:
            # Flat forward extrapolation beyond longest pillar
            t_last = self.times[-1]
            t_prev = self.times[-2] if len(self.times) > 1 else 0.0
            fwd_last = -(self.log_dfs[-1] - self.log_dfs[-2]) / (t_last - t_prev) if len(self.times) > 1 else -self.log_dfs[-1] / t_last
            return self.dfs[-1] * math.exp(-fwd_last * (t - t_last))

        # Binary search for bracket [t_i, t_{i+1}]
        idx = bisect.bisect_right(self.times, t) - 1
        t1, t2 = self.times[idx], self.times[idx + 1]
        log_df1, log_df2 = self.log_dfs[idx], self.log_dfs[idx + 1]

        # Log-linear interpolation
        weight = (t - t1) / (t2 - t1)
        log_df_t = log_df1 + weight * (log_df2 - log_df1)
        return math.exp(log_df_t)

    def zero_rate(self, t: float, compounding: str = "continuous") -> float:
        """Returns the zero rate for tenor t.

        Parameters
        ----------
        t : float
            Tenor in years.
        compounding : str
            'continuous', 'annual', or 'semiannual'.
        """
        if t <= 1e-12:
            t = 1e-6
        df = self.discount_factor(t)

        if compounding == "continuous":
            return -math.log(df) / t
        elif compounding == "annual":
            return math.pow(df, -1.0 / t) - 1.0
        elif compounding == "semiannual":
            return 2.0 * (math.pow(df, -1.0 / (2.0 * t)) - 1.0)
        else:
            raise ValueError(f"Unknown compounding convention: {compounding}")

    def forward_rate(self, t1: float, t2: float, compounding: str = "simple") -> float:
        """Returns the forward rate between t1 and t2.

        Parameters
        ----------
        t1, t2 : float
            Start and end tenors in years (t2 > t1).
        compounding : str
            'simple' (money market / Libor / SOFR convention) or 'continuous'.
        """
        if t2 <= t1:
            raise ValueError(f"Forward end time t2 ({t2}) must be strictly greater than t1 ({t1})")

        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        dt = t2 - t1

        if compounding == "simple":
            return (df1 / df2 - 1.0) / dt
        elif compounding == "continuous":
            return (math.log(df1) - math.log(df2)) / dt
        else:
            raise ValueError(f"Unknown compounding convention: {compounding}")

    def bump(self, bump_bps: float) -> "YieldCurve":
        """Returns a new YieldCurve shifted by a parallel bump in basis points."""
        shift = bump_bps / 10000.0
        bumped_dfs = [df * math.exp(-shift * t) for t, df in zip(self.times, self.dfs)]
        return YieldCurve(
            pillar_times=self.times,
            discount_factors=bumped_dfs,
            curve_name=f"{self.curve_name}_bumped_{bump_bps:+.1f}bp",
        )

    def __repr__(self) -> str:
        return f"<YieldCurve '{self.curve_name}' pillars={len(self.times)} max_tenor={self.times[-1]}y>"
