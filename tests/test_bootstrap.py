"""Tests for Yield Curve representation and Multi-Instrument Bootstrapping

validating exact par repricing and log-linear discount factor interpolation.
"""

import math
import pytest
from src.market.curve import YieldCurve
from src.market.bootstrap import (
    DepositQuote,
    SwapQuote,
    YieldCurveBootstrapper,
)


class TestYieldCurve:
    """Test standalone YieldCurve discount factor and rate queries."""

    @pytest.fixture
    def curve(self):
        pillars = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        # Rates roughly 4.0% to 4.5%
        dfs = [
            math.exp(-0.040 * 0.25),
            math.exp(-0.041 * 0.50),
            math.exp(-0.042 * 1.00),
            math.exp(-0.043 * 2.00),
            math.exp(-0.044 * 5.00),
            math.exp(-0.045 * 10.0),
        ]
        return YieldCurve(pillar_times=pillars, discount_factors=dfs, curve_name="TEST_CURVE")

    def test_discount_factor_at_pillars(self, curve):
        """Discount factors at exact pillar dates must match input."""
        assert math.isclose(curve.discount_factor(0.0), 1.0, abs_tol=1e-12)
        assert math.isclose(curve.discount_factor(1.0), math.exp(-0.042 * 1.0), abs_tol=1e-12)
        assert math.isclose(curve.discount_factor(5.0), math.exp(-0.044 * 5.0), abs_tol=1e-12)

    def test_log_linear_interpolation(self, curve):
        """Discount factors between pillars must follow exact log-linear decay."""
        # Test at t = 1.5 (between t=1.0 and t=2.0)
        df_1 = curve.discount_factor(1.0)
        df_2 = curve.discount_factor(2.0)
        df_1_5 = curve.discount_factor(1.5)
        
        # In log-linear interpolation: ln(DF(1.5)) = 0.5 * ln(DF(1.0)) + 0.5 * ln(DF(2.0))
        expected_df = math.exp(0.5 * math.log(df_1) + 0.5 * math.log(df_2))
        assert math.isclose(df_1_5, expected_df, abs_tol=1e-12)

    def test_forward_rate_arbitrage_free(self, curve):
        """Forward rate F(t1, t2) = (DF(t1)/DF(t2) - 1)/(t2 - t1)."""
        t1, t2 = 1.0, 2.0
        df1 = curve.discount_factor(t1)
        df2 = curve.discount_factor(t2)
        expected_fwd = (df1 / df2 - 1.0) / (t2 - t1)
        assert math.isclose(curve.forward_rate(t1, t2), expected_fwd, abs_tol=1e-12)

    def test_parallel_bump(self, curve):
        """100 bps upward bump should decrease discount factor by exp(-0.01 * t)."""
        bumped = curve.bump(100.0)  # +100 bps = +0.01
        for t in [1.0, 3.0, 5.0]:
            expected = curve.discount_factor(t) * math.exp(-0.01 * t)
            assert math.isclose(bumped.discount_factor(t), expected, abs_tol=1e-8)


class TestYieldCurveBootstrapper:
    """Test curve bootstrapping from deposit rates and par interest rate swaps."""

    @pytest.fixture
    def market_quotes(self):
        deposits = [
            DepositQuote(tenor_years=0.25, rate=0.0400),  # 3M @ 4.00%
            DepositQuote(tenor_years=0.50, rate=0.0420),  # 6M @ 4.20%
            DepositQuote(tenor_years=1.00, rate=0.0450),  # 1Y @ 4.50%
        ]
        swaps = [
            SwapQuote(tenor_years=2.0, par_rate=0.0460, fixed_frequency=1.0),
            SwapQuote(tenor_years=3.0, par_rate=0.0475, fixed_frequency=1.0),
            SwapQuote(tenor_years=5.0, par_rate=0.0490, fixed_frequency=1.0),
            SwapQuote(tenor_years=7.0, par_rate=0.0505, fixed_frequency=1.0),
            SwapQuote(tenor_years=10.0, par_rate=0.0520, fixed_frequency=1.0),
        ]
        return deposits, swaps

    def test_bootstrap_exact_reprice(self, market_quotes):
        """Assert all input deposit and swap instruments reprice to par."""
        deposits, swaps = market_quotes
        bootstrapper = YieldCurveBootstrapper()
        curve = bootstrapper.bootstrap(deposits, swaps, curve_name="USD_SOFR_BOOTSTRAP")

        # 1. Check deposit repricing: DF(t) = 1 / (1 + r*t)
        for dep in deposits:
            expected_df = 1.0 / (1.0 + dep.rate * dep.tenor_years)
            actual_df = curve.discount_factor(dep.tenor_years)
            assert math.isclose(actual_df, expected_df, abs_tol=1e-9)

        # 2. Check par swap repricing: Par Rate = (1 - DF(Tn)) / sum(tau_i * DF(Ti))
        for swap in swaps:
            # Generate payment schedule
            n_payments = int(swap.tenor_years * swap.fixed_frequency)
            dt = 1.0 / swap.fixed_frequency
            annuity = sum(dt * curve.discount_factor(i * dt) for i in range(1, n_payments + 1))
            df_tn = curve.discount_factor(swap.tenor_years)
            model_par_rate = (1.0 - df_tn) / annuity
            assert math.isclose(model_par_rate, swap.par_rate, abs_tol=1e-8)

    def test_monotonicity_and_positive_forwards(self, market_quotes):
        """Assert discount factors are monotonically decreasing and forward rates are

        positive.
        """
        deposits, swaps = market_quotes
        bootstrapper = YieldCurveBootstrapper()
        curve = bootstrapper.bootstrap(deposits, swaps)

        times = [0.1 * i for i in range(1, 101)]
        prev_df = 1.0
        for t in times:
            df = curve.discount_factor(t)
            assert df < prev_df, f"Discount factor non-monotonic at t={t}: {df} >= {prev_df}"
            assert df > 0.0, f"Discount factor negative at t={t}: {df}"
            prev_df = df

        for i in range(len(times) - 1):
            fwd = curve.forward_rate(times[i], times[i+1])
            assert fwd > 0.0, f"Negative forward rate between {times[i]} and {times[i+1]}: {fwd}"
