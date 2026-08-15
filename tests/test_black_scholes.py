"""Tests for Black-Scholes pricing engine and analytical Greeks validation against

Hull benchmarks (Options, Futures, and Other Derivatives, 10th Ed).
"""

import math
import pytest
from src.instruments.base import OptionType, PositionSide
from src.instruments.options import EuropeanOption
from src.instruments.forwards import Forward
from src.engines.black_scholes import BlackScholesEngine
from src.market.market_data import MarketData


@pytest.fixture
def hull_market():
    """Standard market data from Hull Chapter 15 Example:

    S0 = 42, r = 10%, sigma = 20%, q = 0.0
    """
    return MarketData(
        as_of_date="2026-08-15",
        spots={"STOCK": 42.0},
        risk_free_rate=0.10,
        dividend_yields={"STOCK": 0.0},
        flat_volatilities={"STOCK": 0.20},
    )


@pytest.fixture
def bs_engine():
    return BlackScholesEngine()


class TestBlackScholesHullBenchmarks:
    """Test against worked benchmark examples from John C. Hull."""

    def test_hull_chapter_15_call_option(self, hull_market, bs_engine):
        """Hull 10th Ed Ch 15 Example: S=42, K=40, r=0.10, sigma=0.20, T=0.5.

        Benchmark Call NPV = 4.7594
        """
        call = EuropeanOption(
            underlying="STOCK",
            strike=40.0,
            expiry=0.5,
            option_type=OptionType.CALL,
        )
        result = bs_engine.price(call, hull_market)
        
        # Verify NPV
        assert math.isclose(result.npv, 4.7594, abs_tol=1e-3)
        assert result.engine_name == "BlackScholesEngine"
        assert result.currency == "USD"
        
        # Verify d1 and d2
        d1 = result.details["d1"]
        d2 = result.details["d2"]
        assert math.isclose(d1, 0.7693, abs_tol=1e-3)
        assert math.isclose(d2, 0.6278, abs_tol=1e-3)

    def test_hull_chapter_15_put_option(self, hull_market, bs_engine):
        """Hull 10th Ed Ch 15 Example: S=42, K=40, r=0.10, sigma=0.20, T=0.5.

        Benchmark Put NPV = 0.8080
        """
        put = EuropeanOption(
            underlying="STOCK",
            strike=40.0,
            expiry=0.5,
            option_type=OptionType.PUT,
        )
        result = bs_engine.price(put, hull_market)
        assert math.isclose(result.npv, 0.8080, abs_tol=1e-3)

    def test_hull_analytical_greeks(self, hull_market, bs_engine):
        """Verify analytical Delta, Gamma, Vega, Theta, Rho."""
        call = EuropeanOption(
            underlying="STOCK",
            strike=40.0,
            expiry=0.5,
            option_type=OptionType.CALL,
        )
        put = EuropeanOption(
            underlying="STOCK",
            strike=40.0,
            expiry=0.5,
            option_type=OptionType.PUT,
        )
        
        call_res = bs_engine.price(call, hull_market)
        put_res = bs_engine.price(put, hull_market)
        
        # Delta: N(d1) for Call, N(d1) - 1 for Put
        assert math.isclose(call_res.greeks["delta"], 0.7791, abs_tol=1e-3)
        assert math.isclose(put_res.greeks["delta"], -0.2209, abs_tol=1e-3)
        
        # Gamma: identical for Call and Put
        assert math.isclose(call_res.greeks["gamma"], 0.0492, abs_tol=1e-3)
        assert math.isclose(put_res.greeks["gamma"], 0.0492, abs_tol=1e-3)
        
        # Vega: S * sqrt(T) * N'(d1) = 8.8134
        assert math.isclose(call_res.greeks["vega"], 8.8134, abs_tol=1e-3)
        assert math.isclose(put_res.greeks["vega"], 8.8134, abs_tol=1e-3)

    def test_dividend_paying_stock_benchmark(self, bs_engine):
        """Hull Ch 17: S=50, K=50, r=0.05, q=0.02, sigma=0.25, T=0.5."""
        div_market = MarketData(
            as_of_date="2026-08-15",
            spots={"DIV_STOCK": 50.0},
            risk_free_rate=0.05,
            dividend_yields={"DIV_STOCK": 0.02},
            flat_volatilities={"DIV_STOCK": 0.25},
        )
        call = EuropeanOption(
            underlying="DIV_STOCK",
            strike=50.0,
            expiry=0.5,
            option_type=OptionType.CALL,
        )
        put = EuropeanOption(
            underlying="DIV_STOCK",
            strike=50.0,
            expiry=0.5,
            option_type=OptionType.PUT,
        )
        
        call_res = bs_engine.price(call, div_market)
        put_res = bs_engine.price(put, div_market)
        
        # Exact theoretical: Call = 3.8415, Put = 3.1044
        assert math.isclose(call_res.npv, 3.8415, abs_tol=1e-3)
        assert math.isclose(put_res.npv, 3.1044, abs_tol=1e-3)

    def test_forward_pricing(self, hull_market, bs_engine):
        """Forward contract pricing: PV = S*e^(-qT) - K*e^(-rT)."""
        forward = Forward(
            underlying="STOCK",
            strike_price=42.0,
            maturity=0.5,
        )
        res = bs_engine.price(forward, hull_market)
        # Expected PV = 42.0 - 42.0 * exp(-0.10 * 0.5) = 42.0 - 42.0 * 0.951229 = 2.04838
        expected_pv = 42.0 - 42.0 * math.exp(-0.10 * 0.5)
        assert math.isclose(res.npv, expected_pv, rel_tol=1e-7)

    def test_short_position_negates_npv(self, hull_market, bs_engine):
        """Short position must have exactly negative NPV."""
        call_long = EuropeanOption(
            underlying="STOCK",
            strike=40.0,
            expiry=0.5,
            option_type=OptionType.CALL,
            position_side=PositionSide.LONG,
        )
        call_short = EuropeanOption(
            underlying="STOCK",
            strike=40.0,
            expiry=0.5,
            option_type=OptionType.CALL,
            position_side=PositionSide.SHORT,
        )
        long_res = bs_engine.price(call_long, hull_market)
        short_res = bs_engine.price(call_short, hull_market)
        assert math.isclose(long_res.npv, -short_res.npv, rel_tol=1e-12)
