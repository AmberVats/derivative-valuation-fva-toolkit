"""Tests for Risk and Sensitivities Calculator.

Validates that Numerical Finite Difference Greeks agree with Analytical Black-Scholes Greeks
and validates Parallel and Tenor DV01 calculations.
"""

import math
import pytest
from src.instruments.base import OptionType
from src.instruments.options import EuropeanOption
from src.instruments.swaps import InterestRateSwap
from src.engines.black_scholes import BlackScholesEngine
from src.engines.dcf import DiscountedCashFlowEngine
from src.market.market_data import MarketData
from src.market.curve import YieldCurve
from src.risk.sensitivities import SensitivitiesCalculator


@pytest.fixture
def risk_market():
    times = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    dfs = [math.exp(-0.045 * t) for t in times]
    curve = YieldCurve(pillar_times=times, discount_factors=dfs, curve_name="USD_SOFR")

    return MarketData(
        as_of_date="2026-08-15",
        spots={"TECH": 150.0},
        risk_free_rate=0.045,
        dividend_yields={"TECH": 0.015},
        flat_volatilities={"TECH": 0.28},
        yield_curves={"DISCOUNT": curve},
    )


class TestSensitivitiesCalculator:
    """Test finite difference vs analytical Greeks and interest rate DV01."""

    def test_finite_difference_matches_analytical_delta(self, risk_market):
        """Numerical finite difference Delta must match analytical BS Delta to <0.01%."""
        option = EuropeanOption(
            underlying="TECH",
            strike=150.0,
            expiry=0.5,
            option_type=OptionType.CALL,
        )
        bs_engine = BlackScholesEngine()
        calc = SensitivitiesCalculator(pricing_engine=bs_engine)

        analytical_delta = bs_engine.price(option, risk_market).greeks["delta"]
        numerical_delta = calc.calculate_delta(option, risk_market, bump_pct=0.001)

        assert math.isclose(numerical_delta, analytical_delta, rel_tol=1e-4)

    def test_finite_difference_matches_analytical_gamma(self, risk_market):
        """Numerical finite difference Gamma must match analytical BS Gamma."""
        option = EuropeanOption(
            underlying="TECH",
            strike=145.0,
            expiry=0.75,
            option_type=OptionType.PUT,
        )
        bs_engine = BlackScholesEngine()
        calc = SensitivitiesCalculator(pricing_engine=bs_engine)

        analytical_gamma = bs_engine.price(option, risk_market).greeks["gamma"]
        numerical_gamma = calc.calculate_gamma(option, risk_market, bump_pct=0.005)

        assert math.isclose(numerical_gamma, analytical_gamma, rel_tol=5e-4)

    def test_finite_difference_matches_analytical_vega(self, risk_market):
        """Numerical finite difference Vega must match analytical BS Vega."""
        option = EuropeanOption(
            underlying="TECH",
            strike=150.0,
            expiry=1.0,
            option_type=OptionType.CALL,
        )
        bs_engine = BlackScholesEngine()
        calc = SensitivitiesCalculator(pricing_engine=bs_engine)

        analytical_vega = bs_engine.price(option, risk_market).greeks["vega"]
        numerical_vega = calc.calculate_vega(option, risk_market, bump_vol=0.001)

        assert math.isclose(numerical_vega, analytical_vega, rel_tol=1e-4)

    def test_swap_parallel_dv01(self, risk_market):
        """Verify parallel 1bp curve bump revaluation for Interest Rate Swap."""
        dcf_engine = DiscountedCashFlowEngine()
        
        # Determine exact par rate
        temp_swap = InterestRateSwap(
            fixed_rate=0.045,
            tenor_years=5.0,
            payment_frequency=2.0,
            receive_fixed=True,
            notional=10_000_000.0,
        )
        par_rate = dcf_engine.price(temp_swap, risk_market).details["par_swap_rate"]

        par_swap = InterestRateSwap(
            fixed_rate=par_rate,
            tenor_years=5.0,
            payment_frequency=2.0,
            receive_fixed=True,
            notional=10_000_000.0,
        )
        calc = SensitivitiesCalculator(pricing_engine=dcf_engine)

        dv01_report = calc.calculate_dv01(par_swap, risk_market, bump_bps=1.0)
        
        # When rates rise by 1bp (+1bp bump), receiving fixed becomes less valuable:
        # V(r) > V(r + 1bp) => DV01 = V(r) - V(r + 1bp) > 0 for receiver swap
        assert math.isclose(dv01_report["base_npv"], 0.0, abs_tol=1e-4)
        assert dv01_report["dv01_usd"] > 0.0
        assert dv01_report["bumped_up_npv"] < dv01_report["base_npv"]

    def test_full_sensitivities_suite(self, risk_market):
        """Compute full risk report dictionary."""
        option = EuropeanOption(
            underlying="TECH",
            strike=150.0,
            expiry=0.5,
            option_type=OptionType.CALL,
        )
        calc = SensitivitiesCalculator(pricing_engine=BlackScholesEngine())
        report = calc.full_risk_report(option, risk_market)

        assert "delta" in report
        assert "gamma" in report
        assert "vega" in report
        assert "theta" in report
        assert "rho" in report
        assert "dv01" in report
