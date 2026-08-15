"""Tests for Interest Rate Swaps and Discounted Cash Flow (DCF) pricing engine.

Validates fixed/float leg decomposition, par swap rate calculation, and zero NPV at par.
"""

import math
import pytest
from src.instruments.swaps import InterestRateSwap
from src.engines.dcf import DiscountedCashFlowEngine
from src.market.market_data import MarketData
from src.market.curve import YieldCurve


@pytest.fixture
def flat_curve():
    """Flat 5.0% yield curve."""
    times = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
    dfs = [math.exp(-0.05 * t) for t in times]
    return YieldCurve(pillar_times=times, discount_factors=dfs, curve_name="FLAT_5PCT")


@pytest.fixture
def swap_market(flat_curve):
    return MarketData(
        as_of_date="2026-08-15",
        risk_free_rate=0.05,
        yield_curves={"DISCOUNT": flat_curve},
    )


@pytest.fixture
def dcf_engine():
    return DiscountedCashFlowEngine()


class TestInterestRateSwapsDCF:
    """Test InterestRateSwap instrument and DCF engine."""

    def test_par_swap_has_zero_npv(self, flat_curve, swap_market, dcf_engine):
        """A swap struck at the par swap rate must have NPV = 0.0."""
        # Calculate theoretical par rate for 5-year annual swap on 5% continuous flat curve
        # Annuity = sum(DF(t)) for t=1..5
        annuity = sum(flat_curve.discount_factor(t) for t in range(1, 6))
        par_rate = (1.0 - flat_curve.discount_factor(5.0)) / annuity

        swap = InterestRateSwap(
            fixed_rate=par_rate,
            tenor_years=5.0,
            payment_frequency=1.0,  # annual
            receive_fixed=True,
            notional=10_000_000.0,
        )

        res = dcf_engine.price(swap, swap_market)
        assert math.isclose(res.npv, 0.0, abs_tol=1e-4)
        assert math.isclose(res.details["par_swap_rate"], par_rate, abs_tol=1e-8)

    def test_receiver_vs_payer_swap_symmetry(self, swap_market, dcf_engine):
        """Receiver swap NPV must equal minus Payer swap NPV."""
        receiver_swap = InterestRateSwap(
            fixed_rate=0.040,  # Below market (in the money for payer, out of money for receiver)
            tenor_years=5.0,
            payment_frequency=2.0,  # semi-annual
            receive_fixed=True,
            notional=5_000_000.0,
        )
        payer_swap = InterestRateSwap(
            fixed_rate=0.040,
            tenor_years=5.0,
            payment_frequency=2.0,
            receive_fixed=False,
            notional=5_000_000.0,
        )

        rec_res = dcf_engine.price(receiver_swap, swap_market)
        pay_res = dcf_engine.price(payer_swap, swap_market)

        assert math.isclose(rec_res.npv, -pay_res.npv, rel_tol=1e-10)
        assert rec_res.details["fixed_leg_pv"] == pay_res.details["fixed_leg_pv"]
        assert rec_res.details["float_leg_pv"] == pay_res.details["float_leg_pv"]

    def test_pv01_dv01_interest_rate_sensitivity(self, flat_curve, swap_market, dcf_engine):
        """Test PV01 (annuity) calculation and 1bp DV01 impact."""
        swap = InterestRateSwap(
            fixed_rate=0.050,
            tenor_years=10.0,
            payment_frequency=2.0,  # semiannual (20 payments)
            receive_fixed=True,
            notional=1_000_000.0,
        )
        res = dcf_engine.price(swap, swap_market)
        pv01 = res.details["annuity_pv01"]  # PV of 1bp coupon over 10 years per unit notional
        assert pv01 > 0.0
        # For $1M notional, 1bp shift on fixed coupon changes fixed leg by notional * dt * sum(DF) * 1e-4
        dt = 0.5
        expected_dv01_fixed = 1_000_000.0 * 0.0001 * sum(dt * flat_curve.discount_factor(i * dt) for i in range(1, 21))
        assert math.isclose(res.details["dv01_fixed_leg"], expected_dv01_fixed, rel_tol=1e-6)
