"""Tests for Fair Value Adjustments (Bid-Offer Reserve, FVA, CVA) and Audit Trail

Framework.
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
from src.adjustments.bid_offer import BidOfferReserve
from src.adjustments.fva import FundingValuationAdjustment
from src.adjustments.cva import CreditValuationAdjustment
from src.adjustments.audit import AuditTrailManager


@pytest.fixture
def xva_market():
    times = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    dfs = [math.exp(-0.045 * t) for t in times]
    curve = YieldCurve(pillar_times=times, discount_factors=dfs, curve_name="USD_SOFR")

    return MarketData(
        as_of_date="2026-08-15",
        spots={"STOCK_A": 100.0, "STOCK_B": 200.0},
        risk_free_rate=0.045,
        dividend_yields={"STOCK_A": 0.01, "STOCK_B": 0.02},
        flat_volatilities={"STOCK_A": 0.25, "STOCK_B": 0.30},
        yield_curves={"DISCOUNT": curve},
        funding_spread_bps=50.0,  # 50 bps funding cost
        cds_spread_bps=150.0,     # 150 bps counterparty CDS
        recovery_rate=0.40,       # 40% recovery => LGD = 60%
        bid_ask_spreads_bps={"STOCK_A": 10.0, "STOCK_B": 20.0},
    )


class TestFairValueAdjustments:
    """Test Bid-Offer reserve, FVA, and CVA modules."""

    def test_bid_offer_reserve_calculation_and_netting(self, xva_market):
        """Test bid-offer reserve calculation with delta aggregation and netting."""
        # 10 Long Calls on STOCK_A (K=100) and 5 Short Calls on STOCK_A (K=100) -> Net 5 Long Calls
        call_long = EuropeanOption(
            underlying="STOCK_A", strike=100.0, expiry=1.0, option_type=OptionType.CALL, notional=1000.0
        )
        call_short = EuropeanOption(
            underlying="STOCK_A", strike=100.0, expiry=1.0, option_type=OptionType.CALL, notional=500.0
        )
        
        pricer = BlackScholesEngine()
        res_long = pricer.price(call_long, xva_market)
        res_short = pricer.price(call_short, xva_market)

        delta_long = res_long.greeks["delta"]
        delta_short = res_short.greeks["delta"]
        net_delta = delta_long - delta_short  # 500 * delta_unit

        bo_module = BidOfferReserve(methodology_version="1.0.0")
        result = bo_module.calculate(
            positions=[(call_long, 1.0), (call_short, -1.0)],
            market_data=xva_market,
            pricing_engine=pricer,
        )

        # Expected reserve = 0.5 * |Net Delta * Spot| * (Spread / 10000)
        spot = xva_market.get_spot("STOCK_A")
        spread = xva_market.get_bid_ask_spread("STOCK_A")
        expected_reserve = 0.5 * abs(net_delta * spot) * spread

        assert math.isclose(result.amount_usd, expected_reserve, rel_tol=1e-5)
        assert result.methodology_version == "1.0.0"
        assert result.audit_hash is not None

    def test_fva_positive_funding_cost(self, xva_market):
        """Test FVA calculation for positive expected exposure profile."""
        call = EuropeanOption(
            underlying="STOCK_A", strike=100.0, expiry=1.0, option_type=OptionType.CALL, notional=100.0
        )
        pricer = BlackScholesEngine()
        fva_module = FundingValuationAdjustment(methodology_version="1.2.0")

        res = fva_module.calculate_instrument(
            instrument=call,
            market_data=xva_market,
            pricing_engine=pricer,
        )

        # Positive exposure derivative must have positive funding cost
        assert res.amount_usd > 0.0
        assert "expected_exposure_profile" in res.breakdown
        assert len(res.breakdown["expected_exposure_profile"]) > 0
        assert res.parameters["funding_spread_bps"] == 50.0

    def test_cva_credit_risk_reserve(self, xva_market):
        """Test CVA calculation with hazard rate and default probability."""
        swap = InterestRateSwap(
            fixed_rate=0.055,  # In the money receiver swap (receiving 5.5% fixed vs paying 4.5% float)
            tenor_years=5.0,
            payment_frequency=2.0,
            receive_fixed=True,
            notional=1_000_000.0,
        )
        pricer = DiscountedCashFlowEngine()
        cva_module = CreditValuationAdjustment(methodology_version="2.0.0")

        res = cva_module.calculate_instrument(
            instrument=swap,
            market_data=xva_market,
            pricing_engine=pricer,
        )

        assert res.amount_usd > 0.0
        assert res.parameters["recovery_rate"] == 0.40
        assert res.parameters["loss_given_default"] == 0.60
        assert res.parameters["cds_spread_bps"] == 150.0
        assert "marginal_default_probabilities" in res.breakdown

    def test_audit_trail_manager_logging(self, xva_market):
        """Verify audit trail record registration, hash generation, and summary report."""
        audit_mgr = AuditTrailManager()
        bo_module = BidOfferReserve()
        call = EuropeanOption(underlying="STOCK_A", strike=100.0, expiry=1.0, notional=100.0)

        bo_res = bo_module.calculate([(call, 1.0)], xva_market, BlackScholesEngine())
        audit_mgr.record_adjustment(bo_res)

        records = audit_mgr.get_records()
        assert len(records) == 1
        rec = records[0]
        assert rec["adjustment_name"] == "BidOfferReserve"
        assert rec["amount_usd"] == bo_res.amount_usd
        assert rec["audit_hash"] == bo_res.audit_hash

        md_summary = audit_mgr.export_markdown_table()
        assert "BidOfferReserve" in md_summary
        assert "Methodology Version" in md_summary
