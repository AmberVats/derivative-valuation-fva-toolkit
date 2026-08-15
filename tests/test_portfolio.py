"""Tests for Portfolio multi-book valuation, risk aggregation, and net fair value."""

import math
import pytest
from src.instruments.base import OptionType
from src.instruments.options import EuropeanOption
from src.instruments.forwards import Forward
from src.instruments.swaps import InterestRateSwap
from src.market.market_data import MarketData
from src.market.curve import YieldCurve
from src.portfolio.portfolio import Portfolio
from src.adjustments.audit import AuditTrailManager


@pytest.fixture
def portfolio_market():
    times = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    dfs = [math.exp(-0.045 * t) for t in times]
    curve = YieldCurve(pillar_times=times, discount_factors=dfs, curve_name="USD_SOFR")

    return MarketData(
        as_of_date="2026-08-15",
        spots={"AAPL": 200.0, "SPX": 5000.0},
        risk_free_rate=0.045,
        dividend_yields={"AAPL": 0.01, "SPX": 0.015},
        flat_volatilities={"AAPL": 0.25, "SPX": 0.16},
        yield_curves={"DISCOUNT": curve},
        funding_spread_bps=40.0,
        cds_spread_bps=100.0,
        recovery_rate=0.40,
        bid_ask_spreads_bps={"AAPL": 5.0, "SPX": 2.0},
    )


class TestPortfolioValuation:
    """Test Portfolio class multi-book aggregation and valuation."""

    def test_portfolio_evaluation_and_net_fair_value(self, portfolio_market):
        portfolio = Portfolio("TestMultiAssetDesk")

        call = EuropeanOption("AAPL", strike=200.0, expiry=0.5, option_type=OptionType.CALL, notional=100.0)
        fwd = Forward("SPX", strike_price=5050.0, maturity=1.0, notional=10.0)
        swap = InterestRateSwap(fixed_rate=0.045, tenor_years=5.0, payment_frequency=2.0, receive_fixed=True, notional=1_000_000.0)

        portfolio.add_position(call, quantity=10.0, book_name="EQUITY_BOOK", trader="TRADER_1")
        portfolio.add_position(fwd, quantity=5.0, book_name="INDEX_BOOK", trader="TRADER_2")
        portfolio.add_position(swap, quantity=1.0, book_name="RATES_BOOK", trader="TRADER_3")

        assert len(portfolio.books) == 3
        assert len(portfolio.get_positions_by_book("EQUITY_BOOK")) == 1

        audit_mgr = AuditTrailManager()
        summary = portfolio.evaluate(portfolio_market, audit_manager=audit_mgr)

        assert summary["num_positions"] == 3
        assert summary["num_books"] == 3
        assert summary["base_npv_usd"] != 0.0
        assert summary["total_adjustments_usd"] > 0.0
        assert summary["net_fair_value_usd"] == summary["base_npv_usd"] - summary["total_adjustments_usd"]
        assert len(summary["positions_table"]) == 3
        assert len(audit_mgr.get_records()) == 3  # BO, FVA, CVA
