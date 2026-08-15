"""Shared pytest fixtures for derivative valuation and FVA testing."""

import pytest
from src.market.market_data import MarketData
from src.market.curve import YieldCurve


@pytest.fixture
def sample_market_data():
    """Standard multi-asset market data fixture."""
    return MarketData(
        as_of_date="2026-08-15",
        spots={
            "AAPL": 220.0,
            "SPX": 5500.0,
            "EURUSD": 1.0850,
            "US_TREASURY_10Y": 100.0,
        },
        risk_free_rate=0.045,
        dividend_yields={
            "AAPL": 0.005,
            "SPX": 0.015,
            "EURUSD": 0.030,
        },
        flat_volatilities={
            "AAPL": 0.22,
            "SPX": 0.16,
            "EURUSD": 0.08,
        },
        funding_spread_bps=45.0,  # 45 bps funding spread
        cds_spread_bps=120.0,     # 120 bps counterparty CDS spread
        recovery_rate=0.40,       # 40% recovery rate
    )


@pytest.fixture
def sample_yield_curve():
    """Standard log-linear discount yield curve fixture."""
    times = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0]
    # Realistic SOFR / OIS discount factors
    discount_factors = [
        0.9880, 0.9760, 0.9520, 0.9070, 0.8650, 0.7890, 0.7180, 0.6250, 0.2850
    ]
    return YieldCurve(pillar_times=times, discount_factors=discount_factors, curve_name="USD_SOFR_DISCOUNT")
