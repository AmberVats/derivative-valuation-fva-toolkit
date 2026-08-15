"""Tests for Put-Call Parity validation across diverse moneyness and market

conditions.
Parity Theorem: C - P = S0 * e^(-q*T) - K * e^(-r*T)
"""

import math
import pytest
from src.instruments.base import OptionType
from src.instruments.options import EuropeanOption
from src.engines.black_scholes import BlackScholesEngine
from src.market.market_data import MarketData


class TestPutCallParity:
    """Validate Put-Call parity holds strictly across parameter space."""

    @pytest.mark.parametrize("spot", [20.0, 50.0, 100.0, 250.0])
    @pytest.mark.parametrize("strike", [25.0, 50.0, 75.0, 120.0])
    @pytest.mark.parametrize("rate", [0.01, 0.05, 0.12])
    @pytest.mark.parametrize("dividend", [0.0, 0.02, 0.05])
    @pytest.mark.parametrize("vol", [0.10, 0.25, 0.60])
    @pytest.mark.parametrize("expiry", [0.1, 0.5, 1.0, 2.5])
    def test_put_call_parity_exact(self, spot, strike, rate, dividend, vol, expiry):
        """Assert C - P == S*exp(-qT) - K*exp(-rT) within floating point

        precision.
        """
        market = MarketData(
            as_of_date="2026-08-15",
            spots={"ASSET": spot},
            risk_free_rate=rate,
            dividend_yields={"ASSET": dividend},
            flat_volatilities={"ASSET": vol},
        )
        engine = BlackScholesEngine()

        call = EuropeanOption(
            underlying="ASSET",
            strike=strike,
            expiry=expiry,
            option_type=OptionType.CALL,
        )
        put = EuropeanOption(
            underlying="ASSET",
            strike=strike,
            expiry=expiry,
            option_type=OptionType.PUT,
        )

        c_res = engine.price(call, market)
        p_res = engine.price(put, market)

        lhs = c_res.npv - p_res.npv
        rhs = spot * math.exp(-dividend * expiry) - strike * math.exp(-rate * expiry)

        assert math.isclose(lhs, rhs, abs_tol=1e-10)
