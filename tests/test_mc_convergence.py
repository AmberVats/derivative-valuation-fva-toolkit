"""Tests for Monte Carlo Pricing Engine convergence to analytical Black-Scholes

and variance reduction via Antithetic Variates.
"""

import math
import pytest
from src.instruments.base import OptionType
from src.instruments.options import EuropeanOption
from src.engines.black_scholes import BlackScholesEngine
from src.engines.monte_carlo import MonteCarloEngine
from src.market.market_data import MarketData


@pytest.fixture
def market_setup():
    return MarketData(
        as_of_date="2026-08-15",
        spots={"EQUITY": 100.0},
        risk_free_rate=0.05,
        dividend_yields={"EQUITY": 0.01},
        flat_volatilities={"EQUITY": 0.20},
    )


class TestMonteCarloConvergence:
    """Validate Monte Carlo pricing converges to analytical Black-Scholes benchmark."""

    def test_mc_within_confidence_interval(self, market_setup):
        """Assert Black-Scholes analytical price falls within MC 99% confidence interval."""
        option = EuropeanOption(
            underlying="EQUITY",
            strike=100.0,
            expiry=1.0,
            option_type=OptionType.CALL,
        )

        bs_engine = BlackScholesEngine()
        bs_res = bs_engine.price(option, market_setup)
        bs_price = bs_res.npv

        # 100,000 paths with antithetic variates
        mc_engine = MonteCarloEngine(num_paths=100_000, antithetic=True, seed=12345)
        mc_res = mc_engine.price(option, market_setup)

        mc_price = mc_res.npv
        std_err = mc_res.details["standard_error"]
        ci_99_lower = mc_res.details["ci_99_lower"]
        ci_99_upper = mc_res.details["ci_99_upper"]

        # Assert analytical price is contained in 99% CI
        assert ci_99_lower <= bs_price <= ci_99_upper, (
            f"BS price {bs_price:.4f} outside MC 99% CI [{ci_99_lower:.4f}, {ci_99_upper:.4f}] "
            f"(MC price: {mc_price:.4f}, SE: {std_err:.4f})"
        )

    def test_mc_convergence_rate_order_sqrt_n(self, market_setup):
        """Assert pricing error strictly decreases as path count increases from

        1,000 to 200,000.
        """
        option = EuropeanOption(
            underlying="EQUITY",
            strike=105.0,  # OTM Call
            expiry=0.75,
            option_type=OptionType.CALL,
        )
        bs_price = BlackScholesEngine().price(option, market_setup).npv

        path_counts = [1_000, 10_000, 100_000]
        errors = []
        standard_errors = []

        for paths in path_counts:
            engine = MonteCarloEngine(num_paths=paths, antithetic=True, seed=42)
            res = engine.price(option, market_setup)
            err = abs(res.npv - bs_price)
            errors.append(err)
            standard_errors.append(res.details["standard_error"])

        # High sample SE must be smaller than low sample SE
        assert standard_errors[2] < standard_errors[1] < standard_errors[0]
        # At 100k paths, error should be under 5 cents on a $100 stock option
        assert errors[2] < 0.05

    def test_antithetic_variates_variance_reduction(self, market_setup):
        """Assert antithetic sampling produces lower standard error than standard

        MC.
        """
        option = EuropeanOption(
            underlying="EQUITY",
            strike=100.0,
            expiry=1.0,
            option_type=OptionType.CALL,
        )

        n_paths = 50_000
        mc_standard = MonteCarloEngine(num_paths=n_paths, antithetic=False, seed=999)
        mc_antithetic = MonteCarloEngine(num_paths=n_paths, antithetic=True, seed=999)

        res_std = mc_standard.price(option, market_setup)
        res_anti = mc_antithetic.price(option, market_setup)

        se_std = res_std.details["standard_error"]
        se_anti = res_anti.details["standard_error"]

        # Antithetic SE should be strictly lower
        assert se_anti < se_std
