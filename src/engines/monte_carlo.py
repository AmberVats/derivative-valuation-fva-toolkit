"""Vectorized Monte Carlo simulation engine with variance reduction and error

analytics.
"""

import math
from typing import Optional, TYPE_CHECKING
import numpy as np

from src.engines.base import PricingEngine, PricingResult

if TYPE_CHECKING:
    from src.instruments.options import EuropeanOption
    from src.instruments.forwards import Forward
    from src.instruments.swaps import InterestRateSwap
    from src.market.market_data import MarketData


class MonteCarloEngine(PricingEngine):
    """Vectorized Monte Carlo pricing engine for path-dependent and vanilla

    derivatives. Supports Antithetic Variates for variance reduction and computes
    exact standard errors and confidence intervals.
    """

    def __init__(
        self,
        num_paths: int = 100_000,
        antithetic: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        if num_paths <= 0:
            raise ValueError(f"num_paths must be positive, got {num_paths}")
        self.num_paths = int(num_paths)
        self.antithetic = antithetic
        self.seed = seed

    @property
    def engine_name(self) -> str:
        return "MonteCarloEngine"

    def visit_european_option(
        self, option: "EuropeanOption", market: "MarketData"
    ) -> PricingResult:
        """Monte Carlo simulation of terminal spot price and payoff under

        risk-neutral measure.
        """
        s0 = market.get_spot(option.underlying)
        k = option.strike
        t = option.expiry
        r = market.get_zero_rate(t)
        q = market.get_dividend_yield(option.underlying)
        sigma = market.get_volatility(option.underlying, strike=k, expiry=t)

        rng = np.random.default_rng(self.seed)
        df_r = math.exp(-r * t)
        drift = (r - q - 0.5 * sigma * sigma) * t
        diffusion = sigma * math.sqrt(t)

        if self.antithetic:
            n_sims = self.num_paths // 2
            z = rng.standard_normal(n_sims)
            # Pair 1 & Pair 2
            st_1 = s0 * np.exp(drift + diffusion * z)
            st_2 = s0 * np.exp(drift - diffusion * z)

            if option.is_call:
                payoff_1 = np.maximum(st_1 - k, 0.0)
                payoff_2 = np.maximum(st_2 - k, 0.0)
            else:
                payoff_1 = np.maximum(k - st_1, 0.0)
                payoff_2 = np.maximum(k - st_2, 0.0)

            # Antithetic pair average
            payoffs = 0.5 * (payoff_1 + payoff_2)
            n_samples = n_sims
        else:
            z = rng.standard_normal(self.num_paths)
            st = s0 * np.exp(drift + diffusion * z)
            if option.is_call:
                payoffs = np.maximum(st - k, 0.0)
            else:
                payoffs = np.maximum(k - st, 0.0)
            n_samples = self.num_paths

        mean_payoff = float(np.mean(payoffs))
        sample_std = float(np.std(payoffs, ddof=1))
        standard_error = df_r * (sample_std / math.sqrt(n_samples))

        unit_npv = df_r * mean_payoff
        multiplier = option.side_multiplier * option.notional
        total_npv = multiplier * unit_npv

        # Confidence intervals (95% and 99%)
        ci_95_half = 1.96 * standard_error
        ci_99_half = 2.576 * standard_error

        details = {
            "num_paths": self.num_paths,
            "antithetic": self.antithetic,
            "standard_error": standard_error,
            "ci_95_lower": unit_npv - ci_95_half,
            "ci_95_upper": unit_npv + ci_95_half,
            "ci_99_lower": unit_npv - ci_99_half,
            "ci_99_upper": unit_npv + ci_99_half,
            "drift": drift,
            "diffusion": diffusion,
            "discount_factor": df_r,
        }

        return PricingResult(
            npv=total_npv,
            currency=option.currency,
            engine_name=self.engine_name,
            as_of_date=market.as_of_date,
            details=details,
        )

    def visit_forward(
        self, forward: "Forward", market: "MarketData"
    ) -> PricingResult:
        """Monte Carlo simulation for forward contract."""
        s0 = market.get_spot(forward.underlying)
        k = forward.strike_price
        t = forward.maturity
        r = market.get_zero_rate(t)
        q = market.get_dividend_yield(forward.underlying)
        sigma = market.get_volatility(forward.underlying, expiry=t)

        rng = np.random.default_rng(self.seed)
        df_r = math.exp(-r * t)
        drift = (r - q - 0.5 * sigma * sigma) * t
        diffusion = sigma * math.sqrt(t)

        z = rng.standard_normal(self.num_paths)
        st = s0 * np.exp(drift + diffusion * z)
        payoffs = st - k

        mean_payoff = float(np.mean(payoffs))
        sample_std = float(np.std(payoffs, ddof=1))
        standard_error = df_r * (sample_std / math.sqrt(self.num_paths))

        unit_npv = df_r * mean_payoff
        multiplier = forward.side_multiplier * forward.notional
        total_npv = multiplier * unit_npv

        return PricingResult(
            npv=total_npv,
            currency=forward.currency,
            engine_name=self.engine_name,
            as_of_date=market.as_of_date,
            details={
                "num_paths": self.num_paths,
                "standard_error": standard_error,
            },
        )

    def visit_interest_rate_swap(
        self, swap: "InterestRateSwap", market: "MarketData"
    ) -> PricingResult:
        """Monte Carlo engine forwards linear swap to DiscountedCashFlowEngine

        for exact analytic performance.
        """
        from src.engines.dcf import DiscountedCashFlowEngine
        return DiscountedCashFlowEngine().visit_interest_rate_swap(swap, market)
