"""Analytical Black-Scholes-Merton pricing engine with exact closed-form Greeks."""

import math
from typing import TYPE_CHECKING
from scipy.stats import norm

from src.engines.base import PricingEngine, PricingResult
from src.instruments.base import PositionSide

if TYPE_CHECKING:
    from src.instruments.options import EuropeanOption
    from src.instruments.forwards import Forward
    from src.instruments.swaps import InterestRateSwap
    from src.market.market_data import MarketData


class BlackScholesEngine(PricingEngine):
    """Closed-form analytical Black-Scholes-Merton engine with continuous dividend

    yield and exact partial derivative Greeks.
    """

    @property
    def engine_name(self) -> str:
        return "BlackScholesEngine"

    def visit_european_option(
        self, option: "EuropeanOption", market: "MarketData"
    ) -> PricingResult:
        """Closed-form European option pricing and analytical Greeks."""
        s = market.get_spot(option.underlying)
        k = option.strike
        t = option.expiry
        r = market.get_zero_rate(t)
        q = market.get_dividend_yield(option.underlying)
        sigma = market.get_volatility(option.underlying, strike=k, expiry=t)

        df_r = math.exp(-r * t)
        df_q = math.exp(-q * t)
        sqrt_t = math.sqrt(t)

        vol_time = sigma * sqrt_t
        if vol_time <= 1e-12:
            # Degenerate case (zero vol or zero time)
            forward_val = s * df_q - k * df_r
            if option.is_call:
                unit_npv = max(forward_val, 0.0)
            else:
                unit_npv = max(-forward_val, 0.0)
            return PricingResult(
                npv=option.side_multiplier * option.notional * unit_npv,
                currency=option.currency,
                engine_name=self.engine_name,
                as_of_date=market.as_of_date,
            )

        d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_time
        d2 = d1 - vol_time

        n_d1 = float(norm.cdf(d1))
        n_d2 = float(norm.cdf(d2))
        n_minus_d1 = float(norm.cdf(-d1))
        n_minus_d2 = float(norm.cdf(-d2))
        n_prime_d1 = float(norm.pdf(d1))

        if option.is_call:
            unit_npv = s * df_q * n_d1 - k * df_r * n_d2
            delta = df_q * n_d1
            theta = (
                -(s * df_q * n_prime_d1 * sigma) / (2.0 * sqrt_t)
                - r * k * df_r * n_d2
                + q * s * df_q * n_d1
            )
            rho = k * t * df_r * n_d2
        else:
            unit_npv = k * df_r * n_minus_d2 - s * df_q * n_minus_d1
            delta = -df_q * n_minus_d1
            theta = (
                -(s * df_q * n_prime_d1 * sigma) / (2.0 * sqrt_t)
                + r * k * df_r * n_minus_d2
                - q * s * df_q * n_minus_d1
            )
            rho = -k * t * df_r * n_minus_d2

        gamma = (df_q * n_prime_d1) / (s * vol_time)
        vega = s * df_q * sqrt_t * n_prime_d1  # per 100% vol

        # Scale by position side and notional
        multiplier = option.side_multiplier * option.notional
        total_npv = multiplier * unit_npv

        greeks = {
            "delta": multiplier * delta,
            "gamma": multiplier * gamma,
            "vega": multiplier * vega,
            "theta": multiplier * theta,
            "rho": multiplier * rho,
            "vega_1pct": multiplier * (vega * 0.01),
            "theta_1day": multiplier * (theta / 365.0),
        }

        details = {
            "d1": d1,
            "d2": d2,
            "spot": s,
            "strike": k,
            "risk_free_rate": r,
            "dividend_yield": q,
            "volatility": sigma,
            "expiry_years": t,
            "df_rate": df_r,
            "df_dividend": df_q,
            "unit_npv": unit_npv,
        }

        return PricingResult(
            npv=total_npv,
            currency=option.currency,
            engine_name=self.engine_name,
            as_of_date=market.as_of_date,
            greeks=greeks,
            details=details,
        )

    def visit_forward(
        self, forward: "Forward", market: "MarketData"
    ) -> PricingResult:
        """Closed-form forward contract pricing: PV = S*e^(-qT) - K*e^(-rT)."""
        s = market.get_spot(forward.underlying)
        k = forward.strike_price
        t = forward.maturity
        r = market.get_zero_rate(t)
        q = market.get_dividend_yield(forward.underlying)

        df_r = math.exp(-r * t)
        df_q = math.exp(-q * t)

        unit_npv = s * df_q - k * df_r
        multiplier = forward.side_multiplier * forward.notional
        total_npv = multiplier * unit_npv

        greeks = {
            "delta": multiplier * df_q,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": multiplier * (r * k * df_r - q * s * df_q),
            "rho": multiplier * (k * t * df_r),
        }

        details = {
            "spot": s,
            "strike_price": k,
            "forward_price": s * math.exp((r - q) * t),
            "df_rate": df_r,
            "df_dividend": df_q,
            "maturity_years": t,
        }

        return PricingResult(
            npv=total_npv,
            currency=forward.currency,
            engine_name=self.engine_name,
            as_of_date=market.as_of_date,
            greeks=greeks,
            details=details,
        )

    def visit_interest_rate_swap(
        self, swap: "InterestRateSwap", market: "MarketData"
    ) -> PricingResult:
        """Black-Scholes engine does not price Interest Rate Swaps.

        Use DiscountedCashFlowEngine instead.
        """
        raise NotImplementedError(
            "BlackScholesEngine cannot price InterestRateSwap. "
            "Please use DiscountedCashFlowEngine for linear interest rate derivatives."
        )
