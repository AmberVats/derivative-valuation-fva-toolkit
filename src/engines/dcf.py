"""Discounted Cash Flow (DCF) pricing engine for linear interest rate and forward

derivatives.
"""

from typing import Any, Dict, List, TYPE_CHECKING
from src.engines.base import PricingEngine, PricingResult

if TYPE_CHECKING:
    from src.instruments.options import EuropeanOption
    from src.instruments.forwards import Forward
    from src.instruments.swaps import InterestRateSwap
    from src.market.market_data import MarketData


class DiscountedCashFlowEngine(PricingEngine):
    """Discounted Cash Flow engine supporting Interest Rate Swaps, Forwards,

    and fixed income cash flow schedules using multi-curve discounting.
    """

    @property
    def engine_name(self) -> str:
        return "DiscountedCashFlowEngine"

    def visit_european_option(
        self, option: "EuropeanOption", market: "MarketData"
    ) -> PricingResult:
        raise NotImplementedError(
            "DiscountedCashFlowEngine cannot price non-linear European Options. "
            "Please use BlackScholesEngine or MonteCarloEngine."
        )

    def visit_forward(
        self, forward: "Forward", market: "MarketData"
    ) -> PricingResult:
        """DCF valuation for forward contracts: PV = DF(T) * (ForwardPrice - Strike)."""
        t = forward.maturity
        s = market.get_spot(forward.underlying)
        q = market.get_dividend_yield(forward.underlying)
        k = forward.strike_price

        df_t = market.get_discount_factor(t)
        # Forward price F = S * exp(-q*t) / DF(t)
        forward_price = (s * math.exp(-q * t)) / df_t if df_t > 0 else s
        unit_pv = df_t * (forward_price - k)

        multiplier = forward.side_multiplier * forward.notional
        total_npv = multiplier * unit_pv

        return PricingResult(
            npv=total_npv,
            currency=forward.currency,
            engine_name=self.engine_name,
            as_of_date=market.as_of_date,
            details={
                "forward_price": forward_price,
                "discount_factor": df_t,
                "unit_pv": unit_pv,
            },
        )

    def visit_interest_rate_swap(
        self, swap: "InterestRateSwap", market: "MarketData"
    ) -> PricingResult:
        """Exact DCF pricing of fixed and floating legs of an Interest Rate Swap."""
        schedule = swap.payment_schedule
        dt = 1.0 / swap.payment_frequency

        cash_flows: List[Dict[str, Any]] = []
        fixed_pv_sum = 0.0
        float_pv_sum = 0.0
        annuity = 0.0

        prev_t = 0.0
        for t in schedule:
            tau = t - prev_t
            df_t = market.get_discount_factor(t)
            df_prev = market.get_discount_factor(prev_t)

            # Implied forward index rate: L = (P(prev) / P(curr) - 1) / tau
            fwd_rate = (df_prev / df_t - 1.0) / tau if df_t > 0 else 0.0

            # Cash amounts
            fixed_cf = swap.notional * swap.fixed_rate * tau
            float_cf = swap.notional * (fwd_rate + swap.float_spread) * tau

            fixed_cf_pv = fixed_cf * df_t
            float_cf_pv = float_cf * df_t

            fixed_pv_sum += fixed_cf_pv
            float_pv_sum += float_cf_pv
            annuity += tau * df_t

            cash_flows.append({
                "period_end": t,
                "tau": tau,
                "df": df_t,
                "fwd_rate": fwd_rate,
                "fixed_cf": fixed_cf,
                "fixed_pv": fixed_cf_pv,
                "float_cf": float_cf,
                "float_pv": float_cf_pv,
            })

            prev_t = t

        # Float leg algebraically equals: Notional * (DF(0) - DF(T_n)) + spread * annuity
        # Under single-curve pricing, fixed vs float net NPV:
        if swap.receive_fixed:
            npv = fixed_pv_sum - float_pv_sum
        else:
            npv = float_pv_sum - fixed_pv_sum

        df_last = market.get_discount_factor(swap.maturity)
        par_swap_rate = (1.0 - df_last) / annuity if annuity > 0 else swap.fixed_rate

        # Sensitivities
        # DV01 of fixed leg per 1bp (0.0001) shift in fixed rate
        dv01_fixed = swap.notional * 0.0001 * annuity
        dv01_net = dv01_fixed if swap.receive_fixed else -dv01_fixed

        details = {
            "fixed_leg_pv": fixed_pv_sum,
            "float_leg_pv": float_pv_sum,
            "annuity_pv01": annuity,
            "par_swap_rate": par_swap_rate,
            "dv01_fixed_leg": dv01_fixed,
            "cash_flows": cash_flows,
            "num_periods": len(schedule),
        }

        greeks = {
            "dv01": dv01_net,
            "pv01": annuity * swap.notional,
        }

        return PricingResult(
            npv=npv,
            currency=swap.currency,
            engine_name=self.engine_name,
            as_of_date=market.as_of_date,
            greeks=greeks,
            details=details,
        )
