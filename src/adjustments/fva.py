"""Funding Valuation Adjustment (FVA) methodology and exposure profile integration."""

from copy import deepcopy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
from src.adjustments.base import AdjustmentResult, FairValueAdjustment

if TYPE_CHECKING:
    from src.instruments.base import Instrument
    from src.engines.base import PricingEngine
    from src.market.market_data import MarketData


class FundingValuationAdjustment(FairValueAdjustment):
    """Calculates Funding Valuation Adjustment (FVA) on uncollateralized or

    partially collateralized derivative portfolios.

    Methodology:
    Integrates the Expected Exposure (EE) profile discounted at the risk-free rate
    multiplied by the bank's net funding spread over OIS:
        FVA = sum_m [ EE(t_m) * s_F * DF(t_m) * delta_t_m ]
    where s_F is the funding cost spread (Funding Cost Adjustment - FCA).
    """

    def __init__(self, methodology_version: str = "1.2.0") -> None:
        super().__init__(methodology_version=methodology_version)

    @property
    def name(self) -> str:
        return "FundingValuationAdjustment"

    @property
    def description(self) -> str:
        return (
            "Expected lifetime funding cost of uncollateralised derivative positive exposures "
            "integrated across discrete simulation time buckets."
        )

    def calculate_instrument(
        self,
        instrument: "Instrument",
        market_data: "MarketData",
        pricing_engine: "PricingEngine",
        time_steps: int = 10,
        funding_spread_bps: Optional[float] = None,
    ) -> AdjustmentResult:
        """Calculate FVA for a single instrument."""
        return self.calculate(
            positions=[(instrument, 1.0)],
            market_data=market_data,
            pricing_engine=pricing_engine,
            time_steps=time_steps,
            funding_spread_bps=funding_spread_bps,
        )

    def calculate(
        self,
        positions: Sequence[Tuple["Instrument", float]],
        market_data: "MarketData",
        pricing_engine: "PricingEngine",
        time_steps: int = 10,
        funding_spread_bps: Optional[float] = None,
    ) -> AdjustmentResult:
        """Calculates portfolio-level Funding Valuation Adjustment (FVA).

        Parameters
        ----------
        positions : Sequence[Tuple[Instrument, float]]
            List of (instrument, quantity) tuples.
        market_data : MarketData
            Market parameters containing funding spread and discount curves.
        pricing_engine : PricingEngine
            Pricing engine to evaluate forward exposure profiles.
        time_steps : int
            Number of discrete integration buckets.
        funding_spread_bps : Optional[float]
            Optional override for funding spread in bps.

        Returns
        -------
        AdjustmentResult
            Calculated FVA amount, parameters, and time-series profile.
        """
        spread_bps = funding_spread_bps if funding_spread_bps is not None else market_data.funding_spread_bps
        s_f = spread_bps / 10000.0

        # Find maximum maturity across portfolio
        max_maturity = max((inst.maturity for inst, _ in positions), default=1.0)
        if max_maturity <= 1e-4:
            max_maturity = 1.0

        dt = max_maturity / float(time_steps)
        time_grid = [round(i * dt, 4) for i in range(1, time_steps + 1)]

        profile: List[Dict[str, Any]] = []
        total_fva = 0.0

        prev_t = 0.0
        for t in time_grid:
            delta_t = t - prev_t
            df_t = market_data.get_discount_factor(t)

            # Evaluate expected exposure at time t
            # For linear/vanilla instruments, expected forward value under forward measure:
            # We decay remaining maturity to simulate forward profile
            bucket_pv = 0.0
            for inst, qty in positions:
                if inst.maturity > t:
                    inst_fwd = deepcopy(inst)
                    if hasattr(inst_fwd, "expiry"):
                        inst_fwd.expiry = max(1e-4, inst_fwd.expiry - t)
                    elif hasattr(inst_fwd, "_maturity"):
                        inst_fwd._maturity = max(1e-4, inst_fwd._maturity - t)
                    elif hasattr(inst_fwd, "_tenor_years"):
                        inst_fwd._tenor_years = max(1e-4, inst_fwd._tenor_years - t)

                    res_fwd = pricing_engine.price(inst_fwd, market_data)
                    bucket_pv += res_fwd.npv * qty

            # Expected positive exposure (EE+)
            ee_positive = max(bucket_pv, 0.0)
            ee_negative = max(-bucket_pv, 0.0)

            # Incremental FCA = EE+ * s_F * DF(t) * delta_t
            incremental_fva = ee_positive * s_f * df_t * delta_t
            total_fva += incremental_fva

            profile.append({
                "time_years": t,
                "delta_t": delta_t,
                "discount_factor": df_t,
                "expected_exposure": bucket_pv,
                "ee_positive": ee_positive,
                "ee_negative": ee_negative,
                "incremental_fva_usd": incremental_fva,
            })

            prev_t = t

        parameters = {
            "funding_spread_bps": spread_bps,
            "funding_spread_decimal": s_f,
            "time_steps": time_steps,
            "max_maturity_years": max_maturity,
        }

        breakdown = {
            "expected_exposure_profile": profile,
            "total_fva_usd": total_fva,
        }

        return AdjustmentResult(
            adjustment_name=self.name,
            amount_usd=total_fva,
            methodology_version=self.methodology_version,
            as_of_date=market_data.as_of_date,
            parameters=parameters,
            breakdown=breakdown,
            notes="Integrated expected exposure across forward discrete simulation buckets.",
        )
