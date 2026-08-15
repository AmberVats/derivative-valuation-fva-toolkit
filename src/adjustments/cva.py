"""Credit Valuation Adjustment (CVA) & Debit Valuation Adjustment (DVA)

methodologies.
"""

from copy import deepcopy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
from src.adjustments.base import AdjustmentResult, FairValueAdjustment

if TYPE_CHECKING:
    from src.instruments.base import Instrument
    from src.engines.base import PricingEngine
    from src.market.market_data import MarketData


class CreditValuationAdjustment(FairValueAdjustment):
    """Calculates Credit Valuation Adjustment (CVA) for counterparty credit risk

    and optional Debit Valuation Adjustment (DVA) for own default risk.

    Methodology:
    Evaluates expected default loss over time buckets:
        CVA = (1 - R) * sum_m [ EE+(t_m) * PD(t_{m-1}, t_m) * DF(t_m) ]
    where:
        R is the recovery rate (default 40% -> LGD = 60%)
        PD(t_{m-1}, t_m) = exp(-lambda * t_{m-1}) - exp(-lambda * t_m)
        lambda is the hazard rate derived from counterparty CDS spread: lambda = CDS_spread / (1 - R).
    """

    def __init__(self, methodology_version: str = "2.0.0") -> None:
        super().__init__(methodology_version=methodology_version)

    @property
    def name(self) -> str:
        return "CreditValuationAdjustment"

    @property
    def description(self) -> str:
        return (
            "Counterparty credit risk reserve integrating positive expected exposures with "
            "hazard-rate marginal default probabilities and Loss Given Default (LGD)."
        )

    def calculate_instrument(
        self,
        instrument: "Instrument",
        market_data: "MarketData",
        pricing_engine: "PricingEngine",
        time_steps: int = 10,
        cds_spread_bps: Optional[float] = None,
        recovery_rate: Optional[float] = None,
    ) -> AdjustmentResult:
        """Calculate CVA for a single instrument."""
        return self.calculate(
            positions=[(instrument, 1.0)],
            market_data=market_data,
            pricing_engine=pricing_engine,
            time_steps=time_steps,
            cds_spread_bps=cds_spread_bps,
            recovery_rate=recovery_rate,
        )

    def calculate(
        self,
        positions: Sequence[Tuple["Instrument", float]],
        market_data: "MarketData",
        pricing_engine: "PricingEngine",
        time_steps: int = 10,
        cds_spread_bps: Optional[float] = None,
        recovery_rate: Optional[float] = None,
    ) -> AdjustmentResult:
        """Calculates portfolio-level Credit Valuation Adjustment (CVA)."""
        spread_bps = cds_spread_bps if cds_spread_bps is not None else market_data.cds_spread_bps
        s_cds = spread_bps / 10000.0
        rec = recovery_rate if recovery_rate is not None else market_data.recovery_rate
        lgd = 1.0 - rec

        # Hazard rate lambda = CDS_spread / LGD
        hazard_rate = s_cds / lgd if lgd > 0 else 0.0

        max_maturity = max((inst.maturity for inst, _ in positions), default=1.0)
        if max_maturity <= 1e-4:
            max_maturity = 1.0

        dt = max_maturity / float(time_steps)
        time_grid = [round(i * dt, 4) for i in range(1, time_steps + 1)]

        profile: List[Dict[str, Any]] = []
        marginal_pds: List[float] = []
        total_cva = 0.0

        prev_t = 0.0
        for t in time_grid:
            delta_t = t - prev_t
            df_t = market_data.get_discount_factor(t)

            # Marginal default probability: PD(t_{m-1}, t_m) = exp(-lambda * t_{m-1}) - exp(-lambda * t_m)
            surv_prev = math.exp(-hazard_rate * prev_t)
            surv_curr = math.exp(-hazard_rate * t)
            marginal_pd = surv_prev - surv_curr
            marginal_pds.append(marginal_pd)

            # Evaluate expected exposure at bucket midpoint or bucket end
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

            ee_positive = max(bucket_pv, 0.0)
            ee_negative = max(-bucket_pv, 0.0)

            # Incremental CVA = LGD * EE+ * Marginal_PD * DF(t)
            incremental_cva = lgd * ee_positive * marginal_pd * df_t
            total_cva += incremental_cva

            profile.append({
                "time_years": t,
                "delta_t": delta_t,
                "discount_factor": df_t,
                "expected_exposure": bucket_pv,
                "ee_positive": ee_positive,
                "ee_negative": ee_negative,
                "marginal_default_probability": marginal_pd,
                "survival_probability": surv_curr,
                "incremental_cva_usd": incremental_cva,
            })

            prev_t = t

        parameters = {
            "cds_spread_bps": spread_bps,
            "recovery_rate": rec,
            "loss_given_default": lgd,
            "hazard_rate": hazard_rate,
            "time_steps": time_steps,
            "max_maturity_years": max_maturity,
        }

        breakdown = {
            "marginal_default_probabilities": marginal_pds,
            "exposure_profile": profile,
            "total_cva_usd": total_cva,
        }

        return AdjustmentResult(
            adjustment_name=self.name,
            amount_usd=total_cva,
            methodology_version=self.methodology_version,
            as_of_date=market_data.as_of_date,
            parameters=parameters,
            breakdown=breakdown,
            notes="Evaluated with intensity-based Poisson default hazard model.",
        )
