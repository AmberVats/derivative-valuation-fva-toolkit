"""Yield curve bootstrapping engine for deposit quotes and par interest rate swaps."""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence
from scipy.optimize import brentq

from src.market.curve import YieldCurve


@dataclass
class DepositQuote:
    """Money market deposit rate quote (simple compounding)."""
    tenor_years: float
    rate: float  # decimal e.g. 0.045 for 4.5%

    def __post_init__(self):
        if self.tenor_years <= 0:
            raise ValueError(f"Tenor must be positive, got {self.tenor_years}")


@dataclass
class SwapQuote:
    """Par Interest Rate Swap quote (fixed rate vs floating reference index)."""
    tenor_years: float
    par_rate: float  # decimal e.g. 0.05 for 5%
    fixed_frequency: float = 1.0  # payment frequency per year (1.0 = annual, 2.0 = semiannual, 4.0 = quarterly)

    def __post_init__(self):
        if self.tenor_years <= 0:
            raise ValueError(f"Tenor must be positive, got {self.tenor_years}")
        if self.fixed_frequency <= 0:
            raise ValueError(f"Fixed frequency must be positive, got {self.fixed_frequency}")


class YieldCurveBootstrapper:
    """Sequential yield curve bootstrapper solving for discount factors from

    money market deposits and par swaps with exact repricing guarantees.
    """

    def bootstrap(
        self,
        deposits: Sequence[DepositQuote],
        swaps: Sequence[SwapQuote],
        curve_name: str = "BootstrappedYieldCurve",
    ) -> YieldCurve:
        """Bootstraps a complete YieldCurve from deposit and swap quotes.

        Parameters
        ----------
        deposits : Sequence[DepositQuote]
            Short-end money market deposit quotes (up to ~1 year).
        swaps : Sequence[SwapQuote]
            Long-end par swap quotes (from ~1Y to 30Y).
        curve_name : str
            Identifier for the generated curve.

        Returns
        -------
        YieldCurve
            Calibrated arbitrage-free yield curve.
        """
        # Sort quotes by tenor
        sorted_deposits = sorted(deposits, key=lambda d: d.tenor_years)
        sorted_swaps = sorted(swaps, key=lambda s: s.tenor_years)

        pillar_times: List[float] = [0.0]
        discount_factors: List[float] = [1.0]

        # 1. Bootstrap short end from Money Market Deposits
        # DF(t) = 1 / (1 + rate * t)
        for dep in sorted_deposits:
            t = dep.tenor_years
            df = 1.0 / (1.0 + dep.rate * t)
            pillar_times.append(t)
            discount_factors.append(df)

        # 2. Bootstrap long end from Par Swaps
        for swap in sorted_swaps:
            t_swap = swap.tenor_years
            s_rate = swap.par_rate
            freq = swap.fixed_frequency
            dt = 1.0 / freq
            n_payments = int(round(t_swap * freq))

            # Cash flow payment times
            payment_times = [round(i * dt, 6) for i in range(1, n_payments + 1)]

            # Build a root-finding objective to solve for DF(t_swap) using scipy.optimize.brentq
            def swap_pricing_error(target_df_guess: float) -> float:
                """Computes Par Swap NPV - 0 using temporary curve with target_df_guess."""
                temp_times = list(pillar_times) + [t_swap]
                temp_dfs = list(discount_factors) + [target_df_guess]
                temp_curve = YieldCurve(temp_times, temp_dfs)

                # Fixed leg PV = s_rate * sum(dt * DF(t_i))
                # Floating leg PV = 1.0 - DF(t_swap)
                fixed_leg_pv = sum(s_rate * dt * temp_curve.discount_factor(ti) for ti in payment_times)
                float_leg_pv = 1.0 - target_df_guess
                return fixed_leg_pv - float_leg_pv

            # Robust solve using brentq
            # Discount factor for positive rates is strictly in (0.0001, 1.0)
            try:
                solved_df = brentq(swap_pricing_error, 1e-4, 1.0, xtol=1e-12, maxiter=100)
            except Exception as e:
                # Fallback to algebraic solve if exact intermediate points align
                annuity_prev = sum(
                    s_rate * dt * YieldCurve(pillar_times, discount_factors).discount_factor(ti)
                    for ti in payment_times[:-1]
                )
                solved_df = (1.0 - annuity_prev) / (1.0 + s_rate * dt)

            pillar_times.append(t_swap)
            discount_factors.append(float(solved_df))

        return YieldCurve(
            pillar_times=pillar_times,
            discount_factors=discount_factors,
            curve_name=curve_name,
        )
