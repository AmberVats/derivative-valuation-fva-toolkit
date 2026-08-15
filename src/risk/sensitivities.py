"""Risk sensitivities calculator implementing finite-difference Greeks,

analytical reconciliations, and interest rate curve DV01.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING
from copy import deepcopy

if TYPE_CHECKING:
    from src.instruments.base import Instrument
    from src.engines.base import PricingEngine
    from src.market.market_data import MarketData


class SensitivitiesCalculator:
    """Computes first and second-order price sensitivities (Delta, Gamma, Vega,

    Theta, Rho, DV01) using central finite differences and model revaluations.
    """

    def __init__(self, pricing_engine: "PricingEngine") -> None:
        self.pricing_engine = pricing_engine

    def calculate_delta(
        self,
        instrument: "Instrument",
        market: "MarketData",
        bump_pct: float = 0.001,  # 0.10% bump
    ) -> float:
        """Central finite-difference Delta: dV / dS."""
        underlying = getattr(instrument, "underlying", None)
        if not underlying or underlying not in market.spots:
            return 0.0

        s0 = market.get_spot(underlying)
        ds = s0 * bump_pct

        market_up = market.bump_spot(underlying, bump_pct)
        market_down = market.bump_spot(underlying, -bump_pct)

        pv_up = self.pricing_engine.price(instrument, market_up).npv
        pv_down = self.pricing_engine.price(instrument, market_down).npv

        return (pv_up - pv_down) / (2.0 * ds)

    def calculate_gamma(
        self,
        instrument: "Instrument",
        market: "MarketData",
        bump_pct: float = 0.005,  # 0.50% bump
    ) -> float:
        """Central finite-difference Gamma: d2V / dS2."""
        underlying = getattr(instrument, "underlying", None)
        if not underlying or underlying not in market.spots:
            return 0.0

        s0 = market.get_spot(underlying)
        ds = s0 * bump_pct

        market_up = market.bump_spot(underlying, bump_pct)
        market_down = market.bump_spot(underlying, -bump_pct)

        pv_base = self.pricing_engine.price(instrument, market).npv
        pv_up = self.pricing_engine.price(instrument, market_up).npv
        pv_down = self.pricing_engine.price(instrument, market_down).npv

        return (pv_up - 2.0 * pv_base + pv_down) / (ds * ds)

    def calculate_vega(
        self,
        instrument: "Instrument",
        market: "MarketData",
        bump_vol: float = 0.001,  # 0.10% vol bump
    ) -> float:
        """Central finite-difference Vega: dV / dsigma (scaled to 100% vol)."""
        underlying = getattr(instrument, "underlying", None)
        if not underlying or underlying not in market.flat_volatilities:
            return 0.0

        market_up = market.bump_volatility(underlying, bump_vol)
        market_down = market.bump_volatility(underlying, -bump_vol)

        pv_up = self.pricing_engine.price(instrument, market_up).npv
        pv_down = self.pricing_engine.price(instrument, market_down).npv

        return (pv_up - pv_down) / (2.0 * bump_vol)

    def calculate_rho(
        self,
        instrument: "Instrument",
        market: "MarketData",
        bump_rate: float = 0.0001,  # 1 bp
    ) -> float:
        """Central finite-difference Rho: dV / dr."""
        market_up = market.bump_rate(bump_bps=1.0)
        market_down = market.bump_rate(bump_bps=-1.0)

        pv_up = self.pricing_engine.price(instrument, market_up).npv
        pv_down = self.pricing_engine.price(instrument, market_down).npv

        return (pv_up - pv_down) / (2.0 * bump_rate)

    def calculate_theta(
        self,
        instrument: "Instrument",
        market: "MarketData",
        decay_days: float = 1.0,
    ) -> float:
        """Theta: - dV / dt (1 day decay scaled to 1 year)."""
        dt = decay_days / 365.0
        if not hasattr(instrument, "maturity") or instrument.maturity <= dt:
            return 0.0

        pv_base = self.pricing_engine.price(instrument, market).npv

        # Create decayed instrument copy
        inst_decayed = deepcopy(instrument)
        if hasattr(inst_decayed, "expiry"):
            inst_decayed.expiry = max(1e-6, inst_decayed.expiry - dt)
        elif hasattr(inst_decayed, "_maturity"):
            inst_decayed._maturity = max(1e-6, inst_decayed._maturity - dt)
        elif hasattr(inst_decayed, "_tenor_years"):
            inst_decayed._tenor_years = max(1e-6, inst_decayed._tenor_years - dt)

        pv_decayed = self.pricing_engine.price(inst_decayed, market).npv

        # Theta is negative of change over time elapsed
        return -(pv_base - pv_decayed) / dt

    def calculate_dv01(
        self,
        instrument: "Instrument",
        market: "MarketData",
        bump_bps: float = 1.0,
        curve_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Calculates Dollar Value of 01 (DV01) for interest rate shifts.

        Convention: DV01 = V(base) - V(base + 1bp), i.e., the dollar loss if rates rise by 1bp.
        """
        base_res = self.pricing_engine.price(instrument, market)
        base_pv = base_res.npv

        market_up = market.bump_rate(bump_bps=bump_bps, curve_name=curve_name)
        market_down = market.bump_rate(bump_bps=-bump_bps, curve_name=curve_name)

        pv_up = self.pricing_engine.price(instrument, market_up).npv
        pv_down = self.pricing_engine.price(instrument, market_down).npv

        dv01_one_way = base_pv - pv_up
        dv01_central = (pv_down - pv_up) / 2.0

        return {
            "base_npv": base_pv,
            "bumped_up_npv": pv_up,
            "bumped_down_npv": pv_down,
            "dv01_usd": dv01_one_way,
            "dv01_central_usd": dv01_central,
            "bump_bps": bump_bps,
        }

    def full_risk_report(
        self,
        instrument: "Instrument",
        market: "MarketData",
    ) -> Dict[str, Any]:
        """Generates comprehensive risk report with all first and second order

        sensitivities.
        """
        base_res = self.pricing_engine.price(instrument, market)
        delta = self.calculate_delta(instrument, market)
        gamma = self.calculate_gamma(instrument, market)
        vega = self.calculate_vega(instrument, market)
        theta = self.calculate_theta(instrument, market)
        rho = self.calculate_rho(instrument, market)
        dv01_data = self.calculate_dv01(instrument, market)

        report = {
            "instrument_id": instrument.id,
            "instrument_type": instrument.instrument_type,
            "notional": instrument.notional,
            "currency": instrument.currency,
            "base_npv": base_res.npv,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "vega_1pct": vega * 0.01,
            "theta": theta,
            "theta_1day": theta / 365.0,
            "rho": rho,
            "dv01": dv01_data["dv01_usd"],
            "analytical_greeks": base_res.greeks,
        }
        return report
