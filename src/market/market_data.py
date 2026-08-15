"""Market Data container and state provider for pricing and risk engines."""

from copy import deepcopy
import math
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.market.curve import YieldCurve


class MarketData:
    """Encapsulates observable market parameters (spots, curves, volatilities,

    spreads) as of a specific valuation date.
    """

    def __init__(
        self,
        as_of_date: str = "2026-08-15",
        spots: Optional[Dict[str, float]] = None,
        risk_free_rate: float = 0.05,
        dividend_yields: Optional[Dict[str, float]] = None,
        flat_volatilities: Optional[Dict[str, float]] = None,
        yield_curves: Optional[Dict[str, "YieldCurve"]] = None,
        funding_spread_bps: float = 50.0,
        cds_spread_bps: float = 100.0,
        recovery_rate: float = 0.40,
        bid_ask_spreads_bps: Optional[Dict[str, float]] = None,
    ) -> None:
        self.as_of_date = as_of_date
        self.spots = spots or {}
        self.risk_free_rate = float(risk_free_rate)
        self.dividend_yields = dividend_yields or {}
        self.flat_volatilities = flat_volatilities or {}
        self.yield_curves: Dict[str, "YieldCurve"] = yield_curves or {}
        self.funding_spread_bps = float(funding_spread_bps)
        self.cds_spread_bps = float(cds_spread_bps)
        self.recovery_rate = float(recovery_rate)
        self.bid_ask_spreads_bps = bid_ask_spreads_bps or {}

    def get_spot(self, underlying: str) -> float:
        """Retrieve spot price for underlying asset."""
        if underlying not in self.spots:
            raise KeyError(f"Spot price not found for underlying: {underlying}")
        return float(self.spots[underlying])

    def get_dividend_yield(self, underlying: str) -> float:
        """Retrieve continuous dividend yield or borrow/repo cost for underlying."""
        return float(self.dividend_yields.get(underlying, 0.0))

    def get_volatility(
        self,
        underlying: str,
        strike: Optional[float] = None,
        expiry: Optional[float] = None,
    ) -> float:
        """Retrieve implied volatility for underlying asset."""
        if underlying in self.flat_volatilities:
            return float(self.flat_volatilities[underlying])
        raise KeyError(f"No volatility data registered for underlying: {underlying}")

    def get_discount_factor(self, t: float, curve_name: Optional[str] = None) -> float:
        """Get discount factor P(0, t) for tenor t."""
        if t <= 0.0:
            return 1.0
        if curve_name and curve_name in self.yield_curves:
            return self.yield_curves[curve_name].discount_factor(t)
        if "DISCOUNT" in self.yield_curves:
            return self.yield_curves["DISCOUNT"].discount_factor(t)
        # Default fallback to continuous flat risk-free rate compounding
        return math.exp(-self.risk_free_rate * t)

    def get_zero_rate(self, t: float, curve_name: Optional[str] = None) -> float:
        """Get continuously compounded zero rate for tenor t."""
        if t <= 0.0:
            return self.risk_free_rate
        if curve_name and curve_name in self.yield_curves:
            return self.yield_curves[curve_name].zero_rate(t)
        if "DISCOUNT" in self.yield_curves:
            return self.yield_curves["DISCOUNT"].zero_rate(t)
        return self.risk_free_rate

    def get_funding_spread(self) -> float:
        """Net funding spread in decimal (e.g., 50 bps -> 0.0050)."""
        return self.funding_spread_bps / 10000.0

    def get_cds_spread(self) -> float:
        """Counterparty CDS spread in decimal (e.g., 100 bps -> 0.0100)."""
        return self.cds_spread_bps / 10000.0

    def get_bid_ask_spread(self, asset: str, default_bps: float = 10.0) -> float:
        """Bid-ask spread in decimal for the given asset."""
        bps = self.bid_ask_spreads_bps.get(asset, default_bps)
        return bps / 10000.0

    def bump_spot(self, underlying: str, bump_pct: float) -> "MarketData":
        """Create a new MarketData copy with bumped spot price (e.g.

        bump_pct=0.01 for +1%).
        """
        md = deepcopy(self)
        if underlying in md.spots:
            md.spots[underlying] = md.spots[underlying] * (1.0 + bump_pct)
        return md

    def bump_volatility(self, underlying: str, bump_abs: float) -> "MarketData":
        """Create a new MarketData copy with bumped volatility (e.g.

        bump_abs=0.01 for +1 vol point).
        """
        md = deepcopy(self)
        if underlying in md.flat_volatilities:
            md.flat_volatilities[underlying] = max(0.0001, md.flat_volatilities[underlying] + bump_abs)
        return md

    def bump_rate(self, bump_bps: float, curve_name: Optional[str] = None) -> "MarketData":
        """Create a new MarketData copy with parallel yield curve / rate bump (e.g.

        +1 bp = +0.0001).
        """
        md = deepcopy(self)
        rate_shift = bump_bps / 10000.0
        md.risk_free_rate += rate_shift
        for name, curve in md.yield_curves.items():
            if curve_name is None or name == curve_name:
                md.yield_curves[name] = curve.bump(bump_bps)
        return md
