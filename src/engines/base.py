"""Base interface and return structures for Pricing Engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.instruments.base import Instrument
    from src.instruments.options import EuropeanOption
    from src.instruments.forwards import Forward
    from src.instruments.swaps import InterestRateSwap
    from src.market.market_data import MarketData


@dataclass
class PricingResult:
    """Standardized output container for valuation and risk analytics."""
    npv: float
    currency: str = "USD"
    engine_name: str = "BaseEngine"
    as_of_date: str = "2026-08-15"
    greeks: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PricingResult(npv={self.npv:,.4f} {self.currency}, "
            f"engine='{self.engine_name}', greeks={list(self.greeks.keys())})"
        )


class PricingEngine(ABC):
    """Abstract Base Class for pricing engines implementing the Visitor pattern."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Identifying name of the engine."""
        pass

    def price(self, instrument: "Instrument", market: "MarketData") -> PricingResult:
        """Universal entry point delegating to double-dispatch via instrument.accept."""
        return instrument.accept(self, market)

    @abstractmethod
    def visit_european_option(
        self, option: "EuropeanOption", market: "MarketData"
    ) -> PricingResult:
        """Calculate NPV and analytics for European vanilla options."""
        pass

    @abstractmethod
    def visit_forward(
        self, forward: "Forward", market: "MarketData"
    ) -> PricingResult:
        """Calculate NPV and analytics for Forward contracts."""
        pass

    @abstractmethod
    def visit_interest_rate_swap(
        self, swap: "InterestRateSwap", market: "MarketData"
    ) -> PricingResult:
        """Calculate NPV and cash flows for Interest Rate Swaps."""
        pass
