"""Abstract Base Class for financial instruments applying the Visitor / Double

Dispatch pattern.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from src.engines.base import PricingEngine, PricingResult
    from src.market.market_data import MarketData


class OptionType(str, Enum):
    """Option exercise classification."""
    CALL = "CALL"
    PUT = "PUT"


class PositionSide(str, Enum):
    """Position direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class Instrument(ABC):
    """Abstract Base Class for all financial derivative instruments.

    Implements the Visitor / Double Dispatch pattern to decouple instrument
    specifications from pricing engine implementations.
    """

    def __init__(
        self,
        instrument_id: Optional[str] = None,
        notional: float = 1.0,
        currency: str = "USD",
        position_side: PositionSide = PositionSide.LONG,
    ) -> None:
        self.id = instrument_id or str(uuid.uuid4())[:8]
        self.notional = float(notional)
        self.currency = currency
        self.position_side = position_side

    @property
    def side_multiplier(self) -> float:
        """Returns +1.0 for LONG positions, -1.0 for SHORT positions."""
        return 1.0 if self.position_side == PositionSide.LONG else -1.0

    @abstractmethod
    def payoff(self, spot: float) -> float:
        """Terminal payoff function for a given underlying spot price at maturity."""
        pass

    @abstractmethod
    def accept(self, pricer: "PricingEngine", market: "MarketData") -> "PricingResult":
        """Accepts a PricingEngine visitor to evaluate NPV and risk analytics."""
        pass

    @property
    @abstractmethod
    def maturity(self) -> float:
        """Time to maturity in years (fractional)."""
        pass

    @property
    @abstractmethod
    def instrument_type(self) -> str:
        """String representation of instrument type."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id} notional={self.notional} ccy={self.currency}>"
