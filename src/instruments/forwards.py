"""Forward contracts implementation."""

from typing import Optional, TYPE_CHECKING
from src.instruments.base import Instrument, PositionSide

if TYPE_CHECKING:
    from src.engines.base import PricingEngine, PricingResult
    from src.market.market_data import MarketData


class Forward(Instrument):
    """Linear forward contract to buy/sell an underlying asset at strike_price on

    maturity date.
    """

    def __init__(
        self,
        underlying: str,
        strike_price: float,
        maturity: float,
        instrument_id: Optional[str] = None,
        notional: float = 1.0,
        currency: str = "USD",
        position_side: PositionSide = PositionSide.LONG,
    ) -> None:
        super().__init__(
            instrument_id=instrument_id,
            notional=notional,
            currency=currency,
            position_side=position_side,
        )
        if strike_price < 0:
            raise ValueError(f"Strike price cannot be negative, got {strike_price}")
        if maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {maturity}")

        self.underlying = underlying
        self.strike_price = float(strike_price)
        self._maturity = float(maturity)

    @property
    def maturity(self) -> float:
        return self._maturity

    @property
    def instrument_type(self) -> str:
        return "Forward"

    def payoff(self, spot: float) -> float:
        """Terminal payoff: S - K for LONG, K - S for SHORT."""
        return self.side_multiplier * self.notional * (spot - self.strike_price)

    def accept(self, pricer: "PricingEngine", market: "MarketData") -> "PricingResult":
        """Double dispatch to pricer."""
        return pricer.visit_forward(self, market)

    def __repr__(self) -> str:
        return (
            f"<Forward on {self.underlying} K={self.strike_price} "
            f"T={self.maturity} side={self.position_side.value}>"
        )
