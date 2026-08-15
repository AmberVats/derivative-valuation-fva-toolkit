"""European vanilla options implementation."""

from typing import Optional, TYPE_CHECKING
from src.instruments.base import Instrument, OptionType, PositionSide

if TYPE_CHECKING:
    from src.engines.base import PricingEngine, PricingResult
    from src.market.market_data import MarketData


class EuropeanOption(Instrument):
    """Plain vanilla European Option on a single underlying asset."""

    def __init__(
        self,
        underlying: str,
        strike: float,
        expiry: float,
        option_type: OptionType = OptionType.CALL,
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
        if strike <= 0:
            raise ValueError(f"Strike price must be positive, got {strike}")
        if expiry <= 0:
            raise ValueError(f"Expiry must be positive, got {expiry}")

        self.underlying = underlying
        self.strike = float(strike)
        self.expiry = float(expiry)
        self.option_type = option_type

    @property
    def maturity(self) -> float:
        return self.expiry

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        return self.option_type == OptionType.PUT

    @property
    def instrument_type(self) -> str:
        return f"European_{self.option_type.value}"

    def payoff(self, spot: float) -> float:
        """Terminal payoff per unit of notional at expiry:

        Call: max(S - K, 0)
        Put:  max(K - S, 0)
        """
        if self.is_call:
            raw_payoff = max(spot - self.strike, 0.0)
        else:
            raw_payoff = max(self.strike - spot, 0.0)
        return self.side_multiplier * self.notional * raw_payoff

    def accept(self, pricer: "PricingEngine", market: "MarketData") -> "PricingResult":
        """Double dispatch to pricer."""
        return pricer.visit_european_option(self, market)

    def __repr__(self) -> str:
        return (
            f"<EuropeanOption {self.option_type.value} on {self.underlying} "
            f"K={self.strike} T={self.expiry} side={self.position_side.value}>"
        )
