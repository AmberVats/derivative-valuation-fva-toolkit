"""Interest Rate Swap instrument implementation."""

from typing import List, Optional, TYPE_CHECKING
from src.instruments.base import Instrument, PositionSide

if TYPE_CHECKING:
    from src.engines.base import PricingEngine, PricingResult
    from src.market.market_data import MarketData


class InterestRateSwap(Instrument):
    """Vanilla Fixed-for-Floating Interest Rate Swap (IRS).

    Parameters
    ----------
    fixed_rate : float
        Annualized fixed coupon rate (e.g., 0.045 for 4.5%).
    tenor_years : float
        Swap maturity in years (e.g. 5.0).
    float_spread : float
        Floating leg spread in decimal (e.g., 0.0).
    payment_frequency : float
        Number of fixed and float coupon payments per year (e.g. 2.0 = semiannual, 1.0 = annual, 4.0 = quarterly).
    receive_fixed : bool
        True for Receiver Swap (receive fixed rate, pay floating index).
        False for Payer Swap (pay fixed rate, receive floating index).
    notional : float
        Swap principal notional (e.g., 10,000,000.0).
    currency : str
        Currency code (e.g. 'USD').
    """

    def __init__(
        self,
        fixed_rate: float,
        tenor_years: float,
        float_spread: float = 0.0,
        payment_frequency: float = 2.0,
        receive_fixed: bool = True,
        instrument_id: Optional[str] = None,
        notional: float = 1_000_000.0,
        currency: str = "USD",
    ) -> None:
        position_side = PositionSide.LONG if receive_fixed else PositionSide.SHORT
        super().__init__(
            instrument_id=instrument_id,
            notional=notional,
            currency=currency,
            position_side=position_side,
        )
        if tenor_years <= 0:
            raise ValueError(f"Tenor must be positive, got {tenor_years}")
        if payment_frequency <= 0:
            raise ValueError(f"Payment frequency must be positive, got {payment_frequency}")

        self.fixed_rate = float(fixed_rate)
        self._tenor_years = float(tenor_years)
        self.float_spread = float(float_spread)
        self.payment_frequency = float(payment_frequency)
        self.receive_fixed = receive_fixed

    @property
    def maturity(self) -> float:
        return self._tenor_years

    @property
    def instrument_type(self) -> str:
        direction = "Receiver" if self.receive_fixed else "Payer"
        return f"InterestRateSwap_{direction}"

    @property
    def payment_schedule(self) -> List[float]:
        """Generates the list of coupon payment dates (in years)."""
        dt = 1.0 / self.payment_frequency
        n_payments = int(round(self._tenor_years * self.payment_frequency))
        return [round(i * dt, 6) for i in range(1, n_payments + 1)]

    def payoff(self, spot: float) -> float:
        """Single period approximate net cash flow under spot rate."""
        dt = 1.0 / self.payment_frequency
        if self.receive_fixed:
            net_rate = self.fixed_rate - (spot + self.float_spread)
        else:
            net_rate = (spot + self.float_spread) - self.fixed_rate
        return self.notional * net_rate * dt

    def accept(self, pricer: "PricingEngine", market: "MarketData") -> "PricingResult":
        """Double dispatch to pricer."""
        return pricer.visit_interest_rate_swap(self, market)

    def __repr__(self) -> str:
        role = "ReceiveFixed" if self.receive_fixed else "PayFixed"
        return (
            f"<InterestRateSwap {role} rate={self.fixed_rate:.4%} "
            f"T={self._tenor_years}y freq={self.payment_frequency}/yr "
            f"notional={self.notional:,.0f} {self.currency}>"
        )
