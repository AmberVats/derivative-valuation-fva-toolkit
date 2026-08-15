"""Composite pricing engine routing instruments to the appropriate specialist

engine.
"""

from typing import Optional, TYPE_CHECKING
from src.engines.base import PricingEngine, PricingResult
from src.engines.black_scholes import BlackScholesEngine
from src.engines.dcf import DiscountedCashFlowEngine
from src.engines.monte_carlo import MonteCarloEngine

if TYPE_CHECKING:
    from src.instruments.base import Instrument
    from src.instruments.options import EuropeanOption
    from src.instruments.forwards import Forward
    from src.instruments.swaps import InterestRateSwap
    from src.market.market_data import MarketData


class CompositePricingEngine(PricingEngine):
    """Unified multi-asset pricing engine routing non-linear equity/FX options

    to Black-Scholes / Monte Carlo and interest rate swaps/bonds to DCF.
    """

    def __init__(
        self,
        options_engine: Optional[PricingEngine] = None,
        rates_engine: Optional[PricingEngine] = None,
    ) -> None:
        self.options_engine = options_engine or BlackScholesEngine()
        self.rates_engine = rates_engine or DiscountedCashFlowEngine()

    @property
    def engine_name(self) -> str:
        return f"CompositeEngine({self.options_engine.engine_name}+{self.rates_engine.engine_name})"

    def visit_european_option(
        self, option: "EuropeanOption", market: "MarketData"
    ) -> PricingResult:
        return self.options_engine.visit_european_option(option, market)

    def visit_forward(
        self, forward: "Forward", market: "MarketData"
    ) -> PricingResult:
        return self.options_engine.visit_forward(forward, market)

    def visit_interest_rate_swap(
        self, swap: "InterestRateSwap", market: "MarketData"
    ) -> PricingResult:
        return self.rates_engine.visit_interest_rate_swap(swap, market)
