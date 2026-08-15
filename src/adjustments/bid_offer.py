"""Prudent Valuation & Fair Value Bid-Offer Reserve calculation engine."""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
from src.adjustments.base import AdjustmentResult, FairValueAdjustment

if TYPE_CHECKING:
    from src.instruments.base import Instrument
    from src.engines.base import PricingEngine
    from src.market.market_data import MarketData


class BidOfferReserve(FairValueAdjustment):
    """Calculates the Fair Value Bid-Offer Reserve (Closeout Cost) across a

    portfolio.

    Methodology:
    Estimates the expected cost of liquidating / unwinding net market risk exposure
    at prevailing market bid-offer spreads:
        Reserve_k = 0.5 * |Net_Exposure_k| * Bid_Ask_Spread_k
    where Net_Exposure for equities/FX is Net Delta in currency, and for interest rates
    is Net DV01.
    """

    def __init__(self, methodology_version: str = "1.0.0") -> None:
        super().__init__(methodology_version=methodology_version)

    @property
    def name(self) -> str:
        return "BidOfferReserve"

    @property
    def description(self) -> str:
        return (
            "Prudent Valuation Bid-Offer closeout reserve on net portfolio market risk exposures "
            "evaluated at half the asset-specific bid-ask spread."
        )

    def calculate(
        self,
        positions: Sequence[Tuple["Instrument", float]],
        market_data: "MarketData",
        pricing_engine: "PricingEngine",
        custom_spreads_bps: Optional[Dict[str, float]] = None,
    ) -> AdjustmentResult:
        """Calculates portfolio-level Bid-Offer Reserve.

        Parameters
        ----------
        positions : Sequence[Tuple[Instrument, float]]
            List of (instrument, quantity/multiplier) tuples.
        market_data : MarketData
            Market parameters containing spot prices and spreads.
        pricing_engine : PricingEngine
            Pricing engine to evaluate position Greeks and sensitivities.
        custom_spreads_bps : Optional[Dict[str, float]]
            Overrides for asset bid-ask spreads in basis points.

        Returns
        -------
        AdjustmentResult
            Calculated reserve amount and detailed netting breakdown.
        """
        asset_deltas: Dict[str, float] = defaultdict(float)
        asset_gross_deltas: Dict[str, float] = defaultdict(float)
        asset_notionals: Dict[str, float] = defaultdict(float)
        position_details: List[Dict[str, Any]] = []

        total_reserve = 0.0
        gross_reserve = 0.0

        for instrument, qty in positions:
            res = pricing_engine.price(instrument, market_data)
            underlying = getattr(instrument, "underlying", "RATES")
            
            # Extract delta or sensitivity
            if "delta" in res.greeks:
                pos_delta = res.greeks["delta"] * qty
            elif "dv01" in res.greeks:
                pos_delta = res.greeks["dv01"] * qty
            else:
                pos_delta = instrument.notional * qty

            asset_deltas[underlying] += pos_delta
            asset_gross_deltas[underlying] += abs(pos_delta)
            asset_notionals[underlying] += abs(instrument.notional * qty)

            position_details.append({
                "instrument_id": instrument.id,
                "instrument_type": instrument.instrument_type,
                "underlying": underlying,
                "quantity": qty,
                "npv": res.npv * qty,
                "delta": pos_delta,
            })

        breakdown_by_asset = {}
        spreads_used = {}

        for asset, net_delta in asset_deltas.items():
            if custom_spreads_bps and asset in custom_spreads_bps:
                spread = custom_spreads_bps[asset] / 10000.0
            else:
                spread = market_data.get_bid_ask_spread(asset, default_bps=10.0)

            spreads_used[asset] = spread * 10000.0  # bps

            if asset in market_data.spots:
                spot = market_data.get_spot(asset)
                # Dollar exposure = Delta * Spot
                net_dollar_exposure = net_delta * spot
                gross_dollar_exposure = asset_gross_deltas[asset] * spot
            else:
                # Interest rate or generic exposure
                net_dollar_exposure = net_delta
                gross_dollar_exposure = asset_gross_deltas[asset]

            # Reserve = 0.5 * |Exposure| * spread
            net_res = 0.5 * abs(net_dollar_exposure) * spread
            gross_res = 0.5 * abs(gross_dollar_exposure) * spread

            total_reserve += net_res
            gross_reserve += gross_res

            breakdown_by_asset[asset] = {
                "net_delta": net_delta,
                "gross_delta": asset_gross_deltas[asset],
                "net_dollar_exposure": net_dollar_exposure,
                "bid_ask_spread_bps": spread * 10000.0,
                "net_reserve_usd": net_res,
                "gross_reserve_usd": gross_res,
                "netting_benefit_usd": gross_res - net_res,
            }

        parameters = {
            "num_positions": len(positions),
            "num_assets": len(asset_deltas),
            "spreads_bps": spreads_used,
            "gross_reserve_usd": gross_reserve,
            "netting_benefit_usd": gross_reserve - total_reserve,
        }

        breakdown = {
            "by_asset": breakdown_by_asset,
            "positions": position_details,
        }

        return AdjustmentResult(
            adjustment_name=self.name,
            amount_usd=total_reserve,
            methodology_version=self.methodology_version,
            as_of_date=market_data.as_of_date,
            parameters=parameters,
            breakdown=breakdown,
            notes="Calculated per Basel / EBA Prudent Valuation closeout methodology.",
        )
