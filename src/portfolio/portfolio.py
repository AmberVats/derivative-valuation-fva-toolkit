"""Portfolio aggregation, multi-book valuation, and comprehensive risk & XVA

reporting.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

from src.instruments.base import Instrument
from src.engines.base import PricingEngine, PricingResult
from src.engines.black_scholes import BlackScholesEngine
from src.engines.dcf import DiscountedCashFlowEngine
from src.engines.composite import CompositePricingEngine
from src.market.market_data import MarketData
from src.adjustments.bid_offer import BidOfferReserve
from src.adjustments.fva import FundingValuationAdjustment
from src.adjustments.cva import CreditValuationAdjustment
from src.adjustments.audit import AuditTrailManager


@dataclass
class PortfolioPosition:
    """Represents a trading position within a specific Product Control book."""
    instrument: Instrument
    quantity: float = 1.0
    book_name: str = "TRADING_BOOK_1"
    trader: str = "DESK_QUANT"


class Portfolio:
    """Multi-book derivative portfolio managing batch valuations, risk aggregations,

    and fair value adjustments.
    """

    def __init__(self, name: str = "DerivativeTradingPortfolio") -> None:
        self.name = name
        self.positions: List[PortfolioPosition] = []

    def add_position(
        self,
        instrument: Instrument,
        quantity: float = 1.0,
        book_name: str = "MAIN_BOOK",
        trader: str = "QUANT_DESK",
    ) -> None:
        """Add a trade position to the portfolio."""
        self.positions.append(
            PortfolioPosition(
                instrument=instrument,
                quantity=float(quantity),
                book_name=book_name,
                trader=trader,
            )
        )

    @property
    def books(self) -> List[str]:
        """List unique book names."""
        return list(sorted({pos.book_name for pos in self.positions}))

    def get_positions_by_book(self, book_name: str) -> List[PortfolioPosition]:
        """Retrieve positions assigned to a specific book."""
        return [pos for pos in self.positions if pos.book_name == book_name]

    def evaluate(
        self,
        market_data: MarketData,
        audit_manager: Optional[AuditTrailManager] = None,
        pricing_engine_override: Optional[PricingEngine] = None,
    ) -> Dict[str, Any]:
        """Execute full portfolio valuation, risk aggregation, and fair value

        adjustments.
        """
        composite_engine = pricing_engine_override or CompositePricingEngine()

        pos_records = []
        total_npv = 0.0
        book_npvs: Dict[str, float] = defaultdict(float)
        net_greeks: Dict[str, float] = defaultdict(float)

        raw_positions_list = []

        for pos in self.positions:
            inst = pos.instrument
            qty = pos.quantity
            raw_positions_list.append((inst, qty))

            res = composite_engine.price(inst, market_data)
            pos_npv = res.npv * qty
            total_npv += pos_npv
            book_npvs[pos.book_name] += pos_npv

            # Aggregate Greeks
            for greek_name, val in res.greeks.items():
                net_greeks[greek_name] += val * qty

            pos_records.append({
                "instrument_id": inst.id,
                "type": inst.instrument_type,
                "book": pos.book_name,
                "quantity": qty,
                "notional": inst.notional * qty,
                "npv_usd": pos_npv,
                "delta": res.greeks.get("delta", 0.0) * qty,
                "gamma": res.greeks.get("gamma", 0.0) * qty,
                "vega": res.greeks.get("vega", 0.0) * qty,
                "dv01": res.greeks.get("dv01", 0.0) * qty,
            })

            if audit_manager:
                audit_manager.record_pricing(
                    instrument_id=inst.id,
                    engine_name=res.engine_name,
                    npv=pos_npv,
                    currency=inst.currency,
                    as_of_date=market_data.as_of_date,
                )

        # 2. Calculate Fair Value Adjustments
        bo_calc = BidOfferReserve()
        fva_calc = FundingValuationAdjustment()
        cva_calc = CreditValuationAdjustment()

        bo_res = bo_calc.calculate(raw_positions_list, market_data, composite_engine)
        fva_res = fva_calc.calculate(raw_positions_list, market_data, composite_engine)
        cva_res = cva_calc.calculate(raw_positions_list, market_data, composite_engine)

        if audit_manager:
            audit_manager.record_adjustment(bo_res)
            audit_manager.record_adjustment(fva_res)
            audit_manager.record_adjustment(cva_res)

        total_adjustments = bo_res.amount_usd + fva_res.amount_usd + cva_res.amount_usd
        fair_value_net = total_npv - total_adjustments

        summary = {
            "portfolio_name": self.name,
            "as_of_date": market_data.as_of_date,
            "num_positions": len(self.positions),
            "num_books": len(self.books),
            "base_npv_usd": total_npv,
            "bid_offer_reserve_usd": bo_res.amount_usd,
            "fva_usd": fva_res.amount_usd,
            "cva_usd": cva_res.amount_usd,
            "total_adjustments_usd": total_adjustments,
            "net_fair_value_usd": fair_value_net,
            "net_greeks": dict(net_greeks),
            "book_npvs": dict(book_npvs),
            "positions_table": pd.DataFrame(pos_records),
            "adjustments": {
                "bid_offer": bo_res,
                "fva": fva_res,
                "cva": cva_res,
            },
        }
        return summary
