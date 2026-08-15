"""Command-line interface (CLI) for derivative valuation, yield curve bootstrapping,

and Fair Value Adjustment audit reporting.
"""

import argparse
import sys
from tabulate import tabulate

from src.instruments.base import OptionType
from src.instruments.options import EuropeanOption
from src.instruments.forwards import Forward
from src.instruments.swaps import InterestRateSwap
from src.engines.composite import CompositePricingEngine
from src.market.market_data import MarketData
from src.market.bootstrap import DepositQuote, SwapQuote, YieldCurveBootstrapper
from src.portfolio.portfolio import Portfolio
from src.adjustments.audit import AuditTrailManager


def run_full_valuation_demo(export_json_path: str = None) -> None:
    """Runs a complete end-to-end derivative valuation and FVA workflow."""
    print("=" * 80)
    print("  DERIVATIVE VALUATION & FAIR VALUE ADJUSTMENT (XVA) TOOLKIT")
    print("  Product Control Analytics & Quantitative Valuation Framework")
    print("=" * 80)

    # 1. Curve Bootstrapping
    print("\n[Step 1/5] Bootstrapping Multi-Pillar USD SOFR Yield Curve...")
    deposits = [
        DepositQuote(tenor_years=0.0833, rate=0.0410),  # 1M
        DepositQuote(tenor_years=0.25, rate=0.0425),    # 3M
        DepositQuote(tenor_years=0.50, rate=0.0440),    # 6M
        DepositQuote(tenor_years=1.00, rate=0.0460),    # 1Y
    ]
    swaps = [
        SwapQuote(tenor_years=2.0, par_rate=0.0470, fixed_frequency=2.0),
        SwapQuote(tenor_years=3.0, par_rate=0.0485, fixed_frequency=2.0),
        SwapQuote(tenor_years=5.0, par_rate=0.0500, fixed_frequency=2.0),
        SwapQuote(tenor_years=7.0, par_rate=0.0515, fixed_frequency=2.0),
        SwapQuote(tenor_years=10.0, par_rate=0.0530, fixed_frequency=2.0),
        SwapQuote(tenor_years=30.0, par_rate=0.0550, fixed_frequency=2.0),
    ]
    bootstrapper = YieldCurveBootstrapper()
    usd_curve = bootstrapper.bootstrap(deposits, swaps, curve_name="USD_SOFR_DISCOUNT")

    curve_rows = [
        [f"{t:.2f}y", f"{usd_curve.discount_factor(t):.6f}", f"{usd_curve.zero_rate(t):.4%}"]
        for t in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    ]
    print(tabulate(curve_rows, headers=["Pillar Tenor", "Discount Factor P(0, t)", "Zero Rate (Cont.)"], tablefmt="github"))

    # 2. Market Data Setup
    print("\n[Step 2/5] Initializing Market State Provider...")
    market = MarketData(
        as_of_date="2026-08-15",
        spots={
            "AAPL": 220.0,
            "NVDA": 125.0,
            "SPX": 5500.0,
            "EURUSD": 1.0850,
        },
        risk_free_rate=0.045,
        dividend_yields={
            "AAPL": 0.005,
            "NVDA": 0.001,
            "SPX": 0.015,
            "EURUSD": 0.030,
        },
        flat_volatilities={
            "AAPL": 0.24,
            "NVDA": 0.38,
            "SPX": 0.16,
            "EURUSD": 0.08,
        },
        yield_curves={"DISCOUNT": usd_curve},
        funding_spread_bps=45.0,
        cds_spread_bps=120.0,
        recovery_rate=0.40,
        bid_ask_spreads_bps={"AAPL": 5.0, "NVDA": 8.0, "SPX": 2.0, "EURUSD": 1.5},
    )
    print("Market Data successfully loaded.")

    # 3. Portfolio Construction
    print("\n[Step 3/5] Constructing Trading Books & Derivatives Portfolio...")
    portfolio = Portfolio(name="Global_PC_Derivatives_Book")

    # Equity Options Book
    portfolio.add_position(
        EuropeanOption("AAPL", strike=220.0, expiry=0.5, option_type=OptionType.CALL, notional=1000.0),
        quantity=50.0,
        book_name="EQUITY_DERIVATIVES",
        trader="J_DOE",
    )
    portfolio.add_position(
        EuropeanOption("AAPL", strike=210.0, expiry=0.5, option_type=OptionType.PUT, notional=1000.0),
        quantity=-20.0,
        book_name="EQUITY_DERIVATIVES",
        trader="J_DOE",
    )
    portfolio.add_position(
        EuropeanOption("NVDA", strike=130.0, expiry=1.0, option_type=OptionType.CALL, notional=2000.0),
        quantity=25.0,
        book_name="EQUITY_DERIVATIVES",
        trader="M_SMITH",
    )

    # Rates & Linear Book
    portfolio.add_position(
        Forward("SPX", strike_price=5550.0, maturity=1.0, notional=500.0),
        quantity=10.0,
        book_name="MACRO_LINEAR",
        trader="A_KUMAR",
    )
    portfolio.add_position(
        InterestRateSwap(fixed_rate=0.052, tenor_years=5.0, payment_frequency=2.0, receive_fixed=True, notional=50_000_000.0),
        quantity=1.0,
        book_name="RATES_DERIVATIVES",
        trader="R_CHEN",
    )

    # 4. Valuation & Fair Value Adjustments
    print("\n[Step 4/5] Executing Valuations, Risk Sensitivities, and XVA Adjustments...")
    audit_manager = AuditTrailManager()
    summary = portfolio.evaluate(market, audit_manager=audit_manager)

    print("\n--- Positions Breakdown ---")
    pos_df = summary["positions_table"]
    print(tabulate(pos_df, headers="keys", tablefmt="github", showindex=False, floatfmt=",.2f"))

    print("\n--- Valuation Summary & Fair Value Adjustments ---")
    fva_summary = [
        ["Gross Unadjusted NPV", f"${summary['base_npv_usd']:,.2f}"],
        ["Bid-Offer Reserve (Closeout Cost)", f"-${summary['bid_offer_reserve_usd']:,.2f}"],
        ["Funding Valuation Adjustment (FVA)", f"-${summary['fva_usd']:,.2f}"],
        ["Credit Valuation Adjustment (CVA)", f"-${summary['cva_usd']:,.2f}"],
        ["Total Adjustments (Reserves)", f"-${summary['total_adjustments_usd']:,.2f}"],
        ["NET FAIR VALUE (Balance Sheet)", f"${summary['net_fair_value_usd']:,.2f}"],
    ]
    print(tabulate(fva_summary, headers=["Line Item", "Amount (USD)"], tablefmt="github"))

    # 5. Regulatory Audit Trail Output
    print("\n[Step 5/5] Regulatory Audit Trail & Governance Report:")
    print(audit_manager.export_markdown_table())

    if export_json_path:
        audit_manager.export_json(export_json_path)
        print(f"\n[Audit] Saved full JSON audit logs to: {export_json_path}")

    print("\n" + "=" * 80)
    print("  Valuation & Fair Value Adjustment pipeline completed successfully!")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derivative Valuation & Fair Value Adjustment (XVA) Toolkit"
    )
    parser.add_argument("--demo", action="store_true", help="Run full valuation demo")
    parser.add_argument("--audit-json", type=str, help="Path to export audit JSON")
    args = parser.parse_args()

    run_full_valuation_demo(export_json_path=args.audit_json)


if __name__ == "__main__":
    main()
