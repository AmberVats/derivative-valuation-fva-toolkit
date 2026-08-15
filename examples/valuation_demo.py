"""Interactive valuation and pricing demo showcasing OOP double dispatch, multi-engine

revaluation, Greeks, and Fair Value Adjustments.
"""

from tabulate import tabulate
from src.instruments.options import EuropeanOption
from src.instruments.base import OptionType
from src.instruments.swaps import InterestRateSwap
from src.engines.black_scholes import BlackScholesEngine
from src.engines.monte_carlo import MonteCarloEngine
from src.engines.dcf import DiscountedCashFlowEngine
from src.market.market_data import MarketData
from src.market.bootstrap import DepositQuote, SwapQuote, YieldCurveBootstrapper


def main():
    print("=== MULTI-ENGINE REVALUATION & BENCHMARK DEMO ===")
    
    # Bootstrap curve
    deposits = [DepositQuote(0.5, 0.045), DepositQuote(1.0, 0.048)]
    swaps = [SwapQuote(2.0, 0.050), SwapQuote(5.0, 0.052), SwapQuote(10.0, 0.055)]
    curve = YieldCurveBootstrapper().bootstrap(deposits, swaps, "USD_DISCOUNT")
    
    market = MarketData(
        as_of_date="2026-08-15",
        spots={"TECH": 100.0},
        risk_free_rate=0.05,
        dividend_yields={"TECH": 0.01},
        flat_volatilities={"TECH": 0.25},
        yield_curves={"DISCOUNT": curve},
    )

    # 1. Price the same option with two different engines: Black-Scholes vs Monte Carlo
    option = EuropeanOption("TECH", strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    bs_engine = BlackScholesEngine()
    mc_engine = MonteCarloEngine(num_paths=200_000, antithetic=True, seed=42)

    bs_res = bs_engine.price(option, market)
    mc_res = mc_engine.price(option, market)

    print("\n1. Black-Scholes vs Monte Carlo Comparison on European Call (S=100, K=100, T=1y):")
    comp_table = [
        ["Black-Scholes (Analytical)", f"${bs_res.npv:.4f}", "Exact", "N/A"],
        [
            "Monte Carlo (200k paths)",
            f"${mc_res.npv:.4f}",
            f"SE: ${mc_res.details['standard_error']:.4f}",
            f"[{mc_res.details['ci_99_lower']:.4f}, {mc_res.details['ci_99_upper']:.4f}]",
        ],
    ]
    print(tabulate(comp_table, headers=["Engine", "NPV", "Precision / Error", "99% Confidence Interval"], tablefmt="github"))

    # 2. Interest Rate Swap DCF
    swap = InterestRateSwap(fixed_rate=0.052, tenor_years=5.0, payment_frequency=2.0, receive_fixed=True, notional=10_000_000.0)
    dcf_engine = DiscountedCashFlowEngine()
    swap_res = dcf_engine.price(swap, market)

    print("\n2. Interest Rate Swap Valuation (5Y Semiannual Receiver Swap, $10M Notional):")
    swap_table = [
        ["Fixed Leg PV", f"${swap_res.details['fixed_leg_pv']:,.2f}"],
        ["Float Leg PV", f"${swap_res.details['float_leg_pv']:,.2f}"],
        ["Net Swap NPV", f"${swap_res.npv:,.2f}"],
        ["Par Swap Rate", f"{swap_res.details['par_swap_rate']:.4%}"],
        ["Fixed Leg DV01 (per 1bp)", f"${swap_res.details['dv01_fixed_leg']:,.2f}"],
    ]
    print(tabulate(swap_table, headers=["Metric", "Value"], tablefmt="github"))


if __name__ == "__main__":
    main()
