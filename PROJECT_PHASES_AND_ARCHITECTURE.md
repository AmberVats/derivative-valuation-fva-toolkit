# Comprehensive Project Guide: Derivative Valuation & Fair Value Adjustment (XVA) Toolkit

### Product Control Analytics & Quantitative Valuation Architecture
**Author:** Amber Vats  
**Repository:** [https://github.com/AmberVats/derivative-valuation-fva-toolkit](https://github.com/AmberVats/derivative-valuation-fva-toolkit)  
**Standard Compliance:** IFRS 13 (Fair Value Measurement) & Basel Committee Prudent Valuation (PVA)

---

## 1. Project Background & Business Context

In institutional investment banks like HSBC, the **Product Control Analytics & Valuations** division is tasked with independent price verification (IPV), valuation methodology governance, and daily balance sheet fair value adjustments.

Trading desks capture derivatives at unadjusted mid-market prices. However, in reality:
1. **Closing out positions incurs bid-ask liquidity costs** (Bid-Offer Reserve).
2. **Uncollateralized trades consume funding liquidity** over their lifetime (Funding Valuation Adjustment - FVA).
3. **Counterparties carry risk of default** before maturity (Credit Valuation Adjustment - CVA).

This toolkit provides an autonomous, decoupled valuation and risk engine that:
- Prices multi-asset vanilla and non-linear derivatives across equity, FX, and interest rate asset classes.
- Boots multi-pillar discount curves from cash deposits and par swap quotes.
- Calculates first and second-order Greeks alongside interest rate curve DV01.
- Computes regulatory fair value adjustments with deterministic SHA-256 parameter hashing for auditability.

---

## 2. Detailed Breakdown of Project Phases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PROJECT ROADMAP                                 │
└─────────────────────────────────────────────────────────────────────────────┘
  Phase 1: Environment, Tooling & Dependency Architecture
     │
  Phase 2: Domain Modeling & Visitor / Double-Dispatch Pattern
     │
  Phase 3: Closed-Form Black-Scholes-Merton Analytical Engine
     │
  Phase 4: Yield Curve Bootstrapping & Log-Linear Interpolation
     │
  Phase 5: Linear Interest Rate Swaps & Multi-Curve DCF Engine
     │
  Phase 6: Vectorized Monte Carlo Simulation with Antithetic Variates
     │
  Phase 7: Central Finite-Difference Greeks & Curve DV01
     │
  Phase 8: Modular Fair Value Adjustments (Bid-Offer, FVA, CVA)
     │
  Phase 9: Multi-Book Portfolio Aggregation & Fair Value Waterfall
     │
  Phase 10: Benchmark Regression Validation & Executive HTML Reporting
```

---

### Phase 1: Environment, Tooling & Packaging Setup
- **Objective**: Establish reproducible virtual environment, pinned dependencies, CI pipeline, and package structure.
- **Key Modules**:
  - `pyproject.toml`: Modern packaging specification defining package metadata, pytest configuration, and build backend.
  - `requirements.txt`: Pinned scientific stack (`numpy`, `scipy`, `pandas`, `pyyaml`, `pytest`, `pytest-cov`, `tabulate`, `matplotlib`).
  - `.github/workflows/ci.yml`: Matrix automated testing across Python 3.10, 3.11, 3.12, and 3.13.

---

### Phase 2: Domain Modeling & The Visitor / Double-Dispatch Pattern
- **Objective**: Design an extensible OOP instrument hierarchy decoupled from pricing algorithms.
- **Key Design Pattern**: **Visitor / Double Dispatch**
  ```python
  class Instrument(ABC):
      @abstractmethod
      def accept(self, pricer: PricingEngine, market: MarketData) -> PricingResult:
          pass

  class EuropeanOption(Instrument):
      def accept(self, pricer: PricingEngine, market: MarketData) -> PricingResult:
          return pricer.visit_european_option(self, market)
  ```
- **Why this is critical for Product Control**:
  Trade representations are immutable business objects. When Independent Model Review (IMR) mandates a new pricing engine (e.g. transitioning from analytical Black-Scholes to Local Volatility PDE), **zero modifications** are made to trade capture or booking models.

---

### Phase 3: Analytical Black-Scholes Engine & Greeks Formulation
- **Objective**: Implement exact closed-form Black-Scholes-Merton valuation with continuous dividend yield $q$ and analytical partial derivatives.
- **Mathematical Formulations**:
  $$\begin{aligned}
  d_1 &= \frac{\ln(S / K) + \left(r - q + \frac{1}{2}\sigma^2\right)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T} \\
  C &= S e^{-qT} N(d_1) - K e^{-rT} N(d_2) \\
  P &= K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)
  \end{aligned}$$
- **Analytical Greeks**:
  - Delta ($\Delta$): $e^{-qT} N(d_1)$ for Call, $-e^{-qT} N(-d_1)$ for Put.
  - Gamma ($\Gamma$): $\frac{e^{-qT} N'(d_1)}{S \sigma \sqrt{T}}$.
  - Vega ($\mathcal{V}$): $S e^{-qT} \sqrt{T} N'(d_1)$.
  - Theta ($\Theta$): Time decay per year and per day.
  - Rho ($\rho$): Interest rate sensitivity per 100% and 1bp.

---

### Phase 4: Yield Curve Bootstrapping & Log-Linear Interpolation
- **Objective**: Calibrate arbitrage-free discount curves from market cash deposits and par swap quotes.
- **Interpolation Methodology**: **Log-Linear on Discount Factors**
  $$P(0, t) = P(0, t_i) \left(\frac{P(0, t_{i+1})}{P(0, t_i)}\right)^{\frac{t - t_i}{t_{i+1} - t_i}}$$
  - Corresponds to piecewise constant instantaneous forward rates:
    $$f(t) = \frac{\ln P(0, t_i) - \ln P(0, t_{i+1})}{t_{i+1} - t_i} > 0$$
  - Guarantees strictly positive forward rates and eliminates artificial cubic oscillations.
- **Root Solving**: Utilizes SciPy's `brentq` to reprice par swaps to within $< 10^{-10}$ precision.

---

### Phase 5: Linear Interest Rate Swaps & Multi-Curve DCF Engine
- **Objective**: Price vanilla Fixed-for-Floating Interest Rate Swaps (IRS) with explicit cash flow scheduling.
- **Formulation**:
  $$\text{Fixed Leg PV} = \sum_{i=1}^n N \cdot R_{\text{fixed}} \cdot \tau_i \cdot P(0, t_i)$$
  $$\text{Float Leg PV} = \sum_{j=1}^m N \cdot (L(t_{j-1}, t_j) + s) \cdot \tau_j \cdot P(0, t_j)$$
  $$\text{Par Swap Rate } S_{\text{par}} = \frac{P(0, t_0) - P(0, t_n)}{\sum_{i=1}^n \tau_i P(0, t_i)}$$
- **Verification**: Asserts par swap NPV equals $0.00$ within floating point tolerance.

---

### Phase 6: Stochastic Monte Carlo Simulation & Variance Reduction
- **Objective**: Build a high-performance vectorized stochastic simulator with statistical confidence bounds.
- **Variance Reduction**: **Antithetic Variates Sampling**
  $$S_T^{(1)} = S_0 \exp\left(\mu T + \sigma \sqrt{T} Z\right), \quad S_T^{(2)} = S_0 \exp\left(\mu T - \sigma \sqrt{T} Z\right)$$
  $$Y_i = \frac{1}{2}\left[\text{Payoff}(S_T^{(1)}) + \text{Payoff}(S_T^{(2)})\right]$$
- **Convergence**: Computes standard error $\text{SE} = e^{-rT} \frac{s_Y}{\sqrt{N}}$ and verifies that Black-Scholes analytical prices reside within the 99% Confidence Interval ($\pm 2.58 \cdot \text{SE}$).

---

### Phase 7: Central Finite-Difference Greeks & Curve DV01
- **Objective**: Universal, model-agnostic risk engine using central difference approximations:
  $$\Delta = \frac{V(S + h) - V(S - h)}{2h}, \quad \Gamma = \frac{V(S + h) - 2V(S) + V(S - h)}{h^2}$$
- **Parallel & Key Rate DV01**:
  Evaluates 1 basis point (0.0001) parallel curve bumps:
  $$\text{DV01} = V(r) - V(r + 1\text{bp})$$

---

### Phase 8: Fair Value Adjustments (XVA & Reserves) Framework
- **1. Bid-Offer Reserve (Closeout Cost)**:
  $$\text{Reserve}_k = \frac{1}{2} \cdot \left|\text{Net Delta}_k \cdot S_k\right| \cdot \text{Spread}_k$$
  Demonstrates netting benefit across trading books.
- **2. Funding Valuation Adjustment (FVA)**:
  $$\text{FVA} = \sum_{m=1}^M \text{EE}^+(t_m) \cdot s_F \cdot P(0, t_m) \cdot \Delta t_m$$
  Integrated across discrete forward simulation buckets.
- **3. Credit Valuation Adjustment (CVA)**:
  $$\text{CVA} = (1 - R) \sum_{m=1}^M \text{EE}^+(t_m) \cdot \text{PD}(t_{m-1}, t_m) \cdot P(0, t_m)$$
  With Poisson hazard rate $\lambda = \frac{s_{\text{CDS}}}{1 - R}$ and marginal $\text{PD} = e^{-\lambda t_{m-1}} - e^{-\lambda t_m}$.
- **4. Deterministic SHA-256 Audit Hashing**:
  Every adjustment generates an immutable hash of its input parameters for model governance.

---

### Phase 9: Multi-Book Portfolio Aggregation & Accounting Waterfall
- Aggregates positions across desks (`EQUITY_DERIVATIVES`, `MACRO_LINEAR`, `RATES_DERIVATIVES`).
- Produces the executive **Balance Sheet Fair Value Waterfall**:
  $$\text{Gross Mid-Market NPV} - \text{Bid-Offer Reserve} - \text{FVA} - \text{CVA} = \mathbf{\text{Net Fair Value}}$$

---

### Phase 10: Validation Suite, Textbook Regression & HTML Reporting
- **1,757 automated tests** testing edge cases, boundary conditions, and textbook worked examples.
- **Standalone HTML Report Generator** (`src/report/html_report.py`) producing executive dashboards with KPI cards, risk matrices, and governance logs.

---

## 3. Benchmark Verification Summary Table

| Benchmark Case | Published Literature / Formula | Target | Model Output | Discrepancy | Status |
|---|---|---|---|---|---|
| **Hull Ch 15 European Call** | John Hull 10th Ed Section 15.6 | `4.7594` | `4.7594` | $2.24 \times 10^{-5}$ | **PASS** |
| **Hull Ch 15 European Put** | John Hull 10th Ed Section 15.6 | `0.8080` | `0.8086` | $5.99 \times 10^{-4}$ | **PASS** |
| **Hull Ch 15 Call Delta** | $N(d_1)$ | `0.7791` | `0.7791` | $3.13 \times 10^{-5}$ | **PASS** |
| **Hull Ch 15 Vega** | $S \sqrt{T} N'(d_1)$ | `8.8134` | `8.8134` | $1.51 \times 10^{-5}$ | **PASS** |
| **Put-Call Parity Identity** | $C - P = S_0 e^{-qT} - K e^{-rT}$ | `3.950823` | `3.950823` | $< 10^{-14}$ | **PASS** |
| **Monte Carlo Convergence** | 200k paths with Antithetic sampling | `4.7594 (BS)` | `4.7679 (MC)` | $0.0085$ ($\le 2.58 \cdot \text{SE}$) | **PASS** |
| **5Y Swap Par Repricing** | Bootstrapped SOFR curve | `0.052000` | `0.052000` | $< 10^{-14}$ | **PASS** |
| **Finite Difference $\Delta$** | Central difference $\Delta S = 0.1\%$ | `0.779131` | `0.779129` | $2.25 \times 10^{-6}$ | **PASS** |
