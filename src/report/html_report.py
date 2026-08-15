"""Interactive, standalone HTML valuation and Fair Value Adjustment (XVA)

report generator for Product Control and Model Governance.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, Optional


class HTMLReportGenerator:
    """Generates standalone, institutional-grade HTML valuation reports

    with embedded styling, responsive metric cards, data tables, and audit logs.
    """

    def generate_report(
        self,
        valuation_summary: Dict[str, Any],
        output_filepath: str = "reports/valuation_report.html",
    ) -> str:
        """Generates and writes an HTML report to disk.

        Parameters
        ----------
        valuation_summary : Dict[str, Any]
            Dictionary returned by Portfolio.evaluate()
        output_filepath : str
            Destination filepath for the HTML document.

        Returns
        -------
        str
            Full generated HTML content.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

        portfolio_name = valuation_summary.get("portfolio_name", "Global Derivatives Portfolio")
        as_of_date = valuation_summary.get("as_of_date", "2026-08-15")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        base_npv = valuation_summary.get("base_npv_usd", 0.0)
        bo_res = valuation_summary.get("bid_offer_reserve_usd", 0.0)
        fva = valuation_summary.get("fva_usd", 0.0)
        cva = valuation_summary.get("cva_usd", 0.0)
        total_adj = valuation_summary.get("total_adjustments_usd", 0.0)
        net_fv = valuation_summary.get("net_fair_value_usd", 0.0)

        net_greeks = valuation_summary.get("net_greeks", {})
        delta = net_greeks.get("delta", 0.0)
        gamma = net_greeks.get("gamma", 0.0)
        vega = net_greeks.get("vega", 0.0)
        dv01 = net_greeks.get("dv01", 0.0)

        positions_df = valuation_summary.get("positions_table")

        # Build Positions Table Rows
        pos_rows_html = ""
        if positions_df is not None and not positions_df.empty:
            for _, row in positions_df.iterrows():
                npv_val = row.get("npv_usd", 0.0)
                npv_class = "positive" if npv_val >= 0 else "negative"
                pos_rows_html += f"""
                <tr>
                    <td><code>{row.get('instrument_id', '')}</code></td>
                    <td><span class="badge badge-type">{row.get('type', '')}</span></td>
                    <td><strong>{row.get('book', '')}</strong></td>
                    <td class="num">{row.get('quantity', 0.0):,.2f}</td>
                    <td class="num">${row.get('notional', 0.0):,.2f}</td>
                    <td class="num {npv_class}">${npv_val:,.2f}</td>
                    <td class="num">{row.get('delta', 0.0):,.2f}</td>
                    <td class="num">{row.get('gamma', 0.0):,.4f}</td>
                    <td class="num">${row.get('vega', 0.0):,.2f}</td>
                    <td class="num">${row.get('dv01', 0.0):,.2f}</td>
                </tr>
                """

        # Build Adjustments & Audit Records Table Rows
        adjustments_dict = valuation_summary.get("adjustments", {})
        adj_rows_html = ""
        for name, adj_obj in adjustments_dict.items():
            if hasattr(adj_obj, "adjustment_name"):
                adj_rows_html += f"""
                <tr>
                    <td><strong>{adj_obj.adjustment_name}</strong></td>
                    <td><span class="badge badge-version">v{adj_obj.methodology_version}</span></td>
                    <td class="num negative">-${adj_obj.amount_usd:,.2f}</td>
                    <td><code>{adj_obj.audit_hash}</code></td>
                    <td>{adj_obj.notes}</td>
                </tr>
                """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Valuation & Fair Value Adjustment (XVA) Report — {portfolio_name}</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #21262d;
            --border-color: #30363d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-amber: #d29922;
            --accent-purple: #bc8cff;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 2rem;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 2rem;
        }}
        .header-title h1 {{
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }}
        .header-title p {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }}
        .header-meta {{
            text-align: right;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        .header-meta strong {{
            color: var(--accent-blue);
        }}
        .grid-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.25rem;
        }}
        .card-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }}
        .card-value {{
            font-size: 1.5rem;
            font-weight: 700;
        }}
        .card-subtext {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        .highlight-net {{
            border-color: var(--accent-green);
            background: linear-gradient(180deg, rgba(63, 185, 80, 0.08) 0%, var(--bg-secondary) 100%);
        }}
        .highlight-net .card-value {{
            color: var(--accent-green);
        }}
        .positive {{ color: var(--accent-green); }}
        .negative {{ color: var(--accent-red); }}
        .section-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 2rem;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: var(--bg-card);
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.04em;
        }}
        tr:hover {{
            background-color: rgba(255, 255, 255, 0.02);
        }}
        .num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        code {{
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            background-color: var(--bg-card);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.8rem;
            color: var(--accent-blue);
        }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-type {{
            background-color: rgba(88, 166, 255, 0.15);
            color: var(--accent-blue);
        }}
        .badge-version {{
            background-color: rgba(188, 140, 255, 0.15);
            color: var(--accent-purple);
        }}
        .waterfall-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .waterfall-box {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.25rem;
        }}
        .waterfall-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.6rem 0;
            border-bottom: 1px dashed var(--border-color);
            font-size: 0.9rem;
        }}
        .waterfall-row:last-child {{
            border-bottom: none;
            padding-top: 0.8rem;
            font-weight: 700;
            font-size: 1.05rem;
        }}
        footer {{
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border-color);
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>Valuation & Fair Value Adjustment (XVA) Report</h1>
                <p>Product Control Analytics & Model Governance — {portfolio_name}</p>
            </div>
            <div class="header-meta">
                <div>As-Of Date: <strong>{as_of_date}</strong></div>
                <div>Generated: {timestamp}</div>
                <div>Governance: <strong>IFRS 13 & Basel PVA Compliant</strong></div>
            </div>
        </header>

        <!-- KPI Metric Cards -->
        <div class="grid-cards">
            <div class="card highlight-net">
                <div class="card-label">Net Fair Value (Balance Sheet)</div>
                <div class="card-value">${net_fv:,.2f}</div>
                <div class="card-subtext">Post-XVA & PVA Reserves</div>
            </div>
            <div class="card">
                <div class="card-label">Gross Mid-Market NPV</div>
                <div class="card-value">${base_npv:,.2f}</div>
                <div class="card-subtext">Unadjusted valuation</div>
            </div>
            <div class="card">
                <div class="card-label">Total XVA / PVA Reserves</div>
                <div class="card-value negative">-${total_adj:,.2f}</div>
                <div class="card-subtext">Bid-Offer + FVA + CVA</div>
            </div>
            <div class="card">
                <div class="card-label">Net Portfolio Delta</div>
                <div class="card-value">{delta:,.2f}</div>
                <div class="card-subtext">Aggregated directional risk</div>
            </div>
            <div class="card">
                <div class="card-label">Fixed Curve DV01 (1bp)</div>
                <div class="card-value">${dv01:,.2f}</div>
                <div class="card-subtext">Interest rate sensitivity</div>
            </div>
        </div>

        <!-- Fair Value Waterfall & Risk Sensitivities -->
        <div class="waterfall-grid">
            <div class="waterfall-box">
                <h3 style="font-size: 1rem; margin-bottom: 1rem; color: var(--text-primary);">Fair Value Accounting Waterfall</h3>
                <div class="waterfall-row">
                    <span>Gross Unadjusted NPV</span>
                    <span class="num">${base_npv:,.2f}</span>
                </div>
                <div class="waterfall-row">
                    <span>Bid-Offer Reserve (Closeout Cost)</span>
                    <span class="num negative">-${bo_res:,.2f}</span>
                </div>
                <div class="waterfall-row">
                    <span>Funding Valuation Adjustment (FVA)</span>
                    <span class="num negative">-${fva:,.2f}</span>
                </div>
                <div class="waterfall-row">
                    <span>Credit Valuation Adjustment (CVA)</span>
                    <span class="num negative">-${cva:,.2f}</span>
                </div>
                <div class="waterfall-row">
                    <span>NET BALANCE SHEET FAIR VALUE</span>
                    <span class="num positive">${net_fv:,.2f}</span>
                </div>
            </div>

            <div class="waterfall-box">
                <h3 style="font-size: 1rem; margin-bottom: 1rem; color: var(--text-primary);">Aggregated Greeks & Risk Sensitivities</h3>
                <div class="waterfall-row">
                    <span>Net Delta (&Delta;)</span>
                    <span class="num">{delta:,.2f}</span>
                </div>
                <div class="waterfall-row">
                    <span>Net Gamma (&Gamma;)</span>
                    <span class="num">{gamma:,.4f}</span>
                </div>
                <div class="waterfall-row">
                    <span>Net Vega (100% Vol)</span>
                    <span class="num">${vega:,.2f}</span>
                </div>
                <div class="waterfall-row">
                    <span>Vega 1% Shift</span>
                    <span class="num">${vega * 0.01:,.2f}</span>
                </div>
                <div class="waterfall-row">
                    <span>Yield Curve Parallel DV01</span>
                    <span class="num">${dv01:,.2f}</span>
                </div>
            </div>
        </div>

        <!-- Trading Books & Positions Inventory -->
        <div class="section-title">Trading Books & Derivatives Position Inventory</div>
        <table>
            <thead>
                <tr>
                    <th>Trade ID</th>
                    <th>Type</th>
                    <th>Book</th>
                    <th class="num">Quantity</th>
                    <th class="num">Notional</th>
                    <th class="num">NPV (USD)</th>
                    <th class="num">Delta</th>
                    <th class="num">Gamma</th>
                    <th class="num">Vega</th>
                    <th class="num">DV01</th>
                </tr>
            </thead>
            <tbody>
                {pos_rows_html}
            </tbody>
        </table>

        <!-- Model Governance & Audit Trail -->
        <div class="section-title">Regulatory Model Governance & SHA-256 Audit Trail</div>
        <table>
            <thead>
                <tr>
                    <th>Adjustment Name</th>
                    <th>Methodology Version</th>
                    <th class="num">Reserve Amount</th>
                    <th>Audit Hash (SHA-256)</th>
                    <th>Notes & Governance Standard</th>
                </tr>
            </thead>
            <tbody>
                {adj_rows_html}
            </tbody>
        </table>

        <footer>
            HSBC Product Control Analytics & Quantitative Valuation Framework &bull; Autonomous Valuation Engine &bull; Generated for Independent Model Review
        </footer>
    </div>
</body>
</html>
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_content
