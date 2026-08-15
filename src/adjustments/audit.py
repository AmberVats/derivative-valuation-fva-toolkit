"""Audit Trail logging, persistence, and reporting for valuations and Fair Value

Adjustments.
"""

import json
from typing import Any, Dict, List, Optional
from tabulate import tabulate
from src.adjustments.base import AdjustmentResult


class AuditTrailManager:
    """Central registry and audit trail generator for independent model review

    and regulatory governance.
    """

    def __init__(self) -> None:
        self._adjustment_records: List[AdjustmentResult] = []
        self._pricing_records: List[Dict[str, Any]] = []

    def record_adjustment(self, result: AdjustmentResult) -> None:
        """Register a fair value adjustment result."""
        self._adjustment_records.append(result)

    def record_pricing(
        self, instrument_id: str, engine_name: str, npv: float, currency: str, as_of_date: str
    ) -> None:
        """Register an instrument pricing event."""
        self._pricing_records.append({
            "instrument_id": instrument_id,
            "engine_name": engine_name,
            "npv": npv,
            "currency": currency,
            "as_of_date": as_of_date,
        })

    def get_records(self) -> List[Dict[str, Any]]:
        """Retrieve all adjustment records as dictionaries."""
        return [r.to_dict() for r in self._adjustment_records]

    def export_json(self, filepath: Optional[str] = None) -> str:
        """Export audit log to formatted JSON."""
        data = {
            "adjustments": [r.to_dict() for r in self._adjustment_records],
            "pricings": self._pricing_records,
        }
        json_str = json.dumps(data, indent=2, default=str)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def export_markdown_table(self) -> str:
        """Generate formatted GitHub Markdown table for README or executive reports."""
        if not self._adjustment_records:
            return "_No audit records logged._"

        headers = [
            "Adjustment Name",
            "Methodology Version",
            "Amount (USD)",
            "As-Of Date",
            "Audit Hash",
            "Key Parameter",
        ]
        rows = []
        for r in self._adjustment_records:
            key_param = ""
            if "spreads_bps" in r.parameters:
                key_param = f"Spreads: {list(r.parameters['spreads_bps'].values())} bps"
            elif "funding_spread_bps" in r.parameters:
                key_param = f"Funding: {r.parameters['funding_spread_bps']} bps"
            elif "cds_spread_bps" in r.parameters:
                key_param = f"CDS: {r.parameters['cds_spread_bps']} bps, Rec: {r.parameters.get('recovery_rate', 0.4):.0%}"

            rows.append([
                r.adjustment_name,
                f"v{r.methodology_version}",
                f"${r.amount_usd:,.2f}",
                r.as_of_date,
                f"`{r.audit_hash}`",
                key_param,
            ])

        return tabulate(rows, headers=headers, tablefmt="github")
