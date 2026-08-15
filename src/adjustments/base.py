"""Base framework and audit container for Fair Value Adjustments (XVA & Reserves)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, Optional


@dataclass
class AdjustmentResult:
    """Standardized output container for fair value adjustment calculations and

    regulatory audit trail.
    """
    adjustment_name: str
    amount_usd: float
    methodology_version: str
    as_of_date: str
    calculation_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    parameters: Dict[str, Any] = field(default_factory=dict)
    breakdown: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    audit_hash: str = ""

    def __post_init__(self) -> None:
        if not self.audit_hash:
            # Generate deterministic SHA-256 hash of audit inputs
            payload = {
                "name": self.adjustment_name,
                "amount": round(self.amount_usd, 6),
                "version": self.methodology_version,
                "as_of_date": self.as_of_date,
                "parameters": self.parameters,
            }
            raw = json.dumps(payload, sort_keys=True, default=str)
            self.audit_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adjustment_name": self.adjustment_name,
            "amount_usd": self.amount_usd,
            "methodology_version": self.methodology_version,
            "as_of_date": self.as_of_date,
            "calculation_timestamp": self.calculation_timestamp,
            "parameters": self.parameters,
            "breakdown": self.breakdown,
            "notes": self.notes,
            "audit_hash": self.audit_hash,
        }

    def __repr__(self) -> str:
        return (
            f"AdjustmentResult({self.adjustment_name} = ${self.amount_usd:,.2f} "
            f"v{self.methodology_version} hash={self.audit_hash})"
        )


class FairValueAdjustment(ABC):
    """Abstract Base Class for modular Fair Value Adjustment models."""

    def __init__(self, methodology_version: str = "1.0.0") -> None:
        self.methodology_version = methodology_version

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifying name of the adjustment methodology."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Methodology documentation overview."""
        pass
