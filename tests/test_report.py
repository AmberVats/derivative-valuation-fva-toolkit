"""Tests for HTML valuation and Fair Value Adjustment report generation."""

import math
import os
import pytest
from src.instruments.base import OptionType
from src.instruments.options import EuropeanOption
from src.market.market_data import MarketData
from src.portfolio.portfolio import Portfolio
from src.report.html_report import HTMLReportGenerator


@pytest.fixture
def test_market():
    return MarketData(
        as_of_date="2026-08-15",
        spots={"STOCK": 100.0},
        risk_free_rate=0.05,
        dividend_yields={"STOCK": 0.01},
        flat_volatilities={"STOCK": 0.20},
    )


class TestHTMLReportGeneration:
    """Test HTMLReportGenerator output content and structure."""

    def test_html_report_creates_file_with_metrics(self, test_market, tmp_path):
        portfolio = Portfolio("TestHTMLPortfolio")
        call = EuropeanOption("STOCK", strike=100.0, expiry=0.5, option_type=OptionType.CALL, notional=100.0)
        portfolio.add_position(call, quantity=1.0, book_name="MAIN_BOOK")

        summary = portfolio.evaluate(test_market)
        report_file = str(tmp_path / "test_report.html")

        generator = HTMLReportGenerator()
        html_str = generator.generate_report(summary, output_filepath=report_file)

        assert os.path.exists(report_file)
        assert "<!DOCTYPE html>" in html_str
        assert "Valuation & Fair Value Adjustment (XVA) Report" in html_str
        assert "TestHTMLPortfolio" in html_str
        assert "MAIN_BOOK" in html_str
        assert "Net Fair Value" in html_str
        assert "Bid-Offer Reserve" in html_str
        assert "Funding Valuation Adjustment" in html_str
        assert "Credit Valuation Adjustment" in html_str
        assert "SHA-256" in html_str
