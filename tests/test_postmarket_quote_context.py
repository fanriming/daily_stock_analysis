"""Closing-price input remains usable when live quotes are intentionally disabled."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.agent.executor import AgentExecutor
from src.agent.orchestrator import AgentOrchestrator
from src.agent.tools.data_tools import _handle_get_realtime_quote
from src.analysis_context_pack_overview import extract_analysis_context_pack_overview, render_analysis_context_pack_overview
from src.analysis_context_pack_prompt import format_analysis_context_pack_prompt_section
from src.config import Config
from src.core.pipeline import StockAnalysisPipeline
from src.services.analysis_context_builder import AnalysisContextBuilder


def artifacts(*, enabled=False, phase="postmarket", bar_date="2026-10-16", expected_date="2026-10-16", close=13.82):
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.config = SimpleNamespace(enable_realtime_quote=enabled)
    pipeline.query_source = "api"
    return pipeline._build_agent_analysis_artifacts(
        code="603883.SH", stock_name="老百姓", market="cn", query_id="test-closing-quote",
        phase={"phase": phase, "effective_daily_bar_date": expected_date, "is_partial_bar": phase == "intraday"},
        initial_context={}, fundamental_context=None,
        base_context={
            "date": bar_date,
            "today": {"date": bar_date, "close": close, "open": 14.0, "high": 14.1, "low": 13.7,
                      "volume": 1000.0, "amount": 13820.0, "pct_chg": -1.0, "data_source": "TushareFetcher"},
            "yesterday": {"date": "2026-10-15", "close": 14.0},
        },
    )


@pytest.mark.parametrize("phase", ["postmarket", "non_trading", "premarket"])
def test_disabled_live_quotes_use_matching_completed_close_without_missing_penalty(phase):
    pack = AnalysisContextBuilder.build(artifacts(phase=phase))
    quote = pack.blocks["quote"]
    assert quote.status.value == "available"
    assert quote.items["price"].value == 13.82
    assert quote.source == "TushareFetcher"
    assert quote.metadata["price_basis"] == "daily_close"
    assert quote.metadata["trade_date"] == "2026-10-16"
    assert quote.timestamp is None  # A trade date is not a fabricated quote timestamp.
    assert pack.data_quality.block_scores["quote"] == 100
    assert "quote: missing" not in pack.data_quality.limitations
    for lang in ("zh", "en", "ko"):
        overview = render_analysis_context_pack_overview(pack, report_language=lang)
        block = next(block for block in overview["blocks"] if block["key"] == "quote")
        assert "2026-10-16" in block["label"]
        assert block["missing_reasons"] == []
        assert extract_analysis_context_pack_overview({"analysis_context_pack_overview": overview})["blocks"][0] == block
        summary = format_analysis_context_pack_prompt_section(pack, report_language=lang)
        assert "13.82" not in summary  # Keep the existing low-sensitivity summary contract.
        assert "realtime_quote_missing" not in summary
        assert "不得仅因关闭实时行情" in summary if lang == "zh" else "do not lower confidence solely" in summary


@pytest.mark.parametrize("kwargs", [
    {"bar_date": "2026-10-15"},
    {"bar_date": "2026-10-19"},
    {"phase": "intraday"},
])
def test_closing_price_does_not_hide_stale_or_intraday_limitations(kwargs):
    pack = AnalysisContextBuilder.build(artifacts(**kwargs))
    assert pack.blocks["quote"].status.value == "stale"
    assert "quote: stale" in pack.data_quality.limitations
    summary = format_analysis_context_pack_prompt_section(pack)
    assert "不得为高" in summary
    assert "不得仅因关闭实时行情" not in summary


@pytest.mark.parametrize("kwargs", [
    {"phase": "unknown"}, {"expected_date": None},
])
def test_unverified_closing_date_remains_partial(kwargs):
    assert AnalysisContextBuilder.build(artifacts(**kwargs)).blocks["quote"].status.value == "partial"


@pytest.mark.parametrize("close", [None, 0, -1, float("nan"), float("inf"), True, "invalid"])
def test_invalid_close_remains_missing(close):
    block = AnalysisContextBuilder.build(artifacts(close=close)).blocks["quote"]
    assert block.status.value == "missing"
    assert block.items["quote"].missing_reason == "daily_bars_missing"


def test_missing_daily_data_and_bad_date_cannot_be_promoted():
    original = artifacts()
    for context in ({}, {"data_missing": True, **original.base_context}, {"today": {"close": 13.82, "date": "bad-date"}}):
        assert AnalysisContextBuilder.build(replace(original, base_context=context)).blocks["quote"].status.value == "missing"


def test_enabled_quote_failure_is_not_hidden_by_closing_price():
    block = AnalysisContextBuilder.build(artifacts(enabled=True)).blocks["quote"]
    assert block.status.value == "missing"
    assert block.items["quote"].missing_reason == "realtime_quote_missing"


def test_stale_live_quote_falls_back_to_verified_daily_close():
    """A stale quote and a switched-off quote must reach the same verdict."""
    stale_quote = {"source": "TushareFetcher", "price": 13.82, "is_stale": True}
    pack = AnalysisContextBuilder.build(
        replace(artifacts(enabled=True), realtime_quote=stale_quote)
    )
    quote = pack.blocks["quote"]
    assert quote.status.value == "available"
    assert quote.metadata["price_basis"] == "daily_close"
    assert quote.metadata["daily_close_reason"] == "realtime_stale"
    assert quote.metadata["realtime_quote_enabled"] is True
    assert quote.items["price"].value == 13.82
    assert "quote_stale" in quote.warnings
    assert "quote: stale" not in pack.data_quality.limitations
    assert pack.data_quality.block_scores["quote"] == 100
    summary = format_analysis_context_pack_prompt_section(pack, report_language="zh")
    assert "实时行情已过期" in summary
    assert "不得仅因实时行情不可用而降低置信度" in summary


def test_stale_live_quote_without_verified_bar_date_remains_partial():
    stale_quote = {"source": "TushareFetcher", "price": 13.82, "is_stale": True}
    pack = AnalysisContextBuilder.build(
        replace(artifacts(enabled=True, expected_date=None), realtime_quote=stale_quote)
    )
    assert pack.blocks["quote"].status.value == "partial"
    assert "quote_stale" in pack.blocks["quote"].warnings


def test_stale_live_quote_with_mismatched_bar_date_stays_stale():
    stale_quote = {"source": "TushareFetcher", "price": 13.82, "is_stale": True}
    pack = AnalysisContextBuilder.build(
        replace(artifacts(enabled=True, bar_date="2026-10-15"), realtime_quote=stale_quote)
    )
    assert pack.blocks["quote"].status.value == "stale"
    assert "quote: stale" in pack.data_quality.limitations


def test_closing_overview_label_falls_back_without_trade_date():
    """A quote block without trade_date must not render an empty bracket."""
    pack = AnalysisContextBuilder.build(artifacts())
    block = pack.blocks["quote"]
    block.metadata = {k: v for k, v in block.metadata.items() if k != "trade_date"}

    for lang, expected in (("zh", "行情"), ("en", "quote")):
        overview = render_analysis_context_pack_overview(pack, report_language=lang)
        rendered = next(b for b in overview["blocks"] if b["key"] == "quote")
        assert rendered["label"] == expected
        assert "()" not in rendered["label"]


def test_single_and_multi_agent_receive_closing_basis_from_real_pipeline_outputs():
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    summary, overview = pipeline._build_analysis_context_pack_outputs(
        artifacts(), report_language="zh", code="603883.SH", query_id="test-closing-quote",
    )
    context = {"stock_code": "603883.SH", "analysis_context_pack_summary": summary}
    single = AgentExecutor.__new__(AgentExecutor)._build_user_message("分析股票", context)
    multi = AgentOrchestrator.__new__(AgentOrchestrator)._build_context("分析股票", context)
    assert "收盘价已满足" in single
    assert "收盘价已满足" in multi.meta["analysis_context_pack_summary"]
    assert overview["blocks"][0]["status"] == "available"
    assert multi.get_data("realtime_quote") is None


def test_realtime_tool_reports_disabled_without_initializing_or_calling_provider():
    with patch("src.config.get_config", return_value=Config(enable_realtime_quote=False)), \
            patch("src.agent.tools.data_tools._get_fetcher_manager") as manager:
        result = _handle_get_realtime_quote("603883.SH")
    manager.assert_not_called()
    assert result["status"] == "disabled"
    assert result["retriable"] is False
    assert "not a provider failure" in result["note"]
    assert "price" not in result


def test_realtime_tool_keeps_actual_provider_failure_when_enabled():
    manager = Mock()
    manager.get_realtime_quote.return_value = None
    with patch("src.config.get_config", return_value=Config(enable_realtime_quote=True)), \
            patch("src.agent.tools.data_tools._get_fetcher_manager", return_value=manager):
        result = _handle_get_realtime_quote("603883.SH")
    manager.get_realtime_quote.assert_called_once_with("603883.SH")
    assert "error" in result
    assert "disabled" not in result
