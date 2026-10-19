import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
import requests

from src.search_service import BailianMCPSearchProvider, SearchService, SearchResult, SearchResponse, _mcp_post
from src.config import Config, DEFAULT_BAILIAN_MCP_URL
from src.core.config_registry import get_field_definition, get_registered_field_keys


def _response(payload=None, *, status=200, content_type="application/json", headers=None):
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.headers = {"Content-Type": content_type, **(headers or {})}
    response.encoding = "utf-8"
    response.content = json.dumps(payload or {}).encode()
    return response


def _mcp_sequence(pages, *, sse=False, session_id=None):
    init = {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
    tool = {
        "jsonrpc": "2.0", "id": 2,
        "result": {"content": [{"type": "text", "text": json.dumps({"pages": pages})}]},
    }
    init_response = _response(init, headers={"Mcp-Session-Id": session_id} if session_id else {})
    if sse:
        init_response.headers["Content-Type"] = "text/event-stream"
        init_response.content = f"event: message\ndata: {json.dumps(init)}\n\n".encode()
        tool_response = _response({}, content_type="text/event-stream")
        tool_response.content = f"data: {json.dumps(tool)}\n\n".encode()
    else:
        tool_response = _response(tool)
    return [init_response, _response({}, status=202), tool_response]


@pytest.mark.parametrize("sse,session_id", [(False, None), (True, "session-1")])
def test_bailian_mcp_parses_json_sse_and_optional_session(sse, session_id):
    pages = [{
        "title": "公司公告", "snippet": "摘要", "url": "https://example.com/2026/10/02/a",
        "hostname": "example.com",
    }]
    provider = BailianMCPSearchProvider(["secret"], timeout_seconds=7)
    with patch("src.search_service._mcp_post", side_effect=_mcp_sequence(pages, sse=sse, session_id=session_id)) as post:
        result = provider.search("药明康德 最新消息", max_results=3, days=3)

    assert result.success is True
    assert result.results[0].published_date == "2026-10-02"
    assert post.call_args_list[-1].kwargs["payload"]["params"]["arguments"]["count"] == 3
    if session_id:
        assert post.call_args_list[-1].kwargs["headers"]["Mcp-Session-Id"] == session_id


def test_bailian_rejects_tool_error_invalid_url_and_unknown_date(caplog):
    provider = BailianMCPSearchProvider(["secret"])
    error = {"jsonrpc": "2.0", "id": 2, "result": {"isError": True, "content": []}}
    responses = _mcp_sequence([])
    responses[-1] = _response(error)
    with patch("src.search_service._mcp_post", side_effect=responses):
        failed = provider.search("query")
    assert failed.success is False
    assert "secret" not in (failed.error_message or "")
    assert "secret" not in caplog.text

    pages = [
        {"title": "bad", "url": "file:///tmp/a", "snippet": "x"},
        {"title": "list", "url": "https://example.com/news", "snippet": "正文含 2026-10-02"},
    ]
    with patch("src.search_service._mcp_post", side_effect=_mcp_sequence(pages)):
        valid = provider.search("query")
    assert [item.title for item in valid.results] == ["list"]
    assert valid.results[0].published_date is None


def test_bailian_is_highest_priority_and_survives_topic_subprocess_rebuild():
    service = SearchService(
        bailian_keys=["b"], anspire_keys=["a"], bocha_keys=["c"],
        bailian_mcp_url="https://example.com/mcp", bailian_mcp_timeout_seconds=9,
    )
    assert service._providers[0].name == "Bailian"
    assert service._providers[1].name == "Anspire"
    rebuilt = SearchService(**service._constructor_kwargs)
    assert rebuilt._providers[0].name == "Bailian"
    assert rebuilt._constructor_kwargs["bailian_mcp_timeout_seconds"] == 9


def test_anspire_search_first_keeps_anspire_ahead_of_bailian():
    service = SearchService(
        bailian_keys=["b"], anspire_keys=["a"], bocha_keys=["c"],
        anspire_search_first=True,
    )
    assert service._providers[0].name == "Anspire"
    assert service._providers[1].name == "Bailian"
    # The topic-news subprocess rebuilds SearchService from the constructor
    # kwargs, so the ordering has to survive that round trip too.
    rebuilt = SearchService(**service._constructor_kwargs)
    assert rebuilt._providers[0].name == "Anspire"
    assert rebuilt._providers[1].name == "Bailian"


def test_anspire_search_first_without_bailian_still_leads():
    service = SearchService(anspire_keys=["a"], bocha_keys=["c"], anspire_search_first=True)
    assert service._providers[0].name == "Anspire"


def test_bailian_rejects_redirect_oversize_and_mismatched_id():
    redirect = _response({}, status=302)
    with pytest.raises(RuntimeError, match="重定向"):
        BailianMCPSearchProvider._decode_mcp_response(redirect, 1)
    oversized = _response({})
    oversized.content = b"x" * (BailianMCPSearchProvider.MAX_RESPONSE_BYTES + 1)
    with pytest.raises(RuntimeError, match="大小限制"):
        BailianMCPSearchProvider._decode_mcp_response(oversized, 1)
    mismatch = _response({"jsonrpc": "2.0", "id": 99, "result": {}})
    with pytest.raises(RuntimeError, match="ID 不匹配"):
        BailianMCPSearchProvider._decode_mcp_response(mismatch, 1)


@pytest.mark.parametrize("payload", [
    [], {"jsonrpc": "1.0", "id": 1},
    {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "secret"}},
])
def test_bailian_invalid_rpc_response_is_sanitized(payload):
    with pytest.raises(RuntimeError) as error:
        BailianMCPSearchProvider._decode_mcp_response(_response(payload), 1)
    assert "secret" not in str(error.value)


def test_bailian_reports_rate_limit_without_leaking_credentials(caplog):
    provider = BailianMCPSearchProvider(["top-secret"])
    with patch("src.search_service._mcp_post", return_value=_response({}, status=429)):
        result = provider.search("query")
    assert result.success is False
    assert "HTTP 429" in (result.error_message or "")
    assert "top-secret" not in caplog.text


def test_bailian_timeout_becomes_provider_failure(caplog):
    provider = BailianMCPSearchProvider(["top-secret"])
    with patch("src.search_service.requests.post", side_effect=requests.Timeout("top-secret")):
        result = provider.search("query")
    assert result.success is False
    assert "Timeout" in (result.error_message or "")
    assert "top-secret" not in (result.error_message or "")
    assert "top-secret" not in caplog.text


@pytest.mark.parametrize("tool_result", [
    {"content": "invalid"},
    {"content": [{"type": "text", "text": '{"status":"failed","pages":[]}'}]},
])
def test_bailian_invalid_tool_payload_is_diagnosable(tool_result):
    responses = _mcp_sequence([])
    responses[-1] = _response({"jsonrpc": "2.0", "id": 2, "result": tool_result})
    with patch("src.search_service._mcp_post", side_effect=responses):
        response = BailianMCPSearchProvider(["secret"]).search("query")
    assert response.success is False
    assert response.error_message
    assert "secret" not in response.error_message


def test_bailian_old_risk_result_is_removed_by_dsa_freshness_filter():
    service = SearchService(bailian_keys=["secret"], news_max_age_days=3)
    old = service._providers[0]._published_date({
        "url": "https://example.com/2022/05/18/risk",
    })
    filtered = service._filter_news_response(
        SearchResponse(
            query="risk", provider="Bailian", success=True,
            results=[SearchResult("old", "old risk", "https://example.com/2022/05/18/risk", "example", old)],
        ),
        search_days=3, max_results=5, log_scope="test",
    )
    assert filtered.results == []


@pytest.mark.parametrize("first", ["failed", "undated", "old", "future"])
def test_stock_news_falls_back_after_bailian_failure_or_unusable_dates(first):
    service = SearchService(bailian_keys=["b"], bocha_keys=["c"])
    today = datetime.now().date()
    dates = {"undated": None, "old": "2022-05-18", "future": (today + timedelta(days=5)).isoformat()}
    pages = [{"title": "药明康德业绩公告", "snippet": "药明康德披露最新财报营收。",
              "url": "https://example.com/article", "published_date": dates.get(first)}]
    good = SearchResponse(query="news", provider="Bocha", success=True, results=[
        SearchResult("药明康德业绩公告", "药明康德披露最新财报营收。", "https://example.com/current",
                     "example", today.isoformat()),
    ])
    sequence = [_response({}, status=401)] if first == "failed" else _mcp_sequence(pages)
    with patch("src.search_service._mcp_post", side_effect=sequence), patch.object(service._providers[1], "search", return_value=good) as fallback:
        result = service.search_stock_news("603259", "药明康德")
    assert result.provider == "Bocha"
    assert len(result.results) == 1
    fallback.assert_called_once()


def test_comprehensive_intel_preserves_rotation():
    service = SearchService(bailian_keys=["b"], bocha_keys=["c"])
    with patch.object(service._providers[0], "search", return_value=SearchResponse("q", [], "Bailian", True)) as bailian, patch.object(service._providers[1], "search", return_value=SearchResponse("q", [], "Bocha", True)) as bocha, patch("src.search_service.time.sleep"):
        result = service.search_comprehensive_intel("603259", "药明康德", max_searches=6)
    assert len(result) == 6
    assert bailian.call_count == bocha.call_count == 3
    assert [r.provider for r in result.values()] == ["Bailian", "Bocha"] * 3


def test_publication_dates_do_not_use_incidental_or_listing_dates():
    extract = BailianMCPSearchProvider._published_date
    assert extract({"url": "https://example.com/news", "snippet": "发布日期：2026年10月1日 正文"}) == "2026-10-01"
    assert extract({"url": "https://example.com/quotes/2026/10/01", "snippet": "发布日期：2026年10月1日"}) is None
    assert extract({"url": "https://example.com/quotes/a", "published_date": "2026-10-01"}) is None
    assert extract({"url": "https://stock.cngold.org/c/2026-09-30/c10790863.html",
                    "title": "今日药明康德股票最新行情走势查询"}) is None
    assert extract({"url": "https://example.com/2026/02/31/article"}) is None
    assert extract({"url": "https://example.com/article?date=/2026/10/01/"}) is None


def test_configuration_loading_and_sensitive_schema():
    with patch("src.config.setup_env"), patch.dict("os.environ", {
        "BAILIAN_API_KEYS": "b1, b2", "BAILIAN_MCP_URL": "",
        "BAILIAN_MCP_TIMEOUT_SECONDS": "200",
    }, clear=True):
        config = Config._load_from_env()
    assert config.bailian_api_keys == ["b1", "b2"]
    assert config.bailian_mcp_url == DEFAULT_BAILIAN_MCP_URL
    assert config.bailian_mcp_timeout_seconds == 120
    assert config.has_search_capability_enabled()
    schema = get_field_definition("BAILIAN_API_KEYS")
    assert schema["is_sensitive"] is True
    assert schema["ui_control"] == "password"
    from api.v1.schemas.system_config import SystemConfigFieldSchema
    assert "BAILIAN_MCP_URL" not in get_registered_field_keys()
    for key in ("BAILIAN_API_KEYS", "BAILIAN_MCP_TIMEOUT_SECONDS"):
        SystemConfigFieldSchema.model_validate(get_field_definition(key))


def test_streaming_limit_and_shared_deadline():
    response = _response()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content = Mock(return_value=iter([b"x" * 5, b"y" * 5]))
    with patch("src.search_service.requests.post", return_value=response) as post, patch("src.search_service.time.monotonic", return_value=10), patch.object(BailianMCPSearchProvider, "MAX_RESPONSE_BYTES", 8):
        with pytest.raises(RuntimeError, match="大小限制"):
            _mcp_post("https://example.com/mcp", headers={}, payload={"id": 1}, deadline=15)
    assert post.call_args.kwargs["stream"] is True
    assert post.call_args.kwargs["allow_redirects"] is False
    assert post.call_args.kwargs["timeout"] == 5
    with patch("src.search_service.time.monotonic", return_value=16), patch("src.search_service.requests.post") as expired_post:
        with pytest.raises(RuntimeError, match="总超时预算"):
            _mcp_post("https://example.com/mcp", headers={}, payload={"id": 1}, deadline=15)
    expired_post.assert_not_called()


def _plain_stream_response(body: bytes):
    """A non-SSE response whose stream never terminates on its own."""
    response = _response()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.headers["Content-Length"] = str(len(body))

    def _chunks():
        yield body[:3]
        yield body[3:]
        raise AssertionError("non-SSE stream was not terminated at Content-Length")

    response.iter_content = Mock(side_effect=lambda chunk_size: _chunks())
    return response


def test_plain_mcp_response_stops_at_content_length_and_is_readable():
    body = b'{"jsonrpc":"2.0","id":2,"result":{}}'
    response = _plain_stream_response(body)

    with patch("src.search_service.requests.post", return_value=response) as post, \
         patch("src.search_service.time.monotonic", return_value=10):
        result = _mcp_post("https://example.com/mcp", headers={}, payload={"id": 2}, deadline=15)

    assert post.call_args.kwargs["stream"] is True
    # The bounded bytes are exposed through the public Response.content path,
    # not through the private _content attribute.
    assert result.content == body
    assert result.status_code == 200


def test_sse_mcp_response_stops_at_first_complete_message():
    body = b'data: {"jsonrpc":"2.0","id":2,"result":{}}\n\n'
    response = _response(content_type="text/event-stream")
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.headers.pop("Content-Length", None)

    def _chunks():
        yield body
        raise AssertionError("SSE stream was not terminated at the event boundary")

    response.iter_content = Mock(side_effect=lambda chunk_size: _chunks())

    with patch("src.search_service.requests.post", return_value=response), \
         patch("src.search_service.time.monotonic", return_value=10):
        result = _mcp_post("https://example.com/mcp", headers={}, payload={"id": 2}, deadline=15)

    assert result.content == body


def test_encoded_mcp_response_ignores_content_length_and_reads_to_eof():
    """iter_content yields decompressed bytes, which exceed the compressed length.

    The break fires as soon as the decompressed total reaches the compressed
    length, i.e. before the body is fully read, so every gzipped reply would
    fail with "invalid JSON" and point at the vendor. The stream must instead
    run to its natural end.
    """
    body = b'{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{}"}]}}'
    response = _response()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.headers["Content-Length"] = str(len(body) // 2)
    response.headers["Content-Encoding"] = "gzip"

    def _chunks():
        for start in range(0, len(body), 8):
            yield body[start:start + 8]

    response.iter_content = Mock(side_effect=lambda chunk_size: _chunks())

    with patch("src.search_service.requests.post", return_value=response), \
         patch("src.search_service.time.monotonic", return_value=10):
        result = _mcp_post("https://example.com/mcp", headers={}, payload={"id": 2}, deadline=15)

    assert result.content == body


def test_mislabeled_sse_payload_stops_at_event_boundary():
    """A proxy may relabel SSE as JSON; the payload decides, not the header."""
    body = b'data: {"jsonrpc":"2.0","id":2,"result":{}}\n\n'
    response = _response(content_type="application/json")
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.headers.pop("Content-Length", None)

    def _chunks():
        yield body
        raise AssertionError("SSE payload under a JSON content-type was not terminated")

    response.iter_content = Mock(side_effect=lambda chunk_size: _chunks())

    with patch("src.search_service.requests.post", return_value=response), \
         patch("src.search_service.time.monotonic", return_value=10):
        result = _mcp_post("https://example.com/mcp", headers={}, payload={"id": 2}, deadline=15)

    assert result.content == body

    response = _response(content_type="text/event-stream")
    response.content = b'data: {"jsonrpc":"2.0","method":"progress"}\n\nevent: message\ndata: {"jsonrpc":"2.0",\ndata: "id":2,"result":{}}\n\n'
    assert BailianMCPSearchProvider._decode_mcp_response(response, 2)["result"] == {}


@pytest.mark.parametrize("value", ["http://example.com/mcp", "https://user:secret@example.com/mcp", "https://example.com/mcp?token=secret", "https://example.com:bad/mcp"])
def test_settings_and_provider_reject_invalid_endpoints(value):
    from src.services.system_config_service import SystemConfigService
    schema = get_field_definition("BAILIAN_MCP_URL")
    issues = SystemConfigService._validate_value("BAILIAN_MCP_URL", value, schema)
    assert any(issue["severity"] == "error" for issue in issues)
    assert "secret" not in str(issues)
    with pytest.raises(ValueError):
        BailianMCPSearchProvider(["secret"], mcp_url=value)
