# -*- coding: utf-8 -*-
"""get_thinking_extra_body 的模型清单行为 + AGENT_LLM_EXTRA_BODY env 兜底解析。"""
import os
import sys
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.llm_adapter import (  # noqa: E402
    LLMToolAdapter,
    _agent_llm_extra_body,
    get_thinking_extra_body,
)


def _clear_env() -> None:
    os.environ.pop("AGENT_LLM_EXTRA_BODY", None)


def _adapter() -> LLMToolAdapter:
    config = MagicMock()
    config.llm_model_list = []
    config.gemini_api_key = ""
    config.anthropic_api_key = ""
    config.openai_api_key = ""
    config.openai_base_url = ""
    config.openai_model = ""
    config.llm_temperature = 0.7
    config.llm_config_path = ""
    return LLMToolAdapter(config=config)


def _completion_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="ok", tool_calls=None, reasoning_content=None,
        ))]
    )


def _call_and_capture(model: str):
    """Run _call_litellm_model and return the kwargs litellm actually received."""
    captured = {}

    def _completion(**kwargs):
        captured.update(kwargs)
        return _completion_response()

    with patch("src.agent.llm_adapter.litellm.completion", side_effect=_completion):
        _adapter()._call_litellm_model(
            [{"role": "user", "content": "hi"}], [], model,
        )
    return captured


def test_disable_thinking_list_returns_disabled_payload():
    _clear_env()
    assert get_thinking_extra_body("deepseek-v4-flash") == {
        "thinking": {"type": "disabled"}
    }
    assert get_thinking_extra_body("deepseek-v4.1-flash") == {
        "thinking": {"type": "disabled"}
    }


def test_model_variant_matches_via_prefix():
    # _model_matches 的 startswith(key + "-") 语义：条目模型的带日期变体继承同一清单行为。
    _clear_env()
    assert get_thinking_extra_body("deepseek-v4-flash-20260101") == {
        "thinking": {"type": "disabled"}
    }


def test_v41_entry_not_shadowed_by_v4_entry():
    # deepseek-v4.1-flash 不满足 startswith("deepseek-v4-flash-")，靠独立条目命中——
    # 固化该语义，防止未来把清单合并成单一前缀条目时静默漏掉 v4.1。
    _clear_env()
    assert "deepseek-v4.1-flash".startswith("deepseek-v4-flash-") is False
    assert get_thinking_extra_body("deepseek-v4.1-flash") == {
        "thinking": {"type": "disabled"}
    }


def test_opt_in_model_returns_enabled_payload():
    _clear_env()
    assert get_thinking_extra_body("deepseek-chat") == {
        "thinking": {"type": "enabled"}
    }


def test_auto_thinking_models_return_none():
    _clear_env()
    for model in ("deepseek-reasoner", "deepseek-r1", "qwq"):
        assert get_thinking_extra_body(model) is None


def test_unknown_model_returns_none():
    _clear_env()
    assert get_thinking_extra_body("gpt-5.5") is None


def test_agent_extra_body_helper_parses_env():
    # A failed assertion must not leak the value into the rest of the session.
    try:
        _clear_env()
        assert _agent_llm_extra_body() == {}

        os.environ["AGENT_LLM_EXTRA_BODY"] = '{"enable_thinking": false}'
        assert _agent_llm_extra_body() == {"enable_thinking": False}

        os.environ["AGENT_LLM_EXTRA_BODY"] = "not-json"
        assert _agent_llm_extra_body() == {}

        os.environ["AGENT_LLM_EXTRA_BODY"] = "[1, 2]"
        assert _agent_llm_extra_body() == {}

        os.environ["AGENT_LLM_EXTRA_BODY"] = ""
        assert _agent_llm_extra_body() == {}
    finally:
        _clear_env()


def test_agent_extra_body_env_overrides_model_payload():
    # 端到端验证合并真的进入 litellm 请求，而不是只在测试里重算一遍 dict.update。
    try:
        _clear_env()
        captured = _call_and_capture("deepseek-v4-flash")
        assert captured["extra_body"] == {"thinking": {"type": "disabled"}}

        os.environ["AGENT_LLM_EXTRA_BODY"] = '{"thinking": {"type": "enabled"}, "custom": 1}'
        captured = _call_and_capture("deepseek-v4-flash")
        assert captured["extra_body"] == {"thinking": {"type": "enabled"}, "custom": 1}

        os.environ["AGENT_LLM_EXTRA_BODY"] = '{"other": true}'
        captured = _call_and_capture("deepseek-v4-flash")
        assert captured["extra_body"] == {"thinking": {"type": "disabled"}, "other": True}
    finally:
        _clear_env()


def test_agent_extra_body_reaches_models_outside_thinking_tables():
    _clear_env()
    os.environ["AGENT_LLM_EXTRA_BODY"] = '{"enable_thinking": false}'
    try:
        captured = _call_and_capture("some-unknown-model")
        assert captured["extra_body"] == {"enable_thinking": False}
    finally:
        _clear_env()


def test_no_extra_body_sent_when_env_unset_and_model_unknown():
    _clear_env()
    captured = _call_and_capture("some-unknown-model")
    assert "extra_body" not in captured


def test_invalid_agent_extra_body_is_reported_without_the_value(caplog):
    """The raw env value must never reach the log, only its length."""
    secret_shaped = '{"api_key": "sk-should-not-be-logged-1234567890"}'
    try:
        _clear_env()
        os.environ["AGENT_LLM_EXTRA_BODY"] = secret_shaped[:-1]  # truncated -> invalid JSON
        with caplog.at_level(logging.WARNING, logger="src.agent.llm_adapter"):
            assert _agent_llm_extra_body() == {}
        assert "sk-should-not-be-logged" not in caplog.text
        assert "长度" in caplog.text
    finally:
        _clear_env()


def test_non_dict_agent_extra_body_is_reported(caplog):
    try:
        _clear_env()
        os.environ["AGENT_LLM_EXTRA_BODY"] = "[1, 2]"
        with caplog.at_level(logging.WARNING, logger="src.agent.llm_adapter"):
            assert _agent_llm_extra_body() == {}
        assert "必须是 JSON 对象" in caplog.text
    finally:
        _clear_env()
