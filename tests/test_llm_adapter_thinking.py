# -*- coding: utf-8 -*-
"""get_thinking_extra_body 的模型清单行为 + AGENT_LLM_EXTRA_BODY env 兜底解析。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.llm_adapter import (  # noqa: E402
    _agent_llm_extra_body,
    get_thinking_extra_body,
)


def _clear_env() -> None:
    os.environ.pop("AGENT_LLM_EXTRA_BODY", None)


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
    _clear_env()


def test_agent_extra_body_env_overrides_model_payload():
    # _call_litellm_model 的合并语义：env 结果覆盖清单 payload（同 key 时 env 赢）。
    _clear_env()
    payload = get_thinking_extra_body("deepseek-v4-flash")

    os.environ["AGENT_LLM_EXTRA_BODY"] = '{"thinking": {"type": "enabled"}, "custom": 1}'
    merged = dict(payload or {})
    merged.update(_agent_llm_extra_body())
    assert merged == {"thinking": {"type": "enabled"}, "custom": 1}

    os.environ["AGENT_LLM_EXTRA_BODY"] = '{"other": true}'
    merged = dict(payload or {})
    merged.update(_agent_llm_extra_body())
    assert merged == {"thinking": {"type": "disabled"}, "other": True}
    _clear_env()
