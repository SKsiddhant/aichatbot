import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dynamic_chatbot"))
import dynamic_ai_chatbot as dac  # noqa: E402


class FakeClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("boom")
        msg = SimpleNamespace(content=f"reply{len(self.calls)}")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


@pytest.fixture
def bot(tmp_path):
    b = dac.ConversationManager(api_key="test", history_file=str(tmp_path / "h.json"))
    b.client = FakeClient()
    return b


def test_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(dac, "DEFAULT_API_KEY", None)
    with pytest.raises(ValueError):
        dac.ConversationManager(history_file=str(tmp_path / "h.json"))


def test_history_is_sent_so_bot_has_memory(bot):
    bot.chat_completion("first")
    bot.chat_completion("second")
    sent = bot.client.calls[1]["messages"]
    assert [m["content"] for m in sent][1:] == ["first", "reply1", "second"]
    assert sent[0]["role"] == "system"


def test_system_message_argument_is_respected(tmp_path):
    b = dac.ConversationManager(api_key="k", system_message="Be brief.", history_file=str(tmp_path / "h.json"))
    assert b.conversation_history[0]["content"] == "Be brief."


def test_api_failure_rolls_back_user_message(bot):
    bot.client = FakeClient(fail=True)
    assert bot.chat_completion("hi") is None
    assert [m["role"] for m in bot.conversation_history] == ["system"]


def test_zero_temperature_is_kept(bot):
    bot.chat_completion("hi", temperature=0)
    assert bot.client.calls[0]["temperature"] == 0


def test_unknown_persona_does_not_crash_or_change_prompt(bot, capsys):
    before = bot.system_message
    bot.set_persona("nope")
    assert bot.system_message == before and "not found" in capsys.readouterr().out


def test_persona_updates_history(bot):
    bot.set_persona("angry_assistant")
    assert bot.conversation_history[0]["content"] == bot.system_messages["angry_assistant"]


def test_token_budget_drops_oldest_but_keeps_system(bot, monkeypatch):
    monkeypatch.setattr(bot, "count_tokens", lambda text: 10)
    bot.token_budget = 35
    for i in range(5):
        bot.chat_completion(f"m{i}")
    assert bot.conversation_history[0]["role"] == "system"
    assert len(bot.conversation_history) <= 3


def test_history_persists_roundtrip(bot, tmp_path):
    bot.chat_completion("hello")
    again = dac.ConversationManager(api_key="k", history_file=bot.history_file)
    assert [m["content"] for m in again.conversation_history][1:] == ["hello", "reply1"]
