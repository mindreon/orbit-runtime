"""The language an agent writes in is decided from the user's words, and said outright in the system prompt.

How it can go wrong, written down before the code:
  - a Chinese request with many English product names is read as English, or an English one with a Chinese name as Chinese;
  - a short or mixed text is decided anyway, and the agent is told the wrong language;
  - code, paths and links in a request swing the result;
  - Japanese or Korean is taken for Chinese;
  - a review of the leader's work, whose prompt is the workflow's English wording, is told English for a Chinese task;
  - the line is missing for a member, or replaces the general rule when the worker is not sure.
"""

import asyncio

import pytest
from agentscope.state import AgentState
from orbit_contracts.models import OpenSessionInput
from orbit_worker.agent_config import AgentConfig
from orbit_worker.chat_model import ModelConfig
from orbit_worker.language import PROMPT_LINES, detect_language, language_line
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import MemoryStateStore, SessionBlob
from orbit_worker.task_activities import _language_of
from orbit_worker.task_stream import TaskStreamContext, TeamTurn, streaming_for


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("请用 React 和 FastAPI 做一个待办事项应用，要有登录和增删改查", "zh"),
        ("帮我写一个 Todo App，前端 React，后端 FastAPI，用 SQLite 存储", "zh"),
        ("Build a todo app with React and FastAPI, with login and CRUD", "en"),
        ("Please ask 张伟 to review the pull request before we merge it on Friday", "en"),
        ("好的", ""),
        ("ok", ""),
        ("", ""),
        ("日本語でお願いします。このアプリを作ってください、よろしく", ""),
        ("이 앱을 만들어 주세요 please build this application for me", ""),
        ("改一下 the login page 的样式 please", ""),
    ],
)
def test_the_language_is_decided_only_when_the_text_is_clear(text: str, expected: str) -> None:
    assert detect_language(text) == expected


def test_code_paths_and_links_do_not_decide_the_language() -> None:
    chinese = "请修复这个登录问题，然后看一下下面的链接 ```python\n" + "def handler(request, response):\n    return response\n" * 5 + "```\n看 https://example.com/some/long/path"
    assert detect_language(chinese) == "zh"
    english = "Fix the bug in /workspace/src/app/main.py and see `用户名` handling"
    assert detect_language(english) == "en"


def test_the_texts_are_taken_together() -> None:
    assert detect_language("ok", "请帮我把登录页做得更好看一些") == "zh"


def test_the_line_names_the_language_and_unknown_has_none() -> None:
    assert "Simplified Chinese (简体中文)" in language_line("zh") and "English" in language_line("en")
    assert language_line("") == "" and language_line("fr") == ""
    assert set(PROMPT_LINES) == {"zh", "en"}


def _runtime() -> AgentRuntime:
    return AgentRuntime(store=MemoryStateStore(), model_config=ModelConfig(mode="mock", name="base"))


def _blob(language: str = "") -> SessionBlob:
    return SessionBlob(
        session_id="s1", task_id="t1", state_version=1, agent_state=AgentState().model_dump(mode="json"),
        permission_preset="workspace-write", language=language,
    )


def _prompt(blob: SessionBlob, **context: object) -> str:
    stream = TaskStreamContext(tenant_id="t", task_id="task", attempt_id="att", activity_attempt=1, **context)  # type: ignore[arg-type]
    with streaming_for(stream):
        return asyncio.run(_runtime()._agent(blob)._get_system_prompt())


def test_every_kind_of_agent_is_told_the_language_outright() -> None:
    line = language_line("zh")
    assert line in _prompt(_blob(), language="zh"), "a single agent"
    member = TeamTurn(role="m", leader=False, leader_role="l")
    leader = TeamTurn(role="l", leader=True, leader_role="l", members=(("m", "builds"),))
    assert line in _prompt(_blob(), language="zh", team=member), "a team member"
    assert line in _prompt(_blob(), language="zh", team=leader), "a team leader"
    assert line in _prompt(_blob("zh")), "a session that was set to it, whatever this turn saw"
    assert line in _prompt(_blob("zh"), language="en"), "the session's language is kept"


def test_when_the_worker_is_not_sure_only_the_general_rule_stays() -> None:
    prompt = _prompt(_blob(), language="")
    assert "in the language the user wrote in" in prompt
    assert not any(line in prompt for line in PROMPT_LINES.values())


def test_the_language_is_kept_in_the_session_it_opens() -> None:
    async def run() -> str:
        runtime = _runtime()
        stream = TaskStreamContext(
            tenant_id="t", task_id="task", attempt_id="att-1", activity_attempt=1, agent=AgentConfig(), language="zh"
        )
        with streaming_for(stream):
            opened = await runtime.open_session(OpenSessionInput(room_id="task", turn_id="att-1:open"))
        return await runtime.session_language(opened.session_id)

    assert asyncio.run(run()) == "zh"


async def test_a_review_of_the_leaders_work_keeps_the_leaders_language() -> None:
    runtime = _runtime()
    await runtime._store.put(_blob("zh").model_copy(update={"session_id": "att-leader"}))
    # The review's own prompt is the workflow's English wording around a few Chinese titles: it must not decide.
    review_goal = "领队复盘（第 1 轮）: the 2 task(s) you created have finished. What each one reported: weigh these results."
    review = {"goal": review_goal, "continue_from": "att-leader"}
    assert await _language_of(runtime, review, "", "") == "zh"
    assert await runtime.session_language("att-missing") == ""
    # With no session to inherit from, and for a later turn of the same attempt, the text itself decides.
    assert await _language_of(runtime, {"goal": "Build a todo app with React"}, "", "") == "en"
    assert await _language_of(runtime, {**review, "goal": "请帮我做一个待办应用并写好文档说明"}, "", "att-review") == "zh"
