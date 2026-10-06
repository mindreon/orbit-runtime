"""What an expert's bundle contributes to an attempt (ADR-0013): SOUL.md before AGENTS.md, and the bundle's own skills.

How it can go wrong, written down before the code:
  - the soul comes after the instructions, or is lost when there are none;
  - soul and instructions together go over the limit and the soul pushes the instructions out entirely;
  - a bundle skill is asked from the catalog endpoint, for the wrong tenant or version, or is not staged like a catalog one;
  - a bundle skill control does not offer stops the attempt or hides the catalog skills;
  - a spec without the new keys (an older expert) changes behaviour, or crashes.
"""

import asyncio
from pathlib import Path

from agentscope.state import AgentState
from aiohttp import web
from orbit_worker.agent_config import (
    MAX_INSTRUCTIONS_CHARS,
    AgentConfig,
    agent_config_from_spec,
    with_task_config,
)
from orbit_worker.chat_model import ModelConfig
from orbit_worker.runtime import AgentRuntime
from orbit_worker.skills import (
    ChainSkillSource,
    ControlSkillSource,
    ExpertScopedSource,
    SkillBundle,
    bundle_skill_ids,
    staged_skills,
)
from orbit_worker.store import MemoryStateStore, SessionBlob
from orbit_worker.task_stream import TaskStreamContext, streaming_for

TOKEN = "t0ken"


def _prompt(config: AgentConfig) -> str:
    runtime = AgentRuntime(store=MemoryStateStore(), model_config=ModelConfig(mode="mock", name="base"))
    blob = SessionBlob(
        session_id="s1", task_id="t1", state_version=1, agent_state=AgentState().model_dump(mode="json"),
        permission_preset="workspace-write",
    )
    context = TaskStreamContext(tenant_id="t", task_id="task", attempt_id="att", activity_attempt=1, agent=config)
    with streaming_for(context):
        return asyncio.run(runtime._agent(blob)._get_system_prompt())


def test_the_soul_comes_before_the_instructions() -> None:
    config = agent_config_from_spec({"soul": "Warm and brief.", "instructions": "Write release notes."})
    prompt = _prompt(config)
    assert prompt.index("Warm and brief.") < prompt.index("Write release notes.")
    assert prompt.endswith("Write release notes.")


def test_a_team_prompt_still_follows_the_instructions() -> None:
    config = agent_config_from_spec({"soul": "S", "instructions": "I"})
    assert config.soul == "S" and config.instructions == "I"
    assert with_task_config(config, None) is config


def test_a_soul_alone_is_still_in_the_prompt() -> None:
    assert _prompt(agent_config_from_spec({"soul": "Only a soul."})).endswith("Only a soul.")


def test_soul_and_instructions_share_one_limit() -> None:
    config = agent_config_from_spec({"soul": "s" * 15_000, "instructions": "i" * 15_000})
    assert len(config.soul) + len(config.instructions) == MAX_INSTRUCTIONS_CHARS
    assert config.soul == "s" * 15_000
    config = agent_config_from_spec({"soul": "s" * 30_000, "instructions": "i"})
    assert len(config.soul) == MAX_INSTRUCTIONS_CHARS and config.instructions == ""


def test_a_spec_from_before_bundles_reads_as_it_did() -> None:
    config = agent_config_from_spec({"instructions": "Do it.", "skills": ["h/s"]})
    assert (config.soul, config.bundle_skills, config.bundle_ref) == ("", (), "")
    assert config.skills == ("h/s",)
    assert agent_config_from_spec({"soul": 3, "bundle_skills": "nope"}) == AgentConfig()


def test_the_bundle_skills_are_read_in_load_order() -> None:
    config = agent_config_from_spec({"bundle_skills": ["zeta", "alpha", "zeta", "", 7]})
    assert config.bundle_skills == ("zeta", "alpha")


# ---- fetching and staging ---------------------------------------------------------------------------------------


class Catalog:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fetch(self, skill_id: str) -> SkillBundle | None:
        self.calls.append(skill_id)
        return SkillBundle(
            id=skill_id, name="Cat", description="d", files=(("SKILL.md", "---\nname: Cat\n---\ncatalog"),)
        )


async def _control(seen: list[tuple[str, str, str]]) -> tuple[web.AppRunner, str]:
    async def handler(request: web.Request) -> web.Response:
        seen.append((request.path, request.query.get("tenant_id", ""), request.headers.get("Authorization", "")))
        if request.match_info["name"] == "gone":
            return web.json_response({"error": "NOT_FOUND"}, status=404)
        return web.json_response(
            {
                "id": "expert_x@2/pdf", "name": "pdf", "description": "",
                "files": [
                    {"path": "SKILL.md", "body": "---\nname: pdf\ndescription: makes pdfs\n---\nbody"},
                    {"path": "t/template.md", "body": "t"},
                ],
            }
        )

    app = web.Application()
    app.router.add_get("/internal/experts/{expert}/{version}/skills/{name}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"http://127.0.0.1:{port}"


async def test_bundle_skills_are_fetched_for_the_tenant_and_version_and_staged_like_catalog_ones(tmp_path) -> None:
    seen: list[tuple[str, str, str]] = []
    runner, base = await _control(seen)
    catalog = Catalog()
    try:
        source = ExpertScopedSource(ChainSkillSource(catalog, ControlSkillSource(base, TOKEN)), "tenant-a", "expert_x@2")
        ids = ("h/cat", *bundle_skill_ids(("pdf", "gone")))
        async with staged_skills(source, ids, root=tmp_path) as skills:
            names = [skill.name for skill in skills]
            assert names == ["Cat", "pdf"]  # the one control does not offer is skipped
            staged = skills[1]
            assert staged.description == "makes pdfs"
            assert (Path(staged.dir) / "t" / "template.md").read_text() == "t"
    finally:
        await runner.cleanup()
    assert catalog.calls == ["h/cat"]  # a bundle skill never goes to the catalog sources
    assert seen[0] == ("/internal/experts/expert_x/2/skills/pdf", "tenant-a", f"Bearer {TOKEN}")


async def test_a_bad_expert_ref_or_name_is_not_sent_to_control() -> None:
    seen: list[tuple[str, str, str]] = []
    runner, base = await _control(seen)
    try:
        control = ControlSkillSource(base, TOKEN)
        assert await control.fetch_expert("t", "no-version", "pdf") is None
        assert await control.fetch_expert("t", "e@1", "../x") is None
        assert await control.fetch_expert("t", "e@1", "a/b") is None
    finally:
        await runner.cleanup()
    assert seen == []


async def test_without_a_control_source_a_bundle_skill_is_skipped(tmp_path) -> None:
    source = ExpertScopedSource(Catalog(), "t", "e@1")
    async with staged_skills(source, bundle_skill_ids(("pdf",)), root=tmp_path) as skills:
        assert skills == ()
