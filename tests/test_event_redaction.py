"""S-M-4: redaction of outgoing events, including streamed assistant.delta.

assistant.delta streams as it arrives, and a secret split across chunks is
still redacted. (S-M-3 is the startup exit on a missing API key.)
"""

import pytest
from agentscope.event import TextBlockDeltaEvent, TextBlockEndEvent
from agentscope.message import TextBlock
from agentscope.model import ChatResponse
from orbit_contracts.models import OpenSessionInput, RunTurnInput
from orbit_worker import runtime as runtime_module
from orbit_worker.events import MemoryEventIngest
from orbit_worker.mock_model import MockChatModel
from orbit_worker.runtime import AgentRuntime
from orbit_worker.secrets import REDACTED, redact_text, redact_value
from orbit_worker.store import MemoryStateStore
from orbit_worker.turn_events import TurnEvents

# Longer than one assistant.delta batch, so the first chunk is released alone.
FILLER = "word " * 45
CJK_FILLER = "这是一段没有任何空格的中文说明文字，" * 12


class ChunkedModel(MockChatModel):
    """Streams fixed text chunks, like a provider sending deltas."""

    def __init__(self, chunks: list[str]) -> None:
        super().__init__()
        self.stream = True
        self._chunks = chunks

    async def _call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        del model_name, messages, tools, tool_choice, kwargs

        async def deltas():
            for chunk in self._chunks:
                # One block id for every delta, as providers stream one text block.
                yield ChatResponse(content=[TextBlock(id="text-1", text=chunk)], is_last=False)

        return deltas()


HEX40 = "0123456789abcdef0123456789abcdef01234567"


def test_secret_key_names_and_token_prefixes_are_redacted() -> None:
    """S-M-4: secret-named keys (L1) and provider token prefixes (L2)."""

    names = [
        "apiKey",
        "x-api-key",
        "client_secret",
        "private_key",
        "refresh_token",
        "cookie",
        "Authorization",
        "password",
        "token",
        "secret",
    ]
    redacted = redact_value({**{name: "v4lue" for name in names}, "name": "orbit"})
    assert redacted == {**{name: REDACTED for name in names}, "name": "orbit"}
    assert redact_text("X-Api-Key: v4lue and clientSecret=v4lue") == (
        f"X-Api-Key: {REDACTED} and clientSecret={REDACTED}"
    )

    tokens = [
        "ghp_" + "A1b2C3d4" * 4,
        "github_pat_11ABCDEFG0_abcdefghijklmnop",
        "glpat-abcdefghij1234567890",
        "AKIAIOSFODNN7EXAMPLE",
        "AIzaSyD-abcdefghijklmnopqrstuvwxyz12345",
        "xoxb-1234-5678-abcdefghij",
    ]
    for token in tokens:
        assert redact_text(f"use {token} now") == f"use {REDACTED} now"
    assert redact_text(f"github token {HEX40}") == f"github token {REDACTED}"
    # A bare commit SHA is not a secret.
    assert redact_text(f"commit {HEX40}") == f"commit {HEX40}"


def test_chinese_text_streams_before_the_block_ends() -> None:
    """S-M-4: the streaming redactor still releases text that has no spaces."""

    # A frozen clock, so only the 200-character batch size releases a delta.
    events = TurnEvents({}, clock=lambda: 0.0)
    chunk = "模型正在逐字输出一段没有任何空格的中文回答，" * 10
    streamed = []
    for _ in range(3):
        streamed += events.observe(TextBlockDeltaEvent(reply_id="r", block_id="b", delta=chunk))
    assert len(streamed) >= 2
    streamed += events.observe(TextBlockEndEvent(reply_id="r", block_id="b"))
    assert all(kind == "assistant.delta" for kind, _ in streamed)
    assert "".join(str(fields["delta"]) for _, fields in streamed) == chunk * 3


@pytest.mark.parametrize(
    ("chunks", "secret"),
    [
        # The prefix that marks the secret is only in the previous chunk.
        ([FILLER + "send it with Bearer ", "q7wz19kx then stop"], "q7wz19kx"),
        ([FILLER + "config api_key= ", "hunter2 then stop"], "hunter2"),
        # The token itself is cut in two.
        ([FILLER + "the key is sk-live-", "4f9a2b77c then stop"], "sk-live-4f9a2b77c"),
        ([CJK_FILLER + "密钥是 sk-live-", "4f9a2b77c，然后停"], "sk-live-4f9a2b77c"),
        # Neither half is long enough to look random on its own.
        (
            [FILLER + "copy Zk8Qw3Rt7Yp2Lm9X", "c4Vb6Nj1Hg5Fd0Sa then stop"],
            "Zk8Qw3Rt7Yp2Lm9Xc4Vb6Nj1Hg5Fd0Sa",
        ),
    ],
    ids=[
        "S-M-4-bearer-prefix",
        "S-M-4-key-prefix",
        "S-M-4-split-token",
        "S-M-4-cjk-split-token",
        "S-M-4-split-random-token",
    ],
)
@pytest.mark.asyncio
async def test_secret_split_across_delta_chunks_is_redacted(
    chunks: list[str], secret: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-M-4: a secret split across two assistant.delta chunks is redacted."""

    monkeypatch.setattr(runtime_module, "build_chat_model", lambda config: ChunkedModel(chunks))
    ingest = MemoryEventIngest()
    runtime = AgentRuntime(MemoryStateStore(), ingest=ingest)
    opened = await runtime.open_session(OpenSessionInput(room_id="room-1", turn_id="open-1"))

    result = await runtime.run_turn(
        RunTurnInput(
            room_id="room-1",
            session_id=opened.session_id,
            turn_id="turn-1",
            message="hello",
            state_version=opened.state_version,
        )
    )

    assert result.status == "completed"
    deltas = [event for event in ingest.events if event.type == "assistant.delta"]
    assert len(deltas) >= 2, "the secret must straddle two assistant.delta events"
    assert [event.seq for event in deltas] == list(range(len(deltas)))
    assert len({event.block_id for event in deltas}) == 1
    assert all(event.turn_id == "turn-1" and event.agent_id == "main" for event in deltas)
    streamed = "".join(event.delta for event in deltas)
    assert streamed == redact_text("".join(chunks))
    assert REDACTED in streamed
    for event in ingest.events:
        wire = event.model_dump_json(by_alias=True)
        assert secret not in wire
        assert secret[:5] not in wire
    # The second chunk alone does not look like a secret.
    if not secret.startswith("sk-"):
        assert redact_text(chunks[1]) == chunks[1]
