"""An artifact's media type does not depend on the host's mime table."""

import mimetypes

import pytest
from orbit_worker.sandbox import media_type_of


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("report.md", "text/markdown"), ("README.MARKDOWN", "text/markdown"), ("notes.txt", "text/plain"),
        ("data.json", "application/json"), ("t.csv", "text/csv"), ("c.yaml", "application/yaml"), ("c.yml", "application/yaml"),
        ("index.html", "text/html"), ("a.py", "text/x-python"), ("a.ts", "text/typescript"), ("a.tsx", "text/typescript"),
        ("a.js", "text/javascript"), ("q.sql", "application/sql"), ("run.sh", "application/x-sh"), ("a.pdf", "application/pdf"),
        ("a.png", "image/png"), ("a.jpg", "image/jpeg"), ("a.svg", "image/svg+xml"),
        ("a.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("a.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("a.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        ("dir/sub/deep.md", "text/markdown"),
    ],
)
def test_common_files_have_a_fixed_media_type(name: str, expected: str) -> None:
    assert media_type_of(name) == expected


def test_the_hosts_table_cannot_change_a_known_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mimetypes, "guess_type", lambda name: (None, None))  # a host that has never heard of .md
    assert media_type_of("report.md") == "text/markdown"


def test_other_files_fall_back_to_the_hosts_table_then_to_octet_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mimetypes, "guess_type", lambda name: ("audio/x-test", None) if name.endswith(".zzz") else (None, None))
    assert media_type_of("a.zzz") == "audio/x-test"
    assert media_type_of("a.unknownext") == "application/octet-stream"
