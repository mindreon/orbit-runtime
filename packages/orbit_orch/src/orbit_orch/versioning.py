"""How Orbit versions Temporal workflow code.

A change that adds, removes, or reorders a command in history (an activity,
a timer, a child workflow, a signal wait, or a version marker) must be
introduced behind ``workflow.patched(change_id)``.

Rules:

- ``change_id`` is ``orbit-<workflow>-<short-name>`` and is never reused.
- Keep the old branch until every open execution that started before the
  change has closed. Then delete the branch and the patch id in a later
  change.
- Do not edit the order of existing awaits on the unpatched path.

The ids below are the current control-compatible surface. New executions
record them so a later change can branch off this history.
"""

ROOM_CONTROL_SURFACE = "orbit-room-control-surface"
AGENT_RUN_SURFACE = "orbit-agent-run-surface"
CLOUD_JOB_SURFACE = "orbit-cloud-job-surface"
