"""Headroom as an MCP server — any MCP-compatible agent routes its actions through the guard.

This is the *cooperative* integration (Level 1 on the no-bypass ladder): a host agent (Claude
Code, Cursor, a custom agent) is configured to call these tools before it acts. Two tools:

  - ``submit_action_for_review(kind, tool, target, args)`` — classify a proposed action. Returns
    the verdict; SAFE means "go ahead", BLOCKED means "do not", and APPROVAL_REQUIRED returns
    ``{status: "pending", action_id}`` and queues it for a human.
  - ``check_review(action_id)`` — poll the human's decision (pending / approved / rejected /
    expired). The host agent proceeds on "approved", aborts on "rejected"/"expired".

It writes to the SAME audit + pending SQLite DB as the dashboard (``HEADROOM_DB``), so an
external agent's actions appear there and a human approves/rejects them in the same UI.

Run as a server:  python -m headroom.mcp_server   (stdio transport)
"""

from __future__ import annotations

import os
import uuid

from . import db
from .policy.guardian import classify, default_judge
from .types import ActionKind, ProposedAction

AUDIT_DB = os.getenv("HEADROOM_DB", "headroom.db")
TTL_MS = int(os.getenv("APPROVAL_TTL_MS", "120000"))
MCP_THREAD_PREFIX = "mcp-"  # marks externally-submitted actions (no LangGraph run to resume)

_UNSET = object()


def review_action(
    kind: str, tool: str, target: str, args: dict | None = None, judge=_UNSET
) -> dict:
    """Classify a proposed action, record it to the shared audit/pending DB, return the verdict.

    Returns ``{status, verdict, reason[, action_id]}`` where status is
    ``allow`` (SAFE) / ``blocked`` (BLOCKED) / ``pending`` (APPROVAL_REQUIRED, with an action_id).
    """
    db.init_db(AUDIT_DB)
    action = ProposedAction(kind=ActionKind(kind), tool=tool, target=target, args=args or {})
    decision = classify(action, default_judge() if judge is _UNSET else judge)
    verdict = decision.verdict.value
    thread_id = MCP_THREAD_PREFIX + uuid.uuid4().hex[:8]

    db.append_audit(
        AUDIT_DB,
        event="PROPOSED",
        thread_id=thread_id,
        action_id=action.id,
        kind=kind,
        target=target,
        verdict=verdict,
        reason=decision.reason,
        rule_id=decision.rule_id,
        source=decision.source.value,
    )

    result = {
        "verdict": verdict,
        "reason": decision.reason,
        "action_id": action.id,
        "source": decision.source.value,  # rule | llm | fail_safe — who decided
    }
    if verdict == "APPROVAL_REQUIRED":
        db.add_pending(
            AUDIT_DB,
            action_id=action.id,
            thread_id=thread_id,
            kind=kind,
            target=target,
            args=action.args,
            reason=decision.reason,
            rule_id=decision.rule_id,
            source=decision.source.value,
            ttl_ms=TTL_MS,
        )
        result["status"] = "pending"
    else:
        result["status"] = "blocked" if verdict == "BLOCKED" else "allow"
    return result


def get_review(action_id: str) -> dict:
    """Poll a pending action's human decision: pending / approved / rejected / expired / unknown."""
    db.init_db(AUDIT_DB)
    row = db.get_pending(AUDIT_DB, action_id)
    if row is None:
        return {"action_id": action_id, "status": "unknown"}
    return {"action_id": action_id, "status": row["status"].lower()}


def build_server():
    """Build the FastMCP server exposing the guard as two tools."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("Headroom")

    @mcp.tool()
    def submit_action_for_review(
        kind: str, tool: str, target: str, args: dict | None = None
    ) -> dict:
        """Submit a proposed agent action for safety review before executing it.

        kind: one of read, list, write, shell, git, create_pr, deploy.
        tool: the tool name (e.g. run_shell). target: the path / command / environment.
        Returns the verdict: status 'allow' (safe), 'blocked' (do not run), or 'pending'
        (a human must approve — poll check_review with the returned action_id).
        """
        return review_action(kind, tool, target, args)

    @mcp.tool()
    def check_review(action_id: str) -> dict:
        """Poll a human's decision on a pending action: pending / approved / rejected / expired."""
        return get_review(action_id)

    return mcp


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
