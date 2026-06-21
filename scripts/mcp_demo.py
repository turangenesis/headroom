"""Demo: act as an EXTERNAL agent plugging into Headroom over MCP.

Launches the Headroom MCP server (stdio) and, as a separate "host agent", submits three actions
through it — a safe read, a destructive shell command, and a production deploy — printing the
guard's verdict for each. This is exactly what a real client (Claude Code / Cursor) does after a
~4-line config change; here we prove the loop end-to-end with no real client needed.

Run (self-contained):  python scripts/mcp_demo.py
    The deploy lands as 'pending'; the script polls check_review once and exits. Throwaway DB.

Run (LIVE, two terminals — the impressive version):
    1)  uvicorn headroom.api:app                       # dashboard at http://localhost:8000
    2)  HEADROOM_DB=headroom.db python scripts/mcp_demo.py
    The 'pending' deploy now appears in the dashboard's approval panel. Click Approve/Reject and
    the script — the external agent — watches check_review flip and proceeds. That is the full
    human-in-the-loop, end to end across MCP.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

# Make `headroom` importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ACTIONS = [
    {"kind": "read", "tool": "read_file", "target": "src/index.ts"},
    {"kind": "shell", "tool": "run_shell", "target": "rm -rf /"},
    # Fuzzy / ambiguous-middle: NO deterministic rule covers "install a dependency". With an
    # ANTHROPIC_API_KEY set this falls to the LLM judge (source=llm); without a key it fail-safes
    # to APPROVAL_REQUIRED (source=fail_safe). This is the slice the guard actually reasons about.
    {"kind": "shell", "tool": "run_shell", "target": "npm install lodash"},
    {"kind": "deploy", "tool": "deploy", "target": "production"},
]


def _unwrap(result) -> dict:
    """Pull the structured dict out of an MCP tool-call result, across SDK shapes."""
    data = getattr(result, "structuredContent", None)
    if isinstance(data, dict):
        return data.get("result", data)
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
    return {"raw": str(result)}


async def _await_human_decision(session, action_id: str, live: bool, timeout_s: int = 90) -> None:
    """A pending action needs a human. In LIVE mode, poll check_review until a human
    approves/rejects on the dashboard (or we time out → fail-safe deny). Otherwise poll once."""
    if not live:
        poll = _unwrap(await session.call_tool("check_review", {"action_id": action_id}))
        print(
            f"          (action_id={action_id} — check_review: {poll['status']}; run LIVE with "
            f"HEADROOM_DB=headroom.db + the dashboard to approve it for real)"
        )
        return

    print(
        f"          action_id={action_id} — waiting for a human on the dashboard ",
        end="",
        flush=True,
    )
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        poll = _unwrap(await session.call_tool("check_review", {"action_id": action_id}))
        if poll["status"] != "pending":
            print(
                f"\n          -> human {poll['status'].upper()} it; the agent proceeds accordingly."
            )
            return
        print(".", end="", flush=True)
        await asyncio.sleep(2)
    print("\n          -> timed out (fail-safe: treated as not approved).")


async def main() -> None:
    # Share the dashboard's DB when HEADROOM_DB is set, so the pending action shows up in the live
    # UI for a human to approve. Otherwise use a throwaway DB (fully self-contained).
    shared_db = os.environ.get("HEADROOM_DB")
    db_path = shared_db or os.path.join(tempfile.mkdtemp(), "mcp_demo.db")
    # If ANTHROPIC_API_KEY is set (shell env or .env), the fuzzy action hits the real LLM judge
    # (source=llm); otherwise it fail-safes to APPROVAL_REQUIRED. The three rule-covered actions
    # never call the LLM, so a run costs at most one tiny judge call (~$0.0002).
    env = {**os.environ, "HEADROOM_DB": db_path}
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "headroom.mcp_server"], env=env
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = [t.name for t in (await session.list_tools()).tools]
        print(f"connected to Headroom MCP server. tools: {tools}")
        if shared_db:
            print(f"LIVE: sharing dashboard DB ({db_path}) — approvals happen in the browser")
        print()

        for a in ACTIONS:
            res = _unwrap(await session.call_tool("submit_action_for_review", a))
            status = res.get("status", "?").upper()
            src = res.get("source", "?")  # rule | llm | fail_safe — who decided
            print(
                f"  {a['kind']:7} {a['target']:28} -> {status:9}[{src:9}] {res.get('reason', '')}"
            )
            if res.get("status") == "pending":
                await _await_human_decision(session, res["action_id"], live=bool(shared_db))

    print("\nThat is the whole integration: a host agent asks before it acts; the guard answers.")


if __name__ == "__main__":
    asyncio.run(main())
