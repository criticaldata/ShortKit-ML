"""Tests for the checked-in Copilot / MCP workspace configuration."""

import json
from pathlib import Path


def test_vscode_mcp_config_matches_project_entrypoint():
    root = Path(__file__).resolve().parents[1]
    claude_config = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))
    vscode_config = json.loads((root / ".vscode" / "mcp.json").read_text(encoding="utf-8"))

    claude_server = claude_config["mcpServers"]["shortkit-ml"]
    vscode_server = (vscode_config.get("mcpServers") or vscode_config.get("servers"))["shortkit-ml"]

    assert claude_server["type"] == "stdio"
    assert vscode_server["type"] == "stdio"
    assert claude_server["command"] == vscode_server["command"] == ".venv/bin/python"
    assert claude_server["args"] == vscode_server["args"] == [
        "-m",
        "shortcut_detect.mcp_server",
    ]
