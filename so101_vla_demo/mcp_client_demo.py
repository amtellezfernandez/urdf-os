"""
SO101 VLA MCP client demo (no Claude required).

This script acts as an MCP client and exercises the same tools as the MCP server
(via an in-process MCP transport for portability):
  - list_cameras / get_camera_frame
  - connect_robot("mock") for safe testing
  - (optional) load a real SmolVLA checkpoint and run 1-2 policy steps

Examples:
  # Smoke test tools (no policy load)
  /home/USER/miniconda3/envs/lerobot/bin/python -m so101_vla_demo.mcp_client_demo

  # Run a real checkpoint in mock-robot mode (downloads from Hugging Face if needed)
  /home/USER/miniconda3/envs/lerobot/bin/python -m so101_vla_demo.mcp_client_demo \\
    --policy Gurkinator/smolvla_so101_policy --steps 2
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import asyncio
import contextlib
import importlib
from pathlib import Path
from typing import Any

from fastmcp import Client


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_for_server(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_repo_root() / "src")

    # Cameras for the MCP server's camera tools (independent of robot_interface cameras).
    if args.camera_sources:
        env["SO101_CAMERA_SOURCES"] = args.camera_sources
    if args.camera_names:
        env["SO101_CAMERA_NAMES"] = args.camera_names

    # Robot connection. Default to mock to avoid moving hardware unexpectedly.
    env["USE_MOCK_ROBOT"] = "false"
    if args.port:
        env["SO101_PORT"] = args.port

    # Optional policy configuration (can be overridden via set_policy tool too).
    if args.policy:
        env["SMOLVLA_POLICY_ID"] = args.policy
    if args.policy_camera_map:
        env["SO101_POLICY_CAMERA_MAP"] = args.policy_camera_map

    return env


def _save_first_image_result(call_result: Any, out_path: Path) -> bool:
    """
    Best-effort decode of the first image payload in a CallToolResult.

    fastmcp returns content items that may contain base64-encoded bytes.
    We keep this very defensive because fastmcp versions differ slightly.
    """
    content = getattr(call_result, "content", None)
    if not isinstance(content, list) or not content:
        return False

    for item in content:
        # Common shapes: {"type":"image","data":"...base64...","mimeType":"image/jpeg"}
        if isinstance(item, dict):
            data = item.get("data")
            if isinstance(data, (bytes, bytearray)):
                out_path.write_bytes(bytes(data))
                return True
            if isinstance(data, str):
                try:
                    out_path.write_bytes(base64.b64decode(data))
                    return True
                except Exception:  # noqa: BLE001
                    continue
        # Some versions wrap objects with attrs.
        data = getattr(item, "data", None)
        if isinstance(data, (bytes, bytearray)):
            out_path.write_bytes(bytes(data))
            return True
        if isinstance(data, str):
            try:
                out_path.write_bytes(base64.b64decode(data))
                return True
            except Exception:  # noqa: BLE001
                continue
    return False


def _tool_result_to_value(call_result: Any) -> Any:
    """
    Convert a CallToolResult into a plain Python value when possible.

    Many MCP servers return structured results as a single JSON string inside a TextContent item.
    """
    content = getattr(call_result, "content", None)
    if not isinstance(content, list) or not content:
        return None

    # Prefer the first text item; fall back to raw content list.
    for item in content:
        text = None
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
        else:
            if getattr(item, "type", None) == "text":
                text = getattr(item, "text", None)
        if isinstance(text, str):
            try:
                import json

                return json.loads(text)
            except Exception:  # noqa: BLE001
                return text
    return content


async def _run(args: argparse.Namespace) -> int:
    # Some environments disallow local socket creation; for maximum portability,
    # we test via an in-process MCP transport (same tools, no HTTP/stdio).
    env = _env_for_server(args)

    report: dict[str, Any] = {"robot": args.robot, "policy": args.policy or None, "transport": "inprocess"}

    @contextlib.contextmanager
    def _temp_env(overrides: dict[str, str]):
        old = dict(os.environ)
        os.environ.update(overrides)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(old)

    with _temp_env(env):
        from so100_vla_demo import mcp_server as mcp_mod

        importlib.reload(mcp_mod)

        async with Client(mcp_mod.mcp, timeout=60, init_timeout=60) as client:
            tools = await client.list_tools()
            report["tools"] = [t.name for t in tools]

            report["cameras"] = _tool_result_to_value(await client.call_tool("list_cameras"))

            frame_result = await client.call_tool("get_camera_frame", {"camera_name": "wrist"})
            frame_path = Path(args.save_frame).expanduser().resolve()
            report["saved_frame"] = str(frame_path)
            report["saved_frame_ok"] = _save_first_image_result(frame_result, frame_path)

            report["connect_robot"] = _tool_result_to_value(
                await client.call_tool("connect_robot", {})
            )
            report["skills"] = _tool_result_to_value(await client.call_tool("list_skills"))

            if args.policy:
                report["set_policy"] = _tool_result_to_value(
                    await client.call_tool("set_policy", {"policy_name": "smolvla", "policy_id": args.policy})
                )
                started = _tool_result_to_value(
                    await client.call_tool(
                        "start_skill",
                        {
                            "skill_name": "smolvla",
                            "instruction": "pick up the red cup",
                            "max_steps": int(max(1, args.steps)),
                        },
                    )
                )
                report["start_skill"] = started
                skill_id = started.get("skill_id") if isinstance(started, dict) else None

                samples: list[Any] = []
                if isinstance(skill_id, str) and skill_id:
                    for _ in range(5):
                        await asyncio.sleep(1.0)
                        status = _tool_result_to_value(await client.call_tool("get_skill_status", {"skill_id": skill_id}))
                        samples.append(status)
                        if isinstance(status, dict) and status.get("status") in {"completed", "stopped", "error"}:
                            break
                    # If still running, request stop (useful when the checkpoint is still downloading/loading).
                    if samples and isinstance(samples[-1], dict) and samples[-1].get("status") == "running":
                        report["stop_skill"] = _tool_result_to_value(await client.call_tool("stop_skill", {"skill_id": skill_id}))
                report["skill_status_samples"] = samples

    # Print report as plain text (CLI-friendly).
    print("MCP client demo report:")
    for k, v in report.items():
        print(f"- {k}: {v}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SO101 VLA MCP client demo")
    parser.add_argument("--port", default=None, help="Serial port for real robot (default: SO101_PORT env)")
    parser.add_argument("--camera-sources", default=None, help='Comma-separated sources (e.g. "/dev/video4,/dev/video6" or "0,2")')
    parser.add_argument("--camera-names", default=None, help='Comma-separated names (e.g. "wrist,overhead")')
    parser.add_argument("--policy", default="", help="SmolVLA policy checkpoint (HF repo_id or local path)")
    parser.add_argument(
        "--policy-camera-map",
        default='{"front":"overhead","wrist":"wrist"}',
        help='JSON mapping policy camera names -> robot camera names',
    )
    parser.add_argument("--steps", type=int, default=1, help="Max steps for start_skill")
    parser.add_argument("--save-frame", default="mcp_demo_frame.jpg", help="Save one camera frame to this path")
    args = parser.parse_args()

    return int(asyncio.run(_run(args)))


if __name__ == "__main__":
    raise SystemExit(main())
