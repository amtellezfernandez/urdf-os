#!/usr/bin/env python3
"""Simple MCP client for interacting with the SO100 VLA server via SSE transport."""

import asyncio
import json
import sys
import base64
from pathlib import Path

from fastmcp import Client

SERVER_URL = "http://localhost:8765/sse"


def _tool_result_to_value(call_result):
    """Convert a CallToolResult into a plain Python value."""
    content = getattr(call_result, "content", None)
    if not isinstance(content, list) or not content:
        return None

    for item in content:
        text = None
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
        else:
            if getattr(item, "type", None) == "text":
                text = getattr(item, "text", None)
        if isinstance(text, str):
            try:
                return json.loads(text)
            except Exception:
                return text
    return content


def _save_image_from_result(call_result, out_path: Path) -> bool:
    """Extract and save image from result."""
    content = getattr(call_result, "content", None)
    if not isinstance(content, list) or not content:
        return False

    for item in content:
        data = None
        if isinstance(item, dict):
            data = item.get("data")
        else:
            data = getattr(item, "data", None)

        if isinstance(data, (bytes, bytearray)):
            out_path.write_bytes(bytes(data))
            return True
        if isinstance(data, str):
            try:
                out_path.write_bytes(base64.b64decode(data))
                return True
            except Exception:
                continue
    return False


async def run_command(cmd: str, args: list):
    """Run an MCP command."""
    async with Client(SERVER_URL, timeout=60, init_timeout=60) as client:
        if cmd == "list_tools":
            tools = await client.list_tools()
            print("Available tools:")
            for t in tools:
                print(f"  - {t.name}: {t.description[:80] if t.description else 'No description'}...")

        elif cmd == "list_cameras":
            result = await client.call_tool("list_cameras")
            value = _tool_result_to_value(result)
            print(f"Cameras: {value}")

        elif cmd == "list_serial_ports":
            result = await client.call_tool("list_serial_ports")
            value = _tool_result_to_value(result)
            print("Serial ports:")
            if isinstance(value, list):
                for port in value:
                    print(f"  {port}")
            else:
                print(f"  {value}")

        elif cmd == "get_camera_frame":
            cam_name = args[0] if args else "wrist"
            result = await client.call_tool("get_camera_frame", {"camera_name": cam_name})
            out_path = Path(f"frame_{cam_name}.jpg")
            if _save_image_from_result(result, out_path):
                print(f"Saved frame to {out_path}")
            else:
                print(f"Could not save frame. Result: {result}")

        elif cmd == "get_robot_camera_frame":
            cam_name = args[0] if args else "wrist"
            result = await client.call_tool("get_robot_camera_frame", {"camera_name": cam_name})
            out_path = Path(f"robot_frame_{cam_name}.jpg")
            if _save_image_from_result(result, out_path):
                print(f"Saved robot camera frame to {out_path}")
            else:
                print(f"Could not save frame. Result: {result}")

        elif cmd == "connect_robot":
            port = args[0] if args else "auto"
            result = await client.call_tool("connect_robot", {"port": port})
            value = _tool_result_to_value(result)
            print(f"Connect robot: {value}")

        elif cmd == "disconnect_robot":
            result = await client.call_tool("disconnect_robot")
            value = _tool_result_to_value(result)
            print(f"Disconnect: {value}")

        elif cmd == "get_robot_state":
            result = await client.call_tool("get_robot_state")
            value = _tool_result_to_value(result)
            print(f"Robot state: {json.dumps(value, indent=2)}")

        elif cmd == "enable_motion":
            enable = args[0].lower() in ("true", "1", "yes") if args else False
            result = await client.call_tool("enable_motion", {"enable": enable})
            value = _tool_result_to_value(result)
            print(f"Enable motion: {value}")

        elif cmd == "list_skills":
            result = await client.call_tool("list_skills")
            value = _tool_result_to_value(result)
            print("Skills:")
            if isinstance(value, list):
                for skill in value:
                    print(f"  - {skill.get('name')}: {skill.get('description', '')[:60]}...")
            else:
                print(f"  {value}")

        elif cmd == "start_skill":
            skill_name = args[0] if args else "home"
            instruction = args[1] if len(args) > 1 else ""
            max_steps = int(args[2]) if len(args) > 2 else 100
            result = await client.call_tool("start_skill", {
                "skill_name": skill_name,
                "instruction": instruction,
                "max_steps": max_steps
            })
            value = _tool_result_to_value(result)
            print(f"Start skill: {value}")

        elif cmd == "get_skill_status":
            skill_id = args[0] if args else ""
            result = await client.call_tool("get_skill_status", {"skill_id": skill_id})
            value = _tool_result_to_value(result)
            print(f"Skill status: {value}")

        elif cmd == "stop_skill":
            skill_id = args[0] if args else ""
            result = await client.call_tool("stop_skill", {"skill_id": skill_id})
            value = _tool_result_to_value(result)
            print(f"Stop skill: {value}")

        else:
            print(f"Unknown command: {cmd}")
            print("Available commands:")
            print("  list_tools, list_cameras, list_serial_ports, get_camera_frame <cam>,")
            print("  connect_robot <port>, disconnect_robot, get_robot_state,")
            print("  enable_motion <true|false>, list_skills,")
            print("  start_skill <name> [instruction] [max_steps],")
            print("  get_skill_status <id>, stop_skill <id>")


def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <command> [args...]")
        print("Commands: list_tools, list_cameras, list_serial_ports, get_camera_frame,")
        print("          connect_robot, disconnect_robot, get_robot_state, enable_motion,")
        print("          list_skills, start_skill, get_skill_status, stop_skill")
        return

    cmd = sys.argv[1]
    args = sys.argv[2:]
    asyncio.run(run_command(cmd, args))


if __name__ == "__main__":
    main()
