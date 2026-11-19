"""Built-in tools for Buddy."""
from __future__ import annotations

import datetime
import json
import logging
import subprocess
import base64
from typing import Annotated, Literal

from buddy.tools.registry import tool

logger = logging.getLogger(__name__)

_config = None  # Set by init_builtins()
_memory_store = None  # Set by init_builtins()


def init_builtins(config, memory_store=None) -> None:
    global _config, _memory_store
    _config = config
    _memory_store = memory_store


def _parse_yaml_time(val) -> str:
    """Convert YAML time values: PyYAML reads 10:30 as int 630."""
    if isinstance(val, int):
        hours = val // 60
        minutes = val % 60
        return f"{hours:02d}:{minutes:02d}"
    return str(val)


def _get_day_name(day: str) -> str:
    """Normalize day input to capitalized day name."""
    today = datetime.date.today()
    if day.lower() == "today":
        return today.strftime("%A")
    if day.lower() == "tomorrow":
        return (today + datetime.timedelta(days=1)).strftime("%A")
    return day.capitalize()


def _load_yaml_file(filename: str) -> dict:
    """Load a YAML file from memory dir with time fixing."""
    import yaml

    if _config is None:
        return {}
    path = _config.memory_dir / filename
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def _fix_times_in_schedule(entries: list) -> list:
    """Fix time values in schedule entries."""
    fixed = []
    for entry in entries:
        if isinstance(entry, dict):
            e = dict(entry)
            if "time" in e:
                e["time"] = _parse_yaml_time(e["time"])
            fixed.append(e)
        else:
            fixed.append(entry)
    return fixed


@tool
def get_time() -> str:
    """Get the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


@tool
def get_schedule(day: Annotated[str, "Day name or 'today'/'tomorrow'"] = "today") -> str:
    """Get the class schedule for a given day."""
    day_name = _get_day_name(day)
    data = _load_yaml_file("timetable.yaml")
    if not data:
        return "No timetable found. Add timetable.yaml to memory directory."
    schedule = data.get(day_name, [])
    if not schedule:
        return f"No classes scheduled for {day_name}."
    schedule = _fix_times_in_schedule(schedule)
    lines = [f"Schedule for {day_name}:"]
    for entry in schedule:
        if isinstance(entry, dict):
            t = entry.get("time", "?")
            c = entry.get("class", entry.get("name", "?"))
            r = entry.get("room", "")
            line = f"  {t} - {c}"
            if r:
                line += f" ({r})"
            lines.append(line)
    return "\n".join(lines)


@tool
def next_class() -> str:
    """Find the next upcoming class."""
    now = datetime.datetime.now()
    day_name = now.strftime("%A")
    current_time = now.strftime("%H:%M")

    data = _load_yaml_file("timetable.yaml")
    if not data:
        return "No timetable found."

    # Check today first
    schedule = _fix_times_in_schedule(data.get(day_name, []))
    for entry in schedule:
        if isinstance(entry, dict):
            t = entry.get("time", "00:00")
            if t > current_time:
                c = entry.get("class", entry.get("name", "?"))
                r = entry.get("room", "")
                result = f"Next class: {c} at {t}"
                if r:
                    result += f" in {r}"
                return result

    # Check subsequent days
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    today_idx = days.index(day_name) if day_name in days else 0
    for offset in range(1, 8):
        next_day = days[(today_idx + offset) % 7]
        schedule = _fix_times_in_schedule(data.get(next_day, []))
        if schedule:
            entry = schedule[0]
            if isinstance(entry, dict):
                t = entry.get("time", "?")
                c = entry.get("class", entry.get("name", "?"))
                r = entry.get("room", "")
                result = f"Next class: {c} on {next_day} at {t}"
                if r:
                    result += f" in {r}"
                return result

    return "No upcoming classes found."


@tool
def get_mess_menu(
    meal: Annotated[str, "Meal: breakfast/lunch/snacks/dinner or morning/afternoon/evening/night. Empty=current/next."] = "",
    day: Annotated[str, "Day name or 'today'/'tomorrow'. Empty=today."] = "today",
) -> str:
    """Get the mess/dining menu for a meal."""
    aliases = {
        "morning": "breakfast",
        "afternoon": "lunch",
        "evening": "snacks",
        "night": "dinner",
    }

    now = datetime.datetime.now()
    day_name = _get_day_name(day)
    hour = now.hour

    if not meal:
        # Determine current/next meal from time
        if hour < 10:
            meal = "breakfast"
        elif hour < 14:
            meal = "lunch"
        elif hour < 17:
            meal = "snacks"
        else:
            meal = "dinner"
    else:
        meal = aliases.get(meal.lower(), meal.lower())

    data = _load_yaml_file("mess.yaml")
    if not data:
        return "No mess menu found. Add mess.yaml to memory directory."

    day_menu = data.get(day_name, {})
    if not day_menu:
        return f"No menu found for {day_name}."

    menu_item = day_menu.get(meal, "")
    if not menu_item:
        return f"No {meal} menu found for {day_name}."

    return f"{meal.capitalize()} on {day_name}: {menu_item}"


@tool
def open_url(url: Annotated[str, "URL to open in browser"]) -> str:
    """Open a URL in the default browser."""
    try:
        subprocess.Popen(
            ["xdg-open", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Opened {url}"
    except Exception as e:
        return f"Error opening URL: {e}"


@tool
def open_classroom() -> str:
    """Open Google Classroom in the browser."""
    return open_url("https://classroom.google.com")


@tool
def open_app(app_name: Annotated[str, "Application name from allowlist"]) -> str:
    """Open an application from the configured allowlist."""
    if _config is None:
        return "Error: Not configured."
    allowed = _config.tools.allowed_apps
    if app_name.lower() not in [a.lower() for a in allowed]:
        return f"Error: '{app_name}' is not in the allowed apps list: {', '.join(allowed)}"
    try:
        subprocess.Popen(
            [app_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Opened {app_name}"
    except FileNotFoundError:
        return f"Error: '{app_name}' not found. Is it installed?"
    except Exception as e:
        return f"Error opening {app_name}: {e}"


@tool
def look_at_screen() -> str:
    """Take a screenshot and describe what's on screen using vision."""
    import shutil
    import tempfile

    # Capture screenshot
    screenshot_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            screenshot_path = f.name

        if shutil.which("grim"):
            subprocess.run(["grim", screenshot_path], check=True, timeout=5)
        elif shutil.which("maim"):
            subprocess.run(["maim", screenshot_path], check=True, timeout=5)
        else:
            return "Error: No screenshot tool found. Install grim (Wayland) or maim (X11)."

        # Convert to JPEG for smaller payload
        from PIL import Image

        img = Image.open(screenshot_path)
        jpeg_path = screenshot_path.replace(".png", ".jpg")
        img.save(jpeg_path, "JPEG", quality=85)

        with open(jpeg_path, "rb") as f:
            img_bytes = f.read()

        # Send to vision endpoint
        import httpx

        if _config is None:
            return "Error: Not configured."

        b64_image = base64.b64encode(img_bytes).decode()
        data_uri = f"data:image/jpeg;base64,{b64_image}"

        response = httpx.post(
            f"{_config.server.base_url}/v1/chat/completions",
            json={
                "model": "lfm3",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe what you see on screen."},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                "max_tokens": 512,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error capturing/analyzing screen: {e}"
    finally:
        import os
        if screenshot_path and os.path.exists(screenshot_path):
            os.unlink(screenshot_path)
        jpeg = screenshot_path.replace(".png", ".jpg") if screenshot_path else None
        if jpeg and os.path.exists(jpeg):
            os.unlink(jpeg)
