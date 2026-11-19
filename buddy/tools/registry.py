"""Tool registry with decorator-based schema generation."""
from __future__ import annotations

import asyncio
import functools
import importlib
import inspect
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

logger = logging.getLogger(__name__)

_PYTHON_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to JSON Schema."""
    origin = get_origin(annotation)

    # Literal -> enum
    if origin is Literal:
        values = list(get_args(annotation))
        base = type(values[0]) if values else str
        return {"type": _PYTHON_TO_JSON.get(base, "string"), "enum": values}

    # list[X]
    if origin is list:
        args = get_args(annotation)
        if args:
            return {"type": "array", "items": _type_to_schema(args[0])}
        return {"type": "array"}

    # dict[str, X]
    if origin is dict:
        return {"type": "object"}

    # Union (Optional)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _type_to_schema(args[0])
        return {"type": "string"}

    # Basic types
    if annotation in _PYTHON_TO_JSON:
        return {"type": _PYTHON_TO_JSON[annotation]}

    return {"type": "string"}


def _extract_annotated(annotation: Any) -> tuple[Any, str]:
    """Extract base type and description from Annotated[T, 'desc']."""
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base = args[0]
        desc = ""
        for a in args[1:]:
            if isinstance(a, str):
                desc = a
                break
        return base, desc
    return annotation, ""


def _build_schema(func: Callable) -> dict[str, Any]:
    """Build OpenAI-compatible function schema from type hints."""
    hints = get_type_hints(func, include_extras=True)
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        ann = hints.get(name, str)
        base_type, desc = _extract_annotated(ann)
        prop = _type_to_schema(base_type)
        if desc:
            prop["description"] = desc

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif param.default is not None:
            prop["default"] = param.default

    schema: dict[str, Any] = {
        "type": "function",
        "name": func.__name__,
        "description": doc,
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }
    if required:
        schema["parameters"]["required"] = required

    return schema


def _coerce_arg(value: Any, target_type: Any) -> Any:
    """Coerce a value to the target type."""
    base, _ = _extract_annotated(target_type)
    origin = get_origin(base)

    if origin is Literal:
        # Check if value is valid
        valid = get_args(base)
        if value in valid:
            return value
        # Try type coercion
        for v in valid:
            if str(value) == str(v):
                return v
        return value

    if base is bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if base is int:
        return int(value)
    if base is float:
        return float(value)
    if base is str:
        return str(value)
    if origin is list and isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return [value]

    return value


class ToolRegistry:
    """Registry of callable tools with JSON schema generation."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._timeout = 30.0

    def register(self, func: Callable, *, name: str | None = None) -> Callable:
        """Register a function as a tool."""
        tool_name = name or func.__name__
        self._tools[tool_name] = func
        schema = _build_schema(func)
        if name:
            schema["name"] = name
        self._schemas[tool_name] = schema
        logger.debug("Registered tool: %s", tool_name)
        return func

    def tool(self, func: Callable | None = None, *, name: str | None = None) -> Any:
        """Decorator to register a tool."""
        if func is None:
            return functools.partial(self.tool, name=name)
        return self.register(func, name=name)

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Return all tool schemas."""
        return list(self._schemas.values())

    def get_schema(self, name: str) -> dict[str, Any] | None:
        """Return schema for a specific tool."""
        return self._schemas.get(name)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    async def run_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name, returning result or error string."""
        if name not in self._tools:
            return f"Error: Unknown tool '{name}'"

        func = self._tools[name]
        hints = get_type_hints(func, include_extras=True)

        # Coerce arguments
        coerced: dict[str, Any] = {}
        sig = inspect.signature(func)
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            if pname in args:
                target = hints.get(pname, str)
                try:
                    coerced[pname] = _coerce_arg(args[pname], target)
                except (ValueError, TypeError) as e:
                    return f"Error: Invalid argument '{pname}': {e}"
            elif param.default is not inspect.Parameter.empty:
                coerced[pname] = param.default
            else:
                return f"Error: Missing required argument '{pname}'"

        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(**coerced), timeout=self._timeout)
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, functools.partial(func, **coerced)),
                    timeout=self._timeout,
                )
            return str(result)
        except asyncio.TimeoutError:
            return f"Error: Tool '{name}' timed out after {self._timeout}s"
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return f"Error: {type(e).__name__}: {e}"

    def load_plugins(self, plugin_dir: Path) -> int:
        """Load plugin tools from .py files in directory. Returns count loaded."""
        if not plugin_dir.is_dir():
            return 0
        count = 0
        for path in sorted(plugin_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"buddy_plugin_{path.stem}", path,
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    # Inject the global registry's tool decorator
                    mod.tool = self.tool  # type: ignore
                    sys.modules[spec.name] = mod
                    spec.loader.exec_module(mod)
                    count += 1
                    logger.info("Loaded plugin: %s", path.name)
            except Exception:
                logger.exception("Failed to load plugin: %s", path.name)
        return count

    def set_timeout(self, seconds: float) -> None:
        self._timeout = seconds


# Global default registry
_default_registry = ToolRegistry()


def tool(func: Callable | None = None, *, name: str | None = None) -> Any:
    """Module-level decorator that registers with the default registry."""
    if func is None:
        return functools.partial(tool, name=name)
    return _default_registry.register(func, name=name)


def get_registry() -> ToolRegistry:
    """Get the default tool registry."""
    return _default_registry
