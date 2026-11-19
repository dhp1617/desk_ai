"""Example user plugin tool for Buddy.
Save in ~/.config/buddy/tools/dice.py
"""
from __future__ import annotations

import random
from typing import Annotated

from buddy.tools.registry import tool


@tool
def roll_dice(sides: Annotated[int, "Number of sides on the die, e.g. 6 or 20"] = 6) -> str:
    """Roll a virtual polyhedral die and return the outcome."""
    if sides < 2:
        return "A die must have at least 2 sides."
    result = random.randint(1, sides)
    return f"🎲 Rolled a d{sides}: {result}"
