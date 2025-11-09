from enum import Enum
from typing import Tuple

Coord = Tuple[int, int]

class Phase(str, Enum):
    PLAN = "PLAN"
    EXECUTE_TASK = "EXECUTE_TASK"
    RETURN_TO_BASE = "RETURN_TO_BASE"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"

def at_same(a: Coord, b: Coord) -> bool:
    """Check if two coordinates are the same"""
    return a[0] == b[0] and a[1] == b[1]

def at_base(pos: Coord, base: Coord, tol: int = 0) -> bool:
    """Check if position is at base with tolerance (grid tolerance: 0=exact, 1=within 1 cell)"""
    return max(abs(pos[0] - base[0]), abs(pos[1] - base[1])) <= tol

