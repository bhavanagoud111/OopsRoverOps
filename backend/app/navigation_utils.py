"""Navigation utilities for deterministic exit criteria and loop detection"""

from typing import Tuple, Set, Optional
from app.models.schemas import RoverPosition

def is_at_target(curr: Tuple[int, int], target: Optional[Tuple[int, int]], tol: float = 1e-6) -> bool:
    """
    Check if current position is at target with tolerance.
    Handles both int and float coordinates.
    
    Args:
        curr: Current position (x, y)
        target: Target position (x, y) or None
        tol: Tolerance for float comparison (default 1e-6)
    
    Returns:
        True if at target, False otherwise
    """
    if target is None:
        return False
    
    cx, cy = curr
    tx, ty = target
    
    # Handle both int and float coordinates
    return abs(cx - tx) <= tol and abs(cy - ty) <= tol

def calculate_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

