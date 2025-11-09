from collections import deque
from typing import Deque, List, Tuple, Optional

Coord = Tuple[int, int]

class RouteQueue:
    """FIFO of coordinates with peek and pop; immutable once set (versioned)."""
    
    def __init__(self):
        self._q: Deque[Coord] = deque()
        self.version: int = 0
    
    def load(self, path: List[Coord]) -> None:
        """Load a new path into the queue, clearing old one and incrementing version"""
        self._q.clear()
        if path:
            # Skip first node if it's the current position (we want to move forward)
            self._q.extend(path[1:] if len(path) > 1 else path)
        self.version += 1
    
    def peek(self) -> Optional[Coord]:
        """Peek at the next coordinate without removing it"""
        return self._q[0] if self._q else None
    
    def pop(self) -> Optional[Coord]:
        """Pop and return the next coordinate"""
        return self._q.popleft() if self._q else None
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return not self._q
    
    def discard_until(self, pos: Coord) -> None:
        """If current pos equals head (or passed it), drop it."""
        if self._q and self._q[0] == pos:
            self._q.popleft()
    
    def size(self) -> int:
        """Get the number of coordinates in the queue"""
        return len(self._q)

