from typing import List, Tuple, Set, Optional
from collections import deque
import heapq

Coord = Tuple[int, int]

def astar(
    width: int,
    height: int,
    start: Coord,
    goal: Coord,
    obstacles: List[Tuple[int, int]],
    safety: int = 0,
    diag: bool = True,
    turn_penalty: float = 0.10
) -> Optional[List[Coord]]:
    """
    A* pathfinding algorithm
    
    Args:
        width: Grid width
        height: Grid height
        start: Starting coordinate (x, y)
        goal: Goal coordinate (x, y)
        obstacles: List of obstacle coordinates
        safety: Safety radius (cells to avoid around obstacles)
        diag: Allow diagonal moves
        turn_penalty: Penalty for turning (not used in simple version)
    
    Returns:
        List of coordinates from start to goal, or None if no path found
    """
    # Convert obstacles to set for fast lookup
    obstacle_set: Set[Coord] = set(obstacles)
    
    # Expand obstacles with safety radius
    expanded_obstacles: Set[Coord] = set(obstacles)
    if safety > 0:
        for ox, oy in obstacles:
            for dx in range(-safety, safety + 1):
                for dy in range(-safety, safety + 1):
                    nx, ny = ox + dx, oy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        expanded_obstacles.add((nx, ny))
    
    # Check if start or goal is blocked
    if start in expanded_obstacles or goal in expanded_obstacles:
        return None
    
    # Heuristic function (Manhattan distance)
    def heuristic(pos: Coord) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # Get neighbors
    def get_neighbors(pos: Coord) -> List[Coord]:
        x, y = pos
        neighbors = []
        
        # Orthogonal moves
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if (nx, ny) not in expanded_obstacles:
                    neighbors.append((nx, ny))
        
        # Diagonal moves
        if diag:
            for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if (nx, ny) not in expanded_obstacles:
                        neighbors.append((nx, ny))
        
        return neighbors
    
    # A* algorithm
    open_set = [(0, start)]  # (f_score, position)
    came_from: dict[Coord, Optional[Coord]] = {start: None}
    g_score: dict[Coord, float] = {start: 0}
    f_score: dict[Coord, float] = {start: heuristic(start)}
    closed_set: Set[Coord] = set()
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        if current == goal:
            # Reconstruct path
            path: List[Coord] = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        
        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue
            
            # Cost to move to neighbor (1 for orthogonal, sqrt(2) for diagonal)
            move_cost = 1.0
            if abs(neighbor[0] - current[0]) == 1 and abs(neighbor[1] - current[1]) == 1:
                move_cost = 1.414  # sqrt(2) for diagonal
            
            tentative_g = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return None

