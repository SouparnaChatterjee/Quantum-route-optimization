"""
Classical Traffic Grid Simulator for comparison
"""
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Vehicle:
    def __init__(self, id: int, start: Tuple[int, int], dest: Tuple[int, int], grid_size: int):
        self.id = id
        self.x, self.y = start
        self.dest_x, self.dest_y = dest
        self.grid_size = grid_size
        self.path = [start]
        self.stuck_count = 0
        
    def manhattan_distance(self) -> int:
        return abs(self.x - self.dest_x) + abs(self.y - self.dest_y)
    
    def at_destination(self) -> bool:
        return self.x == self.dest_x and self.y == self.dest_y

class TrafficSimulator:
    def __init__(self, grid_size: int = 10, num_vehicles: int = 5):
        self.grid_size = grid_size
        self.num_vehicles = num_vehicles
        self.vehicles: List[Vehicle] = []
        self.grid = np.zeros((grid_size, grid_size))
        self.collision_count = 0
        self.total_steps = 0
        self.completed_vehicles = 0
        
    def initialize_vehicles(self):
        """Randomly place vehicles on grid"""
        self.vehicles = []
        occupied = set()
        
        for i in range(self.num_vehicles):
            # Find unoccupied start position
            while True:
                start = (np.random.randint(0, self.grid_size), 
                        np.random.randint(0, self.grid_size))
                if start not in occupied:
                    occupied.add(start)
                    break
            
            # Find different destination
            while True:
                dest = (np.random.randint(0, self.grid_size), 
                       np.random.randint(0, self.grid_size))
                if dest != start:
                    break
            
            vehicle = Vehicle(i, start, dest, self.grid_size)
            self.vehicles.append(vehicle)
            
    def get_possible_moves(self, vehicle: Vehicle) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        moves = []
        x, y = vehicle.x, vehicle.y
        
        # Four directions: up, down, left, right
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                moves.append((new_x, new_y))
        
        return moves
    
    def classical_greedy_move(self, vehicle: Vehicle, occupied: set) -> Tuple[int, int]:
        """Classical greedy algorithm: move closer to destination"""
        possible = self.get_possible_moves(vehicle)
        
        # Filter out occupied positions
        valid_moves = [(x, y) for x, y in possible if (x, y) not in occupied]
        
        if not valid_moves:
            return (vehicle.x, vehicle.y)  # Stay put if blocked
        
        # Choose move that minimizes Manhattan distance
        best_move = min(valid_moves, 
                       key=lambda pos: abs(pos[0] - vehicle.dest_x) + abs(pos[1] - vehicle.dest_y))
        
        return best_move
    
    def step(self, use_quantum: bool = False):
        """Perform one simulation step"""
        occupied = {(v.x, v.y) for v in self.vehicles if not v.at_destination()}
        
        for vehicle in self.vehicles:
            if vehicle.at_destination():
                continue
                
            # Remove current position from occupied
            occupied.discard((vehicle.x, vehicle.y))
            
            # Get next move
            next_pos = self.classical_greedy_move(vehicle, occupied)
            
            # Check for collision
            if next_pos in occupied:
                self.collision_count += 1
                vehicle.stuck_count += 1
            else:
                vehicle.x, vehicle.y = next_pos
                vehicle.path.append(next_pos)
                occupied.add(next_pos)
                vehicle.stuck_count = 0
                
                if vehicle.at_destination():
                    self.completed_vehicles += 1
        
        self.total_steps += 1
    
    def get_metrics(self) -> Dict:
        """Return simulation metrics"""
        total_distance = sum(v.manhattan_distance() for v in self.vehicles)
        avg_path_length = np.mean([len(v.path) for v in self.vehicles])
        
        return {
            'total_steps': self.total_steps,
            'collisions': self.collision_count,
            'completed': self.completed_vehicles,
            'remaining_distance': total_distance,
            'avg_path_length': avg_path_length,
            'collision_rate': self.collision_count / max(self.total_steps, 1)
        }
    
    def visualize(self):
        """Visualize current state"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw grid
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Draw vehicles and destinations
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_vehicles))
        
        for i, vehicle in enumerate(self.vehicles):
            # Vehicle position
            ax.plot(vehicle.x, vehicle.y, 'o', color=colors[i], 
                   markersize=15, label=f'Vehicle {i}')
            
            # Destination
            ax.plot(vehicle.dest_x, vehicle.dest_y, 'x', color=colors[i], 
                   markersize=15, markeredgewidth=3)
            
            # Path
            if len(vehicle.path) > 1:
                path_x = [p[0] for p in vehicle.path]
                path_y = [p[1] for p in vehicle.path]
                ax.plot(path_x, path_y, '--', color=colors[i], alpha=0.5)
        
        ax.legend()
        ax.set_title(f'Traffic Simulation - Step {self.total_steps}')
        plt.show()

if __name__ == "__main__":
    print("Testing Classical Traffic Simulator...")
    sim = TrafficSimulator(grid_size=8, num_vehicles=5)
    sim.initialize_vehicles()
    
    print("\nRunning 20 steps...")
    for i in range(20):
        sim.step()
        if (i + 1) % 5 == 0:
            print(f"Step {i+1}: {sim.get_metrics()}")
    
    print("\nFinal visualization:")
    sim.visualize()