"""
Complete traffic simulation environment with collision detection,
congestion tracking, and multi-vehicle coordination.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field
import time

@dataclass
class Vehicle:
    """Represents a vehicle in the traffic grid"""
    id: int
    position: Tuple[int, int]
    destination: Tuple[int, int]
    path: List[Tuple[int, int]] = field(default_factory=list)
    current_step: int = 0
    completed: bool = False
    total_time: int = 0
    waiting_time: int = 0
    collisions: int = 0
    path_length: int = 0
    
    def move_next(self) -> bool:
        """
        Move to next position in path.
        Returns True if moved, False if completed or no path.
        """
        if self.completed or not self.path or self.current_step >= len(self.path):
            return False
        
        self.current_step += 1
        self.total_time += 1
        
        if self.current_step < len(self.path):
            self.position = self.path[self.current_step]
            
            # Check if reached destination
            if self.position == self.destination:
                self.completed = True
                return False
        
        return True
    
    def reset_path(self, new_path: List[Tuple[int, int]]):
        """Reset vehicle path (for re-routing)"""
        self.path = new_path
        self.current_step = 0
        self.path_length = len(new_path)


class TrafficSimulator:
    """
    Complete traffic simulation with multiple vehicles,
    collision detection, and congestion tracking.
    """
    
    def __init__(self, grid_size: int, num_vehicles: int, obstacle_density: float = 0.1):
        """
        Initialize traffic simulator.
        
        Args:
            grid_size: Size of the grid (N x N)
            num_vehicles: Number of vehicles to simulate
            obstacle_density: Fraction of grid cells that are obstacles
        """
        self.grid_size = grid_size
        self.num_vehicles = num_vehicles
        self.vehicles: List[Vehicle] = []
        self.obstacles: Set[Tuple[int, int]] = set()
        
        # Tracking metrics
        self.congestion_map = np.zeros((grid_size, grid_size))
        self.total_collisions = 0
        self.steps_completed = 0
        self.collision_log = []
        
        # Performance tracking
        self.quantum_times = []
        self.classical_times = []
        
        # Generate obstacles
        self._generate_obstacles(obstacle_density)
    
    def _generate_obstacles(self, density: float):
        """Generate random obstacles on the grid"""
        num_obstacles = int(self.grid_size * self.grid_size * density)
        
        while len(self.obstacles) < num_obstacles:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            self.obstacles.add((x, y))
    
    def initialize_vehicles(self):
        """Initialize vehicles with random start/end positions"""
        self.vehicles = []
        
        for i in range(self.num_vehicles):
            # Random start and destination (avoiding obstacles)
            while True:
                start = (
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                )
                if start not in self.obstacles:
                    break
            
            while True:
                dest = (
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                )
                if dest not in self.obstacles and dest != start:
                    break
            
            vehicle = Vehicle(
                id=i,
                position=start,
                destination=dest
            )
            self.vehicles.append(vehicle)
    
    def step(self, use_quantum: bool = False, router=None):
        """
        Execute one simulation step.
        
        Args:
            use_quantum: Whether to use quantum routing
            router: Router object (quantum or classical)
        """
        self.steps_completed += 1
        
        # Update congestion map
        self._update_congestion()
        
        # Move vehicles
        positions_this_step = {}
        
        for vehicle in self.vehicles:
            if vehicle.completed:
                continue
            
            # Re-route if no path or using adaptive routing
            if not vehicle.path or vehicle.current_step >= len(vehicle.path):
                if router:
                    start_time = time.time()
                    if use_quantum:
                        path = router.quantum_route_optimization(
                            vehicle.position,
                            vehicle.destination,
                            self.grid_size,
                            list(self.obstacles)
                        )
                    else:
                        path = router.a_star(
                            vehicle.position,
                            vehicle.destination,
                            self.obstacles,
                            self.congestion_map
                        )
                    elapsed = time.time() - start_time
                    
                    if use_quantum:
                        self.quantum_times.append(elapsed)
                    else:
                        self.classical_times.append(elapsed)
                    
                    vehicle.reset_path(path)
            
            # Move vehicle
            old_pos = vehicle.position
            if vehicle.move_next():
                new_pos = vehicle.position
                
                # Check for collisions
                if new_pos in positions_this_step:
                    # Collision detected
                    self.total_collisions += 1
                    vehicle.collisions += 1
                    vehicle.waiting_time += 1
                    vehicle.position = old_pos  # Stay in place
                    
                    self.collision_log.append({
                        'step': self.steps_completed,
                        'vehicles': [vehicle.id, positions_this_step[new_pos]],
                        'position': new_pos
                    })
                else:
                    positions_this_step[new_pos] = vehicle.id
            else:
                positions_this_step[vehicle.position] = vehicle.id
    
    def _update_congestion(self):
        """Update congestion map based on vehicle positions"""
        # Decay existing congestion
        self.congestion_map *= 0.9
        
        # Add current vehicle positions
        for vehicle in self.vehicles:
            if not vehicle.completed:
                x, y = vehicle.position
                self.congestion_map[x, y] += 1.0
        
        # Normalize
        max_val = self.congestion_map.max()
        if max_val > 0:
            self.congestion_map /= max_val
    
    def get_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        completed = sum(1 for v in self.vehicles if v.completed)
        active = sum(1 for v in self.vehicles if not v.completed)
        
        avg_time = np.mean([v.total_time for v in self.vehicles if v.completed]) if completed > 0 else 0
        avg_waiting = np.mean([v.waiting_time for v in self.vehicles])
        avg_path_length = np.mean([v.path_length for v in self.vehicles if v.path_length > 0])
        
        collision_rate = self.total_collisions / max(self.steps_completed, 1)
        completion_rate = completed / self.num_vehicles
        
        return {
            'steps': self.steps_completed,
            'completed': completed,
            'active': active,
            'collisions': self.total_collisions,
            'collision_rate': collision_rate,
            'completion_rate': completion_rate,
            'avg_completion_time': avg_time,
            'avg_waiting_time': avg_waiting,
            'avg_path_length': avg_path_length,
            'congestion_level': self.congestion_map.mean(),
            'quantum_avg_time': np.mean(self.quantum_times) if self.quantum_times else 0,
            'classical_avg_time': np.mean(self.classical_times) if self.classical_times else 0
        }
    
    def visualize(self, save_path: str = 'traffic_simulation.png'):
        """Create comprehensive visualization of simulation state"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Grid with vehicles and paths
        ax1 = fig.add_subplot(gs[0, 0])
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = -1
        
        # Mark vehicle positions and paths
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_vehicles))
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.completed:
                continue
            
            # Draw path
            if vehicle.path:
                path_array = np.array(vehicle.path)
                ax1.plot(path_array[:, 1], path_array[:, 0], 
                        '--', color=colors[i], alpha=0.3, linewidth=1)
            
            # Mark position
            x, y = vehicle.position
            grid[x, y] = vehicle.id + 1
            ax1.plot(y, x, 'o', color=colors[i], markersize=12, 
                    markeredgecolor='black', markeredgewidth=2)
            
            # Mark destination
            dx, dy = vehicle.destination
            ax1.plot(dy, dx, 's', color=colors[i], markersize=10, alpha=0.5)
        
        ax1.imshow(grid, cmap='gray', alpha=0.3)
        ax1.set_title(f'Traffic Grid (Step {self.steps_completed})', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Y')
        ax1.set_ylabel('X')
        ax1.grid(True, alpha=0.3)
        
        # 2. Congestion heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(self.congestion_map, cmap='hot', interpolation='nearest')
        ax2.set_title('Congestion Heatmap', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Y')
        ax2.set_ylabel('X')
        plt.colorbar(im, ax=ax2, label='Congestion Level')
        
        # 3. Vehicle status
        ax3 = fig.add_subplot(gs[0, 2])
        statuses = ['Completed', 'Active']
        counts = [
            sum(1 for v in self.vehicles if v.completed),
            sum(1 for v in self.vehicles if not v.completed)
        ]
        colors_status = ['#2ecc71', '#3498db']
        ax3.pie(counts, labels=statuses, autopct='%1.1f%%', colors=colors_status, startangle=90)
        ax3.set_title('Vehicle Status', fontsize=12, fontweight='bold')
        
        # 4. Collision history
        ax4 = fig.add_subplot(gs[1, 0])
        if self.collision_log:
            collision_steps = [c['step'] for c in self.collision_log]
            ax4.hist(collision_steps, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Simulation Step')
        ax4.set_ylabel('Collisions')
        ax4.set_title(f'Collision Distribution (Total: {self.total_collisions})', 
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance metrics
        ax5 = fig.add_subplot(gs[1, 1])
        metrics = self.get_metrics()
        
        metric_names = ['Completion\nRate', 'Collision\nRate', 'Avg Wait\nTime', 'Congestion']
        metric_values = [
            metrics['completion_rate'] * 100,
            metrics['collision_rate'] * 10,  # Scale for visibility
            metrics['avg_waiting_time'],
            metrics['congestion_level'] * 100
        ]
        
        bars = ax5.barh(metric_names, metric_values, color=['green', 'red', 'orange', 'purple'], alpha=0.7)
        ax5.set_xlabel('Value')
        ax5.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            ax5.text(val, i, f' {val:.1f}', va='center', fontweight='bold')
        
        # 6. Path efficiency
        ax6 = fig.add_subplot(gs[1, 2])
        vehicle_ids = [v.id for v in self.vehicles if v.path_length > 0]
        path_lengths = [v.path_length for v in self.vehicles if v.path_length > 0]
        waiting_times = [v.waiting_time for v in self.vehicles if v.path_length > 0]
        
        if vehicle_ids:
            x = np.arange(len(vehicle_ids))
            width = 0.35
            ax6.bar(x - width/2, path_lengths, width, label='Path Length', color='steelblue', alpha=0.8)
            ax6.bar(x + width/2, waiting_times, width, label='Wait Time', color='coral', alpha=0.8)
            ax6.set_xlabel('Vehicle ID')
            ax6.set_ylabel('Steps')
            ax6.set_title('Vehicle Efficiency', fontsize=12, fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(vehicle_ids)
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Quantum Traffic Simulation Dashboard', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nVisualization saved to: {save_path}")
    
    def get_vehicle_stats(self) -> List[Dict]:
        """Get detailed statistics for each vehicle"""
        stats = []
        for v in self.vehicles:
            stats.append({
                'id': v.id,
                'completed': v.completed,
                'total_time': v.total_time,
                'waiting_time': v.waiting_time,
                'collisions': v.collisions,
                'path_length': v.path_length,
                'efficiency': v.path_length / max(v.total_time, 1) if v.total_time > 0 else 0
            })
        return stats
    
    def reset(self):
        """Reset simulation to initial state"""
        self.vehicles = []
        self.congestion_map = np.zeros((self.grid_size, self.grid_size))
        self.total_collisions = 0
        self.steps_completed = 0
        self.collision_log = []
        self.quantum_times = []
        self.classical_times = []