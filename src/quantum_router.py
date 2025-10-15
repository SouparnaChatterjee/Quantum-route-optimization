"""
Quantum Traffic Router using Quantum-Inspired Optimization
Implements QAOA-inspired algorithm for route optimization
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class QuantumState:
    """Represents a quantum superposition state for routing"""
    amplitudes: np.ndarray
    probabilities: np.ndarray
    paths: List[Tuple[int, int]]

class QuantumTrafficDecision:
    """
    Quantum-inspired decision maker for traffic routing.
    Uses amplitude amplification and interference patterns.
    """
    
    def __init__(self, beta: float = 0.8, iterations: int = 3):
        """
        Args:
            beta: Quantum interference parameter (0-1)
            iterations: Number of amplitude amplification iterations
        """
        self.beta = beta
        self.iterations = iterations
        self.decision_history = []
        
    def quantum_choose_move(
        self, 
        moves: List[Tuple[int, int]], 
        current: Tuple[int, int], 
        destination: Tuple[int, int],
        congestion_map: Optional[np.ndarray] = None
    ) -> Tuple[int, int]:
        """
        Choose next move using quantum-inspired decision making.
        
        Args:
            moves: List of possible moves (dx, dy)
            current: Current position (x, y)
            destination: Target position (x, y)
            congestion_map: Optional congestion information
            
        Returns:
            Chosen move (dx, dy)
        """
        if not moves:
            return (0, 0)
        
        # Initialize quantum state with uniform superposition
        n = len(moves)
        amplitudes = np.ones(n, dtype=complex) / np.sqrt(n)
        
        # Apply quantum-inspired optimization
        for iteration in range(self.iterations):
            # Oracle: Mark good solutions (closer to destination)
            amplitudes = self._apply_oracle(
                amplitudes, moves, current, destination, congestion_map
            )
            
            # Diffusion: Amplify marked states
            amplitudes = self._apply_diffusion(amplitudes)
        
        # Measure: Convert to probabilities
        probabilities = np.abs(amplitudes) ** 2
        probabilities /= probabilities.sum()
        
        # Store decision info
        self.decision_history.append({
            'moves': moves,
            'probabilities': probabilities.copy()
        })
        
        # Sample from quantum probability distribution
        choice_idx = np.random.choice(n, p=probabilities)
        return moves[choice_idx]
    
    def _apply_oracle(
        self,
        amplitudes: np.ndarray,
        moves: List[Tuple[int, int]],
        current: Tuple[int, int],
        destination: Tuple[int, int],
        congestion_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Oracle operator: Marks good solutions by modifying phase.
        Good solutions = moves that reduce distance to destination.
        """
        current_dist = self._manhattan_distance(current, destination)
        
        for i, move in enumerate(moves):
            next_pos = (current[0] + move[0], current[1] + move[1])
            new_dist = self._manhattan_distance(next_pos, destination)
            
            # Calculate improvement score
            improvement = (current_dist - new_dist) / max(current_dist, 1)
            
            # Factor in congestion if available
            congestion_penalty = 0
            if congestion_map is not None:
                x, y = next_pos
                if 0 <= x < congestion_map.shape[0] and 0 <= y < congestion_map.shape[1]:
                    congestion_penalty = congestion_map[x, y] * 0.3
            
            # Apply phase shift (quantum interference)
            score = improvement - congestion_penalty
            phase = np.exp(1j * self.beta * np.pi * score)
            amplitudes[i] *= phase
        
        return amplitudes
    
    def _apply_diffusion(self, amplitudes: np.ndarray) -> np.ndarray:
        """
        Diffusion operator: Inverts amplitudes around average.
        This amplifies marked states (Grover diffusion).
        """
        mean_amplitude = np.mean(amplitudes)
        amplitudes = 2 * mean_amplitude - amplitudes
        return amplitudes
    
    def _manhattan_distance(
        self, 
        pos1: Tuple[int, int], 
        pos2: Tuple[int, int]
    ) -> float:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def quantum_route_optimization(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        grid_size: int,
        obstacles: List[Tuple[int, int]] = None
    ) -> List[Tuple[int, int]]:
        """
        Find optimal route using quantum-inspired optimization.
        
        Returns:
            List of positions forming the path
        """
        if obstacles is None:
            obstacles = []
        
        path = [start]
        current = start
        max_steps = grid_size * 4  # Prevent infinite loops
        
        for _ in range(max_steps):
            if current == end:
                break
            
            # Get valid moves
            moves = self._get_valid_moves(current, grid_size, obstacles, path)
            if not moves:
                break
            
            # Choose next move using quantum decision
            move = self.quantum_choose_move(moves, current, end)
            current = (current[0] + move[0], current[1] + move[1])
            path.append(current)
        
        return path
    
    def _get_valid_moves(
        self,
        pos: Tuple[int, int],
        grid_size: int,
        obstacles: List[Tuple[int, int]],
        visited: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Get list of valid moves from current position"""
        possible_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        valid_moves = []
        
        for move in possible_moves:
            next_pos = (pos[0] + move[0], pos[1] + move[1])
            
            # Check bounds
            if not (0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size):
                continue
            
            # Check obstacles
            if next_pos in obstacles:
                continue
            
            # Avoid recently visited (allow revisit after 5 steps)
            if next_pos in visited[-5:]:
                continue
            
            valid_moves.append(move)
        
        return valid_moves
    
    def visualize_decision_distribution(
        self, 
        num_samples: int = 1000,
        moves: List[Tuple[int, int]] = None,
        current: Tuple[int, int] = (0, 0),
        destination: Tuple[int, int] = (5, 5)
    ) -> None:
        """Visualize the quantum decision distribution"""
        if moves is None:
            moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Sample decisions
        decisions = []
        for _ in range(num_samples):
            choice = self.quantum_choose_move(moves, current, destination)
            decisions.append(choice)
        
        # Count occurrences
        from collections import Counter
        counts = Counter(decisions)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribution bar chart
        move_labels = [f"{m}" for m in moves]
        move_counts = [counts[m] for m in moves]
        
        ax1.bar(move_labels, move_counts, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Move Direction')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Quantum Decision Distribution\n({num_samples} samples)')
        ax1.grid(True, alpha=0.3)
        
        # Probability evolution
        if len(self.decision_history) > 0:
            recent_history = self.decision_history[-100:]
            prob_evolution = np.array([h['probabilities'] for h in recent_history])
            
            for i in range(prob_evolution.shape[1]):
                ax2.plot(prob_evolution[:, i], label=f'Move {i}', alpha=0.7)
            
            ax2.set_xlabel('Decision Number')
            ax2.set_ylabel('Probability')
            ax2.set_title('Probability Evolution Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quantum_decision_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nDecision Statistics:")
        for move, count in counts.items():
            prob = count / num_samples
            print(f"  Move {move}: {prob:.2%} ({count}/{num_samples})")
    
    def get_quantum_metrics(self) -> Dict:
        """Calculate quantum-specific metrics"""
        if not self.decision_history:
            return {}
        
        recent = self.decision_history[-100:]
        probs = np.array([h['probabilities'] for h in recent])
        
        # Entropy (measure of quantum uncertainty)
        entropy = -np.sum(probs * np.log2(probs + 1e-10), axis=1).mean()
        
        # Quantum coherence (variance in probabilities)
        coherence = np.var(probs, axis=1).mean()
        
        return {
            'entropy': entropy,
            'coherence': coherence,
            'decisions_made': len(self.decision_history)
        }