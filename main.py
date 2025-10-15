"""
Main entry point for Quantum Traffic Simulation
Compares Classical vs Quantum routing algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
from src.quantum_router import QuantumTrafficDecision
from src.traffic_simulator import TrafficSimulator
import time

def test_quantum_decision():
    """Test quantum decision making"""
    print("="*50)
    print("Testing Quantum Decision Maker")
    print("="*50)
    
    qdm = QuantumTrafficDecision()
    
    # Test with sample moves
    moves = [(0,1), (1,0), (1,1), (0,0)]
    current = (0,0)
    dest = (5,5)
    
    print(f"\nCurrent Position: {current}")
    print(f"Destination: {dest}")
    print(f"Possible Moves: {moves}\n")
    
    print("Making 10 quantum decisions:")
    for i in range(10):
        chosen = qdm.quantum_choose_move(moves, current, dest)
        print(f"Decision {i+1}: {chosen}")
    
    print("\nVisualizing decision distribution...")
    qdm.visualize_decision_distribution(num_samples=1000)

def compare_algorithms():
    """Compare Classical vs Quantum-inspired algorithms"""
    print("\n" + "="*50)
    print("Comparing Classical vs Quantum Algorithms")
    print("="*50)
    
    grid_size = 10
    num_vehicles = 5
    num_steps = 50
    
    # Classical simulation
    print("\nRunning Classical Simulation...")
    classical_sim = TrafficSimulator(grid_size, num_vehicles)
    classical_sim.initialize_vehicles()
    
    for _ in range(num_steps):
        classical_sim.step(use_quantum=False)
    
    classical_metrics = classical_sim.get_metrics()
    print(f"Classical Results: {classical_metrics}")
    
    # Visualize
    classical_sim.visualize()
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Total Steps: {num_steps}")
    print(f"Classical Collisions: {classical_metrics['collisions']}")
    print(f"Classical Collision Rate: {classical_metrics['collision_rate']:.2%}")
    print(f"Completed Vehicles: {classical_metrics['completed']}")

def run_full_comparison():
    """Run comprehensive comparison study"""
    print("\n" + "="*50)
    print("Full Comparison Study")
    print("="*50)
    
    grid_sizes = [8, 10, 12]
    vehicle_counts = [3, 5, 7]
    results = []
    
    for grid_size in grid_sizes:
        for num_vehicles in vehicle_counts:
            print(f"\nTesting: Grid {grid_size}x{grid_size}, {num_vehicles} vehicles")
            
            sim = TrafficSimulator(grid_size, num_vehicles)
            sim.initialize_vehicles()
            
            for _ in range(100):
                sim.step()
            
            metrics = sim.get_metrics()
            results.append({
                'grid_size': grid_size,
                'num_vehicles': num_vehicles,
                **metrics
            })
            
            print(f"  Collisions: {metrics['collisions']}, Rate: {metrics['collision_rate']:.2%}")
    
    return results

def main():
    """Main execution function"""
    print("Quantum Traffic Grid Simulation")
    print("================================\n")
    
    while True:
        print("\nSelect an option:")
        print("1. Test Quantum Decision Maker")
        print("2. Run Classical Simulation")
        print("3. Compare Algorithms")
        print("4. Full Comparison Study")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            test_quantum_decision()
        elif choice == '2':
            sim = TrafficSimulator(grid_size=8, num_vehicles=5)
            sim.initialize_vehicles()
            for i in range(30):
                sim.step()
                if (i + 1) % 10 == 0:
                    print(f"Step {i+1}: {sim.get_metrics()}")
            sim.visualize()
        elif choice == '3':
            compare_algorithms()
        elif choice == '4':
            results = run_full_comparison()
            print("\nFull results:", results)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()