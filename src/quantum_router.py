"""
Quantum Decision Making for Traffic Grid (Updated for Qiskit >=0.48)
"""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class QuantumTrafficDecision:
    """Quantum decision maker for car movements"""

    def __init__(self):
        self.backend = AerSimulator()
        self.decision_history = []

    def create_decision_circuit(self, num_options: int) -> QuantumCircuit:
        """Create a quantum circuit to choose between movement options"""
        n_qubits = max(1, int(np.ceil(np.log2(num_options))))
        qr = QuantumRegister(n_qubits, 'option')
        cr = ClassicalRegister(n_qubits, 'decision')
        qc = QuantumCircuit(qr, cr)

        # Superposition
        for i in range(n_qubits):
            qc.h(qr[i])

        # Simple interference
        if n_qubits > 1:
            qc.cx(qr[0], qr[1])
            qc.rz(np.pi/4, qr[0])

        qc.measure(qr, cr)
        return qc

    def quantum_choose_move(self, possible_moves: List[Tuple[int, int]],
                            current_pos: Tuple[int, int],
                            destination: Tuple[int, int]) -> Tuple[int, int]:
        """Use quantum circuit to choose next move"""
        if len(possible_moves) <= 1:
            return possible_moves[0] if possible_moves else current_pos

        qc = self.create_decision_circuit(len(possible_moves))
        compiled_circuit = transpile(qc, self.backend)
        job = self.backend.run(compiled_circuit, shots=1)
        result = job.result()
        counts = result.get_counts()

        measured_value = list(counts.keys())[0]
        move_index = int(measured_value, 2) % len(possible_moves)
        chosen_move = possible_moves[move_index]

        self.decision_history.append({
            'options': possible_moves,
            'quantum_measurement': measured_value,
            'chosen_index': move_index,
            'chosen_move': chosen_move
        })

        return chosen_move

    def create_biased_circuit(self, possible_moves: List[Tuple[int, int]], 
                              distances: List[float]) -> QuantumCircuit:
        """Create a biased quantum circuit towards better moves"""
        n_moves = len(possible_moves)
        n_qubits = max(1, int(np.ceil(np.log2(n_moves))))
        qr = QuantumRegister(n_qubits, 'move')
        cr = ClassicalRegister(n_qubits, 'choice')
        qc = QuantumCircuit(qr, cr)

        for i in range(n_qubits):
            qc.h(qr[i])

        if distances:
            max_dist = max(distances) if max(distances) > 0 else 1
            for i, dist in enumerate(distances[:2**n_qubits]):
                if i < n_moves:
                    angle = np.pi * (1 - dist/max_dist) / 4
                    if n_qubits == 1:
                        qc.ry(angle, qr[0])
                    else:
                        binary = format(i, f'0{n_qubits}b')
                        for j, bit in enumerate(binary):
                            if bit == '0':
                                qc.x(qr[j])
                        qc.ry(angle, qr[-1])
                        for j, bit in enumerate(binary):
                            if bit == '0':
                                qc.x(qr[j])

        qc.measure(qr, cr)
        return qc

    def visualize_decision_distribution(self, num_samples: int = 1000):
        """Visualize the probability distribution of last quantum decision"""
        if not self.decision_history:
            print("No decisions made yet!")
            return

        last_decision = self.decision_history[-1]
        options = last_decision['options']
        qc = self.create_decision_circuit(len(options))
        compiled_circuit = transpile(qc, self.backend)
        job = self.backend.run(compiled_circuit, shots=num_samples)
        result = job.result()
        counts = result.get_counts()
        
        fig = plot_histogram(counts)
        plt.show()
        return counts

# Quick test
if __name__ == "__main__":
    print("Testing Quantum Traffic Decision Maker...")
    qdm = QuantumTrafficDecision()
    moves = [(0,1), (1,0), (1,1)]
    current = (0,0)
    dest = (2,2)
    
    print("\nMaking 5 quantum decisions:")
    for i in range(5):
        chosen = qdm.quantum_choose_move(moves, current, dest)
        print(f"Decision {i+1}: {chosen}")
    
    print("\nVisualizing decision distribution...")
    qdm.visualize_decision_distribution()