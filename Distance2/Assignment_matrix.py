import stim
import numpy as np
from gates_with_error import CircuitBuilder

def assignment_matrix(qubits: list[int], p_1q, p_2q, p_meas, shots: int = 10000, **kwargs) -> np.ndarray:
    """주어진 큐비트들에 대해 할당 행렬(Assignment Matrix)을 계산합니다.

    Args:
        circuit (stim.Circuit): 측정 회로.
        qubits (list[int]): 할당 행렬을 계산할 큐비트 인덱스 리스트.
        shots (int, optional): 샷 수. 기본값은 10000.

    Returns:
        np.ndarray: 할당 행렬.
    """

    num_qubits = len(qubits)
    dim = 2 ** num_qubits
    assignment_matrix = np.zeros((dim, dim))

    for prep_state in range(dim):
        circuit = CircuitBuilder(
            p_1q=p_1q,
            p_2q=p_2q,
            p_meas=p_meas,
            **kwargs
        )

        for i in range(num_qubits):
            if (prep_state >> num_qubits-1-i) & 1:
                circuit.pi_x(qubits[i])

        circuit.measure_arbitrary(qubits,'Z')

        # 샘플링
        sampler = circuit.circuit.compile_sampler()
        measurements = sampler.sample(shots=shots)

        for meas in measurements:
            meas_state = 0
            for i in range(num_qubits):
                if meas[num_qubits-1-i] == 1:
                    meas_state += (1 << i)
            assignment_matrix[meas_state, prep_state] += 1

    # 확률로 변환
    assignment_matrix /= shots
    return assignment_matrix