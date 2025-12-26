import stim
import numpy as np
from gates_with_error import CircuitBuilder
import matplotlib.pyplot as plt

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

def plot_assignment_matrix(matrix, 
                           title="Assignment Matrix (Readout Error)", 
                           show_values=True, 
                           cmap='Blues', 
                           save_path=None):
    """
    Assignment Matrix를 히트맵으로 시각화합니다.
    
    Args:
        matrix (np.ndarray): 시각화할 할당 행렬 (2^N x 2^N).
        title (str): 그래프 제목.
        show_values (bool): 셀 안에 숫자(확률)를 표시할지 여부. (큐비트가 많으면 끄는 게 좋습니다)
        cmap (str): 컬러맵 스타일 (예: 'Blues', 'Reds', 'viridis', 'Oranges').
        save_path (str, optional): 파일로 저장할 경로. None이면 저장하지 않음.
    """
    dim = matrix.shape[0]
    num_qubits = int(np.log2(dim))
    
    # 0, 1, ... 정수를 '0000', '0001' 형태의 비트 문자열로 변환 (MSB -> LSB)
    # 예: 10 -> '1010'
    tick_labels = [f"{i:0{num_qubits}b}" for i in range(dim)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 히트맵 그리기
    # origin='upper'는 (0,0)이 왼쪽 위에 오도록 함 (행렬과 동일한 배치)
    im = ax.imshow(matrix, cmap=cmap, origin='upper', vmin=0, vmax=1)
    
    # 컬러바 추가
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom")
    
    # 축 설정
    ax.set_xticks(np.arange(dim))
    ax.set_yticks(np.arange(dim))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right") # X축 라벨 회전
    ax.set_yticklabels(tick_labels)
    
    # 축 제목
    ax.set_xlabel("Prepared State (Input)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Measured State (Output)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=20)
    
    # 셀 안에 확률 값 텍스트 표시 (옵션)
    # 큐비트 수가 4개(16x16) 이하일 때만 추천합니다.
    if show_values:
        threshold = matrix.max() / 2.
        for i in range(dim): # Measured (Row)
            for j in range(dim): # Prepared (Col)
                # 배경색에 따라 글자색 변경 (어두우면 흰색, 밝으면 검은색)
                text_color = "white" if matrix[i, j] > threshold else "black"
                
                # 값이 너무 작으면 표시 생략하거나 0으로 표시
                val_str = f"{matrix[i, j]:.2f}" if matrix[i, j] >= 0.01 else ""
                if val_str == "0.00" and matrix[i, j] > 0: val_str = "."
                
                ax.text(j, i, val_str, ha="center", va="center", 
                        color=text_color, fontsize=8)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"💾 Assignment Matrix 그래프 저장됨: {save_path}")
        
    plt.show()