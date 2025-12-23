import stim
import matplotlib.pyplot as plt

def plot_qubit_layout(circuit):
    coords = {}
    
    # 1. 회로에서 좌표 정보 추출
    for instruction in circuit:
        if instruction.name == "QUBIT_COORDS":
            # [수정됨] gate_args는 속성이 아니라 메서드(.gate_args_copy())로 호출해야 합니다.
            # 리스트 형태로 반환됩니다. (예: [1.0, 2.0])
            args = instruction.gate_args_copy()
            x, y = args[0], args[1]
            
            # targets_copy()는 타겟 리스트를 반환, 첫 번째 타겟의 value가 큐비트 인덱스
            q_index = instruction.targets_copy()[0].value
            coords[q_index] = (x, y)
    
    if not coords:
        print("회로에 좌표 정보(QUBIT_COORDS)가 없습니다.")
        return

    # 2. 시각화 준비
    fig, ax = plt.subplots(figsize=(8, 8))
    
    indices = list(coords.keys())
    xs = [coords[i][0] for i in indices]
    ys = [coords[i][1] for i in indices]
    
    # 3. 큐비트 종류 추정 (색상 구분)
    colors = []
    
    for i in indices:
        r, c = coords[i]
        # Rotated Surface Code 일반적인 배치 규칙:
        # 합이 짝수면 Data, 홀수면 Ancilla (Z/X 구분은 또 다를 수 있음)
        if r % 2 != 0:
            colors.append('black')  # Data Qubit
        else:
            colors.append('red')    # Ancilla Qubit

    # 4. 산점도 그리기
    ax.scatter(xs, ys, s=600, c=colors, edgecolors='gray', zorder=3)
    
    # 인덱스 텍스트 표시
    for i, idx in enumerate(indices):
        ax.text(xs[i], ys[i], str(idx), color='white', 
                ha='center', va='center', fontweight='bold', fontsize=11, zorder=4)
        
    # 5. 그래프 설정
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
    ax.set_title(f"Distance-3 Surface Code Layout\n(Total {len(indices)} Qubits: {colors.count('black')} Data, {colors.count('red')} Ancilla)")
    ax.set_xlabel("Column Coordinate")
    ax.set_ylabel("Row Coordinate")
    
    # 여백 확보
    margin = 1
    if xs and ys:
        ax.set_xlim(min(xs)-margin, max(xs)+margin)
        ax.set_ylim(min(ys)-margin, max(ys)+margin)
        
        # (선택사항) 좌표계가 위쪽이 0이 되도록 뒤집으려면 아래 주석 해제
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# === 실행 ===
if __name__ == "__main__":
    # Distance 3 회로 생성
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=2)
    
    # 레이아웃 그리기
    plot_qubit_layout(circuit)