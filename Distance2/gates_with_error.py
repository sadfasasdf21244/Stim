import stim
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, SVG
import os

# 큐비트 인덱스 정의 (사용하기 편하게 전역 변수로 설정)
D1, D2, D3, D4 = 0, 1, 2, 3
A1, A2, A3 = 4, 5, 6

# 그룹화 (나중에 반복문 돌리기 편하게)
DATA_QUBITS = [D1, D2, D3, D4]
ANCILLA_QUBITS = [A1, A2, A3]
QUBITS_NAME = {D1: 'D1',
               D2: 'D2',
               D3: 'D3',
               D4: 'D4',
               A1: 'A1',
               A2: 'A2',
               A3: 'A3'}
ANCILLA_INDEX = {A1: 1,
                A2: 0,
                A3: 2}  # measure 되는 순서
ALL_QUBITS = DATA_QUBITS + ANCILLA_QUBITS

coords = {
        D1: (0, 0), A1: (-1, 1), D2: (2, 0),
        A2: (1, 1),
        D3: (0, 2), A3: (3, 1), D4: (2, 2)
    }
coordinate_scalefactor = 1.0
# ==========================================
# 1. 사용자 정의 Builder 클래스 (수정하신 버전)
# ==========================================

class CircuitBuilder:
    def __init__(self, p_1q, p_2q, p_meas, p_1q_z=0, sequence_time = 0, T1 = 10, T2 = 15):
        self.circuit = stim.Circuit()
        self.p_1q = p_1q      
        self.p_2q = p_2q           
        self.p_meas = p_meas   
        self.p_1q_z = p_1q_z
        self.sequence_time = sequence_time
        self.T1 = T1
        self.T2 = T2
        # self.T1_error_rate = 1 - np.exp(- self.sequence_time / self.T1)
        self.T2_error_rate = (1 - np.exp( - self.sequence_time / 10/ self.T2))/2
        
        for q_idx in ALL_QUBITS:
            # (r, c) 튜플의 각 원소(val)에 scale을 곱해서 새로운 리스트를 만듦
            scaled_coords = [val * coordinate_scalefactor for val in coords[q_idx]]
            
            # Stim에 추가 (Stim은 좌표를 리스트 형태로 받습니다)
            self.circuit.append("QUBIT_COORDS", [q_idx], scaled_coords)        # self.circuit.append("QUBIT_COORDS", [D1], [0, 0])
    def tick(self): 
        """회로에 시간 구분선(TICK)을 추가합니다."""
        self.circuit.append("TICK")

    def _add_noise_1q(self, target):
        if self.p_1q > 0:
            self.circuit.append("DEPOLARIZE1", [target], self.p_1q)

    def _add_noise_2q(self, t1, t2):
        if self.p_2q > 0:
            self.circuit.append("DEPOLARIZE2", [t1, t2], self.p_2q)

    # def _add_noise_1q_z(self, target):
    #     if self.p_1q_z > 0:
    #         self.circuit.append("DEPOLARIZE1", [target], self.p_1q_z)
    
    # def _add_T1_noise(self, target):
    #     if self.T1_error_rate > 0:
    #         self.circuit.append("AMPLITUDE_DAMPING", [target], self.T1_error_rate)

    def _add_T2_noise(self, target):    
        if self.T2_error_rate > 0:
            for target_t2 in target:
                self.circuit.append("Z_ERROR", [target_t2], self.T2_error_rate)

    # 1. Pi Rotations
    def pi_x(self, target):
        self.circuit.append("X", [target])
        self._add_noise_1q(target)

    def pi_y(self, target):
        self.circuit.append("Y", [target])
        self._add_noise_1q(target)
        
    def pi_z(self, target):
        self.circuit.append("Z", [target])
        self._add_noise_1q(target)

    # 2. Pi/2 Rotations
    def pi_half_x(self, target): 
        self.circuit.append("SQRT_X", [target])
        self._add_noise_1q(target)

    def minus_pi_half_x(self, target): 
        self.circuit.append("SQRT_X_DAG", [target])
        self._add_noise_1q(target)

    def pi_half_y(self, target): 
        self.circuit.append("SQRT_Y", [target])
        self._add_noise_1q(target)

    def minus_pi_half_y(self, target): 
        self.circuit.append("SQRT_Y_DAG", [target])
        self._add_noise_1q(target)

    def pi_half_z(self, target): 
        self.circuit.append("S", [target])
        self._add_noise_1q_z(target)

    def minus_pi_half_z(self, target): 
        self.circuit.append("S_DAG", [target])
        self._add_noise_1q_z(target)

    # 3. Two Qubit Gate
    def cz(self, control, target):
        self.circuit.append("CZ", [control, target])
        self._add_noise_2q(control, target)

    def measure_z(self, target):
        if self.p_meas > 0:
            self.circuit.append("X_ERROR", [target], self.p_meas)
        self.circuit.append("M", [target])

    def get_circuit(self):
        return self.circuit    
    
    def visualize_circuit_ticks(circuit):
        """
        Stim 회로를 타임라인 형태로 시각화합니다.
        TICK이 추가되어 있어 각 단계별 연산이 구분되어 보입니다.
        """
        print("=== Circuit Timeline Visualization ===")
        # timeline-svg: 전체 시간 흐름을 가로로 보여줌
        display(SVG(circuit.diagram(type="timeline-svg")))

        print("\n=== Slice Visualization (Grid View) ===")
        # timeslice-svg: 큐비트 배치(Grid) 위에서 일어나는 일을 Tick 별로 보여줌
        # flatten_to_ops=True로 하면 복잡한 게이트 분해를 막고 논리적 게이트 위주로 보여줍니다.
        display(SVG(circuit.diagram(type="timeslice-svg")))

    def measure_ancilla(self, is_first_round = False, A2_basis = 'X'):      
        
        # [Step 1] Initialization & Basis Change
        if A2_basis == 'X':
            # Data Qubits Basis Change (Z -> X)
            for q in [D1, D2, D3, D4]: self.pi_half_y(q)
            self._add_T2_noise(ANCILLA_QUBITS)
            self.tick()

        self.minus_pi_half_y(A2) # A2 Basis Change
        self._add_T2_noise([D1, D2, D3, D4, A1, A3])
        self.tick() # --- TICK 1: Preparation Complete ---

        # [Step 2] Entangling Gates (Center A2)
        # 실제 실험에서는 순차적으로 일어날 수 있으나 시각화를 위해 그룹핑
        self.cz(D1, A2)
        self.tick()
        self.cz(D2, A2)
        self.tick()
        self.cz(D3, A2)
        self.tick()
        self.cz(D4, A2)
        self._add_T2_noise([A1, A3])
        self.tick() # --- TICK 2: Center Interactions Complete ---

        # [Step 3] A2 Basis Revert & Noise
        self.pi_half_y(A2)
        self._add_T2_noise([D1, D2, D3, D4, A1, A3])
        self.tick()

        if A2_basis == 'X':
            for q in [D1, D2, D3, D4]: self.minus_pi_half_y(q)
            self._add_T2_noise(ANCILLA_QUBITS)
            self.tick()
        self.measure_z(A2)
        self._add_T2_noise([D1, D2, D3, D4, A1, A3])
        self.tick() # --- TICK 3: A2 Measurement Complete ---

        # [Step 4] Side Ancillas (A1, A3) Interactions
        self.minus_pi_half_y(A1)
        self.minus_pi_half_y(A3)
        self._add_T2_noise([D1, D2, D3, D4, A2])
        self.tick() # --- TICK 4: Side Prep ---

        self.cz(D3, A1)
        self.cz(D2, A3)
        self.tick() # --- TICK 5: Interaction Layer 1 ---
        self.cz(D1, A1)
        self.cz(D4, A3)
        self._add_T2_noise([A2])
        self.tick() # --- TICK 6: Interaction Layer 2 ---

        self.pi_half_y(A1)
        self.pi_half_y(A3)
        self._add_T2_noise([D1, D2, D3, D4, A2])
        self.tick()

        self.measure_z(A1)
        self.measure_z(A3)
        self._add_T2_noise([D1, D2, D3, D4, A2])

        # Detectors
        if is_first_round:
            self.circuit.append("DETECTOR", [stim.target_rec(-3)])
            self.circuit.append("DETECTOR", [stim.target_rec(-2)])
            self.circuit.append("DETECTOR", [stim.target_rec(-1)])
        else:
            self.circuit.append("DETECTOR", [stim.target_rec(-3), stim.target_rec(-6)])
            self.circuit.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-5)])
            self.circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-4)])

    def measure_data(self, basis = 'Z'):
        if basis == 'X':
            self.minus_pi_half_y(D1)
            self.minus_pi_half_y(D2)
            self.minus_pi_half_y(D3)
            self.minus_pi_half_y(D4)
            self.tick() # --- TICK ---
        elif basis == 'Y':
            self.pi_half_x(D1)
            self.pi_half_x(D2)
            self.pi_half_x(D3)
            self.pi_half_x(D4)
            self.tick() # --- TICK ---

        self.measure_z(D1)
        self.measure_z(D2)
        self.measure_z(D3)
        self.measure_z(D4)
        self.tick() # --- TICK ---

        if basis == 'X':
            self.pi_half_y(D1)
            self.pi_half_y(D2)
            self.pi_half_y(D3)
            self.pi_half_y(D4)
            self.tick()
        elif basis == 'Y':
            self.minus_pi_half_x(D1)
            self.minus_pi_half_x(D2)
            self.minus_pi_half_x(D3)
            self.minus_pi_half_x(D4)
            self.tick()

    def measure_arbitrary(self, qubit_list: list[int], basis = 'Z'):
        # Basis Change
        for qubit in qubit_list:
            if basis == 'X': self.minus_pi_half_y(qubit)
            elif basis == 'Y': self.pi_half_x(qubit)
            
            self.tick() # --- TICK ---


        # Measure
        for qubit in qubit_list:
            self.measure_z(qubit)
        
        self.tick() # --- TICK ---

        # Revert Basis (if needed for further rounds, usually end of exp)
        for qubit in qubit_list:
            if basis == 'X': self.pi_half_y(qubit)
            elif basis == 'Y': self.minus_pi_half_x(qubit)

            self.tick() # --- TICK ---

def visualize_circuit_ticks(circuit, save_dir="circuit_plots"):
    """
    Stim 회로를 타임라인/타임슬라이스 형태로 시각화하여 SVG 파일로 저장합니다.
    """
    # 저장 폴더 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📂 폴더 생성: {save_dir}")

    print("=== Circuit Visualization 저장 시작 ===")

    # 1. Timeline View (전체 시간 흐름)
    timeline_svg = str(circuit.diagram(type="timeline-svg"))
    
    # [수정됨] encoding='utf-8' 옵션을 반드시 추가해야 합니다!
    with open(os.path.join(save_dir, "timeline.svg"), "w", encoding="utf-8") as f:
        f.write(timeline_svg)
    print(f"  💾 저장됨: {os.path.join(save_dir, 'timeline.svg')}")

    # 2. Timeslice View (Tick 별 그리드 뷰)
    timeslice_svg = str(circuit.diagram(type="timeslice-svg"))
    
    # [수정됨] encoding='utf-8' 옵션을 반드시 추가해야 합니다!
    with open(os.path.join(save_dir, "timeslice.svg"), "w", encoding="utf-8") as f:
        f.write(timeslice_svg)
    print(f"  💾 저장됨: {os.path.join(save_dir, 'timeslice.svg')}")
    
    print("✨ 모든 시각화 파일 저장 완료! 해당 폴더를 확인하세요.")


def plot_qubit_layout():

    fig, ax = plt.subplots(figsize=(6, 6))

    # 1. 큐비트 및 텍스트 그리기
    # coords의 key(큐비트)와 value(row, col)를 순회
    X = []
    Y = []

    for q_idx, (x, y) in coords.items():

        # 범위 설정을 위해 리스트에 저장
        X.append(x)
        Y.append(y)

        if q_idx in DATA_QUBITS:
            color = 'skyblue'
            label = f"D{q_idx+1}" 
            marker = 'o'
        else:
            color = 'salmon'
            a_num = q_idx - 3 
            label = f"A{a_num}"
            marker = 's'
        
        # zorder를 높여서 선 위에 점이 오도록 함
        ax.scatter(x, y, s=1000, c=color, edgecolors='black', marker=marker, zorder=10)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=11)

    # 2. 연결선 그리기 (Interaction Edges)
    # A2 (Center) <-> Neighbors
    c_x, c_y = coords[A2] # (Row, Col)
    for d in [D1, D2, D3, D4]:
        d_x, d_y = coords[d]
        ax.plot([c_x, d_x], [c_y, d_y], color='black', alpha=0.5, linewidth=2, zorder=1)
    
    # A1 (Top) <-> D1, D3 (세로 연결처럼 보이지만 논리적 연결)
    # 실제 연결: A1(0,1)은 D1(0,0)과 D2(0,2) 사이에 있음 (위쪽 역삼각형 형태 가정 시)
    # 혹은 Distance-2 Z-cut 형태에 따라 연결 정의. 
    # (여기서는 제공해주신 Distance-2 Topology에 맞춰 A1-D1, A1-D2 연결로 가정하거나
    #  코드상의 CZ 연결(D1, D3)을 따름. 작성하신 코드 로직상 A1은 D1, D3와 연결됨)
    a1_x, a1_y = coords[A1]
    for d in [D1, D3]: # 기존 코드 로직 유지
        d_x, d_y = coords[d]
        ax.plot([a1_x, d_x], [a1_y, d_y], color='black', alpha=0.5, linewidth=2, zorder=1)

    # A3 (Bottom) <-> D2, D4
    a3_x, a3_y = coords[A3]
    for d in [D2, D4]: # 기존 코드 로직 유지
        d_x, d_y = coords[d]
        ax.plot([a3_x, d_x], [a3_y, d_y], color='black', alpha=0.5, linewidth=2, zorder=1)

    # 3. 축 설정 (사용자 요청 반영)
    
    # (1) Y축 뒤집기: Matplotlib은 기본적으로 위가 +Y지만, 
    # 행렬/이미지 좌표계처럼 아래로 갈수록 Row가 커지게 설정
    # ax.invert_yaxis()

    # (2) Grid 끄기
    ax.grid(False)

    # (3) 최대/최소값으로 축 범위 지정 (여백 0.5 추가)
    pad = 0.5
    ax.set_xlim(min(X) - pad, max(X) + pad)
    ax.set_ylim(max(Y) + pad, min(Y) - pad) # invert_yaxis를 했으므로 max가 아래쪽

    # 비율 유지 (정사각형)
    ax.set_aspect('equal')
    
    # 축 눈금 제거 (깔끔하게)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 테두리(Spine) 제거하고 싶으면 아래 주석 해제
    # ax.axis('off') 

    ax.set_title("Qubit Layout", fontsize=14)
    plt.tight_layout()
    plt.show()