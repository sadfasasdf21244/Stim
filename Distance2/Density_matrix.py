import numpy as np
import stim
import itertools
from scipy.optimize import minimize
from functools import reduce
from Assignment_matrix import *
from gates_with_error import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from surface2_functions import *

# 사용자 정의 클래스 및 변수들 (위에서 주신 코드와 동일하다고 가정)
# D1~D4, A1~A3 정의 및 CircuitBuilder, assignment_matrix 포함

# ==============================================================================
# 1. 기초 설정 (파울리 행렬 및 프로젝터)
# ==============================================================================
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# 각 기저(Basis)에서 측정했을 때 0 또는 1이 나올 프로젝터
# projectors[basis_index][outcome_index]
# Basis: 0(X), 1(Y), 2(Z)
# Outcome: 0(+), 1(-)
local_projectors = {
    'X': [(I + X) / 2, (I - X) / 2],
    'Y': [(I + Y) / 2, (I - Y) / 2],
    'Z': [(I + Z) / 2, (I - Z) / 2]
}

def tensor_product(ops_list):
    """리스트에 있는 행렬들을 텐서곱"""
    return reduce(np.kron, ops_list)

# ==============================================================================
# 2. 수정된 Tomography Experiment 함수 (Post-selection 포함)
# ==============================================================================
def run_tomography_experiments(target_state_name, p_1q, p_2q, p_meas, shots=1000, **kwargs):
    """
    Args:
        target_state_name (str): '0', '1', '+', '-' 등 준비할 상태 이름
        shots (int): 시뮬레이션 샷 수
        **kwargs: 노이즈 파라미터 (p_1q, p_2q, p_meas 등) 및 CircuitBuilder 인자
        
    Returns:
        measured_data: {(basis_tuple): [count_0, ..., count_15]} 
                       (Ancilla가 모두 0인 샷들만 카운트됨)
    """
    data_qubits = [D1, D2, D3, D4]
    bases = ['X', 'Y', 'Z']
    
    # 3^4 = 81가지 기저 조합 생성
    basis_combinations = list(itertools.product(bases, repeat=4))
    
    measured_data = {}
    
    print(f"Target '{target_state_name}': 총 {len(basis_combinations)}개의 Basis 설정에 대해 실험 시작...")

    for basis_config in basis_combinations:
        # -------------------------------------------------------
        # 1. 회로 생성 및 타겟 상태 준비 (Ancilla 측정 3개 포함)
        # -------------------------------------------------------
        # kwargs에 p_1q, p_2q 등이 포함되어 있어야 함
        builder = CircuitBuilder(
        p_1q=0,   
        p_2q=0,  
        p_meas=0,
        squence_time=0,
        )
        # -------------------------------------------------------
        # 2. 기저 회전 (Data Qubits Measurement Basis Rotation)
        # -------------------------------------------------------
        for i, basis in enumerate(basis_config):
            q = data_qubits[i]
            if basis == 'X':
                builder.minus_pi_half_y(q)       # Z -> X basis
            elif basis == 'Y':
                builder.pi_half_x(q) # Z -> Y basis
            # Z basis는 회전 없음

        # builder.circuit.append("Y_ERROR", D2, 0.4)

        # -------------------------------------------------------
        # 3. 데이터 큐비트 측정 (항상 Z basis로 측정, 측정 4개 추가)
        # -------------------------------------------------------
        builder.measure_arbitrary(DATA_QUBITS, basis = 'Z')
                
        # -------------------------------------------------------
        # 4. 실행 및 Post-selection (핵심 변경 사항)
        # -------------------------------------------------------
        sampler = builder.get_circuit().compile_sampler()
        
        # 전체 측정 결과 샘플링
        # 결과 배열 형태: [shots, 7] (Ancilla 3개 + Data 4개)
        raw_samples = sampler.sample(shots=shots)

        print(np.shape(raw_samples))

        # 카운트 배열 초기화 (16개 상태)
        counts = np.zeros(16)
        valid_shots = 0 # 유효한 샷 개수 확인용
        
        for sample in raw_samples:
            # [A] Post-selection: 처음 3비트(Ancilla)가 모두 0인지 확인
            # measure_ancilla 내부 순서에 따라 sample[0], sample[1], sample[2]가 해당됨
            # ancilla_res = sample[:3]
            
            # # np.any(ancilla_res)가 False여야 모두 0임
            # if not np.any(ancilla_res): 
                valid_shots += 1
                
                # [B] Data Qubits 추출: 인덱스 3부터 끝까지 (D1, D2, D3, D4)
                data_res = sample
                
                # [C] 비트스트링 -> 정수 인덱스 변환
                # D4(LSB) -> D1(MSB) 순서 가정 (sample 순서와 measure_arbitrary 순서 일치 확인 필요)
                idx = 0
                for k, bit in enumerate(data_res):
                    if bit:
                        idx += (1 << 3-k)
                counts[idx] += 1
        
        # (선택 사항) 만약 valid_shots가 너무 적으면 경고 출력
        # if valid_shots < shots * 0.1:
        #     print(f"Warning: Basis {basis_config}에서 유효한 샷이 너무 적습니다 ({valid_shots}/{shots})")

        measured_data[basis_config] = counts
    
    return measured_data
# ==============================================================================
# 3. MLE 최적화 함수 (핵심)
# ==============================================================================
def params_to_rho_4q(params):
    """Cholesky 분해를 이용해 실수 파라미터를 유효한 밀도 행렬로 변환"""
    dim = 16
    # 파라미터 개수: 16(대각) + 120*2(비대각 복소수) = 256
    L = np.zeros((dim, dim), dtype=complex)
    
    idx = 0
    for i in range(dim):
        L[i, i] = params[idx] # 대각선은 실수
        idx += 1
        for j in range(i):
            L[i, j] = params[idx] + 1j * params[idx+1]
            idx += 2
            
    rho = L @ L.conj().T
    return rho / np.trace(rho)

def perform_mle_4q(measured_data, assignment_mat):
    """
    measured_data: run_tomography_experiments의 결과
    assignment_mat: 16x16 행렬 A
    """
    print("MLE 최적화를 시작합니다. (시간이 다소 걸릴 수 있습니다...)")
    
    # 1. Projector 미리 계산 (속도 최적화)
    # 81개 기저 * 16개 결과에 대한 projector map
    projector_map = {} 
    
    # 0~15 정수를 비트 리스트로 (예: 3 -> [1, 1, 0, 0]) *순서 주의*
    outcome_indices = range(16)
    
    for basis_config in measured_data.keys():
        projector_map[basis_config] = []
        # 해당 기저 설정(basis_config)에서 가능한 16가지 결과에 대한 프로젝터 생성
        for outcome_int in outcome_indices:
            ops = []
            for i, basis_char in enumerate(basis_config):
                # i번째 큐비트의 결과 비트 (0 or 1)
                bit = (outcome_int >> i) & 1
                ops.append(local_projectors[basis_char][bit])
            
            # P = P1 (x) P2 (x) P3 (x) P4
            full_proj = tensor_product(ops)
            projector_map[basis_config].append(full_proj)

    # 2. Cost Function 정의
    def cost_func(params):
        rho = params_to_rho_4q(params)
        loss = 0.0
        epsilon = 1e-20
        
        # 81가지 실험 데이터 순회
        for basis_config, counts in measured_data.items():
            # 이 기저에서의 이상적인 확률 벡터 P_ideal 계산 (길이 16)
            p_ideal = np.zeros(16)
            projs = projector_map[basis_config]
            
            for k in range(16):
                p_ideal[k] = np.real(np.trace(projs[k] @ rho))
            
            # Readout Error 적용: P_noisy = A @ P_ideal
            p_noisy = assignment_mat @ p_ideal
            
            # Log Likelihood 계산 (Minimizing Negative LL)
            p_noisy = np.clip(p_noisy, epsilon, 1.0)
            loss -= np.sum(counts * np.log(p_noisy))
            
        return loss

    # 3. 최적화 실행
    # 초기값: Identity에 가까운 상태
    dim = 16
    num_params = 16 + (dim * (dim - 1) // 2) * 2 # 256
    init_params = np.random.rand(num_params) * 0.01
    init_params[0:16] += 1.0 / np.sqrt(dim) # 대각선 초기화
    
    res = minimize(cost_func, init_params,method='SLSQP',  # 방법 변경
    options={
        'maxiter': 200000,
        'ftol': 1e-30, # SLSQP에서의 허용 오차 옵션
        'disp': True
    })
    
    return params_to_rho_4q(res.x)


def plot_density_matrix_3d(rho, title_prefix="Density Matrix"):
    """
    16x16 밀도 행렬의 실수부와 허수부를 3D Bar Plot으로 시각화합니다.
    x, y축 라벨은 0000 ~ 1111 (D4 D3 D2 D1 순서 가정)로 표시됩니다.
    """
    # 4큐비트 차원 (16)
    dim = rho.shape[0] 
    
    # X, Y 좌표 격자 생성 (0 ~ 15)
    _x = np.arange(dim)
    _y = np.arange(dim)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    
    # 바닥 위치 (z=0)
    z = np.zeros_like(x)
    
    # 막대 두께
    dx = dy = 0.6  
    
    # 라벨 생성 (0000 ~ 1111)
    # 예: 3 -> '0011' (D4 D3 D2 D1 순서, Big Endian 표기)
    tick_labels = [f"{i:04b}" for i in range(dim)]
    
    # --- 시각화 설정 (1행 2열: 실수부 / 허수부) ---
    fig = plt.figure(figsize=(18, 8))
    
    # ==========================================
    # 1. 실수부 (Real Part)
    # ==========================================
    ax1 = fig.add_subplot(121, projection='3d')
    dz_real = rho.real.ravel() # 높이 데이터
    
    # 높이에 따른 색상 매핑 (Coolwarm)
    # 최댓값 기준으로 정규화하여 색상 입히기
    offset = dz_real + np.abs(dz_real.min())
    fracs = offset.astype(float) / offset.max()
    norm = plt.Normalize(fracs.min(), fracs.max())
    colors = cm.coolwarm(norm(fracs))

    ax1.bar3d(x, y, z, dx, dy, dz_real, color=colors, shade=True)
    
    ax1.set_title(f"{title_prefix} - Real Part (Re[ρ])")
    ax1.set_xlabel('Ket |i>')
    ax1.set_ylabel('Bra <j|')
    ax1.set_zlabel('Amplitude')
    
    # 축 눈금 설정
    ax1.set_xticks(np.arange(dim) + dx/2)
    ax1.set_yticks(np.arange(dim) + dy/2)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(tick_labels, rotation=-20, ha='left', fontsize=8)
    ax1.set_zlim(np.min(dz_real), np.max(dz_real))

    # ==========================================
    # 2. 허수부 (Imaginary Part)
    # ==========================================
    ax2 = fig.add_subplot(122, projection='3d')
    dz_imag = rho.imag.ravel()
    
    # 허수부는 0인 경우가 많으므로 데이터가 있을 때만 색상 처리
    if np.all(dz_imag == 0):
        colors_imag = 'cyan' # 허수부가 없으면 단색
    else:
        offset_i = dz_imag + np.abs(dz_imag.min())
        fracs_i = offset_i.astype(float) / (offset_i.max() + 1e-9)
        norm_i = plt.Normalize(fracs_i.min(), fracs_i.max())
        colors_imag = cm.viridis(norm_i(fracs_i))

    ax2.bar3d(x, y, z, dx, dy, dz_imag, color=colors_imag, shade=True)
    
    ax2.set_title(f"{title_prefix} - Imaginary Part (Im[ρ])")
    ax2.set_xlabel('Ket |i>')
    ax2.set_ylabel('Bra <j|')
    ax2.set_zlabel('Amplitude')
    
    # 축 눈금 설정
    ax2.set_xticks(np.arange(dim) + dx/2)
    ax2.set_yticks(np.arange(dim) + dy/2)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(tick_labels, rotation=-20, ha='left', fontsize=8)
    ax2.set_zlim(np.min(dz_imag), np.max(dz_imag) if np.max(dz_imag) > 0 else 0.1)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. 실행 예시
# ==============================================================================

def density_matrix(target_state, p_1q, p_2q, p_meas, shots, **kwargs): 

    print("1. Assignment Matrix 계산 중...")
    A_matrix = assignment_matrix(
        qubits=[D1, D2, D3, D4], 
        p_1q=p_1q, 
        p_2q=p_2q, 
        p_meas=p_meas,
        shots=shots, 
        **kwargs
    )

    print(A_matrix)
    # A 행렬은 Z basis 기준입니다. 
    # 회로에서 Basis Change는 측정 '전'에 일어나므로, 측정 자체는 항상 Z basis에서 수행됩니다.
    # 따라서 A 행렬 하나만 구해서 모든 기저 실험에 공통으로 사용하면 됩니다.

    print("2. Tomography 데이터 수집 중 (81 circuits)...")
    tomo_data = run_tomography_experiments(
        target_state_name=target_state,  
        p_1q=p_1q, 
        p_2q=p_2q, 
        p_meas=p_meas,
        shots=shots, 
        **kwargs
    )

    # [Step 3] MLE 수행
    final_rho = perform_mle_4q(tomo_data, A_matrix)

    print(f"\n=== 복원된 {target_state}state의 Density Matrix (Top-left 4x4) ===")
    print(np.round(final_rho[:, :], 3))
    plot_density_matrix_3d(final_rho, title_prefix="Reconstructed State (Example)")
    # Fidelity 확인 (Target: |Phi+> (x) |00>)
    # (타겟 상태 정의는 생략함)
    
    # physical fidelity, probability, rho logical 확인
    logical = {}
    logical[0] = np.zeros(16)
    logical[0][0b0000] = 1/np.sqrt(2)
    logical[0][0b1111] = 1/np.sqrt(2)

    logical[1] = np.zeros(16)
    logical[1][0b0101] = 1/np.sqrt(2)
    logical[1][0b1010] = 1/np.sqrt(2)
    match target_state:
        case '0':
            psi = logical[0]
        case '1':
            psi = logical[1]
        case '+':
            psi = 1/np.sqrt(2)*(logical[0]+logical[1])
        case '-':
            psi = 1/np.sqrt(2)*(logical[0]-logical[1])

    Logical_probability = logical[0].T @ final_rho @ logical[0] + logical[1].T @ final_rho @ logical[1]
    rho_logical = np.zeros([2, 2], dtype=complex)

    for i in range(2):
        for j in range(2):
            rho_logical[i][j] = logical[i].T @ final_rho @ logical[j] / Logical_probability

    Phisical_Fidelity = (psi.T @ final_rho @ psi).real
    print(f"Phisical Fidelity Fphys : {Phisical_Fidelity:.3f}")
    print(f"physical probability PL : {Logical_probability:.3f}")
    print(rho_logical)

    return final_rho