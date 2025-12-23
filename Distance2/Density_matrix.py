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
from functools import reduce

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
def run_tomography_experiments(target_state_name, p_1q, p_2q, p_meas, shots, **kwargs):
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
        builder = state_prep(
        target_state = target_state_name,
        p_1q=p_1q,
        p_2q=p_2q,
        p_meas=p_meas,
        **kwargs
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
            ancilla_res = sample[:3]
            
            # np.any(ancilla_res)가 False여야 모두 0임
            if not np.any(ancilla_res): 
                valid_shots += 1
                
                # [B] Data Qubits 추출: 인덱스 3부터 끝까지 (D1, D2, D3, D4)
                data_res = sample[3:]
                
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
        'ftol': 1e-40, # SLSQP에서의 허용 오차 옵션
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

def get_shadow_tables():
    """
    단일 큐비트 Pauli Shadow 스냅샷 행렬을 미리 계산하여 반환합니다.
    공식: rho_snapshot = 3 * |psi><psi| - I
    """
    I = np.eye(2, dtype=complex)
    
    # 기저별 고유상태(Eigenstates) 정의
    # Z basis
    z0 = np.array([[1], [0]], dtype=complex) # |0>
    z1 = np.array([[0], [1]], dtype=complex) # |1>
    
    # X basis (|0> +/- |1>) / sqrt(2)
    x0 = (z0 + z1) / np.sqrt(2) # |+>
    x1 = (z0 - z1) / np.sqrt(2) # |->
    
    # Y basis (|0> +/- i|1>) / sqrt(2)
    y0 = (z0 + 1j * z1) / np.sqrt(2) # |+i>
    y1 = (z0 - 1j * z1) / np.sqrt(2) # |-i>

    # 스냅샷 행렬 생성 함수: 3 * (|psi><psi|) - I
    def snapshot(state):
        return 3 * (state @ state.conj().T) - I

    # 미리 계산된 룩업 테이블 (속도 최적화)
    # 키: (기저 'X','Y','Z', 결과비트 0,1)
    tables = {
        ('X', 0): snapshot(x0),
        ('X', 1): snapshot(x1),
        ('Y', 0): snapshot(y0),
        ('Y', 1): snapshot(y1),
        ('Z', 0): snapshot(z0),
        ('Z', 1): snapshot(z1),
    }
    return tables

def reconstruct_pauli_shadow_4q(measured_data):
    """
    Pauli Shadow 방식을 이용해 4큐비트 밀도 행렬을 재구성합니다.
    
    Args:
        measured_data: { 'XYZI': [count_0, ..., count_15], ... } 형태의 딕셔너리
        
    Returns:
        rho (16x16 numpy array): 재구성된 밀도 행렬
    """
    print("Pauli Shadow 재구성을 시작합니다. (최적화 과정 없음)")
    
    dim = 16
    rho_accum = np.zeros((dim, dim), dtype=complex)
    total_shots = 0
    
    # 1. 단일 큐비트 스냅샷 룩업 테이블 로드
    shadow_tables = get_shadow_tables()
    
    # 2. 모든 측정 데이터 순회
    for basis_config, counts in measured_data.items():
        # basis_config 예: "XZZI" (4글자)
        
        # counts는 길이 16인 배열 (index 0~15는 측정 결과 0000~1111에 대응)
        for outcome_int, count in enumerate(counts):
            if count == 0:
                continue
                
            # 해당 outcome_int에 대한 4큐비트 스냅샷 생성
            # 스냅샷 = kron(snapshot_q0, snapshot_q1, snapshot_q2, snapshot_q3)
            snapshots = []
            
            for i in range(4): # 4 Qubits
                basis_char = basis_config[i] # 해당 큐비트의 측정 기저 (X, Y, Z)
                
                # outcome_int의 i번째 비트 추출 (0 또는 1)
                # 주의: 큐비트 순서(Little Endian vs Big Endian)에 따라 shift 방향 확인 필요
                # 여기서는 outcome_int >> i 로 i번째 큐비트 값을 가져온다고 가정
                bit = (outcome_int >> 3-i) & 1
                
                # 미리 계산된 테이블에서 행렬 가져오기
                snapshots.append(shadow_tables[(basis_char, bit)])
            
            # 텐서 곱으로 전체 시스템의 스냅샷 생성
            # reduce(np.kron, [A, B, C, D]) -> A (x) B (x) C (x) D
            full_snapshot = reduce(np.kron, snapshots)
            
            # 평균을 위해 누적 (count만큼 가중치)
            rho_accum += full_snapshot * count
            total_shots += count
            
    # 3. 전체 샷 수로 나누어 평균 계산
    rho_est = rho_accum / total_shots
    
    return rho_est

def density_matrix(target_state, p_1q, p_2q, p_meas, shots, with_plot = True, save_directory = "", **kwargs): 
    
    print("1. Assignment Matrix 계산 중...")
    A_matrix = assignment_matrix(
        qubits=[D1, D2, D3, D4], 
        p_1q=p_1q, 
        p_2q=p_2q, 
        p_meas=p_meas,
        shots=shots, 
        **kwargs
    )

    print("2. Tomography 데이터 수집 중 (81 circuits)...")
    tomo_data = run_tomography_experiments(
        target_state_name=target_state,  
        p_1q=p_1q, 
        p_2q=p_2q, 
        p_meas=p_meas,
        shots=shots, 
        **kwargs
    )

    # [Step 3] MLE (Pauli Shadow Reconstruct) 수행
    # 코드 문맥상 reconstruct_pauli_shadow_4q를 사용하는 것으로 보임
    final_rho = reconstruct_pauli_shadow_4q(tomo_data)

    # ---------------------------------------------------------
    # Logical Metrics Calculation
    # ---------------------------------------------------------
    logical = {}
    logical[0] = np.zeros(16)
    logical[0][0b0000] = 1/np.sqrt(2)
    logical[0][0b1111] = 1/np.sqrt(2)

    logical[1] = np.zeros(16)
    logical[1][0b0101] = 1/np.sqrt(2)
    logical[1][0b1010] = 1/np.sqrt(2)
    
    psi = None
    if target_state == '0':
        psi = logical[0]
    elif target_state == '1':
        psi = logical[1]
    elif target_state == '+':
        psi = (logical[0] + logical[1]) / np.sqrt(2)
    elif target_state == '-':
        psi = (logical[0] - logical[1]) / np.sqrt(2)
    else:
        print(f"Warning: Unknown target state '{target_state}'. Using |0>L for fidelity.")
        psi = logical[0]

    # Yield (Physical Probability PL) 계산
    # P_L = <0_L|rho|0_L> + <1_L|rho|1_L> (Unnormalized projectors sum)
    # 주의: logical 벡터들이 normalized 되어 있다면, 아래 식은 P_L을 구하는 올바른 식입니다.
    # 논문 식: P_L = Trace(P_code * rho)
    Logical_probability = (logical[0].T @ final_rho @ logical[0] + 
                           logical[1].T @ final_rho @ logical[1]).real.item()

    # Logical Density Matrix (rho_logical) 계산
    # rho_L = Project / P_L
    rho_logical = np.zeros([2, 2], dtype=complex)
    if Logical_probability > 1e-9:
        for i in range(2):
            for j in range(2):
                val = logical[i].T @ final_rho @ logical[j]
                rho_logical[i][j] = val / Logical_probability
    else:
        print("Warning: Logical probability is too low to normalize.")

    # Physical Fidelity Calculation
    Physical_Fidelity = (psi.T @ final_rho @ psi).real.item()

    print(f"\n--- Metrics ---")
    print(f"Physical Fidelity (F_phys) : {Physical_Fidelity:.3f}")
    print(f"Logical Yield (P_L)        : {Logical_probability:.3f}")
    print("Logical Density Matrix (rho_L):")
    print(np.round(rho_logical, 3))

    # ---------------------------------------------------------
    # Plotting & Saving
    # ---------------------------------------------------------
    # 파일명 생성
    param_str = f"Shots_{shots}_p1q_{p_1q}_p2q_{p_2q}_pmeas_{p_meas}"
    # kwargs에 있는 추가 파라미터(T1, T2 등) 파일명에 추가
    exclude_keys = ['save_directory', 'with_plot']
    if kwargs:
        for key, value in kwargs.items():
            if key not in exclude_keys:
                param_str += f"_{key}_{value}"
    
    filename = f"DensityMat_{target_state}_{param_str}.png"
    
    # Metrics 딕셔너리 포장
    metrics_dict = {
        'F_phys': Physical_Fidelity,
        'P_L': Logical_probability
    }

    # 통합 플롯 함수 호출
    plot_density_matrix_combined(
        rho_phys=final_rho,
        rho_logical=rho_logical,
        metrics=metrics_dict,
        title_prefix=f"Target State |{target_state}⟩",
        save_dir=save_directory,
        filename=filename,
        with_plot=with_plot
    )

    return final_rho

def plot_density_matrix_combined(rho_phys, rho_logical, metrics, title_prefix, save_dir, filename, with_plot):
    """
    Physical Density Matrix (16x16)의 실수/허수부와 
    Logical Density Matrix (2x2)의 실수부를 함께 시각화합니다.
    """
    fig = plt.figure(figsize=(24, 7))
    
    # 텍스트 정보 (Title)
    f_phys = metrics.get('F_phys', 0)
    p_L = metrics.get('P_L', 0)
    main_title = (f"{title_prefix}\n"
                  f"Physical Fidelity ($F_{{phys}}$): {f_phys:.3f} | "
                  f"Yield ($P_L$): {p_L:.3f}")
    fig.suptitle(main_title, fontsize=16, fontweight='bold')

    # ------------------------------------------------
    # Plot 1: Physical Real Part (16x16)
    # ------------------------------------------------
    ax1 = fig.add_subplot(131, projection='3d')
    _plot_3d_bar(ax1, rho_phys.real, "Physical Re[$\\rho$]", 16)

    # ------------------------------------------------
    # Plot 2: Physical Imag Part (16x16)
    # ------------------------------------------------
    ax2 = fig.add_subplot(132, projection='3d')
    _plot_3d_bar(ax2, rho_phys.imag, "Physical Im[$\\rho$]", 16, is_imag=True)

    # ------------------------------------------------
    # Plot 3: Logical Real Part (2x2) - Figure 4c 스타일
    # ------------------------------------------------
    ax3 = fig.add_subplot(133, projection='3d')
    # 논리적 큐비트 라벨
    tick_labels_logical = [r'$|0\rangle_L$', r'$|1\rangle_L$']
    _plot_3d_bar(ax3, rho_logical.real, "Logical Re[$\\rho_L$]", 2, tick_labels=tick_labels_logical)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # 제목 공간 확보

    # ------------------------------------------------
    # 파일 저장 (Save Logic)
    # ------------------------------------------------
    if save_dir and save_dir != "":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"📂 폴더 생성: {save_dir}")
        
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"💾 그래프 저장 완료: {full_path}")

    # ------------------------------------------------
    # 출력 (Show Logic)
    # ------------------------------------------------
    if with_plot:
        plt.show()
    else:
        plt.close(fig) # 메모리 해제

def _plot_3d_bar(ax, matrix_part, title, dim, is_imag=False, tick_labels=None):
    """3D Bar Plot을 그리는 내부 헬퍼 함수"""
    _x = np.arange(dim)
    _y = np.arange(dim)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    z = np.zeros_like(x)
    
    data = matrix_part.ravel()
    dx = dy = 0.6
    
    # 색상 설정
    if is_imag and np.all(data == 0):
        colors = 'cyan'
    else:
        # 값이 너무 작으면 정규화 시 에러 발생 방지
        max_val = np.max(np.abs(data))
        if max_val < 1e-9:
             colors = cm.coolwarm(0.5)
        else:
            offset = data + max_val
            fracs = offset.astype(float) / (2 * max_val)
            norm = plt.Normalize(0, 1)
            colors = cm.coolwarm(norm(fracs))

    ax.bar3d(x, y, z, dx, dy, data, color=colors, shade=True)
    ax.set_title(title)
    
    # 축 라벨 설정
    if tick_labels is None:
        # 기본 16차원 라벨 (0000 ~ 1111)
        tick_labels = [f"{i:04b}" for i in range(dim)]
        
    ax.set_xticks(np.arange(dim) + dx/2)
    ax.set_yticks(np.arange(dim) + dy/2)
    
    # 16개일 때는 글자 크기 줄이고 회전, 2개일 때는 크게
    fontsize = 8 if dim > 4 else 12
    rotation_x = 45 if dim > 4 else 0
    rotation_y = -20 if dim > 4 else 0
    
    ax.set_xticklabels(tick_labels, rotation=rotation_x, ha='right', fontsize=fontsize)
    ax.set_yticklabels(tick_labels, rotation=rotation_y, ha='left', fontsize=fontsize)
    
    # Z축 범위 설정 (Logical은 0~1 사이가 많음)
    z_min, z_max = np.min(data), np.max(data)
    if dim == 2: # Logical
        ax.set_zlim(0, 1.0)
    else:
        ax.set_zlim(z_min, max(z_max, 0.1))