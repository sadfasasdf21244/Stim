#%%
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
import stim
from pymatching import Matching
from qutip import *
from gates_with_error import *
import itertools
import os
from scipy.optimize import curve_fit

SEED = 12345

def state_prep(target_state,         
        p_1q,
        p_2q,   
        p_meas,
        **kwargs
        ):
    builder = CircuitBuilder(
        p_1q=p_1q,   
        p_2q=p_2q,  
        p_meas=p_meas, 
        **kwargs
    )

    if target_state == '0':
        pass 

    elif target_state == '1':
        builder.pi_y(D2)
        builder.pi_y(D4)

    elif target_state == '+':
        builder.pi_half_y(D2)
        builder.pi_half_y(D4)

    elif target_state == '-':
        builder.pi_half_y(D2)
        builder.minus_pi_half_y(D4)

    # state preparation
    builder.measure_ancilla(True)

    return builder

#%%

# state preparation 테스트
def state_prep_test(p_1q, p_2q, p_meas, shots, **kwargs):
    for states in ['0', '1', '+', '-']:
        print(f"=== 상태 준비: |{states}⟩ ===")
        builder = state_prep(states,
                            p_1q,    # 0.5%
                            p_2q,     # 2.0%
                            p_meas,   # 3.0%
                            **kwargs
                            )
        det_sampler = builder.get_circuit().compile_detector_sampler(seed=SEED)
        result = det_sampler.sample(shots=shots)

        count_success = np.sum(~np.any(result, axis=1))
        prob_success = count_success / shots
        print(f"=== 결과 분석 ===")
        print(f"총 실행 횟수 (Shots): {shots}")
        print(f"'000' 발생 횟수: {count_success}")
        print(f"{states} State Prep 성공 확률 (000 비율): {prob_success:.4f} ({prob_success * 100:.2f}%)")
    return

# %%

def Figure3_experiment(target_qubits,
                        ancilla_idx_in_measure_order,
                        input_labels,
                        Shots,
                        p_1q,  
                        p_2q,    
                        p_meas,  
                        **kwargs    
                        ):
    probs = []

    for label in input_labels:
        # 1. 빌더 생성
        builder = CircuitBuilder(
            p_1q=p_1q,    
            p_2q=p_2q,    
            p_meas=p_meas, 
            **kwargs
        )        
        # 2. 상태 준비 (State Preparation)
        # label이 '10'이면 첫번째 타겟 큐비트에 X, 두번째는 그냥 둠.
        for i, char in enumerate(label):
            if char == '1':
                builder.pi_y(target_qubits[i])
        
        # 3. 측정 수행 (First round 모드 -> Active Reset 없이 측정만 수행)
        builder.measure_ancilla(is_first_round=True, A2_basis = 'Z')
        
        # 4. 샘플링
        sampler = builder.get_circuit().compile_sampler(seed=SEED)
        result = sampler.sample(shots=Shots)
        
        ancilla_result = result[:, ancilla_idx_in_measure_order]
        prob_1 = np.sum(ancilla_result) / Shots
        probs.append(prob_1)
    return probs

def Figure3(shots = 10000,
            p_1q=0.005,    # 0.5%
            p_2q=0.02,     # 2.0%
            p_meas=0.03,   # 3.0%
            with_plot = True,
            save_directory = "",
            **kwargs
            ):
    labels_2q = ["".join(x) for x in itertools.product("01", repeat=2)] # 00, 01, 10, 11
    probs_a1 = Figure3_experiment([D1, D3], 1, labels_2q, Shots=shots, p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, **kwargs)

    labels_4q = ["".join(x) for x in itertools.product("01", repeat=4)] # 0000 ~ 1111
    probs_a2 = Figure3_experiment([D1, D2, D3, D4], 0, labels_4q , Shots=shots, p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, **kwargs)

    probs_a3 = Figure3_experiment([D2, D4], 2, labels_2q, Shots=shots, p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, **kwargs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: A1 (D1-D3)
    axes[0].bar(labels_2q, probs_a1, color='skyblue', edgecolor='black')
    axes[0].set_title("A1 Parity Check (D1, D3)")
    axes[0].set_ylabel("P(Ancilla = 1)")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    # 홀수 패리티(01, 10)에서 높게 나와야 함

    # Plot 2: A2 (D1-D2-D3-D4)
    axes[1].bar(labels_4q, probs_a2, color='salmon', edgecolor='black')
    axes[1].set_title("A2 Parity Check (D1~D4)")
    axes[1].set_xticklabels(labels_4q, rotation=45, ha='right')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    # 1의 개수가 홀수인 상태들에서 높게 나와야 함

    # Plot 3: A3 (D2-D4)
    axes[2].bar(labels_2q, probs_a3, color='lightgreen', edgecolor='black')
    axes[2].set_title("A3 Parity Check (D2, D4)")
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_directory != "":
        # (1) 폴더가 없으면 생성
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"📂 폴더를 생성했습니다: {save_directory}")

        # (2) 파라미터를 문자열로 변환하여 파일명 구성
        # 기본 파라미터들
        param_str = f"Shots_{shots}_p1q_{p_1q}_p2q_{p_2q}_pmeas_{p_meas}"
        
        # **kwarg에 있는 추가 파라미터들 (예: T1, T2 등)도 파일명에 추가
        if kwargs:
            for key, value in kwargs.items():
                param_str += f"_{key}_{value}"
                
        filename = f"Figure3_{param_str}.png"
        
        # (3) 전체 경로 생성
        full_path = os.path.join(save_directory, filename)
        
        
        # (4) 저장 (bbox_inches='tight'는 여백이 잘리는 것을 방지)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"💾 그래프가 저장되었습니다: {full_path}")
    
    if with_plot:
        plt.show()
    return    


# %%
def plot_figure5(
                p_1q=0.005, 
                p_2q=0.02, 
                p_meas=0.03, 
                shots=10000, 
                rounds=10,
                with_plot = True,
                save_directory = "",
                **kwargs):
    builder = figure5_experiment(p_1q, p_2q, p_meas, shots, rounds, **kwargs)
    plot_figure5_ab(p_1q, p_2q, p_meas, shots, rounds, with_plot = with_plot, save_directory = save_directory, **kwargs)
    plot_figure5_c(builder, p_1q, p_2q, p_meas, shots, with_plot = with_plot, save_directory = save_directory, **kwargs)
    plot_figure5_d(builder, p_1q, p_2q, p_meas, shots, with_plot = with_plot, save_directory = save_directory, **kwargs)
    return

def figure5_experiment(p_1q, p_2q, p_meas, shots, rounds,  **kwargs):
    state_builders = {}

    state_labels = ['0', '1', '+', '-']

    for state in state_labels:
        print(f"=== 상태 준비: |{state}⟩ ===")
        builder = state_prep(state, p_1q, p_2q, p_meas, **kwargs)

        # 추가 라운드 반복
        for r in range(rounds):
            builder.measure_ancilla()

        state_builders[state] = builder

    return state_builders

def plot_figure5_ab(p_1q, p_2q, p_meas, shots, max_rounds, with_plot=True, 
                    save_directory="", **kwargs):
    # (기존 코드와 동일하되, 파라미터 전달 방식 유지)
    Data = {}
    
    # 1라운드부터 max_rounds까지 반복
    for r in range(0, max_rounds + 1):
        print(f"Processing Round {r} / {max_rounds} ...")
        
        state_builders = figure5_experiment(p_1q, p_2q, p_meas, shots, r, **kwargs)
        
        data_per_round = {}
        state_labels = ['0', '1', '+', '-']

        for state in state_labels:
            builder = state_builders[state]
            
            # Distance-2 Surface Code 측정 기저 설정
            if state in ['0', '1']:
                builder.measure_arbitrary([D1, D2], 'Z') # Z_L = Z1 * Z2 (혹은 Topology에 맞게 수정)
            else:
                builder.measure_arbitrary([D1, D3], 'X') # X_L = X1 * X3

            circuit = builder.get_circuit()
            sampler = circuit.compile_sampler()
            raw_result = sampler.sample(shots=shots)
            
            # 데이터 분리
            data_meas = raw_result[:, -2:] 
            ancilla_meas = raw_result[:, :-2]
            
            # Post-selection 로직
            num_ancillas = len(ANCILLA_QUBITS)
            reshaped_ancilla = ancilla_meas.reshape(shots, -1, num_ancillas)
            
            prep_errors = np.any(reshaped_ancilla[:, 0, :], axis=1)
            detectors = np.logical_xor(reshaped_ancilla[:, 1:, :], reshaped_ancilla[:, :-1, :])
            mid_errors = np.any(detectors, axis=(1, 2))
            valid_mask = ~(prep_errors | mid_errors)
            
            num_valid = np.sum(valid_mask)
            
            if num_valid > 0:
                valid_data = data_meas[valid_mask]
                # Parity 계산 (+1 or -1)
                parity = (1 - 2 * valid_data[:, 0]) * (1 - 2 * valid_data[:, 1])
                expectation_val = np.mean(parity)
            else:
                expectation_val = 0.0

            data_per_round[state] = expectation_val

        Data[r] = data_per_round
        
    # 그래프 그리기 및 피팅 수행
    plot_memory_experiment(p_1q, p_2q, p_meas, shots, Data, with_plot=with_plot, save_directory=save_directory, **kwargs)
    
    return Data

def plot_memory_experiment(p_1q, p_2q, p_meas, shots, data, with_plot=True, save_directory="", **kwargs):
    """
    Fitting 기능을 추가하여 그래프를 그립니다.
    """
    
    # 1. 데이터 추출
    rounds = np.array(sorted(data.keys()))
    
    y_0 = np.array([data[r]['0'] for r in rounds])
    y_1 = np.array([data[r]['1'] for r in rounds])
    y_plus = np.array([data[r]['+'] for r in rounds])
    y_minus = np.array([data[r]['-'] for r in rounds])

    # 2. 그래프 설정
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 피팅을 위한 x축 (부드러운 곡선용)
    x_fit = np.linspace(min(rounds), max(rounds), 100)

    # --------------------------------------------------------------------------
    # (a) Z-basis Memory (|0>, |1>)
    # --------------------------------------------------------------------------
    # Data Plotting
    ax1.plot(rounds, y_0, 'o', color='blue', label=r'State $|0\rangle_L$', markersize=6)
    ax1.plot(rounds, y_1, 's', color='red', label=r'State $|1\rangle_L$', markersize=6)
    
    # Fitting & Plotting Curve
    stats_text_z = ""
    
    # Fit |0> (Expected A ~ 1)
    try:
        popt0, _ = curve_fit(exponential_decay, rounds, y_0, p0=[1.0, 10.0])
        ax1.plot(x_fit, exponential_decay(x_fit, *popt0), '--', color='blue', alpha=0.7)
        err0 = get_error_rate_from_tau(popt0[1]) * 100
        stats_text_z += f"$|0\\rangle_L$: $\\tau={popt0[1]:.1f}$, $\\epsilon_L={err0:.2f}\\%$\n"
    except:
        stats_text_z += f"$|0\\rangle_L$: Fit Failed\n"

    # Fit |1> (Expected A ~ -1)
    try:
        popt1, _ = curve_fit(exponential_decay, rounds, y_1, p0=[-1.0, 10.0])
        ax1.plot(x_fit, exponential_decay(x_fit, *popt1), '--', color='red', alpha=0.7)
        err1 = get_error_rate_from_tau(popt1[1]) * 100
        stats_text_z += f"$|1\\rangle_L$: $\\tau={popt1[1]:.1f}$, $\\epsilon_L={err1:.2f}\\%$"
    except:
        stats_text_z += f"$|1\\rangle_L$: Fit Failed"

    # Settings
    ax1.set_title("(a) Z-basis Memory", fontsize=16)
    ax1.set_xlabel("Number of rounds", fontsize=14)
    ax1.set_ylabel(r"Logical Expectation $\langle Z_L \rangle$", fontsize=14)
    ax1.set_ylim(-1.1, 1.1)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10, loc='upper right')
    
    # Decay Rate Text (Left Center)
    # transform=ax1.transAxes를 쓰면 (0,0)이 왼쪽 아래, (1,1)이 오른쪽 위입니다.
    ax1.text(0.02, 0.5, stats_text_z, transform=ax1.transAxes, 
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # --------------------------------------------------------------------------
    # (b) X-basis Memory (|+>, |->)
    # --------------------------------------------------------------------------
    # Data Plotting
    ax2.plot(rounds, y_plus, '^', color='green', label=r'State $|+\rangle_L$', markersize=6)
    ax2.plot(rounds, y_minus, 'd', color='purple', label=r'State $|-\rangle_L$', markersize=6)

    # Fitting & Plotting Curve
    stats_text_x = ""

    # Fit |+> (Expected A ~ 1)
    try:
        popt_p, _ = curve_fit(exponential_decay, rounds, y_plus, p0=[1.0, 10.0])
        ax2.plot(x_fit, exponential_decay(x_fit, *popt_p), '--', color='green', alpha=0.7)
        err_p = get_error_rate_from_tau(popt_p[1]) * 100
        stats_text_x += f"$|+\\rangle_L$: $\\tau={popt_p[1]:.1f}$, $\\epsilon_L={err_p:.2f}\\%$\n"
    except:
        stats_text_x += f"$|+\\rangle_L$: Fit Failed\n"

    # Fit |-> (Expected A ~ -1)
    try:
        popt_m, _ = curve_fit(exponential_decay, rounds, y_minus, p0=[-1.0, 10.0])
        ax2.plot(x_fit, exponential_decay(x_fit, *popt_m), '--', color='purple', alpha=0.7)
        err_m = get_error_rate_from_tau(popt_m[1]) * 100
        stats_text_x += f"$|-\\rangle_L$: $\\tau={popt_m[1]:.1f}$, $\\epsilon_L={err_m:.2f}\\%$"
    except:
        stats_text_x += f"$|-\\rangle_L$: Fit Failed"
    
    print(f"fitted parameters {popt_p}")
    
    # Settings
    ax2.set_title("(b) X-basis Memory", fontsize=16)
    ax2.set_xlabel("Number of rounds", fontsize=14)
    ax2.set_ylabel(r"Logical Expectation $\langle X_L \rangle$", fontsize=14)
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=10, loc='upper right')

    # Decay Rate Text (Left Center)
    ax2.text(0.02, 0.5, stats_text_x, transform=ax2.transAxes, 
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()

    # 3. 파일 저장
    if save_directory != "":
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"📂 폴더를 생성했습니다: {save_directory}")

        param_str = f"Shots_{shots}_p1q_{p_1q}_p2q_{p_2q}_pmeas_{p_meas}"
        if kwargs:
            for key, value in kwargs.items():
                param_str += f"_{key}_{value}"
                
        filename = f"Figure5ab_Fit_{param_str}.png"
        full_path = os.path.join(save_directory, filename)
        
        plt.savefig(full_path, bbox_inches='tight')
        print(f"💾 그래프가 저장되었습니다: {full_path}")

    if with_plot:
        plt.show()
        
def exponential_decay(n, A, tau):
    """
    지수 감쇠 모델: y = A * exp(-n / tau)
    n: 라운드 수 (Cycle)
    A: 초기 진폭 (Amplitutde, 보통 1 or -1)
    tau: 수명 (Lifetime, decay constant)
    """
    return A * np.exp(-n / tau)

def get_error_rate_from_tau(tau):
    """
    Lifetime(tau)를 Cycle당 에러율(epsilon)로 변환
    Decay model: <O> ~ exp(-n/tau)
    Discrete Error model: <O> ~ (1 - 2*epsilon)^n
    Relation: 1 - 2*epsilon = exp(-1/tau)
    => epsilon = (1 - exp(-1/tau)) / 2
    """
    if tau == 0: return 1.0
    return (1 - np.exp(-1 / tau)) / 2

def plot_memory_experiment(p_1q, p_2q, p_meas, shots, data, with_plot = True, save_directory = "", **kwargs):
    """
    data 구조: data[round][state] = expectation_value
    round: 1 ~ 10
    state: '0', '1', '+', '-'
    """
    
    # 1. 데이터 추출 (Parsing)
    # 라운드 키를 정렬 (1, 2, ..., 10)
    rounds = sorted(data.keys())
    
    # 각 상태별로 리스트 생성
    y_0 = [data[r]['0'] for r in rounds]
    y_1 = [data[r]['1'] for r in rounds]
    y_plus = [data[r]['+'] for r in rounds]
    y_minus = [data[r]['-'] for r in rounds]

    # 2. 그래프 설정 (2개의 서브플롯 생성)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ==========================================================
    # Figure 5 (a): Z-basis Memory (|0>, |1>)
    # Y축: <Z_L>
    # ==========================================================
    # |0> 상태 (Expected +1)
    ax1.plot(rounds, y_0, 'o-', color='blue', label=r'State $|0\rangle_L$', markersize=8)
    # |1> 상태 (Expected -1)
    ax1.plot(rounds, y_1, 's-', color='red', label=r'State $|1\rangle_L$', markersize=8)

    ax1.set_title("(a) Z-basis Memory", fontsize=16)
    ax1.set_xlabel("Number of rounds", fontsize=14)
    ax1.set_ylabel(r"Logical Expectation $\langle Z_L \rangle$", fontsize=14)
    ax1.set_ylim(-1.1, 1.1)  # 기댓값은 -1 ~ 1 사이
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8) # 0 기준선
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=12)

    # ==========================================================
    # Figure 5 (b): X-basis Memory (|+>, |->)
    # Y축: <X_L>
    # ==========================================================
    # |+> 상태 (Expected +1)
    ax2.plot(rounds, y_plus, '^-', color='green', label=r'State $|+\rangle_L$', markersize=8)
    # |-> 상태 (Expected -1)
    ax2.plot(rounds, y_minus, 'd-', color='purple', label=r'State $|-\rangle_L$', markersize=8)

    ax2.set_title("(b) X-basis Memory", fontsize=16)
    ax2.set_xlabel("Number of rounds", fontsize=14)
    ax2.set_ylabel(r"Logical Expectation $\langle X_L \rangle$", fontsize=14)
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()

    if save_directory != "":
        # (1) 폴더가 없으면 생성
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"📂 폴더를 생성했습니다: {save_directory}")

        # (2) 파라미터를 문자열로 변환하여 파일명 구성
        # 기본 파라미터들
        param_str = f"Shots_{shots}_p1q_{p_1q}_p2q_{p_2q}_pmeas_{p_meas}"
        
        # **kwarg에 있는 추가 파라미터들 (예: T1, T2 등)도 파일명에 추가
        if kwargs:
            for key, value in kwargs.items():
                param_str += f"_{key}_{value}"
                
        filename = f"Figure5ab_{param_str}.png"
        
        # (3) 전체 경로 생성
        full_path = os.path.join(save_directory, filename)
        
        
        # (4) 저장 (bbox_inches='tight'는 여백이 잘리는 것을 방지)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"💾 그래프가 저장되었습니다: {full_path}")

    if with_plot:
        plt.show()



# %%det_results, p_1q, p_2q, p_meas, shots, with_plot = with_plot, save_directory = save_directory
def plot_figure5_c(state_builders, p_1q, p_2q, p_meas,  shots, with_plot = True, save_directory = "", **kwargs):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    state_labels = ['0', '1', '+', '-']
    colors = ['tab:green', 'tab:red', 'tab:blue'] # A1:초록, A2:빨강, A3:파랑
    markers = ['s', 'o', '^']

    for idx, state in enumerate(state_labels):
        circuit = state_builders[state].get_circuit()
        det_result = circuit.compile_detector_sampler().sample(shots=shots)
        
        num_ancillas =  len(ANCILLA_QUBITS)
        num_rounds = det_result.shape[1] // num_ancillas # +1 for state prep round
        shots = det_result.shape[0]

        det_reshaped_result = det_result.reshape(shots, -1, num_ancillas)               #reshaped detection result
        reconstructed_result = np.logical_xor.accumulate(det_reshaped_result, axis=1)   #reshaped measurement result 샷 마다 round마다 stabilizer measurement 결과 (0, 1)

        prep_mask = ~np.any(reconstructed_result[:, 0, :], axis=1) #prep된 애들의 마스크
        print(f"State |{state}⟩: {np.sum(prep_mask)}/{shots} clean shots ({np.sum(prep_mask)/shots*100:.1f}%)")
        
        preped_result = reconstructed_result[prep_mask] #prep된 애들의 measurement result
        preped_det_result = det_reshaped_result[prep_mask]  #prep된 애들의 detection result
        preped_shots = preped_result.shape[0]               #prep된 애들의 샷 수

        preped_measurement_prob = np.sum(preped_result, axis=0) / preped_shots #prep된 애들의 measurement 기댓값 (0 또는 1)
        preped_operator_prob = 1 - 2*preped_measurement_prob #prep 된 애들 stabilizer operator의 기댓값

        preped_det_result_prob = np.sum(preped_det_result, axis=0) / preped_shots #prep 된 애들 detection result 기댓값
        
        
        no_error_mask = ~np.any(det_reshaped_result, axis = 2) # 에러 없는 샷 마스크
        no_error_mask_accumulated = np.logical_and.accumulate(no_error_mask, axis=1)
        success_prob = np.sum(no_error_mask_accumulated, axis=0) / shots

        # plot
        ax = axes.flat[idx]
        ax.plot(range(num_rounds), success_prob, color='purple', marker='o',
                markersize=5, alpha=0.8)
        ax.set_yscale('log')
        ax.set_title(f"Success Probability : |{state}⟩$_L$")
        ax.set_xlabel("Syndrome Extraction Round")
        ax.set_ylabel("Success Probability")
        ax.set_xticks(range(num_rounds))
        ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])


    if save_directory != "":
        # (1) 폴더가 없으면 생성
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"📂 폴더를 생성했습니다: {save_directory}")

        # (2) 파라미터를 문자열로 변환하여 파일명 구성
        # 기본 파라미터들
        param_str = f"Shots_{shots}_p1q_{p_1q}_p2q_{p_2q}_pmeas_{p_meas}"
        
        # **kwarg에 있는 추가 파라미터들 (예: T1, T2 등)도 파일명에 추가
        if kwargs:
            for key, value in kwargs.items():
                param_str += f"_{key}_{value}"
                
        filename = f"Figure5c_{param_str}.png"
        
        # (3) 전체 경로 생성
        full_path = os.path.join(save_directory, filename)
        
        
        # (4) 저장 (bbox_inches='tight'는 여백이 잘리는 것을 방지)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"💾 그래프가 저장되었습니다: {full_path}")

    if with_plot:
        plt.show()
    
    return
#%%
def plot_figure5_d(state_builders, p_1q, p_2q, p_meas,  shots, with_plot = True, save_directory = "", **kwargs):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    state_labels = ['0', '1', '+', '-']
    colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:orange'] # 0,1,2,3 errors
    markers = ['s', 'o', '^', 'D']

    for idx, state in enumerate(state_labels):
        circuit = state_builders[state].get_circuit()
        det_result = circuit.compile_detector_sampler().sample(shots=shots)
        
        num_ancillas =  len(ANCILLA_QUBITS)
        num_rounds = det_result.shape[1] // num_ancillas # +1 for state prep round
        shots = det_result.shape[0]

        det_reshaped_result = det_result.reshape(shots, -1, num_ancillas)               #reshaped detection result
        reconstructed_result = np.logical_xor.accumulate(det_reshaped_result, axis=1)   #reshaped measurement result 샷 마다 round마다 stabilizer measurement 결과 (0, 1)

        prep_mask = ~np.any(reconstructed_result[:, 0, :], axis=1) #prep된 애들의 마스크
        print(f"State |{state}⟩: {np.sum(prep_mask)}/{shots} clean shots ({np.sum(prep_mask)/shots*100:.1f}%)")
        
        preped_result = reconstructed_result[prep_mask] #prep된 애들의 measurement result
        preped_det_result = det_reshaped_result[prep_mask]  #prep된 애들의 detection result
        preped_shots = preped_result.shape[0]               #prep된 애들의 샷 수

        preped_measurement_prob = np.sum(preped_result, axis=0) / preped_shots #prep된 애들의 measurement 기댓값 (0 또는 1)
        preped_operator_prob = 1 - 2*preped_measurement_prob #prep 된 애들 stabilizer operator의 기댓값

        preped_det_result_prob = np.sum(preped_det_result, axis=0) / preped_shots #prep 된 애들 detection result 기댓값
        
        
        no_error_mask = ~np.any(det_reshaped_result, axis = 2) # 에러 없는 샷 마스크
        no_error_mask_accumulated = np.logical_and.accumulate(no_error_mask, axis=1)

        Data = []
        for r in range(1, num_rounds):
            survivors_mask = no_error_mask_accumulated[:, r-1]
            num_survivors = np.sum(survivors_mask)

            if num_survivors == 0:
                print(f"Round {r}: 생존자가 없습니다.")
                continue    

            data = []
            for i in range(num_ancillas):
                # 이번 라운드 안실라 i의 결과 (True=Error)
                current_error = det_reshaped_result[:, r, ANCILLA_INDEX[ANCILLA_QUBITS[i]]]
                
                # 조건: (살아남음) AND (이번에 에러)
                new_errors = current_error & survivors_mask
                
                data.append(new_errors)
            
            summed_data = np.sum(data, axis = 0)
            values, counts = np.unique(summed_data, return_counts=True)
            a = np.zeros(4)
            for idx_e, value in enumerate(values):
                if value != 0:
                    a[value] = counts[idx_e]/num_survivors
            a[0] = 1 - np.sum(a)
            Data.append(a)
        Data = np.array(Data) # shape: (rounds-1, 4)

        ax = axes.flat[idx]
        for key in range(4):
            y_data = Data[:, key]
            ax.plot(range(1, num_rounds), y_data,
                    marker=markers[key], 
                    linestyle='-', 
                    color=colors[key % len(colors)], 
                    label=f"{key} Errors", 
                    alpha=0.8)
        ax.set_title(f"Multiple Error Probability : |{state}⟩$_L$")
        ax.set_xlabel("Syndrome Extraction Round")
        ax.set_ylabel("Probability")
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    if save_directory != "":
        # (1) 폴더가 없으면 생성
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"📂 폴더를 생성했습니다: {save_directory}")

        # (2) 파라미터를 문자열로 변환하여 파일명 구성
        # 기본 파라미터들
        param_str = f"Shots_{shots}_p1q_{p_1q}_p2q_{p_2q}_pmeas_{p_meas}"
        
        # **kwarg에 있는 추가 파라미터들 (예: T1, T2 등)도 파일명에 추가
        if kwargs:
            for key, value in kwargs.items():
                param_str += f"_{key}_{value}"
                
        filename = f"Figure5d_{param_str}.png"
        
        # (3) 전체 경로 생성
        full_path = os.path.join(save_directory, filename)
        
        
        # (4) 저장 (bbox_inches='tight'는 여백이 잘리는 것을 방지)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"💾 그래프가 저장되었습니다: {full_path}")

    if with_plot:
        plt.show()

    return
