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
from datetime import datetime  # [중요] 날짜/시간 처리를 위해 모듈 임포트 필요

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
        builder.tick()

    elif target_state == '+':
        builder.pi_half_y(D2)
        builder.pi_half_y(D4)
        builder.tick()

    elif target_state == '-':
        builder.pi_half_y(D2)
        builder.minus_pi_half_y(D4)
        builder.tick()

    # state preparation
    builder.measure_ancilla(is_first_round = True)

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
        det_sampler = builder.get_circuit().compile_detector_sampler()
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

import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from datetime import datetime # datetime 모듈 필요

def Figure3(shots=10000,
            p_1q=0.005,    # 0.5%
            p_2q=0.02,     # 2.0%
            p_meas=0.03,   # 3.0%
            with_plot=True,
            save_directory="",
            **kwargs
            ):
    
    # ----------------------------------------------------------------
    # 내부 함수: 성공 확률 계산 (Assignment Fidelity)
    # ----------------------------------------------------------------
    def calculate_success_rate(labels, probs):
        """
        이상적인 패리티(Odd=1, Even=0)와 측정 결과(prob_1)를 비교하여
        평균 성공 확률을 계산합니다.
        """
        total_success = 0
        for label, prob_1 in zip(labels, probs):
            # 입력 상태의 '1' 개수가 홀수면 Parity=1, 짝수면 Parity=0
            ideal_parity = label.count('1') % 2
            
            if ideal_parity == 1:
                # 홀수 패리티: 1로 측정되어야 성공 (prob_1이 성공 확률)
                total_success += prob_1
            else:
                # 짝수 패리티: 0으로 측정되어야 성공 (1 - prob_1이 성공 확률)
                total_success += (1 - prob_1)
        
        # 전체 상태에 대한 평균 반환
        return total_success / len(labels)

    # 1. 데이터 수집
    # ----------------------------------------------------------------
    labels_2q = ["".join(x) for x in itertools.product("01", repeat=2)] # 00, 01, 10, 11
    probs_a1 = Figure3_experiment([D1, D3], 1, labels_2q, Shots=shots, p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, **kwargs)

    labels_4q = ["".join(x) for x in itertools.product("01", repeat=4)] # 0000 ~ 1111
    probs_a2 = Figure3_experiment([D1, D2, D3, D4], 0, labels_4q , Shots=shots, p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, **kwargs)

    probs_a3 = Figure3_experiment([D2, D4], 2, labels_2q, Shots=shots, p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, **kwargs)

    # 2. 성공 확률 계산
    # ----------------------------------------------------------------
    acc_a1 = calculate_success_rate(labels_2q, probs_a1)
    acc_a2 = calculate_success_rate(labels_4q, probs_a2)
    acc_a3 = calculate_success_rate(labels_2q, probs_a3)

    # 3. Plotting
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: A1 (D1-D3)
    axes[0].bar(labels_2q, probs_a1, color='skyblue', edgecolor='black')
    axes[0].set_title(f"A1 Parity Check (D1, D3)\nSuccess Prob: {acc_a1*100:.1f}%") # 제목에 확률 추가
    axes[0].set_ylabel("P(Ancilla = 1)")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: A2 (D1-D2-D3-D4)
    axes[1].bar(labels_4q, probs_a2, color='salmon', edgecolor='black')
    axes[1].set_title(f"A2 Parity Check (D1~D4)\nSuccess Prob: {acc_a2*100:.1f}%") # 제목에 확률 추가
    axes[1].set_xticklabels(labels_4q, rotation=45, ha='right')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 3: A3 (D2-D4)
    axes[2].bar(labels_2q, probs_a3, color='lightgreen', edgecolor='black')
    axes[2].set_title(f"A3 Parity Check (D2, D4)\nSuccess Prob: {acc_a3*100:.1f}%") # 제목에 확률 추가
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # 4. 파일 저장 (타임스탬프 포함)
    # ----------------------------------------------------------------
    if save_directory != "":
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"📂 폴더를 생성했습니다: {save_directory}")

        param_str = f"Shots_{shots}_p1q_{p_1q}_p2q_{p_2q}_pmeas_{p_meas}"
        
        if kwargs:
            for key, value in kwargs.items():
                param_str += f"_{key}_{value}"
        
        timestamp = datetime.now().strftime("%m%d%H%M")
        filename = f"Figure3_Fit_{param_str}_{timestamp}.png"
        full_path = os.path.join(save_directory, filename)
        
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
    # plot_figure5_ab(p_1q, p_2q, p_meas, shots, rounds, with_plot = with_plot, save_directory = save_directory, **kwargs)
    # plot_figure5_c(builder, p_1q, p_2q, p_meas, shots, with_plot = with_plot, save_directory = save_directory, **kwargs)
    plot_figure5_d(builder, p_1q, p_2q, p_meas, shots, with_plot = with_plot, save_directory = save_directory, **kwargs)
    return

def figure5_experiment(p_1q, p_2q, p_meas, shots, rounds, **kwargs):
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
    sequence_time = kwargs.get('sequence_time', 0.0)
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
                builder.measure_arbitrary([D3, D4], 'Z') # Z_L = Z1 * Z2 (혹은 Topology에 맞게 수정)
            else:
                builder.measure_arbitrary([D2, D4], 'X') # X_L = X1 * X3

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
                parity = (1 - 2 * valid_data[:, -1]) * (1 - 2 * valid_data[:, -2])
                expectation_val = np.mean(parity)
            else:
                expectation_val = 0.0

            data_per_round[state] = expectation_val

        Data[r] = data_per_round
        
    # 그래프 그리기 및 피팅 수행
    plot_memory_experiment(p_1q, p_2q, p_meas, shots, Data, with_plot=with_plot, save_directory=save_directory, **kwargs)
    
    return Data


# (이전과 동일한 exponential_decay, get_error_rate_from_tau 함수가 있다고 가정)

def plot_memory_experiment(p_1q, p_2q, p_meas, shots, data, with_plot=True, save_directory="", **kwargs):
    """
    Fitting 기능을 추가하여 그래프를 그립니다.
    sequence_time이 주어지면 물리적 시간 단위의 Lifetime도 계산하여 표시합니다.
    Args:
        sequence_time (float, optional): 한 라운드(Stabilizer Measurement)에 걸리는 시간 (단위: us 권장).
    """
    sequence_time = kwargs.get('sequence_time', 0.0)
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
    
    # --- Helper function for text formatting ---
    def format_stats(label, popt, seq_time):
        tau_rnd = popt[1]
        err_rate = get_error_rate_from_tau(tau_rnd) * 100
        
        # 기본 텍스트 (Rounds 기준)
        text = f"${label}$: $\\tau_{{rnd}}={tau_rnd:.1f}$"
        
        # Sequence Time이 있으면 물리적 시간 추가
        if seq_time is not None:
            tau_abs = tau_rnd * seq_time
            text += f", $\\tau_{{time}}={tau_abs:.1f}\\mu s$"
            
        text += f", $\\epsilon_L={err_rate:.2f}\\%$\n"
        return text
    # -------------------------------------------

    # Fit |0> (Expected A ~ 1)
    try:
        popt0, _ = curve_fit(exponential_decay, rounds, y_0, p0=[1.0, 10.0])
        ax1.plot(x_fit, exponential_decay(x_fit, *popt0), '--', color='blue', alpha=0.7)
        stats_text_z += format_stats("|0\\rangle_L", popt0, sequence_time)
    except:
        stats_text_z += f"$|0\\rangle_L$: Fit Failed\n"

    # Fit |1> (Expected A ~ -1)
    try:
        popt1, _ = curve_fit(exponential_decay, rounds, y_1, p0=[-1.0, 10.0])
        ax1.plot(x_fit, exponential_decay(x_fit, *popt1), '--', color='red', alpha=0.7)
        stats_text_z += format_stats("|1\\rangle_L", popt1, sequence_time).strip() # 마지막 줄바꿈 제거
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
        stats_text_x += format_stats("|+\\rangle_L", popt_p, sequence_time)
    except:
        stats_text_x += f"$|+\\rangle_L$: Fit Failed\n"

    # Fit |-> (Expected A ~ -1)
    try:
        popt_m, _ = curve_fit(exponential_decay, rounds, y_minus, p0=[-1.0, 10.0])
        ax2.plot(x_fit, exponential_decay(x_fit, *popt_m), '--', color='purple', alpha=0.7)
        stats_text_x += format_stats("|-\\rangle_L", popt_m, sequence_time).strip()
    except:
        stats_text_x += f"$|-\\rangle_L$: Fit Failed"
    
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
        
        # ---------------------------------------------------------
        # [수정된 부분] 타임스탬프 생성 및 적용
        # 현재 시간을 가져와서 '월일시분' (예: 10281630) 형식으로 변환
        # %m: 월, %d: 일, %H: 시(24시간), %M: 분
        timestamp = datetime.now().strftime("%m%d%H%M")
        
        # 파일명 끝에 타임스탬프 추가
        filename = f"Figure5ab_Fit_{param_str}_{timestamp}.png"
        # ---------------------------------------------------------

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


# %%det_results, p_1q, p_2q, p_meas, shots, with_plot = with_plot, save_directory = save_directory
def plot_figure5_c(state_builders, p_1q, p_2q, p_meas,  shots, with_plot = True, save_directory = "", **kwargs):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sequence_time = kwargs.get('sequence_time', 0.0)
    
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

    # 3. 파일 저장
    if save_directory != "":
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"📂 폴더를 생성했습니다: {save_directory}")

        param_str = f"Shots_{shots}_p1q_{p_1q}_p2q_{p_2q}_pmeas_{p_meas}"
        
        if kwargs:
            for key, value in kwargs.items():
                param_str += f"_{key}_{value}"
        
        # ---------------------------------------------------------
        # [수정된 부분] 타임스탬프 생성 및 적용
        # 현재 시간을 가져와서 '월일시분' (예: 10281630) 형식으로 변환
        # %m: 월, %d: 일, %H: 시(24시간), %M: 분
        timestamp = datetime.now().strftime("%m%d%H%M")
        
        # 파일명 끝에 타임스탬프 추가
        filename = f"Figure5c_Fit_{param_str}_{timestamp}.png"
        # ---------------------------------------------------------

        full_path = os.path.join(save_directory, filename)
        
        plt.savefig(full_path, bbox_inches='tight')
        print(f"💾 그래프가 저장되었습니다: {full_path}")

    if with_plot:
        plt.show()
    
    return
#%%
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_figure5_d(state_builders, p_1q, p_2q, p_meas, shots, with_plot = True, save_directory = "", **kwargs):
    sequence_time = kwargs.get('sequence_time', 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    state_labels = ['0', '1', '+', '-']
    colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:orange'] # 0,1,2,3 errors
    markers = ['s', 'o', '^', 'D']

    for idx, state in enumerate(state_labels):
        circuit = state_builders[state].get_circuit()
        det_result = circuit.compile_detector_sampler().sample(shots=shots)
        
        num_ancillas = len(ANCILLA_QUBITS)
        num_rounds = det_result.shape[1] // num_ancillas 
        shots = det_result.shape[0]

        det_reshaped_result = det_result.reshape(shots, -1, num_ancillas)               
        reconstructed_result = np.logical_xor.accumulate(det_reshaped_result, axis=1)   

        prep_mask = ~np.any(reconstructed_result[:, 0, :], axis=1) 
        print(f"State |{state}⟩: {np.sum(prep_mask)}/{shots} clean shots ({np.sum(prep_mask)/shots*100:.1f}%)")
        
        # (중략: preped_result 계산 부분은 Figure 5d 그래프 그리는 데 직접 안 쓰이므로 생략 가능하나 원본 유지)
        
        no_error_mask = ~np.any(det_reshaped_result, axis = 2) # 에러 없는 샷 마스크
        no_error_mask_accumulated = np.logical_and.accumulate(no_error_mask, axis=1)

        Data = []
        # round 1부터 시작
        for r in range(1, num_rounds):
            survivors_mask = no_error_mask_accumulated[:, r-1]
            num_survivors = np.sum(survivors_mask)

            if num_survivors == 0:
                print(f"Round {r}: 생존자가 없습니다.")
                continue    

            data = []
            for i in range(num_ancillas):
                current_error = det_reshaped_result[:, r, ANCILLA_INDEX[ANCILLA_QUBITS[i]]]
                new_errors = current_error & survivors_mask
                data.append(new_errors)
            
            summed_data = np.sum(data, axis = 0)
            values, counts = np.unique(summed_data, return_counts=True)
            a = np.zeros(4) # 0, 1, 2, 3개 에러 확률 담을 배열
            for idx_e, value in enumerate(values):
                if value != 0:
                    a[value] = counts[idx_e]/num_survivors
            a[0] = 1 - np.sum(a) # 나머지는 0개 에러 확률
            Data.append(a)
        
        Data = np.array(Data) # shape: (rounds-1, 4)

        # -----------------------------------------------------------
        # [추가된 부분] No Error Probability 평균 계산
        # -----------------------------------------------------------
        # Data[:, 0]은 각 라운드 별 '에러가 0개일 확률'들의 배열입니다.
        avg_no_error_prob = np.mean(Data[:, 0])
        print(f" -> State |{state}⟩ Avg No Error Prob: {avg_no_error_prob:.4f} ({avg_no_error_prob*100:.2f}%)")
        # -----------------------------------------------------------

        ax = axes.flat[idx]
        for key in range(4):
            y_data = Data[:, key]
            ax.plot(range(1, num_rounds), y_data,
                    marker=markers[key], 
                    linestyle='-', 
                    color=colors[key % len(colors)], 
                    label=f"{key} Errors", 
                    alpha=0.8)
        
        # 제목에 평균 확률 추가
        ax.set_title(f"Multiple Error Probability : |{state}⟩$_L$\n(Avg No-Error: {avg_no_error_prob*100:.2f}%)")
        ax.set_xlabel("Syndrome Extraction Round")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.05, 1.05) # 확률이니까 범위 고정해주는게 보기 좋음
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend() # 범례 위치 자동

    plt.tight_layout()

    if save_directory != "":
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"📂 폴더를 생성했습니다: {save_directory}")

        param_str = f"Shots_{shots}_p1q_{p_1q}_p2q_{p_2q}_pmeas_{p_meas}"
        
        if kwargs:
            for key, value in kwargs.items():
                param_str += f"_{key}_{value}"
        
        timestamp = datetime.now().strftime("%m%d%H%M")
        
        filename = f"Figure5d_Fit_{param_str}_{timestamp}.png"
        full_path = os.path.join(save_directory, filename)
        
        plt.savefig(full_path, bbox_inches='tight')
        print(f"💾 그래프가 저장되었습니다: {full_path}")

    if with_plot:
        plt.show()

    return
