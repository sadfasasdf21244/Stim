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
def state_prep_test(p_1q, p_2q, p_meas, shots, **kwarg):
    for states in ['0', '1', '+', '-']:
        print(f"=== 상태 준비: |{states}⟩ ===")
        builder = state_prep(states,
                            p_1q,    # 0.5%
                            p_2q,     # 2.0%
                            p_meas,   # 3.0%
                            **kwarg
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
                        **kwarg    
                        ):
    probs = []

    for label in input_labels:
        # 1. 빌더 생성
        builder = CircuitBuilder(
            p_1q=p_1q,    
            p_2q=p_2q,    
            p_meas=p_meas, 
            **kwarg    
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
            **kwarg
            ):
    labels_2q = ["".join(x) for x in itertools.product("01", repeat=2)] # 00, 01, 10, 11
    probs_a1 = Figure3_experiment([D1, D3], 1, labels_2q, Shots=shots, p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, **kwarg)

    labels_4q = ["".join(x) for x in itertools.product("01", repeat=4)] # 0000 ~ 1111
    probs_a2 = Figure3_experiment([D1, D2, D3, D4], 0, labels_4q , Shots=shots, p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, **kwarg)

    probs_a3 = Figure3_experiment([D2, D4], 2, labels_2q, Shots=shots, p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, **kwarg)

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
    plt.show()
    return


# %%
def plot_figure5(
                p_1q=0.005, 
                p_2q=0.02, 
                p_meas=0.03, 
                shots=10000000, 
                rounds=10,
                 **kwarg):
    det_results = figure5_experiment(p_1q, p_2q, p_meas, shots, rounds, **kwarg)
    plot_figure5_ab(p_1q, p_2q, p_meas, shots, rounds, **kwarg)
    plot_figure5_c(det_results, shots)
    plot_figure5_d(det_results, shots)
    return

def figure5_experiment(p_1q, p_2q, p_meas, shots, rounds, **kwarg):
    state_builders = {}

    state_labels = ['0', '1', '+', '-']

    for state in state_labels:
        print(f"=== 상태 준비: |{state}⟩ ===")
        builder = state_prep(state, p_1q, p_2q, p_meas, **kwarg)

        # 추가 라운드 반복
        for r in range(rounds):
            builder.measure_ancilla()

        state_builders[state] = builder

    return state_builders

def plot_figure5_ab(p_1q, p_2q, p_meas, shots, max_rounds, **kwarg):
    # 결과를 저장할 딕셔너리
    # 구조: Data[round][state] = expectation_value
    Data = {}
    
    # 1라운드부터 max_rounds까지 반복
    for r in range(0, max_rounds + 1):
        print(f"\n--- Processing Round {r} ---")
        
        # 해당 라운드 수만큼 회로 생성 (0, 1, +, - 네 가지 상태에 대해)
        state_builders = figure5_experiment(p_1q, p_2q, p_meas, shots, r, **kwarg)
        
        data_per_round = {}
        state_labels = ['0', '1', '+', '-']

        for state in state_labels:
            builder = state_builders[state]
            
            # [수정 1] 논리 연산자 정의 (Topology에 맞게 수정 필수!)
            # Distance-2 Surface Code 표준 가정:
            # Z_L = Z1 * Z3 (Vertical)
            # X_L = X1 * X2 (Horizontal)
            if state in ['0', '1']:
                # Z Basis 측정 (D1, D3)
                builder.measure_arbitrary([D1, D2], 'Z')
            else:
                # X Basis 측정 (D1, D2) -> 회로상에서 Basis Change 후 Z측정
                builder.measure_arbitrary([D1, D3], 'X')

            # 회로 컴파일 및 실행
            circuit = builder.get_circuit()
            sampler = circuit.compile_sampler()
            
            # 결과: [Shots, (N_ancilla_rounds * 3) + 2_data]
            raw_result = sampler.sample(shots=shots)
            print(f"raw_results shape = {np.shape(raw_result)}")
            # -----------------------------------------------------------
            # [수정 2] 데이터 분리 및 Detector 계산
            # -----------------------------------------------------------
            # 마지막 2개 비트는 데이터 큐비트 측정 결과
            data_meas = raw_result[:, -2:] 
            
            # 나머지는 안실라 측정 결과
            ancilla_meas = raw_result[:, :-2]
            print(f"ancilla_meas shape = {np.shape(ancilla_meas)}")
            # 안실라 결과 Reshape: (Shots, Rounds+1, 3)
            # Rounds+1인 이유: state_prep(R0) + r번의 추가 라운드
            num_ancillas = len(ANCILLA_QUBITS)
            reshaped_ancilla = ancilla_meas.reshape(shots, -1, num_ancillas)
            print(f"reshaped_ancilla shape = {np.shape(reshaped_ancilla)}")
            # --- Post-selection Logic (핵심) ---
            # 1. Round 0 (Prep): 값이 0이어야 함 (에러 없음)
            # 2. Round N > 0: 이전 라운드와 값이 같아야 함 (M_t XOR M_{t-1} == 0)
            
            # (1) Preparation Error Check (Round 0)
            prep_errors = np.any(reshaped_ancilla[:, 0, :], axis=1)
            
            # (2) Mid-circuit Error Check (Detection Events)
            # 시간 축(axis 1)을 따라 인접한 값끼리 XOR (Diff)
            # detectors shape: (Shots, Rounds, 3)
            detectors = np.logical_xor(reshaped_ancilla[:, 1:, :], reshaped_ancilla[:, :-1, :])
            
            
            # 전체 라운드 중 하나라도 1(변화/에러)이 있으면 True
            mid_errors = np.any(detectors, axis=(1, 2))
            
            # 최종 마스크: Prep 에러도 없고, 중간 에러도 없어야 함
            valid_mask = ~(prep_errors | mid_errors)
            
            # -----------------------------------------------------------
            # 기댓값 계산
            # -----------------------------------------------------------
            num_valid = np.sum(valid_mask)
            
            if num_valid > 0:
                # 살아남은 샷들의 데이터 큐비트 결과 가져오기
                valid_data = data_meas[valid_mask]
                
                # Parity 계산: (1 - 2*D_a) * (1 - 2*D_b)
                # 0 -> +1, 1 -> -1 로 변환하여 곱함
                # Z_L = Z_a * Z_b
                parity = (1 - 2 * valid_data[:, 0]) * (1 - 2 * valid_data[:, 1])
                
                # 평균 (Expectation Value)
                expectation_val = np.mean(parity)
            else:
                print(f"Warning: State |{state}⟩ has 0 valid shots!")
                expectation_val = 0.0

            print(f"  State |{state}⟩: {num_valid}/{shots} ({num_valid/shots*100:.1f}%) kept. <O_L> = {expectation_val:.3f}")
            
            data_per_round[state] = expectation_val

        Data[r] = data_per_round
        
    # 데이터 수집이 끝났으면 그래프 그리기
    plot_memory_experiment(Data)
    
    return Data

def plot_memory_experiment(data):
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
    plt.show()



# def plot_figure5_ab_잘못됨(state_builders):
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))

#     state_labels = ['0', '1', '+', '-']
#     colors = ['tab:green', 'tab:red', 'tab:blue'] # A1:초록, A2:빨강, A3:파랑
#     markers = ['s', 'o', '^']

#     for idx, state in enumerate(state_labels):
#         # 2. 샘플링
#         circuit = state_builders[state].get_circuit()
#         det_result = circuit.compile_detector_sampler().sample(shots=shots)
        
#         num_ancillas =  len(ANCILLA_QUBITS)
#         num_rounds = det_result.shape[1] // num_ancillas # +1 for state prep round
#         shots = det_result.shape[0]

#         det_reshaped_result = det_result.reshape(shots, -1, num_ancillas)               #reshaped detection result
#         reconstructed_result = np.logical_xor.accumulate(det_reshaped_result, axis=1)   #reshaped measurement result 샷 마다 round마다 stabilizer measurement 결과 (0, 1)

#         prep_mask = ~np.any(reconstructed_result[:, 0, :], axis=1) #prep된 애들의 마스크
#         print(f"State |{state}⟩: {np.sum(prep_mask)}/{shots} clean shots ({np.sum(prep_mask)/shots*100:.1f}%)")
        
#         preped_result = reconstructed_result[prep_mask] #prep된 애들의 measurement result
#         preped_det_result = det_reshaped_result[prep_mask]  #prep된 애들의 detection result
#         preped_shots = preped_result.shape[0]               #prep된 애들의 샷 수

#         preped_measurement_prob = np.sum(preped_result, axis=0) / preped_shots #prep된 애들의 measurement 기댓값 (0 또는 1)
#         preped_operator_prob = 1 - 2*preped_measurement_prob #prep 된 애들 stabilizer operator의 기댓값

#         preped_det_result_prob = np.sum(preped_det_result, axis=0) / preped_shots #prep 된 애들 detection result 기댓값
#         preped_det_operator_prob = 1-2*preped_det_result_prob

#         # Plotting
#         ax = axes.flat[idx]
        
#         for i in range(num_ancillas): 
#             ax.plot(range(num_rounds), preped_operator_prob[:, ANCILLA_INDEX[ANCILLA_QUBITS[i]]], 
#                     label=f"Ancilla {QUBITS_NAME[ANCILLA_QUBITS[i]]}", 
#                     color=colors[i], 
#                     marker=markers[i], 
#                     markersize=5, 
#                     alpha=0.8)
#         ax.set_title(f"State Preparation({preped_shots}/{shots} = {preped_shots/shots*100:.1f}% clean shots): |{state}⟩$_L$")
#         ax.set_xlabel("Syndrome Extraction Round")
#         ax.set_ylabel("operator expectation value")
#         ax.set_ylim(-1, 1)
#         ax.set_xticks(range(num_rounds))
#         ax.set_yticks(np.arange(-1, 1, 0.25))
#         ax.grid(True, which='both', linestyle='--', alpha=0.5)

#     plt.legend()
#     plt.tight_layout()
#     plt.show()



# %%
def plot_figure5_c(state_builders,shots):
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


    plt.show()

#%%
def plot_figure5_d(state_builders, shots):
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
    plt.show()

    #     data = np.array(data) # shape: (rounds-1, num_ancillas)

    #     # plot
    #     ax = axes.flat[idx]
    #     for i in range(num_ancillas):
    #         ax.plot(range(1, num_rounds), data[:, i],
    #                 marker=markers[i], 
    #                 linestyle='-', 
    #                 color=colors[i], 
    #                 label=f"Ancilla {QUBITS_NAME[ANCILLA_QUBITS[i]]}", 
    #                 alpha=0.8)
            
    #     ax.set_title(f"Conditional Error Rate : |{state}⟩$_L$")
    #     ax.set_xlabel("Syndrome Extraction Round")
    #     ax.set_ylabel("P(Error at r | No Error < r)")
    #     ax.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    # plt.show()

    return