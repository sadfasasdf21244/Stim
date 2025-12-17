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
        **kwarg
        ):
    builder = TransmonBuilder(
        p_1q=p_1q,   
        p_2q=p_2q,  
        p_meas=p_meas, 
        **kwarg 
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
    builder.measure_all(True)

    return builder

#%%

# state preparation 테스트
def state_prep_test(p_1q, p_2q, p_meas, **kwarg):
    for states in ['0', '1', '+', '-']:
        print(f"=== 상태 준비: |{states}⟩ ===")
        builder = state_prep(states,
                            p_1q,    # 0.5%
                            p_2q,     # 2.0%
                            p_meas,   # 3.0%
                            **kwarg
                            )
        shots = 10000
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
        builder = TransmonBuilder(
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
        builder.measure_all(is_first_round=True, A2_basis = 'Z')
        
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
                shots=1000000, 
                rounds=10,
                 **kwarg):
    det_results = figure5_experiment(p_1q, p_2q, p_meas, shots, rounds, **kwarg)
    plot_figure5_ab(det_results)
    plot_figure5_c(det_results)
    plot_figure5_d(det_results)
    return

def figure5_experiment(p_1q, p_2q, p_meas, shots, rounds, **kwarg):
    det_results = {}

    state_labels = ['0', '1', '+', '-']

    for state in state_labels:
        print(f"=== 상태 준비: |{state}⟩ ===")
        builder = state_prep(state, p_1q, p_2q, p_meas, **kwarg)

        # 추가 라운드 반복
        for r in range(rounds):
            builder.measure_all()

        # 2. 샘플링
        circuit = builder.get_circuit()
        raw_result = circuit.compile_detector_sampler().sample(shots=shots)

        det_results[state] = raw_result

    return det_results

def plot_figure5_ab(det_results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    state_labels = ['0', '1', '+', '-']
    colors = ['tab:green', 'tab:red', 'tab:blue'] # A1:초록, A2:빨강, A3:파랑
    markers = ['s', 'o', '^']

    for idx, state in enumerate(state_labels):
        det_result = det_results[state]
        
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

        # Plotting
        ax = axes.flat[idx]
        
        for i in range(num_ancillas): 
            ax.plot(range(num_rounds), preped_operator_prob[:, ANCILLA_INDEX[ANCILLA_QUBITS[i]]], 
                    label=f"Ancilla {QUBITS_NAME[ANCILLA_QUBITS[i]]}", 
                    color=colors[i], 
                    marker=markers[i], 
                    markersize=5, 
                    alpha=0.8)
        ax.set_title(f"State Preparation({preped_shots}/{shots} = {preped_shots/shots*100:.1f}% clean shots): |{state}⟩$_L$")
        ax.set_xlabel("Syndrome Extraction Round")
        ax.set_ylabel("operator expectation value")
        ax.set_ylim(-1, 1)
        ax.set_xticks(range(num_rounds))
        ax.set_yticks(np.arange(-1, 1, 0.25))
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.show()

    #     #prep 성공한 애들
    #     prep_round = reshaped_result[:, 0, :]
    #     prep_success_mask = ~np.any(prep_round, axis=1)
    #     prep_valid_shots = reshaped_result[prep_success_mask]
    #     num_valid = len(prep_valid_shots)
        
    #     #처음부터 에러 없는 애들
    #     no_error_mask = ~np.any(reshaped_result, axis=2)
    #     no_error_mask_accumulated = np.logical_and.accumulate(no_error_mask, axis=1)
    #     success_prob = np.sum(no_error_mask_accumulated, axis=0) / shots

    #     #에러 없고 그 다음에도 에러 확률
    #     cond_probs = {0: [], 1: [], 2: []} # 안실라 인덱스별 리스트

    #     for r in range(1, actual_rounds):
    #         survivors_mask = no_error_mask_accumulated[:, r-1]
    #         num_survivors = np.sum(survivors_mask)
    #         if num_survivors == 0:
    #             print(f"Round {r}: 생존자가 없습니다.")
    #             for i in range(num_ancillas):
    #                 cond_probs[i].append(0.0)
    #             continue    

    #         for i in range(num_ancillas):
    #             # 이번 라운드 안실라 i의 결과 (True=Error)
    #             current_error = reshaped_result[:, r, i]
                
    #             # 조건: (살아남음) AND (이번에 에러)
    #             new_errors = current_error & survivors_mask
    #             num_new_errors = np.sum(new_errors)
                
    #             # 확률 계산
    #             prob = num_new_errors / num_survivors
    #             cond_probs[i].append(prob)


    #     ax_parity_prob = axes[idx, 0]   
    #     ax_success_prob = axes[idx, 1]
    #     ax_syndrome_prob = axes[idx, 2]

    #     # 성공 확률 계산
    #     ax_success_prob.plot(success_prob)
    #     ax_success_prob.set_title(f"Success Probability per Round for |{state}⟩")
    #     ax_success_prob.set_xlabel("Syndrome Extraction Round")
    #     ax_success_prob.set_ylabel("Success Probability")
    #     ax_success_prob.set_yscale('log')

    #     # 5. 평균 에러율 계산
    #     if num_valid > 0:
    #         avg_errors = np.mean(prep_valid_shots, axis=0)
    #     else:
    #         avg_errors = np.zeros((actual_rounds, num_ancillas))
    #         print(f"Warning: State |{state}⟩ has 0 valid shots!")

    #     print(f"State |{state}⟩: {num_valid}/{shots} clean shots ({num_valid/shots*100:.1f}%)")

    #     # 6. Plotting
    #     x_axis = np.arange(actual_rounds)
        
    #     for i in range(num_ancillas): 
    #         ax_parity_prob.plot(x_axis, avg_errors[:, i], 
    #                 label=ancilla_names[i], 
    #                 color=colors[i], 
    #                 marker=markers[i], 
    #                 markersize=5, 
    #                 alpha=0.8)

    #     ax_parity_prob.set_title(f"State Preparation({num_valid}/{shots} = {num_valid/shots*100:.1f}% clean shots): |{state}⟩$_L$")
    #     ax_parity_prob.set_xlabel("Syndrome Extraction Round")
    #     ax_parity_prob.set_ylabel("Detection Event Probability")
    #     ax_parity_prob.set_ylim(0, np.max(avg_errors) * 1.3 + 0.01)
    #     ax_parity_prob.set_xticks(x_axis)
    #     ax_parity_prob.grid(True, which='both', linestyle='--', alpha=0.5)
        
    #     # Round 0 (필터링 기준) 표시
    #     ax_parity_prob.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    #     if idx == 0:
    #         ax_parity_prob.legend()

    #     rounds_x_cond = range(1, actual_rounds)

    #     for i in range(num_ancillas):
    #         ax_syndrome_prob.plot(rounds_x_cond, cond_probs[i], 
    #                               marker=markers[i], 
    #                               linestyle='-', 
    #                               color=colors[i], 
    #                               label=ancilla_names[i], 
    #                               alpha=0.8)

    #     # 4. 스타일링
    #     ax_syndrome_prob.set_title(f"Conditional Error Rate |{state}⟩")
    #     ax_syndrome_prob.set_xlabel("Syndrome Extraction Round")
    #     ax_syndrome_prob.set_ylabel("P(Error at r | No Error < r)")
        
    #     # Y축 범위: 데이터에 따라 유동적으로, 혹은 0~0.1 정도로 고정
    #     ax_syndrome_prob.grid(True, linestyle='--', alpha=0.5)
        
    #     # 첫 번째 행에만 범례 표시 (깔끔하게)
    #     if idx == 0:
    #         ax_syndrome_prob.legend()

    # plt.tight_layout()
    # plt.show()


# %%
def plot_figure5_c(det_results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    state_labels = ['0', '1', '+', '-']
    colors = ['tab:green', 'tab:red', 'tab:blue'] # A1:초록, A2:빨강, A3:파랑
    markers = ['s', 'o', '^']

    for idx, state in enumerate(state_labels):
        det_result = det_results[state]
        
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
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1])


    plt.show()

#%%
def plot_figure5_d(det_results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    state_labels = ['0', '1', '+', '-']
    colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:orange'] # 0,1,2,3 errors
    markers = ['s', 'o', '^', 'D']

    for idx, state in enumerate(state_labels):
        det_result = det_results[state]
        
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