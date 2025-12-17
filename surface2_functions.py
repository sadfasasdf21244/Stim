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
    results , det_results = figure5_experiment(p_1q, p_2q, p_meas, shots, rounds, **kwarg)
    plot_figure5_ab(results, det_results)
    # plot_figure5_c(results, det_results)
    # plot_figure5_d(results, det_results)
    return

def figure5_experiment(p_1q, p_2q, p_meas, shots, rounds, **kwarg):
    results = {}
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
        raw_result = circuit.compile_detector_sampler(seed=SEED).sample(shots=shots)
        A_result = circuit.compile_sampler(seed=SEED).sample(shots=shots)

        results[state] = raw_result
        det_results[state] = A_result

    return results, det_results

def plot_figure5_ab(results, det_results):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    state_labels = ['0', '1', '+', '-']
    
    ancilla_names = ['A1 (2-body)', 'A2 (4-body)', 'A3 (2-body)']
    ancilla_index = [1, 0, 2]  # A1, A2, A3 순서대로 인덱스
    colors = ['tab:green', 'tab:red', 'tab:blue'] # A1:초록, A2:빨강, A3:파랑
    markers = ['s', 'o', '^']

    for idx, state in enumerate(state_labels):
        raw_result = results[state]
        det_result = det_results[state]
        
        num_ancillas =  len(ANCILLA_QUBITS)
        num_rounds = raw_result.shape[1] // num_ancillas + 1 # +1 for state prep round
        shots = raw_result.shape[0]

        reshaped_result = raw_result.reshape(shots, -1, num_ancillas)
        det_reshaped_result = det_result.reshape(shots, -1, num_ancillas)
        


    #     reshaped_result = raw_result.reshape(shots, -1, len(ANCILLA_QUBITS))
    #     A_reshaped_result = raw_result.reshape(shots, -1, len(ANCILLA_QUBITS))

    #     actual_rounds = reshaped_result.shape[1]
    #     num_ancillas = reshaped_result.shape[2]

    #     print(np.bitwise_xor.accumulate(reshaped_result, axis=1) == A_reshaped_result)

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
