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
#%%

# state prepare
target_state = "-"  # '0', '1', '+', '-' 중 선택

builder = TransmonBuilder(
    p_1q=0.005,    # 0.5%
    p_2q=0.02,     # 2.0%
    p_meas=0.03,   # 3.0%
    p_1q_z=0.0     # 가상 Z 게이트는 에러 0으로 설정
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

#%% preparationg 될 확률
shots = 1000
sampler = builder.get_circuit().compile_sampler()

result = sampler.sample(shots=shots)
count_success = np.sum(np.all(result == 0, axis=1))
prob_success = count_success / shots
print(f"=== 결과 분석 ===")
print(f"총 실행 횟수 (Shots): {shots}")
print(f"'000' 발생 횟수: {count_success}")
print(f"State Prep 성공 확률 (000 비율): {prob_success:.4f} ({prob_success * 100:.2f}%)")
# %%
SHOTS = 1000

def run_parity_experiment(target_qubits, ancilla_idx_in_measure_order, input_labels):
    """
    target_qubits: 상태를 준비할 데이터 큐비트 리스트 (예: [D1, D3])
    ancilla_idx_in_measure_order: measure_all 함수에서 몇 번째로 측정되는지 (A2=0, A1=1, A3=2)
    input_labels: x축 라벨 리스트 (예: ['00', '01'...])
    """
    probs = []

    for label in input_labels:
        # 1. 빌더 생성
        builder = TransmonBuilder(
            p_1q=0.005,    # 0.5%
            p_2q=0.02,     # 2.0%
            p_meas=0.03,   # 3.0%
            p_1q_z=0.0     # 가상 Z 게이트는 에러 0으로 설정
        )        
        # 2. 상태 준비 (State Preparation)
        # label이 '10'이면 첫번째 타겟 큐비트에 X, 두번째는 그냥 둠.
        for i, char in enumerate(label):
            if char == '1':
                builder.circuit.append("X", [target_qubits[i]])
        
        # 3. 측정 수행 (First round 모드 -> Active Reset 없이 측정만 수행)
        builder.measure_all(is_first_round=True)
        
        # 4. 샘플링
        sampler = builder.get_circuit().compile_sampler()
        result = sampler.sample(shots=SHOTS)
        
        # 5. 해당 안실라가 1이 나온 횟수 계산
        # result의 컬럼 순서는 measure_all 내부의 측정 순서: [A2, A1, A3]
        ancilla_result = result[:, ancilla_idx_in_measure_order]
        prob_1 = np.sum(ancilla_result) / SHOTS
        probs.append(prob_1)
        
    return probs

# ---------------------------------------------------------
# 1. A1 실험 (D1, D3) - 2 Body Parity
# ---------------------------------------------------------
labels_2q = ["".join(x) for x in itertools.product("01", repeat=2)] # 00, 01, 10, 11
# measure_all 순서: A2, A1, A3 -> A1은 인덱스 1
probs_a1 = run_parity_experiment([D1, D3], 1, labels_2q)

# ---------------------------------------------------------
# 2. A2 실험 (D1, D2, D3, D4) - 4 Body Parity
# ---------------------------------------------------------
labels_4q = ["".join(x) for x in itertools.product("01", repeat=4)] # 0000 ~ 1111
# measure_all 순서: A2, A1, A3 -> A2는 인덱스 0
probs_a2 = run_parity_experiment([D1, D2, D3, D4], 0, labels_4q)

# ---------------------------------------------------------
# 3. A3 실험 (D2, D4) - 2 Body Parity
# ---------------------------------------------------------
# measure_all 순서: A2, A1, A3 -> A3는 인덱스 2
probs_a3 = run_parity_experiment([D2, D4], 2, labels_2q)

# =========================================================
# 그래프 그리기 (Figure 3 스타일)
# =========================================================
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
# %%
