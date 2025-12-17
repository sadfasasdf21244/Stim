import stim
import numpy as np
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

# ==========================================
# 1. 사용자 정의 Builder 클래스 (수정하신 버전)
# ==========================================

class TransmonBuilder:
    def __init__(self, p_1q, p_2q, p_meas, p_1q_z=0, squence_time = 2, T1 = 10, T2 = 10):
        self.circuit = stim.Circuit()
        self.p_1q = p_1q      
        self.p_2q = p_2q           
        self.p_meas = p_meas   
        self.p_1q_z = p_1q_z
        self.squence_time = squence_time
        self.T1 = T1
        self.T2 = T2
        self.T1_error_rate = 1 - np.exp(- self.squence_time / self.T1)
        self.T2_error_rate = 1 - np.exp( self.squence_time / self.T2)


    def _add_noise_1q(self, target):
        if self.p_1q > 0:
            self.circuit.append("DEPOLARIZE1", [target], self.p_1q)

    def _add_noise_1q_z(self, target):
        if self.p_1q_z > 0:
            self.circuit.append("DEPOLARIZE1", [target], self.p_1q_z)

    def _add_noise_2q(self, t1, t2):
        if self.p_2q > 0:
            self.circuit.append("DEPOLARIZE2", [t1, t2], self.p_2q)
    
    # def _add_T1_noise(self, target):
    #     if self.T1_error_rate > 0:
    #         self.circuit.append("AMPLITUDE_DAMPING", [target], self.T1_error_rate)

    def _add_T2_noise(self, target):    
        if self.T2_error_rate > 0:
            self.circuit.append("DEPOLARIZE1", [target], self.T2_error_rate)

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
    
    def measure_all(self, is_first_round = False, A2_basis = 'X'):
        if A2_basis == 'X':
            self.pi_half_y(D1)
            self.pi_half_y(D2)
            self.pi_half_y(D3)
            self.pi_half_y(D4)
        
        self.minus_pi_half_y(A2) 
        self.cz(D1, A2)
        self.cz(D2, A2)
        self.cz(D3, A2)
        self.cz(D4, A2)
        self.pi_half_y(A2)
        self.measure_z(A2)

        if A2_basis == 'X':
            self.minus_pi_half_y(D1)
            self.minus_pi_half_y(D2)
            self.minus_pi_half_y(D3)
            self.minus_pi_half_y(D4)

        self.minus_pi_half_y(A1)
        self.minus_pi_half_y(A3)
        self.cz(D3, A1)
        self.cz(D4, A3)
        self.cz(D1, A1)
        self.cz(D2, A3)
        self.pi_half_y(A1)
        self.pi_half_y(A3)
        self.measure_z(A1)
        self.measure_z(A3)

        if is_first_round:
            self.circuit.append("DETECTOR", [stim.target_rec(-3)])
            self.circuit.append("DETECTOR", [stim.target_rec(-2)])
            self.circuit.append("DETECTOR", [stim.target_rec(-1)])
        else:
            self.circuit.append("DETECTOR", [stim.target_rec(-3), stim.target_rec(-6)])
            self.circuit.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-5)])
            self.circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-4)])

        for qubit in ALL_QUBITS:
            # self._add_T1_noise([qubit])
            self._add_T2_noise([qubit])