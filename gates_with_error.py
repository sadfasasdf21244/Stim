import stim

# 큐비트 인덱스 정의 (사용하기 편하게 전역 변수로 설정)
D1, D2, D3, D4 = 0, 1, 2, 3
A1, A2, A3 = 4, 5, 6

# 그룹화 (나중에 반복문 돌리기 편하게)
DATA_QUBITS = [D1, D2, D3, D4]
ANCILLA_QUBITS = [A1, A2, A3]
ALL_QUBITS = DATA_QUBITS + ANCILLA_QUBITS

# ==========================================
# 1. 사용자 정의 Builder 클래스 (수정하신 버전)
# ==========================================

class TransmonBuilder:
    def __init__(self, p_1q, p_2q, p_meas, p_1q_z=0):
        self.circuit = stim.Circuit()
        self.p_1q = p_1q      
        self.p_2q = p_2q           
        self.p_meas = p_meas   
        self.p_1q_z = p_1q_z      

    def _add_noise_1q(self, target):
        if self.p_1q > 0:
            self.circuit.append("DEPOLARIZE1", [target], self.p_1q)

    def _add_noise_1q_z(self, target):
        if self.p_1q_z > 0:
            self.circuit.append("DEPOLARIZE1", [target], self.p_1q_z)

    def _add_noise_2q(self, t1, t2):
        if self.p_2q > 0:
            self.circuit.append("DEPOLARIZE2", [t1, t2], self.p_2q)

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
    
    def measure_all(self, is_first_round=False):
        if is_first_round:
            # 첫 라운드는 이전 기록이 없으므로 그냥 하드 리셋(R) 후 시작
            # (시뮬레이션 시작 시점엔 보통 0이므로 R을 생략하기도 하지만, 명시적으로 넣음)
            self.circuit.append("R", ANCILLA_QUBITS)
        else:
            # 2번째 라운드부터는 이전 측정 결과(rec)를 이용해 Active Reset 수행
            # 인덱스 로직: 바로 직전 라운드 끝에서 A2, A1, A3 순으로 측정했다고 가정
            # rec[-3]: A2, rec[-2]: A1, rec[-1]: A3
            self.circuit.append("CX", [stim.target_rec(-3), A2])
            self.circuit.append("CX", [stim.target_rec(-2), A1])
            self.circuit.append("CX", [stim.target_rec(-1), A3])
            # Active Reset 동작 중 발생하는 노이즈 (또는 대기 시간 노이즈)
            self._add_noise_1q(A2)
            self._add_noise_1q(A1)
            self._add_noise_1q(A3)

        self.pi_half_y(D1)
        self.pi_half_y(D2)
        self.pi_half_y(D3)
        self.pi_half_y(D4)

        self.pi_half_y(A2) 
        self.cz(D1, A2)
        self.cz(D2, A2)
        self.cz(D3, A2)
        self.cz(D4, A2)
        self.minus_pi_half_y(A2)
        self.measure_z(A2)

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