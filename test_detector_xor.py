import numpy as np
import stim

def verify_detector_measurement_relation():
    # 1. 테스트용 회로 생성 (사용자님의 measure_all 로직 모사)
    circuit = stim.Circuit()
    
    # 3개의 안실라, 5라운드 진행 가정
    rounds = 5
    ancillas = [0, 1, 2] # A1, A2, A3
    
    for r in range(rounds):
        # (1) 랜덤 노이즈 추가 (에러가 발생해야 검증 의미가 있음)
        circuit.append("X_ERROR", ancillas, 0.2) 
        
        # (2) 측정 (M)
        circuit.append("M", ancillas)
        
        # (3) 디텍터 (DETECTOR)
        # Round 0: 현재 값 그대로 (XOR 0)
        # Round N: 현재 값 XOR 이전 값
        for i, q in enumerate(ancillas):
            if r == 0:
                circuit.append("DETECTOR", [stim.target_rec(-3 + i)])
            else:
                circuit.append("DETECTOR", [stim.target_rec(-3 + i), stim.target_rec(-6 + i)])

    # ----------------------------------------------------------------
    # 2. 데이터 수집 (Seed 고정 필수!)
    # ----------------------------------------------------------------
    shots = 10
    seed = 12345 # 같은 에러 패턴을 만들기 위해 시드 고정
    
    # A. Raw Measurement (M 결과) 가져오기
    sampler = circuit.compile_sampler(seed=seed)
    raw_measurement = sampler.sample(shots=shots)
    
    # B. Detector Event (DETECTOR 결과) 가져오기
    det_sampler = circuit.compile_detector_sampler(seed=seed)
    detector_result = det_sampler.sample(shots=shots)
    
    # ----------------------------------------------------------------
    # 3. Reshape (Shots, Rounds, Ancillas)
    # ----------------------------------------------------------------
    # 안실라 3개
    m_reshaped = raw_measurement.reshape(shots, rounds, 3)
    d_reshaped = detector_result.reshape(shots, rounds, 3)
    
    # ----------------------------------------------------------------
    # 4. 검증: Detector를 누적 XOR 하면 Measurement가 되는가?
    # ----------------------------------------------------------------
    # axis=1 (Round 방향/시간 방향)으로 누적 XOR 수행
    # np.logical_xor.accumulate: True/False 누적 XOR
    reconstructed_measurement = np.logical_xor.accumulate(d_reshaped, axis=1)
    
    # 두 배열이 완전히 같은지 비교
    is_same = np.array_equal(m_reshaped, reconstructed_measurement)
    
    print(f"=== 검증 결과 ===")
    print(f"1. Raw Measurement Shape: {m_reshaped.shape}")
    print(f"2. Detector Result Shape: {d_reshaped.shape}")
    print(f"3. 일치 여부: {'✅ 성공! (완벽히 일치함)' if is_same else '❌ 실패 (다름)'}")
    
    if not is_same:
        print("\n[디버깅] 첫 번째 샷 비교:")
        print("Original Meas:\n", m_reshaped[0].astype(int))
        print("Reconstructed:\n", reconstructed_measurement[0].astype(int))

# 실행
verify_detector_measurement_relation()