import numpy as np

def load_complex_matrix_from_text(file_path):
    """
    NumPy print 형식(대괄호, 불규칙한 공백 포함)의 텍스트 파일을 읽어
    16x16 복소수 행렬로 변환합니다.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. 불필요한 대괄호 제거
    text = text.replace('[', '').replace(']', '')

    # 2. 모든 공백과 줄바꿈 제거 (이 과정이 중요합니다!)
    #    '0.    +0.001j' 처럼 숫자 중간에 있는 공백을 없애서
    #    '0.+0.001j' 처럼 유효한 파이썬 복소수 문자열로 만듭니다.
    text = text.replace(' ', '').replace('\n', '')

    # 3. 구분자 삽입 ('j'를 기준으로 숫자를 나눕니다)
    #    모든 숫자가 복소수(j 포함) 형식이므로 'j' 뒤에 공백을 넣어 쪼갭니다.
    text = text.replace('j', 'j ')

    # 4. 문자열을 복소수로 변환
    #    split()으로 나눈 뒤 빈 문자열은 제외하고 변환
    complex_numbers = [complex(x) for x in text.split(' ') if x]

    # 5. NumPy 배열로 변환 및 Reshape (16x16)
    #    데이터 개수가 256개인지 확인
    if len(complex_numbers) != 256:
        print(f"경고: 데이터 개수가 {len(complex_numbers)}개 입니다. (256개 필요)")
        # 개수에 맞춰서 자동으로 차원 조절 (혹시 모르니)
        dim = int(np.sqrt(len(complex_numbers)))
        return np.array(complex_numbers).reshape(dim, dim)
    
    return np.array(complex_numbers).reshape(16, 16)

# ==============================================================================
# 사용 예시
# ==============================================================================

# 1. 텍스트 파일 이름 지정 (사용자님의 파일 경로)
state = '+'

file_name = 'Distance2/testing/'+state+'state_density_matrix.txt'

# (테스트를 위해 파일이 없으면 생성하는 코드입니다. 실제로는 이 부분 건너뛰세요)
import os

# 2. 불러오기
rho_loaded = load_complex_matrix_from_text(file_name)

logical = {}
logical[0] = np.zeros(16)
logical[0][0b0000] = 1/np.sqrt(2)
logical[0][0b1111] = 1/np.sqrt(2)

logical[1] = np.zeros(16)
logical[1][0b0101] = 1/np.sqrt(2)
logical[1][0b1010] = 1/np.sqrt(2)
match state:
    case '0':
        psi = logical[0]
    case '1':
        psi = logical[1]
    case '+':
        psi = 1/np.sqrt(2)*(logical[0]+logical[1])
    case '-':
        psi = 1/np.sqrt(2)*(logical[0]-logical[1])

Logical_probability = logical[0].T @ rho_loaded @ logical[0] + logical[1].T @ rho_loaded @ logical[1]
rho_logical = np.zeros([2,2])

for i in range(2):
    for j in range(2):
        rho_logical[i][j] = logical[i].T @ rho_loaded @ logical[j] / Logical_probability

Phisical_Fidelity = psi.T @ rho_loaded @ psi
print(f"Phisical Fidelity Fphys : {Phisical_Fidelity:.3f}")
print(f"physical probability PL : {Logical_probability:.3f}")
print(rho_logical)
