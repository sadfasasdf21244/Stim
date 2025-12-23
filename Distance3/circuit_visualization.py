import stim
import os

def save_all_ticks_svg(circuit, output_dir="circuit_plots"):
    """
    Stim 회로의 모든 Tick을 SVG 파일로 저장합니다.
    """
    # 1. 저장할 폴더 생성 (없으면 생성)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 폴더 생성 완료: {output_dir}")
    else:
        print(f"📂 기존 폴더 사용: {output_dir}")

    # 2. 회로 내의 총 TICK 개수 계산
    # Tick이 N개 있으면, 시점(Slice)은 0 ~ N 까지 총 N+1개가 존재합니다.
    total_ticks = sum(1 for instruction in circuit if instruction.name == "TICK")
    
    print(f"총 {total_ticks}개의 TICK을 발견했습니다. (저장할 파일: 0 ~ {total_ticks})")

    # 3. 모든 Tick 순회하며 저장
    for t in range(total_ticks + 1):
        # 다이어그램 생성
        svg_helper = circuit.diagram(type="timeslice-svg", tick=t)
        
        # 파일명 지정 (예: tick_00.svg, tick_01.svg ...)
        # {:02d}는 숫자를 두 자리로 맞춰줍니다 (0 -> 00, 1 -> 01)
        filename = f"tick_{t:02d}.svg"
        file_path = os.path.join(output_dir, filename)
        
        # 파일 쓰기
        with open(file_path, "w") as f:
            f.write(str(svg_helper)) # 꼭 str()로 변환해야 함
            
        print(f"  💾 저장됨: {filename}")

    print("\n✨ 모든 이미지 저장이 완료되었습니다!")

# ==========================================================
# 실행 부분
# ==========================================================
if __name__ == "__main__":
    # 1. Distance-3 Surface Code 생성
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", 
        distance=3, 
        rounds=2
    )

    # 2. 저장 함수 실행
    # 원하는 폴더 경로를 입력하세요 (예: "G:/내 드라이브/QDL/Stim/figures")
    save_path = "Distance3/Figure/ticks" 
    
    save_all_ticks_svg(circuit, output_dir=save_path)