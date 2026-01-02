import cv2
import numpy as np
import pyautogui
import json
import time
import os
import copy
from ultralytics import YOLO
import keyboard

# === [설정] ===
pyautogui.FAILSAFE = True
CONFIG_FILE = "grid_config.json"
MODEL_PATH = "best.pt"
ROWS = 10
COLS = 17

class OneShotVision:
    def __init__(self):
        print("👁️ [1단계] One-Shot 비전 (Config 비율 스케일링) 가동")
        
        if not os.path.exists(CONFIG_FILE):
            raise Exception("❌ grid_config.json이 없습니다!")
        
        # 설정 파일 로드 (여백/비율 기준점)
        with open(CONFIG_FILE, 'r') as f:
            self.cfg = json.load(f)
            
        if not os.path.exists(MODEL_PATH):
            raise Exception(f"❌ 모델({MODEL_PATH})이 없습니다!")
        
        print("🧠 모델 로딩 중...")
        self.model = YOLO(MODEL_PATH)

    def get_matrix(self):
        # 1. 화면 캡처
        screenshot = pyautogui.screenshot()
        img_np = np.array(screenshot)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 2. 게임판 찾기
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        max_cnt = max(contours, key=cv2.contourArea)
        gx, gy, gw, gh = cv2.boundingRect(max_cnt)
        
        # 이미지 추출
        padding = 2
        board_img = img_bgr[gy+padding : gy+gh-padding, gx+padding : gx+gw-padding]
        
        # [중요] 리사이징 안 함 (현재 해상도 그대로 사용)
        
        # 3. 비율(Scale) 계산
        # 현재 화면의 게임판 크기
        cur_h, cur_w = board_img.shape[:2]
        
        # Config 파일 만들 때의 원본 게임판 크기
        ref_w = self.cfg['img_w']
        ref_h = self.cfg['img_h']
        
        # "화면이 얼마나 커졌나/작아졌나?" 비율 계산
        scale_x = cur_w / ref_w
        scale_y = cur_h / ref_h
        
        # 4. Config의 격자 정보를 현재 비율에 맞게 변환
        # (단순 등분이 아니라, Config에 설정된 여백 비율을 그대로 적용)
        
        # 현재 화면 기준 격자 시작점 (Scaled gx, gy)
        real_start_x = self.cfg['gx'] * scale_x
        real_start_y = self.cfg['gy'] * scale_y
        
        # 현재 화면 기준 격자 한 칸 크기 (Scaled cell size)
        # (config의 gw는 전체 격자 폭이므로, 그걸 17로 나누고 스케일 적용)
        real_cell_w = (self.cfg['gw'] / COLS) * scale_x
        real_cell_h = (self.cfg['gh'] / ROWS) * scale_y

        # 5. 모델 추론
        results = self.model(board_img, conf=0.5, iou=0.5, verbose=False)
        
        grid = [[0]*COLS for _ in range(ROWS)]
        
        if results[0].boxes:
            for box in results[0].boxes:
                # bx, by: 현재 이미지 내의 좌표
                bx, by, bw, bh = box.xywh[0].cpu().numpy()
                cls = int(box.cls[0]) + 1 
                
                # [핵심] 변환된(Scaled) 격자 기준으로 인덱스 찾기
                # (bx - 시작점) / 칸크기
                col_idx = int((bx - real_start_x) / real_cell_w)
                row_idx = int((by - real_start_y) / real_cell_h)
                
                if 0 <= row_idx < ROWS and 0 <= col_idx < COLS:
                    # 드래그 좌표 계산 (화면 절대 좌표)
                    screen_cx = gx + padding + bx
                    screen_cy = gy + padding + by
                    
                    # 드래그 박스 크기 (박스의 90%)
                    half_w = (bw * 0.9) / 2
                    half_h = (bh * 0.9) / 2
                    
                    grid[row_idx][col_idx] = {
                        'num': cls,
                        'coords': {
                            'x1': screen_cx - half_w,
                            'y1': screen_cy - half_h,
                            'x2': screen_cx + half_w,
                            'y2': screen_cy + half_h
                        }
                    }
        return grid

# === 이하 Brain, Hand, Main은 기존과 동일 ===
class AppleBrain:
    def solve_simulation(self, initial_grid):
        print("🧠 [시뮬레이션] 전체 경로 미리 계산 중...")
        # 숫자 행렬 생성
        num_map = np.array([[(cell['num'] if cell != 0 else 0) for cell in row] for row in initial_grid])
        total_moves = []
        
        H, W = num_map.shape
        
        r = 0
        while r < H:
            c = 0
            while c < W:
                found = False

                # (r,c)를 좌상단으로 하는 모든 직사각형 탐색
                for rr in range(r, H):
                    for cc in range(c, W):
                        rect = num_map[r:rr+1, c:cc+1]
                        rect_sum = int(rect.sum())

                        if rect_sum == 10:
                            # 드래그 좌표 계산 (좌상단 -> 우하단)
                            # 좌상단 셀과 우하단 셀의 좌표 찾기
                            start_cell = initial_grid[r][c]
                            end_cell = initial_grid[rr][cc]
                            
                            if start_cell != 0 and end_cell != 0:
                                total_moves.append({
                                    'start': start_cell['coords'],
                                    'end': end_cell['coords']
                                })
                            elif start_cell != 0:
                                # 우하단이 0이면 실제 숫자가 있는 마지막 셀 찾기
                                for er in range(rr, r-1, -1):
                                    for ec in range(cc, c-1, -1):
                                        if initial_grid[er][ec] != 0:
                                            total_moves.append({
                                                'start': start_cell['coords'],
                                                'end': initial_grid[er][ec]['coords']
                                            })
                                            break
                                    else:
                                        continue
                                    break
                            
                            # 영역 지우기
                            num_map[r:rr+1, c:cc+1] = 0
                            
                            found = True
                            # 보드가 바뀌었으니 처음부터 다시 탐색
                            r = 0
                            c = -1
                            break

                        # 합이 10을 넘으면 더 넓은 영역 볼 필요 없음
                        if rect_sum > 10:
                            break

                    if found:
                        break

                c += 1
            r += 1
                
        print(f"📋 예측 완료: 총 {len(total_moves)}회의 드래그 순서 생성!")
        return total_moves

class AppleHand:
    def execute(self, moves):
        if not moves: return
        print(f"✋ [실행] {len(moves)}회 연속 드래그 시작")
        for move in moves:
            s = move['start']
            e = move['end']
            pyautogui.moveTo(s['x1'], s['y1'])
            pyautogui.mouseDown()
            pyautogui.moveTo(e['x2'], e['y2'], duration=0.45, tween=pyautogui.easeOutQuad)
            time.sleep(0.05)
            pyautogui.mouseUp()
            time.sleep(0.1)

def main():
    try:
        vision = OneShotVision()
        brain = AppleBrain()
        hand = AppleHand()
        
        print("\n=== 🍎 원샷 시뮬레이션 매크로 ===")
        print("💡 언제든 'ESC' 키를 꾹 누르면 즉시 종료됩니다.")
        print("3초 뒤 시작...")
        time.sleep(3)
        
        # 메인 루프 (ESC 체크를 위해 구조 살짝 변경)
        while True:
            # 1. 긴급 종료 체크
            if keyboard.is_pressed('esc'):
                print("\n🛑 사용자 요청으로 강제 종료합니다.")
                break

            grid = vision.get_matrix()
            if not grid:
                print("❌ 게임판 못 찾음 (종료)")
                break
                
            all_moves = brain.solve_simulation(grid)
            
            if all_moves:
                # hand.execute 안에서도 ESC 체크를 해야 반응이 빠름
                # AppleHand 클래스 수정 없이 여기서 처리하려면 루프를 쪼개야 함
                print(f"✋ [실행] {len(all_moves)}회 드래그 시작")
                for i, move in enumerate(all_moves):
                    if keyboard.is_pressed('esc'):
                        print("\n🛑 드래그 중 강제 종료!")
                        return # 함수 완전히 탈출

                    # --- 드래그 실행 로직 복사 ---
                    s = move['start']
                    e = move['end']
                    pyautogui.moveTo(s['x1'], s['y1'])
                    pyautogui.mouseDown()
                    pyautogui.moveTo(e['x2'], e['y2'], duration=0.45, tween=pyautogui.easeOutQuad)
                    time.sleep(0.05)
                    pyautogui.mouseUp()
                    time.sleep(0.1)
                    # -------------------------
                
                print("🏁 한 사이클 완료. 다시 스캔합니다...")
            else:
                print("🏁 깰 수 있는 사과가 없습니다.")
                break
                
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    main()