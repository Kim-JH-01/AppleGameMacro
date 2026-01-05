import cv2
import numpy as np
import pyautogui
import json
import time
import os
import copy
from ultralytics import YOLO
from abc import ABC, abstractmethod

# === [설정] ===
CONFIG_FILE = "grid_config.json"
MODEL_PATH = "best.pt"
ROWS = 10
COLS = 17

class Vision:
    def __init__(self):
        print("👁️ Vision 모듈 초기화...")
        with open(CONFIG_FILE, 'r') as f: self.cfg = json.load(f)
        self.model = YOLO(MODEL_PATH)

    def get_initial_grid(self):
        print("📸 화면 캡처 및 분석 중...")
        screenshot = pyautogui.screenshot()
        img_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None

        max_cnt = max(contours, key=cv2.contourArea)
        gx, gy, gw, gh = cv2.boundingRect(max_cnt)
        padding = 2
        board_img = img_bgr[gy+padding:gy+gh-padding, gx+padding:gx+gw-padding]
        
        # 디버그용 이미지 복사
        debug_img = board_img.copy()
        
        cur_h, cur_w = board_img.shape[:2]
        scale_x = cur_w / self.cfg['img_w']
        scale_y = cur_h / self.cfg['img_h']
        real_start_x = self.cfg['gx'] * scale_x
        real_start_y = self.cfg['gy'] * scale_y
        real_cell_w = (self.cfg['gw'] / COLS) * scale_x
        real_cell_h = (self.cfg['gh'] / ROWS) * scale_y

        results = self.model(board_img, conf=0.5, iou=0.5, verbose=False)
        grid = [[0]*COLS for _ in range(ROWS)]
        
        print(f"📊 감지된 객체 수: {len(results[0].boxes)}")
        
        if results[0].boxes:
            for box in results[0].boxes:
                bx, by, bw, bh = box.xywh[0].cpu().numpy()
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0]) + 1 
                
                col_idx = int((bx - real_start_x) / real_cell_w)
                row_idx = int((by - real_start_y) / real_cell_h)
                
                if 0 <= row_idx < ROWS and 0 <= col_idx < COLS:
                    grid[row_idx][col_idx] = {'num': cls, 'coords': None}
                    
                    # [디버그 수정] 잘 보이게 그리기
                    # 1. 빨간 박스 그리기
                    cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    
                    # 2. 글자 배경(흰색) 만들기
                    text = str(cls)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # 글자 위치 계산 (박스 중앙)
                    center_x = int(x1 + (x2 - x1) / 2)
                    center_y = int(y1 + (y2 - y1) / 2)
                    text_x = center_x - text_w // 2
                    text_y = center_y + text_h // 2
                    
                    # 흰색 배경 박스 채우기
                    cv2.rectangle(debug_img, (text_x - 2, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), (255, 255, 255), -1)
                    
                    # 3. 검은색 글자 쓰기 (잘 보임)
                    cv2.putText(debug_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        cv2.imwrite("debug_vision.jpg", debug_img)
        print("💾 인식 결과 저장 완료: debug_vision.jpg (흰색 배경 글자로 수정됨)")
        
        return grid, debug_img

class BaseBrain(ABC):
    @abstractmethod
    def solve(self, initial_grid):
        pass

    def get_all_candidates(self, current_map):
        candidates = []
        for r1 in range(ROWS):
            for c1 in range(COLS):
                if current_map[r1][c1] == 0: continue
                for r2 in range(r1, ROWS):
                    for c2 in range(c1, COLS):
                        if r1 == r2 and c1 == c2: continue
                        if current_map[r2][c2] == 0: continue
                        current_sum = 0
                        temp_cells = []
                        valid = True
                        for i in range(r1, r2+1):
                            for j in range(c1, c2+1):
                                val = current_map[i][j]
                                current_sum += val
                                if val > 0: temp_cells.append((i, j))
                            if current_sum > 10: valid = False; break
                        if valid and current_sum == 10:
                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            candidates.append({'area': area, 'size': len(temp_cells), 'cells': temp_cells})
        return candidates

    def apply_move(self, target_map, move):
        for r, c in move['cells']:
            target_map[r][c] = 0

# 전략 1: 단순 탐욕
class SimpleGreedyBrain(BaseBrain):
    def solve(self, initial_grid):
        num_map = [[(cell['num'] if cell != 0 else 0) for cell in row] for row in initial_grid]
        total_score = 0
        while True:
            candidates = self.get_all_candidates(num_map)
            if not candidates: break
            candidates.sort(key=lambda x: (x['area'], x['size']))
            moves_in_pass = 0
            for cand in candidates:
                available = True
                for r,c in cand['cells']:
                    if num_map[r][c] == 0: available = False; break
                if available:
                    self.apply_move(num_map, cand)
                    total_score += cand['size']
                    moves_in_pass += 1
            if moves_in_pass == 0: break
        return total_score

# 전략 2: Depth-1 (1수 앞 예측)
class Depth1Brain(BaseBrain):
    def solve(self, initial_grid):
        num_map = [[(cell['num'] if cell != 0 else 0) for cell in row] for row in initial_grid]
        total_score = 0
        while True:
            candidates = self.get_all_candidates(num_map)
            if not candidates: break
            
            best_move = None
            max_opp = -1
            
            for move in candidates:
                sim_map = [row[:] for row in num_map]
                self.apply_move(sim_map, move)
                opp_count = len(self.get_all_candidates(sim_map))
                
                if opp_count > max_opp:
                    max_opp = opp_count
                    best_move = move
                elif opp_count == max_opp:
                    if best_move is None or \
                       (move['size'] < best_move['size']) or \
                       (move['size'] == best_move['size'] and move['area'] < best_move['area']):
                        best_move = move
                        
            self.apply_move(num_map, best_move)
            total_score += best_move['size']
        return total_score

# 전략 3: Depth-2 (후보군 대폭 증가!)
class Depth2Brain(BaseBrain):
    def solve(self, initial_grid):
        num_map = [[(cell['num'] if cell != 0 else 0) for cell in row] for row in initial_grid]
        total_score = 0
        while True:
            candidates = self.get_all_candidates(num_map)
            if not candidates: break

            best_move = None
            max_score = -1

            # [핵심 수정] 8개 -> 30개로 증가!
            # 속도는 느려지지만(5~10초), 숨겨진 좋은 수를 놓치지 않게 됩니다.
            candidates.sort(key=lambda x: (x['size'], x['area']))
            top_candidates = candidates[:100] 

            for move in top_candidates:
                sim_map_1 = [row[:] for row in num_map]
                self.apply_move(sim_map_1, move)
                score = self.evaluate(sim_map_1)
                
                if score > max_score:
                    max_score = score
                    best_move = move
                elif score == max_score:
                    if best_move is None or \
                       (move['size'] < best_move['size']) or \
                       (move['size'] == best_move['size'] and move['area'] < best_move['area']):
                        best_move = move
            
            # 안전장치: 혹시라도 best_move가 없으면 그냥 1순위 실행
            if best_move is None:
                 best_move = candidates[0]

            self.apply_move(num_map, best_move)
            total_score += best_move['size']
        return total_score

    def evaluate(self, current_map):
        next_moves = self.get_all_candidates(current_map)
        return len(next_moves)

def run_test():
    try:
        vision = Vision()
        print("\n게임 화면을 준비해주세요. 3초 뒤 캡처합니다...")
        time.sleep(3)
        
        initial_grid, _ = vision.get_initial_grid()
        if not initial_grid:
            print("❌ 게임판을 찾을 수 없습니다.")
            return

        print("\n✅ 화면 캡처 완료! 알고리즘 경쟁 시작...\n")
        print("="*60)
        print(f"{'알고리즘 이름':<25} | {'예상 점수 (사과 개수)':<20} | {'소요 시간':<10}")
        print("="*60)

        brains_to_test = [
            ("Simple Greedy (Area)", SimpleGreedyBrain()),
            ("Depth-1 Lookahead", Depth1Brain()),
            ("Depth-2 Lookahead (Big)", Depth2Brain()), # 이름 변경
        ]

        results = []
        for name, brain in brains_to_test:
            grid_copy = copy.deepcopy(initial_grid)
            start_time = time.time()
            score = brain.solve(grid_copy)
            end_time = time.time()
            elapsed = end_time - start_time
            results.append((name, score, elapsed))
            print(f"{name:<25} | {score:<20} | {elapsed:.4f}s")

        print("="*60)
        results.sort(key=lambda x: x[1], reverse=True)
        winner = results[0]
        print(f"\n🏆 최종 승자: [{winner[0]}] - 예상 점수: {winner[1]}점")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    run_test()