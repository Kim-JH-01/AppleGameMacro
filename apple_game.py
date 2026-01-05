import cv2
import numpy as np
import pyautogui
import json
import time
import os
import copy
from ultralytics import YOLO
import keyboard
import math # 거리 계산을 위해 상단에 추가 필요

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




    def solve_simulation(self, initial_grid):
        print("🧠 [시뮬레이션] 전략: 중앙 집중형 클러스터링 (Center-Out)")
        
        virtual_board = copy.deepcopy(initial_grid)
        num_map = [[(cell['num'] if cell != 0 else 0) for cell in row] for row in virtual_board]
        total_moves = []
        
        # 맵의 정중앙 좌표 (행, 열)
        center_r = ROWS / 2
        center_c = COLS / 2
        
        while True:
            candidates = []
            
            # 1. 현재 상태에서 가능한 모든 수 찾기
            for r1 in range(ROWS):
                for c1 in range(COLS):
                    if num_map[r1][c1] == 0: continue
                    
                    for r2 in range(r1, ROWS):
                        for c2 in range(c1, COLS):
                            if r1 == r2 and c1 == c2: continue
                            if num_map[r2][c2] == 0: continue
                            
                            current_sum = 0
                            temp_cells = []
                            valid = True
                            
                            for i in range(r1, r2+1):
                                for j in range(c1, c2+1):
                                    val = num_map[i][j]
                                    current_sum += val
                                    if val > 0: temp_cells.append((i, j))
                                if current_sum > 10: 
                                    valid = False; break
                            
                            if valid and current_sum == 10:
                                # [전략 핵심] 이 드래그의 '중심점'이 맵의 '중앙'에서 얼마나 먼가?
                                # 드래그 박스의 중심 좌표 계산
                                drag_center_r = (r1 + r2) / 2
                                drag_center_c = (c1 + c2) / 2
                                
                                # 피타고라스 정리로 거리 계산 (중앙과의 거리)
                                dist_from_center = math.sqrt(
                                    (drag_center_r - center_r)**2 + 
                                    (drag_center_c - center_c)**2
                                )
                                
                                area = (r2 - r1 + 1) * (c2 - c1 + 1)
                                
                                candidates.append({
                                    'dist': dist_from_center, # 1순위: 중앙과 가까운가?
                                    'area': area,             # 2순위: 면적이 작은가?
                                    'start': initial_grid[r1][c1]['coords'],
                                    'end': initial_grid[r2][c2]['coords'],
                                    'cells': temp_cells
                                })
            
            if not candidates:
                break
            
            # 2. 정렬 (Clustering Logic)
            # 1순위: 중앙에서의 거리 (가까울수록 먼저) -> 가운데부터 파먹음
            # 2순위: 면적 (작을수록 먼저) -> 알뜰하게 먹음
            candidates.sort(key=lambda x: (x['dist'], x['area']))
            
            # 3. 가장 좋은 것 '하나만' 실행하고 다시 스캔 (Greedy Step)
            # 한 번에 여러 개를 예약하지 않고, 하나 깰 때마다 지형이 바뀌는 걸 즉시 반영하여
            # 구멍을 점점 넓혀가는 방식입니다.
            best_move = candidates[0]
            
            # 사과 삭제 (가상 맵 업데이트)
            for r, c in best_move['cells']:
                num_map[r][c] = 0
            
            total_moves.append(best_move)
            
        print(f"📋 예측 완료: 클러스터링 경로 {len(total_moves)}회 생성!")
        return total_moves


    def solve_simulation(self, initial_grid):
        print("🧠 [시뮬레이션] 전략: 밀도 기반 시드 확장 (Density-Based Expansion)")
        
        virtual_board = copy.deepcopy(initial_grid)
        # 계산을 빠르게 하기 위해 숫자만 추출
        num_map = [[(cell['num'] if cell != 0 else 0) for cell in row] for row in virtual_board]
        total_moves = []

        # ---------------------------------------------------------
        # 1. [Seed 탐색] 어디가 가장 '핫플레이스'인지 찾기
        # ---------------------------------------------------------
        density_map = [[0] * COLS for _ in range(ROWS)]
        
        # 전체를 훑으며 "바로 인접한(상하좌우) 짝꿍"이 있는지 카운트
        for r in range(ROWS):
            for c in range(COLS):
                if num_map[r][c] == 0: continue
                val = num_map[r][c]
                
                # 상하좌우 확인
                neighbors = [(-1,0), (1,0), (0,-1), (0,1)]
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        if num_map[nr][nc] != 0 and (val + num_map[nr][nc] == 10):
                            density_map[r][c] += 1 # 짝꿍이 있으면 점수 추가

        # 가장 점수가 높은 좌표 찾기 (Seed)
        max_score = -1
        seed_r, seed_c = ROWS // 2, COLS // 2 # 기본값은 중앙
        
        for r in range(ROWS):
            for c in range(COLS):
                if density_map[r][c] > max_score:
                    max_score = density_map[r][c]
                    seed_r, seed_c = r, c
        
        print(f"📍 전략적 요충지(Seed) 발견: ({seed_r}, {seed_c}) / 밀도 점수: {max_score}")

        # ---------------------------------------------------------
        # 2. [시뮬레이션] Seed 중심으로 퍼져나가기
        # ---------------------------------------------------------
        while True:
            candidates = []
            
            # 가능한 모든 드래그 찾기
            for r1 in range(ROWS):
                for c1 in range(COLS):
                    if num_map[r1][c1] == 0: continue
                    
                    for r2 in range(r1, ROWS):
                        for c2 in range(c1, COLS):
                            if r1 == r2 and c1 == c2: continue
                            if num_map[r2][c2] == 0: continue
                            
                            current_sum = 0
                            temp_cells = []
                            valid = True
                            
                            for i in range(r1, r2+1):
                                for j in range(c1, c2+1):
                                    val = num_map[i][j]
                                    current_sum += val
                                    if val > 0: temp_cells.append((i, j))
                                if current_sum > 10: 
                                    valid = False; break
                            
                            if valid and current_sum == 10:
                                # [전략 핵심] Seed와의 거리 계산
                                drag_center_r = (r1 + r2) / 2
                                drag_center_c = (c1 + c2) / 2
                                
                                dist_from_seed = math.sqrt(
                                    (drag_center_r - seed_r)**2 + 
                                    (drag_center_c - seed_c)**2
                                )
                                
                                area = (r2 - r1 + 1) * (c2 - c1 + 1)
                                
                                candidates.append({
                                    'dist': dist_from_seed, # 1순위: 꿀단지(Seed) 옆인가?
                                    'area': area,           # 2순위: 작게 먹는가?
                                    'start': initial_grid[r1][c1]['coords'],
                                    'end': initial_grid[r2][c2]['coords'],
                                    'cells': temp_cells
                                })
            
            if not candidates:
                break
            
            # 정렬: Seed에서 가깝고(dist), 면적이 작은(area) 순서
            candidates.sort(key=lambda x: (x['dist'], x['area']))
            
            # 가장 좋은 수 하나 실행
            best_move = candidates[0]
            
            # 사과 삭제 (가상 맵 업데이트)
            for r, c in best_move['cells']:
                num_map[r][c] = 0
            
            total_moves.append(best_move)
            
        print(f"📋 예측 완료: 밀도 기반 경로 {len(total_moves)}회 생성!")
        return total_moves


    def solve_simulation(self, initial_grid):
        print("🧠 [시뮬레이션] 전략: '2개짜리 짝' 우선 + 밀도 확장")
        
        virtual_board = copy.deepcopy(initial_grid)
        num_map = [[(cell['num'] if cell != 0 else 0) for cell in row] for row in virtual_board]
        total_moves = []

        # ---------------------------------------------------------
        # 1. [Seed 탐색] 2개짜리 짝(Pair)이 가장 많은 곳 찾기
        # ---------------------------------------------------------
        density_map = [[0] * COLS for _ in range(ROWS)]
        
        for r in range(ROWS):
            for c in range(COLS):
                if num_map[r][c] == 0: continue
                val = num_map[r][c]
                
                # 상하좌우만 검사 (대각선 제외, 가장 확실한 짝꿍)
                neighbors = [(-1,0), (1,0), (0,-1), (0,1)]
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        # 0이 아니고, 둘이 합쳐서 딱 10이 되는 경우 (Pair)
                        if num_map[nr][nc] != 0 and (val + num_map[nr][nc] == 10):
                            density_map[r][c] += 1

        # 가장 짝꿍이 많은 좌표(Seed) 선정
        max_score = -1
        seed_r, seed_c = ROWS // 2, COLS // 2
        
        for r in range(ROWS):
            for c in range(COLS):
                if density_map[r][c] > max_score:
                    max_score = density_map[r][c]
                    seed_r, seed_c = r, c
        
        print(f"📍 꿀단지(Seed) 발견: ({seed_r}, {seed_c}) 주변에 짝꿍 다수 포착!")

        # ---------------------------------------------------------
        # 2. [시뮬레이션] 우선순위 기반 탐색
        # ---------------------------------------------------------
        while True:
            candidates = []
            
            # 가능한 모든 드래그 찾기
            for r1 in range(ROWS):
                for c1 in range(COLS):
                    if num_map[r1][c1] == 0: continue
                    
                    for r2 in range(r1, ROWS):
                        for c2 in range(c1, COLS):
                            if r1 == r2 and c1 == c2: continue
                            if num_map[r2][c2] == 0: continue
                            
                            current_sum = 0
                            temp_cells = []
                            valid = True
                            
                            for i in range(r1, r2+1):
                                for j in range(c1, c2+1):
                                    val = num_map[i][j]
                                    current_sum += val
                                    if val > 0: temp_cells.append((i, j))
                                if current_sum > 10: 
                                    valid = False; break
                            
                            if valid and current_sum == 10:
                                # 거리 계산
                                drag_center_r = (r1 + r2) / 2
                                drag_center_c = (c1 + c2) / 2
                                dist_from_seed = math.sqrt((drag_center_r - seed_r)**2 + (drag_center_c - seed_c)**2)
                                
                                area = (r2 - r1 + 1) * (c2 - c1 + 1)
                                
                                # 사과 개수 (콤보 사이즈)
                                combo_size = len(temp_cells)
                                
                                candidates.append({
                                    'size': combo_size,     # [1순위] 사과 개수 (2개 > 3개)
                                    'dist': dist_from_seed, # [2순위] Seed 거리 (가까운 순)
                                    'area': area,           # [3순위] 면적 (작은 순)
                                    'start': initial_grid[r1][c1]['coords'],
                                    'end': initial_grid[r2][c2]['coords'],
                                    'cells': temp_cells
                                })
            
            if not candidates:
                break
            
            # [핵심] 정렬 기준 적용
            # 1. size (오름차순): 2개짜리를 다 없애야 3개짜리가 나옴
            # 2. dist (오름차순): 가운데부터 밖으로 퍼짐
            candidates.sort(key=lambda x: (x['size'], x['dist'], x['area']))
            
            best_move = candidates[0]
            
            # 가상 맵 업데이트
            for r, c in best_move['cells']:
                num_map[r][c] = 0
            
            total_moves.append(best_move)
            
        print(f"📋 예측 완료: 총 {len(total_moves)}회 (2개짜리 우선 처리)")
        return total_moves


    def solve_simulation(self, initial_grid):
        print("🧠 [시뮬레이션] 전략: '닥터 스트레인지' (미래 예측 롤아웃)")
        
        # 실제 게임 진행 상황을 담을 보드
        virtual_board = copy.deepcopy(initial_grid)
        num_map = [[(cell['num'] if cell != 0 else 0) for cell in row] for row in virtual_board]
        
        total_moves = []
        
        while True:
            # 1. 현재 상태에서 가능한 모든 후보수(Moves) 찾기
            candidates = self.get_all_candidates(num_map, initial_grid)
            
            if not candidates:
                break
            
            # 2. [미래 예측] 각 후보를 뒀을 때, 최종 점수가 몇 점이 될지 시뮬레이션
            best_move = None
            max_future_score = -1
            
            # 모든 후보에 대해 "가상으로 끝까지 플레이" 해봄
            for move in candidates:
                # 맵 복사 (미래를 보기 위한 가상 공간)
                sim_map = [row[:] for row in num_map] 
                
                # 일단 이 수를 둬본다
                self.apply_move(sim_map, move)
                
                # 남은 게임을 '기본 전략(작은것 우선)'으로 끝까지 돌려본다
                future_score = 1 + self.play_rest_of_game(sim_map)
                
                # 이 미래가 점수가 더 높다면 선택
                if future_score > max_future_score:
                    max_future_score = future_score
                    best_move = move
                
                # (최적화) 만약 미래 점수가 압도적으로 높으면 조기 종료 가능하지만, 
                # 정확도를 위해 다 비교합니다.
            
            # 3. 가장 엔딩이 좋았던 수를 실제로 둔다
            self.apply_move(num_map, best_move)
            total_moves.append(best_move)
            
            # 진행 상황 출력 (너무 빠르면 생략 가능)
            # print(f"📍 결정: 예상 최종 점수 {max_future_score}점 루트 선택")

        print(f"📋 예측 완료: 최적 경로 {len(total_moves)}회 생성!")
        return total_moves

    def get_all_candidates(self, current_map, grid_ref):
        """현재 맵에서 가능한 모든 드래그 찾기"""
        candidates = []
        ROWS = len(current_map)
        COLS = len(current_map[0])
        
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
                            if current_sum > 10: 
                                valid = False; break
                        
                        if valid and current_sum == 10:
                            # grid_ref가 None이면(시뮬레이션 중) 좌표 정보 없이 로직만 계산
                            coords = {}
                            if grid_ref:
                                coords = {
                                    'start': grid_ref[r1][c1]['coords'],
                                    'end': grid_ref[r2][c2]['coords']
                                }
                                
                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            candidates.append({
                                'area': area,
                                'size': len(temp_cells),
                                'cells': temp_cells,
                                **coords
                            })
        return candidates

    def play_rest_of_game(self, sim_map):
        """남은 게임을 '가장 효율적인 방식(짝 우선)'으로 빠르게 끝까지 돌려보고 깬 횟수 반환"""
        score = 0
        while True:
            moves = self.get_all_candidates(sim_map, None)
            if not moves: break
            
            # 시뮬레이션 내부 정책: "2개짜리 > 면적 작은거" 우선으로 막 깬다
            # (이게 평균적으로 점수가 잘 나오는 방식이므로 벤치마크로 사용)
            moves.sort(key=lambda x: (x['size'], x['area']))
            
            best = moves[0]
            self.apply_move(sim_map, best)
            score += 1
            
        return score

    def apply_move(self, target_map, move):
        """맵에서 사과 지우기"""
        for r, c in move['cells']:
            target_map[r][c] = 0


    def solve_simulation(self, initial_grid):
        print("🧠 [시뮬레이션] 전략: '1수 앞 예측' (Next-Move Maximization)")
        
        # 실제 게임 진행 상황을 담을 보드
        virtual_board = copy.deepcopy(initial_grid)
        num_map = [[(cell['num'] if cell != 0 else 0) for cell in row] for row in virtual_board]
        
        total_moves = []
        
        while True:
            # 1. 현재 상태에서 가능한 모든 후보수 찾기
            candidates = self.get_all_candidates(num_map, initial_grid)
            
            if not candidates:
                break
            
            # 2. 각 후보를 선택했을 때, '다음에 할 수 있는 것'이 몇 개나 남는지 계산
            best_move = None
            max_next_opportunities = -1
            
            # 만약 후보가 하나뿐이면 계산할 필요 없이 바로 실행
            if len(candidates) == 1:
                best_move = candidates[0]
            else:
                for move in candidates:
                    # 가상으로 이 수를 둬본다 (1-Step Simulation)
                    # 맵 전체 복사 대신 필요한 부분만 잠깐 0으로 만들었다가 복구하는 게 더 빠르지만,
                    # 맵이 작아서 deepcopy도 충분히 빠름
                    sim_map = [row[:] for row in num_map]
                    
                    # 사과 삭제 적용
                    for r, c in move['cells']:
                        sim_map[r][c] = 0
                    
                    # 이 상태에서 다시 한 번 깰 수 있는 게 몇 개인지 센다
                    next_moves = self.get_all_candidates(sim_map, None)
                    opportunity_count = len(next_moves)
                    
                    # 더 많은 기회를 남기는 수를 선택
                    # 기회 수가 같다면? -> 2개짜리(size) 우선, 면적(area) 작은거 우선
                    if opportunity_count > max_next_opportunities:
                        max_next_opportunities = opportunity_count
                        best_move = move
                    elif opportunity_count == max_next_opportunities:
                        # 동점일 경우: 짝(2개) 우선 > 면적 작은거 우선
                        # 현재 best_move와 비교
                        if (move['size'] < best_move['size']) or \
                           (move['size'] == best_move['size'] and move['area'] < best_move['area']):
                            best_move = move
            
            # 3. 결정된 최고의 수를 실제로 실행
            for r, c in best_move['cells']:
                num_map[r][c] = 0
            
            total_moves.append(best_move)

        print(f"📋 예측 완료: 스마트 경로 {len(total_moves)}회 생성!")
        return total_moves

    def get_all_candidates(self, current_map, grid_ref):
        """현재 맵에서 가능한 모든 드래그 찾기"""
        candidates = []
        ROWS = len(current_map)
        COLS = len(current_map[0])
        
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
                            if current_sum > 10: 
                                valid = False; break
                        
                        if valid and current_sum == 10:
                            coords = {}
                            if grid_ref:
                                coords = {
                                    'start': grid_ref[r1][c1]['coords'],
                                    'end': grid_ref[r2][c2]['coords']
                                }
                            
                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            candidates.append({
                                'area': area,
                                'size': len(temp_cells), # 사과 개수
                                'cells': temp_cells,
                                **coords
                            })
        return candidates

class AppleBrain:
    def solve_simulation(self, initial_grid):
        print("🧠 [시뮬레이션] 전략: '2수 앞 예측' (Depth-2 Lookahead)")
        
        virtual_board = copy.deepcopy(initial_grid)
        num_map = [[(cell['num'] if cell != 0 else 0) for cell in row] for row in virtual_board]
        total_moves = []
        
        while True:
            # 현재 상태에서 가능한 모든 후보 찾기
            candidates = self.get_all_candidates(num_map, initial_grid)
            
            if not candidates:
                break
            
            # 후보가 하나뿐이면 고민 없이 실행
            if len(candidates) == 1:
                best_move = candidates[0]
            else:
                # [핵심] 모든 후보에 대해 '2수 앞'까지 시뮬레이션 점수 계산
                best_move = None
                max_score = -1
                
                for move in candidates:
                    # 1단계 시뮬레이션
                    sim_map_1 = [row[:] for row in num_map]
                    self.apply_move_to_map(sim_map_1, move)
                    
                    # 2수 앞의 잠재력(Score) 계산
                    score = self.evaluate_future(sim_map_1, depth=1)
                    
                    # 점수가 더 높거나, 같으면 더 효율적인(짝, 작은면적) 것 선택
                    if score > max_score:
                        max_score = score
                        best_move = move
                    elif score == max_score:
                        # 동점일 경우: 사과 개수 적은 것(2개) > 면적 작은 것 우선
                        if (move['size'] < best_move['size']) or \
                           (move['size'] == best_move['size'] and move['area'] < best_move['area']):
                            best_move = move
            
            # 결정된 최고의 수 실행
            self.apply_move_to_map(num_map, best_move)
            total_moves.append(best_move)

        print(f"📋 예측 완료: 신중한 경로 {len(total_moves)}회 생성!")
        return total_moves

    def evaluate_future(self, current_map, depth):
        """
        재귀적으로 미래를 예측하여 점수를 반환함
        depth: 현재 얼마나 깊이 들어왔는지 (0부터 시작해서 MAX_DEPTH까지)
        """
        # 다음 단계의 후보들 탐색
        next_moves = self.get_all_candidates(current_map, None)
        
        # 더 이상 깰 게 없으면, 현재까지 확보된 '기회 비용'은 0
        if not next_moves:
            return 0
            
        # 목표 깊이에 도달했으면, 현재 남은 '기회의 개수'를 반환 (Depth-1 전략과 동일)
        # 여기서 depth=1이라는 건, 이미 1수(Main Loop) + 1수(Here) = 총 2수를 봤다는 뜻
        if depth >= 1: 
            return len(next_moves)
        
        # 아직 더 깊이 볼 수 있다면, 가장 좋은 다음 수를 찾아봄 (Max Search)
        max_sub_score = 0
        
        # 가지치기(Pruning): 너무 많으면 느려지니까, 상위 5개 정도만 추려서 검사 (속도 최적화)
        # 정렬 기준: 2개짜리 > 면적 작은거
        next_moves.sort(key=lambda x: (x['size'], x['area']))
        top_k_moves = next_moves[:5] 
        
        for move in top_k_moves:
            sim_map_next = [row[:] for row in current_map]
            self.apply_move_to_map(sim_map_next, move)
            
            # 재귀 호출 (depth + 1)
            # 점수 = 1(이번 턴 성공) + 미래 점수
            sub_score = 1 + self.evaluate_future(sim_map_next, depth + 1)
            
            if sub_score > max_sub_score:
                max_sub_score = sub_score
                
        return max_sub_score

    def get_all_candidates(self, current_map, grid_ref):
        candidates = []
        ROWS = len(current_map)
        COLS = len(current_map[0])
        
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
                            if current_sum > 10: 
                                valid = False; break
                        
                        if valid and current_sum == 10:
                            coords = {}
                            if grid_ref:
                                coords = {
                                    'start': grid_ref[r1][c1]['coords'],
                                    'end': grid_ref[r2][c2]['coords']
                                }
                            
                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            candidates.append({
                                'area': area,
                                'size': len(temp_cells),
                                'cells': temp_cells,
                                **coords
                            })
        return candidates

    def apply_move_to_map(self, target_map, move):
        for r, c in move['cells']:
            target_map[r][c] = 0

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