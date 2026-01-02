# AppleGameMacro
YOLOv8 기반 객체 탐지와 시뮬레이션 알고리즘을 활용한 사과 게임 완전 자동화 프로젝트


GitHub **`README.md`** 파일에 바로 복사해서 붙여넣을 수 있도록, 깔끔하고 전문적인 포맷으로 작성해 드립니다.
프로젝트의 핵심 기술(AI 학습 방법, 알고리즘)과 실행 방법이 모두 포함되어 있습니다.

---

### 📋 적용 방법

1. GitHub 저장소 페이지에서 **`Add a README`** 버튼 클릭 (또는 `README.md` 파일 생성)
2. 아래 내용을 **그대로 복사**해서 붙여넣기
3. `[ ]` 로 표시된 부분(데모 GIF, 깃허브 주소 등)만 본인 상황에 맞게 수정
4. **Commit changes** 클릭

---

```markdown
# 🍎 AI Apple Game Auto-Solver (사과 게임 자동화)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green?logo=ultralytics&logoColor=white)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red?logo=opencv&logoColor=white)
![PyAutoGUI](https://img.shields.io/badge/Automation-PyAutoGUI-orange)

> **"One-Shot Vision & Simulation"**: 단 한 번의 화면 인식으로 게임의 끝까지 예측하여 플레이하는 AI 매크로입니다.

## 📝 프로젝트 개요 (Overview)
웹 기반 퍼즐 게임 **'사과 게임' (드래그하여 합이 10이 되면 사과 삭제)**을 완전 자동화한 프로젝트입니다.
단순한 이미지 서칭이 아닌, **Deep Learning(YOLOv8)**을 활용한 객체 탐지와 **시뮬레이션 알고리즘**을 결합하여 사람보다 빠르고 정확하게 게임을 클리어합니다.

---

## 🚀 핵심 기술 (Key Features)

### 1. 👁️ One-Shot Vision (YOLOv8)
* **Full-Shot Detection:** 170개의 사과가 있는 전체 게임판을 **0.1초 만에** 한 번에 인식합니다.
* **Resolution Free:** 게임판의 크기를 실시간으로 측정하여 비율(Ratio) 기반으로 좌표를 계산하므로, **해상도나 창 크기가 변해도 정확하게 동작**합니다.
* **High Performance:** mAP50 **0.995** (정확도 99.5%) 달성.

### 2. 🏗️ Semi-supervised Learning (Auto-Labeling)
데이터셋 구축 비용을 절감하기 위해 **준지도 학습(Semi-supervised Learning)** 기법을 도입했습니다.
1. **Teacher Model:** 기존에 학습된 '단일 사과 분류 모델'을 활용.
2. **Auto Labeling:** 게임판을 격자로 잘라 Teacher 모델로 판별하고, 이를 바탕으로 전체 화면용 라벨(`.txt`)을 자동 생성.
3. **Student Model:** 자동 생성된 41장의 고품질 데이터로 현재의 **YOLOv8n Full-Shot 모델**을 학습.

### 3. 🧠 Simulation Algorithm (Look-ahead)
* **No Re-capture:** 매번 드래그할 때마다 화면을 다시 찍지 않습니다.
* **Virtual Simulation:** 첫 화면을 분석한 뒤, 내부 매트릭스에서 사과가 사라지고 빈 공간이 연결되는 과정을 **가상으로 시뮬레이션**합니다.
* **Optimization:** **'최소 면적 우선(Greedy Area-First)'** 전략을 사용하여 가까운 사과부터 효율적으로 처리합니다.

---

## 🛠️ 설치 및 실행 방법 (Installation)

이 프로젝트는 Python 3.10+ 환경에서 실행됩니다.

### 1. 저장소 복제 (Clone)
```bash
git clone [https://github.com/Kim-JH-01/AppleGameMacro.git](https://github.com/Kim-JH-01/AppleGameMacro.git)
cd AppleGameMacro

```

### 2. 가상환경 생성 및 활성화

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

```

### 3. 필수 라이브러리 설치

```bash
pip install -r requirements.txt

```

### 4. 실행 (Run)

게임 화면을 띄워둔 상태에서 아래 명령어를 실행하세요.

```bash
python apple_game.py

```

> **종료 방법:** 실행 중 **`ESC`** 키를 길게 누르면 즉시 강제 종료됩니다.

---

## 📂 프로젝트 구조 (File Structure)

```
AppleGameMacro/
├── apple_game.py       # 메인 실행 파일 (Vision, Brain, Hand 클래스 포함)
├── best.pt             # 학습 완료된 YOLOv8 모델 가중치
├── grid_config.json    # 게임판 비율 및 여백 설정 파일
├── requirements.txt    # 의존성 라이브러리 목록
└── train_data/         # 학습에 사용된 데이터셋 (Auto-labeled)

```

---

## 📊 성능 (Performance)

* **Detection Accuracy (mAP50):** 0.995
* **Precision:** 0.999 / **Recall:** 1.000
* **Average Score:** ~102 points (Stable Clear)

---

## 🤝 기여 (Contributing)

이 프로젝트는 팀 프로젝트로 진행되었습니다.
Pull Request는 언제나 환영합니다. 변경 사항을 적용하기 전에 이슈(Issue)를 먼저 등록해 주세요.

---

### ⚠️ Disclaimer

이 프로그램은 학습 및 연구 목적으로 개발되었습니다. 과도한 사용으로 인한 게임 서버의 부하 등에 대한 책임은 사용자에게 있습니다.

```

```
