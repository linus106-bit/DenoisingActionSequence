# [Prompt] Flow Matching 기반 2D 경로 디노이징(Denoising) 실험 설계

## 1. 프로젝트 개요 (Context & Goal)
우리는 10x10 격자판(2D Grid World)에서 장애물을 피해 목표 지점(Goal)으로 가는 **'노이즈 섞인 액션 시퀀스'**를 입력받아, 이를 최적의 경로로 **복원(Denoising)**하는 모델을 개발하고자 합니다. 

이 프로젝트는 이산적인 액션(0:상, 1:하, 2:좌, 3:우)을 연속적인 임베딩 공간으로 매핑한 뒤, **Flow Matching(FM)** 프레임워크를 적용하여 노이즈 상태($x_0$)에서 정답 경로 상태($x_1$)로 가는 벡터 필드(Velocity Field)를 학습하는 것이 핵심입니다.

---

## 2. 기술 요구사항 (Requirements)
* **언어 및 프레임워크:** Python, PyTorch
* **필수 라이브러리:** `numpy`, `networkx` (데이터 생성 및 최단 경로 계산), `matplotlib` (결과 시각화)
* **주요 알고리즘:** Flow Matching (Optimal Transport Path), A* Search (Ground Truth 생성용)

---

## 3. 상세 구성 요소 (Component Breakdown)

### A. 데이터 생성부 (`data_utils.py`)
1. **Grid 환경:** 10x10 크기의 바이너리 행렬 생성 (0: 통로, 1: 벽). 벽의 비율은 약 20~30%로 설정.
2. **정답 경로($x_1$):** `networkx.shortest_path`를 사용하여 시작점부터 목표점까지의 최단 액션 시퀀스를 생성.
3. **노이즈 경로($x_0$):** 정답 경로에 '무작위 액션 교체', '무의미한 반대 방향 액션 쌍 삽입' 등의 노이즈를 추가.
4. **데이터 규격:** 모든 시퀀스는 고정 길이로 패딩(Padding) 처리하며, `(Map_Tensor, Noisy_Sequence, Clean_Sequence)` 쌍을 반환하는 DataLoader 구현.

### B. 모델 아키텍처 (`model.py`)
1. **Map Encoder:** 10x10 지도를 입력받아 공간적 특징을 추출하는 간단한 CNN 또는 MLP.
2. **Action Embedding:** 0~3의 정수 액션을 고차원 벡터(예: 32 or 64 dim)로 변환.
3. **Flow Matching Transformer:**
    * **Input:** $x_t$ (임베딩된 시퀀스), $t$ (Time step Scalar), Map Context Feature.
    * **Time Embedding:** $t$를 Sinusoidal 또는 MLP를 통해 임베딩하여 시퀀스에 주입.
    * **Conditioning:** Map Features를 Transformer의 각 레이어에 Cross-Attention 또는 필터 방식으로 결합.
    * **Output:** 현재 시점 $t$에서의 속도 벡터 $v_t$ (Velocity) 예측.

### C. Flow Matching 학습 로직 (`train.py`)
1. **Probability Path:** $x_t = (1 - t)x_0 + t x_1$ (Linear Interpolation 사용).
2. **Target Velocity:** $u_t = x_1 - x_0$.
3. **Loss Function:** $MSE(Model(x_t, t, Map) - (x_1 - x_0))$.
4. 학습 시 각 배치마다 $t \in [0, 1]$을 무작위로 샘플링.

### D. 추론 및 시각화 (`eval.py`)
1. **ODE Solver:** Euler Method를 사용하여 $t=0$($x_0$)에서 $t=1$($x_{final}$)까지 $x_{next} = x_t + v_t \cdot \Delta t$로 적분 수행.
2. **Decoding:** 최종 결과 벡터를 Nearest Neighbor 방식으로 0, 1, 2, 3 액션으로 변환.
3. **Visualization:** 원본 노이즈 경로와 모델이 복원한 경로를 10x10 격자 위에 화살표 또는 선으로 시각화.

---

## 4. 에이전트 지시사항 (Instructions)
1. 위 설계를 바탕으로 모듈화된 파이썬 코드를 작성해줘.
2. 특히 **Flow Matching**의 수식($x_t$ 생성 및 $v_t$ 예측)이 정확하게 구현되어야 해.
3. 학습 후, 임의의 노이즈 경로를 넣었을 때 얼마나 깨끗하게 복원되는지 확인할 수 있는 시각화 데모 코드를 포함해줘.
4. 10x10 환경에서 빠르게 학습될 수 있도록 모델은 너무 무겁지 않게 설계해줘.
