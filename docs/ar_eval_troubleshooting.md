# AR eval에서 `goal_reached`가 잘 안 나오는 이유 (코드 기반 진단)

이 문서는 `eval_ar.py`/`train.py`/`data_utils.py` 구현을 기준으로, 100개 샘플 평가에서 목표 도달률이 낮게 나오는 전형적인 원인을 정리합니다.

## 현재 AR 학습 방식 한 줄 요약
- **네, 현재 AR은 Next Token Prediction(다음 액션 토큰 예측)으로 학습합니다.**
- 입력은 `tokens_in=[BOS, y_0, ..., y_{T-1}]`, 타깃은 `target=[y_0, ..., y_T]` 형태의 teacher forcing입니다.
- 손실은 각 시점의 다음 토큰 분류 CE(`ignore_index=PAD`)이며, 토큰 공간은 `{BOS(0), 상/하/좌/우(1~4), EOS(5), PAD(6)}`입니다.

## causal attention 구현 여부
- **네, 구현되어 있습니다.**
- `AutoregressiveTrajectoryTransformer`에서 상삼각 `-inf` 마스크를 만들어(`_causal_mask`) 미래 토큰을 보지 못하게 하고,
  `TransformerEncoder` 호출 시 `mask=`로 전달합니다.
- 따라서 각 시점은 자기 자신과 과거 토큰까지만 참조합니다.

## 1) Train/Eval 분포 차이 (매번 새로운 랜덤 맵)
- 학습 데이터와 평가 데이터 모두 `GridDenoiseDataset`를 새로 샘플링합니다.
- `eval_ar.py`는 체크포인트를 불러온 뒤, 평가용 데이터셋을 다시 생성합니다.
- 즉 **학습에서 본 맵/경로와 다른 분포 샘플**에서 일반화를 바로 요구합니다.

관련 코드:
- `train.py`: `GridDenoiseDataset(n_samples=args.n_samples, ...)`로 학습셋 생성
- `eval_ar.py`: `GridDenoiseDataset(n_samples=args.num_eval_samples, ...)`로 평가셋 재생성

## 2) AR 학습 목표는 token CE라서, trajectory 성공률과 직접 정렬되지 않음
- AR loss는 teacher forcing + token 단위 cross entropy입니다.
- 목적함수는 각 위치 token 예측 정확도이고, 최종적으로 목표 지점 도달(`goal_reached`)을 직접 최적화하지 않습니다.
- 따라서 token 1~2개만 어긋나도 최종 위치가 크게 벗어날 수 있습니다.

관련 코드:
- `train.py`: `ar_loss()`에서 `F.cross_entropy(..., ignore_index=PAD_ACTION)` 사용
- `eval_ar.py`: 평가지표는 `trajectory_metrics(..., goal_reached)` 별도 계산

## 3) Teacher forcing ↔ 생성 시 free-running 간 노출 편향
- 학습 시 입력은 `[BOS, 정답 prefix]`입니다.
- 평가 생성 시에는 모델이 방금 낸 token을 다음 입력으로 사용합니다.
- 초반 1개 오차가 누적되며 경로가 쉽게 붕괴할 수 있습니다.

관련 코드:
- `train.py`: `tokens_in = [BOS, clean[:-1]]`
- `model/autoregressive.py`: `generate()` 루프에서 `tokens = torch.cat([tokens, next_token], ...)`

## 4) AR 경로 생성은 noisy action을 입력으로 사용하지 않음
- 현재 AR은 맵 조건부 planner 형태입니다.
- `data_utils.py`가 생성하는 `noisy_actions`는 AR 학습/평가에서 사용되지 않습니다.
- 만약 사용자가 기대한 것이 "노이즈 시퀀스 복원"이라면, 현재 AR 구현 목표와 다릅니다.

관련 코드:
- `train.py`의 `ar_loss()`는 `clean_actions`만 사용
- `eval_ar.py`도 `model.generate(map_tensor, ...)`만 호출

## 5) 생성 종료 규칙이 조기 종료를 유도할 수 있음
- 디코딩 시 BOS만 금지하고 EOS/PAD는 허용됩니다.
- 모델이 EOS/PAD를 너무 이르게 내면 짧은 경로로 종료되어 goal miss가 증가합니다.

관련 코드:
- `model/autoregressive.py`: `logits[:, BOS_TOKEN_ID] = -inf` (EOS/PAD는 금지 안 함)
- `eval_ar.py`: `trim_at_stop()`이 EOS/PAD를 만나면 시퀀스 절단

## 6) 데이터 난이도 자체가 높은 편
- 벽 비율 20~30%, 최단경로 길이 최소 8, grid 기본 10x10.
- 경로 길이/장애물 조합이 다양해 소용량 모델/짧은 학습으로는 일반화가 어렵습니다.

관련 코드:
- `data_utils.py`: `wall_ratio_range=(0.2,0.3)`, `min_path_len=8`

## 모델 사이즈를 늘리는 것, 효과 있을까?
- **네, 우선순위 높은 시도입니다.** 현재 기본값(`embed_dim=64`, `layers=3`, `heads=4`)은 작아서 복잡한 맵/긴 경로 일반화에 한계가 있을 수 있습니다.
- 권장 탐색 순서(메모리/속도 고려):
  1. `embed_dim 64 -> 128`
  2. `layers 3 -> 4~6`
  3. `heads 4 -> 8` (단, `embed_dim`과 나눠떨어지게)
- 단, 모델만 키우면 과적합/학습 불안정이 생길 수 있어 `n_samples`, `epochs`, `weight_decay`를 함께 조정하는 게 안전합니다.
- 실무적으로는 **"모델 2배 + 데이터 2배"**를 함께 올렸을 때 `goal_reached` 개선 가능성이 가장 큽니다.

예시 커맨드:
```bash
python train.py --model_type autoregressive \
  --embed_dim 128 --layers 5 --heads 8 \
  --n_samples 3000 --epochs 40 \
  --lr 1e-3 --weight_decay 1e-4 \
  --out checkpoints/ar_trajectory_larger.pt
```

## 바로 해볼 개선 우선순위
1. **학습량 증가**: `n_samples`, `epochs`를 먼저 2~4배 확대.
2. **검증셋 분리 + 고정 시드**: 동일한 난이도에서 추세를 보기.
3. **scheduled sampling / DAgger 류**로 노출 편향 완화.
4. **조기 EOS/PAD 억제**: 최소 길이 이전에는 EOS/PAD logit penalty.
5. **목표 정렬 보조 loss**: rollout 기반 goal 보상(또는 differentiable surrogate) 추가.
6. "denoising"이 목표라면 AR 입력에 `noisy_actions` 조건을 추가.

