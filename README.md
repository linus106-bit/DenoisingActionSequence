# Flow Matching 기반 2D Grid 경로 디노이징

`PROMPT.md` 요구사항을 반영해 다음 모듈을 구현했습니다.

- `data_utils.py`: 10x10 grid 생성, shortest path 기반 clean action 생성, noisy action 합성, `Dataset` 제공
- `model.py`: map encoder + action embedding + time embedding + transformer 기반 velocity field 예측기
- `train.py`: Flow Matching 학습 (`x_t=(1-t)x_0+t x_1`, target velocity `u_t=x_1-x_0`, MSE)
  - 학습은 `max_seq_len` 전체 포지션에 대해 loss를 계산해 고정 길이 시퀀스 동작을 유지
  - 학습용 corruption은 timestep `t`에 비례: 각 샘플에서 `floor(valid_len * t)`개 valid action만 랜덤 치환
  - clean data의 PAD 구간도 `{0,1,2,3}`로 노이즈화(`--pad_noise_prob`)하여 PAD 복원까지 학습
  - 첫 번째 학습 step에서 예시 `noisy/clean/pred token`, `t`, token별 MSE 일부, 최종 loss를 디버그 출력
- `eval.py`: Euler 적분으로 denoising, LM 스타일(softmax + multinomial sampling) action decoding, 시각화 저장
  - 평가 시 모델은 **GridMap만 조건으로** `max_seq_len` 길이 내에서 action sequence를 생성
  - 초기 입력은 길이 `max_seq_len`의 랜덤 토큰 시퀀스(0~4)이며 denoising 후 PAD(4) 이전까지만 최종 시퀀스로 사용
  - 적분 중 매 step마다 decode된 action 시퀀스를 콘솔에 출력
  - 시각화는 `['Noisy path', 'one step', f'{args.steps} step']` 3개 패널로 저장하며, Noisy path는 주황색으로 표시

## 실행 예시

```bash
python train.py --n_samples 1500 --grid_size 10 --epochs 25 --out checkpoints/fm_denoiser.pt
python eval.py --ckpt checkpoints/fm_denoiser.pt --steps 25 --grid_size 10 --max_seq_len 40 --plot_dir artifacts/eval_plots
```

## Grid size 변경

- 학습 시 `--grid_size`로 데이터셋 격자 크기를 설정할 수 있습니다. (기본값: `10`)
- 평가 시 `--grid_size`를 주지 않으면 체크포인트에 저장된 학습 설정값(`grid_size`)을 자동 사용합니다.

## 필요 패키지

- `torch`
- `numpy`
- `networkx`
- `matplotlib`

## Padding 규칙

- 액션 시퀀스에서 `4`는 **패딩(PAD)** 값입니다.
- 실제 이동 액션은 `0,1,2,3`이며 `4`는 정지/종료 토큰으로 취급합니다.
- 모델 입력/출력 토큰 공간은 일관되게 `0,1,2,3,4`를 사용합니다.
- 경로 rollout/시각화에서는 `4`가 나오면 해당 시점에서 경로 전개를 종료합니다.

## 데이터 생성 규칙

- `clean_actions`(최단경로)가 `max_seq_len`보다 길면 해당 샘플은 버리고 다시 샘플링합니다.
- 따라서 데이터셋 액션 텐서는 항상 길이 `max_seq_len`이며, 목표 경로 이후 구간은 PAD(`4`)로 채워집니다.
