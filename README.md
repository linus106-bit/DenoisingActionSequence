# Flow Matching 기반 2D Grid 경로 디노이징

`PROMPT.md` 요구사항을 반영해 다음 모듈을 구현했습니다.

- `data_utils.py`: 10x10 grid 생성, shortest path 기반 clean action 생성, noisy action 합성, `Dataset` 제공
- `model.py`: map encoder + action embedding + time embedding + transformer 기반 velocity field 예측기
- `train.py`: Flow Matching 학습 (`x_t=(1-t)x_0+t x_1`, target velocity `u_t=x_1-x_0`, MSE)
- `eval.py`: Euler 적분으로 denoising, LM 스타일(softmax + multinomial sampling) action decoding, 시각화 저장
  - 평가 시 입력 `noisy_actions`는 유효 길이 전체를 랜덤 액션(0~3)으로 채운 **완전 노이즈 시퀀스**를 사용
  - 적분 중 매 step마다 decode된 action 시퀀스를 콘솔에 출력
  - 시각화는 `['Noisy path', 'one step', f'{args.steps} step']` 3개 패널로 저장하며, Noisy path는 주황색으로 표시

## 실행 예시

```bash
python train.py --n_samples 1500 --epochs 25 --out checkpoints/fm_denoiser.pt
python eval.py --ckpt checkpoints/fm_denoiser.pt --steps 25 --plot_out artifacts/denoise_demo.png
```

## 필요 패키지

- `torch`
- `numpy`
- `networkx`
- `matplotlib`

## Padding 규칙

- 액션 시퀀스에서 `-1`은 **패딩(PAD)** 값입니다.
- 실제 액션은 `0,1,2,3`만 사용합니다.
- 모델 내부에서는 `-1`을 PAD 토큰 id(`4`)로 매핑해 임베딩합니다.
- `nn.Embedding(..., padding_idx=4)`를 사용해 PAD 임베딩은 학습 업데이트에서 제외됩니다.
- 경로 rollout/시각화에서는 `-1`이 나오면 해당 시점에서 경로 전개를 종료합니다.
