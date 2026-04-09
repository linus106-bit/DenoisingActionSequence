# Flow Matching 기반 2D Grid 경로 디노이징

`PROMPT.md` 요구사항을 반영해 다음 모듈을 구현했습니다.

- `data_utils.py`: 10x10 grid 생성, shortest path 기반 clean action 생성, noisy action 합성, `Dataset` 제공
- `model.py`: map encoder + action embedding + time embedding + transformer 기반 velocity field 예측기
- `train.py`: Flow Matching 학습 (`x_t=(1-t)x_0+t x_1`, target velocity `u_t=x_1-x_0`, MSE)
- `eval.py`: Euler 적분으로 denoising, LM 스타일(softmax + multinomial sampling) action decoding, 시각화 저장

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
