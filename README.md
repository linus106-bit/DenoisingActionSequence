# Flow Matching 기반 2D Grid 경로 디노이징

`PROMPT.md` 요구사항을 반영해 다음 모듈을 구현했습니다.

- `data_utils.py`: 8x8 grid(기본) 생성, shortest path 기반 clean action 생성, noisy action 합성, `Dataset` 제공
- `model/flow_matching.py`: map encoder + action embedding + time embedding + transformer 기반 velocity field 예측기
- `model/autoregressive.py`: map 조건부 autoregressive trajectory 생성기
- `model/masked_diffusion.py`: mask token 기반 병렬 action 복원 transformer
- `elf_model.py`: ELF 스타일 denoiser/decoder action transformer와 혼합 학습 objective
- `model/common.py`: 모델들이 공유하는 embedding / map encoder / token 상수
- `train.py`: Flow Matching / Autoregressive / Masked Diffusion / ELF 학습 통합 엔트리포인트
  - 학습은 `max_seq_len` 전체 포지션에 대해 loss를 계산해 고정 길이 시퀀스 동작을 유지
  - 학습용 corruption은 noise level `(1-t)`에 비례: 각 샘플에서 `floor(valid_len * (1-t))`개 valid action만 랜덤 치환
  - PAD 위치는 corruption과 loss에서 제외
  - 첫 번째 학습 step에서 예시 `noisy/clean/pred token`, `t`, token별 MSE 일부, 최종 loss를 디버그 출력
- `eval.py`: Flow Matching / Autoregressive / Masked Diffusion / ELF를 모두 지원하는 통합 평가 스크립트
  - `--model_type`에 따라 Euler denoising, autoregressive generate, iterative masked denoising, ELF denoising 중 하나를 실행
  - 공통 데이터 로딩, metric 집계(`valid_token_acc`, `trimmed_exact_match`, `goal_reached` 등), JSON 저장, plot 저장 경로를 공유

## 실행 예시

```bash
python train.py --model_size base --n_samples 1500 --grid_size 10 --epochs 25 --out checkpoints/fm_denoiser.pt
python eval.py --ckpt checkpoints/fm_denoiser.pt --steps 25 --grid_size 10 --max_seq_len 40 --plot_dir artifacts/eval_plots
python train.py --model_type masked_diffusion --model_size base --n_samples 1500 --epochs 25 --out checkpoints/masked_diffusion.pt
python eval.py --model_type masked_diffusion --ckpt checkpoints/masked_diffusion.pt --mask_ratio 1.0 --steps 8
python train.py --model_type elf --model_size base --n_samples 1500 --epochs 25 --out checkpoints/elf_action.pt
# 또는 python train_elf.py --n_samples 1500 --epochs 25
```

> `train.py`는 `--model_type`으로 학습 대상을 통일해서 다룰 수 있습니다.
> - Flow Matching: `--model_type flow_matching`
> - Autoregressive: `--model_type autoregressive`
> - Masked Diffusion: `--model_type masked_diffusion`
> - ELF: `--model_type elf`


## 모델링 방식 비교

- **Flow Matching** (`flow_matching`): clean token embedding과 noisy token embedding 사이의 연속 경로 `x_t`에서 velocity를 예측합니다. 평가 시 Euler 적분으로 noisy sequence를 점진적으로 clean embedding 쪽으로 이동시킨 뒤 token으로 디코딩합니다.
- **Autoregressive** (`autoregressive`): BOS에서 시작해 이전 token들을 조건으로 다음 action token을 순차 생성합니다. teacher forcing cross entropy로 학습하며 PAD는 loss에서 제외합니다.
- **Masked Diffusion** (`masked_diffusion`): clean action sequence의 PAD가 아닌 위치 일부를 `MASK_TOKEN_ID=7`로 치환하고, LLaDA처럼 timestep embedding 없이 map 조건과 양방향 Transformer encoder로 마스킹된 위치의 원래 token을 병렬 복원합니다. loss는 PAD 위치와 마스킹되지 않은 위치를 제외하고, sampled mask probability로 reweight합니다.
- **ELF** (`elf`): `elf_model.py`의 `ELFActionTransformer`를 학습합니다. denoiser step에서는 noisy embedding에서 clean embedding/velocity를 맞추고, decoder step에서는 action token CE를 섞어 학습하며 self-conditioning 설정을 CLI로 조절할 수 있습니다.


## ELF action 모델 학습

`elf_model.py`에 추가된 ELF 스타일 모델을 `train.py --model_type elf`로 직접 학습할 수 있습니다. 학습 루프는 `ELFTrainingConfig`를 CLI 인자로 구성하고, 첫 번째 step에서 CE/L2 loss, decoder step 비율, 예측 token 디버그를 출력합니다.

- `train.py --model_type elf`
  - `ELFActionTransformer` 생성 및 AdamW optimizer 사용
  - `elf_action_loss()`의 mixed denoiser/decoder objective 사용
  - `--elf_decoder_prob`, `--elf_self_cond_prob`, `--elf_grad_clip_norm` 등 ELF 전용 하이퍼파라미터 지원
- `train_elf.py`
  - `train.py --model_type elf`를 호출하는 얇은 래퍼
  - 기본 출력 경로: `checkpoints/elf_action.pt`

실행 예시:

```bash
python train.py --model_type elf --model_size base --n_samples 1500 --epochs 25 --out checkpoints/elf_action.pt
# 또는 python train_elf.py --n_samples 1500 --epochs 25
```

- `eval.py --model_type elf`
  - Gaussian embedding noise에서 시작해 ELF denoiser를 `--steps`만큼 반복 적용한 뒤 action token으로 디코딩합니다.
  - 기본은 embedding similarity head를 사용하고, `--elf_use_decoder_head`를 주면 마지막에 ELF decoder head logits로 token을 선택합니다.
  - `--elf_self_cond_cfg_scale`, `--elf_noise_scale`, `--decode`, `--temperature`, `--save_step_decodes`를 조절할 수 있습니다.
- `eval_elf.py`
  - `eval.py --model_type elf`를 호출하는 얇은 래퍼
  - 기본 체크포인트: `checkpoints/elf_action.pt`

평가 예시:

```bash
python eval.py --model_type elf --ckpt checkpoints/elf_action.pt --steps 25 --num_eval_samples 10
# 또는 python eval_elf.py --steps 25 --num_eval_samples 10
python eval.py --model_type elf --ckpt checkpoints/elf_action.pt --steps 25 --elf_use_decoder_head
```

## Masked Diffusion trajectory 모델

- `model/masked_diffusion.py`의 `MaskedDiffusionTrajectoryTransformer`는 기존 `MapEncoder`, `SinusoidalPositionEmbedding`, token embedding 패턴을 재사용합니다.
- 기본 action vocabulary는 기존 7개 token(`0..6`)에 mask token(`7`)을 더한 `VOCAB_SIZE_WITH_MASK=8`입니다.
- 학습 예시:

```bash
python train.py --model_type masked_diffusion --model_size base --n_samples 1500 --epochs 25 --out checkpoints/masked_diffusion.pt
```

- 평가 예시(전체 valid token mask 후 iterative denoising):

```bash
python eval.py --model_type masked_diffusion --ckpt checkpoints/masked_diffusion.pt --mask_ratio 1.0 --steps 8 --num_eval_samples 10
```

## Autoregressive trajectory 모델

Flow Matching이 정답 시퀀스를 잘 복원하지 못할 때 비교할 수 있도록, map 조건부 **Autoregressive Transformer** 학습/평가 스크립트를 추가했습니다.

- `train.py --model_type autoregressive`
  - teacher forcing 학습 (`input=[BOS, y_0, ..., y_{T-1}]`, `target=[y_0, ..., y_T]`)
  - PAD 토큰은 CE loss에서 제외
  - 출력: `checkpoints/ar_trajectory.pt`
- `train_ar.py`
  - `train.py --model_type autoregressive`를 호출하는 얇은 래퍼(하위 호환용)
- `eval.py --model_type autoregressive`
  - BOS에서 시작해 한 토큰씩 autoregressive 생성
  - `argmax` 또는 `sample` 디코딩 지원
  - 기존 평가 지표(`valid_token_acc`, `trimmed_exact_match`, `goal_reached` 등) JSON 저장
  - 샘플별 예측 경로 시각화(`artifacts/eval_ar_plots/sample_XX.png`) 저장

실행 예시:

```bash
python train.py --model_type autoregressive --model_size base --n_samples 1500 --epochs 25 --out checkpoints/ar_trajectory.pt
# 또는 (호환) python train_ar.py --n_samples 1500 --epochs 25 --out checkpoints/ar_trajectory.pt
python eval.py --model_type autoregressive --ckpt checkpoints/ar_trajectory.pt --decode argmax --num_eval_samples 10
```


## 통합 평가 스크립트

`eval.py` 하나에서 네 모델을 모두 평가합니다. 별도 `eval_ar.py`, `eval_masked_diffusion.py` 파일은 두지 않고, ELF용 `eval_elf.py`만 편의 래퍼로 제공합니다. 기본적으로 `--model_type`만 바꿔 같은 데이터 로딩, metric 집계, JSON 저장, plot 저장 경로를 공유합니다.

```bash
python eval.py --model_type flow_matching --ckpt checkpoints/fm_denoiser.pt --steps 25
python eval.py --model_type autoregressive --ckpt checkpoints/ar_trajectory.pt --decode argmax
python eval.py --model_type masked_diffusion --ckpt checkpoints/masked_diffusion.pt --mask_ratio 1.0 --steps 8
python eval.py --model_type elf --ckpt checkpoints/elf_action.pt --steps 25
```

## 모델 사이즈 설정(config.yaml)

- 루트의 `config.yaml`에서 모델 사이즈 프리셋(`tiny/small/base/large`)을 관리합니다.
- `train.py`는 `--model_size`로 프리셋을 고르고, `--model_config`로 설정 파일 경로를 바꿀 수 있습니다.
- 선택한 프리셋 값(`embed_dim`, `layers`, `heads`)이 CLI 기본값보다 우선 적용됩니다.

예시:

```bash
python train.py --model_size tiny
python train.py --model_size large --model_config config.yaml
```

## Grid size 변경

- 학습 시 `--grid_size`로 데이터셋 격자 크기를 설정할 수 있습니다. (기본값: `8`)
- 평가 시 `--grid_size`를 주지 않으면 체크포인트에 저장된 학습 설정값(`grid_size`)을 자동 사용합니다.

## 필요 패키지

- `torch`
- `numpy`
- `networkx`
- `matplotlib`
- `pyyaml`

## Padding 규칙

- 액션 시퀀스에서 `5`는 EOS, `6`은 **패딩(PAD)** 값입니다.
- 실제 이동 액션은 `1,2,3,4`이며 `0`은 BOS/예약 토큰으로 사용합니다.
- Masked Diffusion 입력에서는 `7`을 mask token으로 사용하므로 모델 vocabulary는 `0..7`입니다.
- 경로 rollout/시각화에서는 EOS(`5`) 또는 PAD(`6`)가 나오면 해당 시점에서 경로 전개를 종료합니다.

## 데이터 생성 규칙

- `clean_actions`(최단경로)가 `max_seq_len`보다 길면 해당 샘플은 버리고 다시 샘플링합니다.
- 따라서 데이터셋 액션 텐서는 항상 길이 `max_seq_len`이며, 목표 경로 이후 구간은 PAD(`6`)로 채워집니다.
