# GPU acceleration report (branch `progression-transfer`)

## Hardware / software audit

- `nvidia-smi`: **NVIDIA GeForce RTX 5060 Laptop GPU**, 8,151 MiB, driver 621.24,
  CUDA UMD 13.5, compute capability **sm_120**, idle at audit (0% util, 0 MiB used by training;
  only a browser process listed).
- Initial environment: Python 3.12.10, PyTorch **2.15.0+cpu**, `torch.cuda.is_available()=False`.
- First GPU wheel `torch 2.14.1+cu126` installed into an isolated venv: CUDA initialised but
  every kernel failed (`no kernel image is available` — cu126 has no Blackwell sm_120 kernels).
- Final: `torch 2.14.1+cu130` in
  `C:\Users\kkr\AppData\Local\Temp\opencode\gpytorch` (system-site-packages so numpy,
  vizdoom, gymnasium remain available). CUDA works; no other agent's environment modified.

## GPU-enabled scripts (all keep CPU fallback)

- `training/device.py` — centralized `get_device(auto|cuda|cpu)`, `describe()`, CLI helper.
- `training/bc.py` — model, dataset, indices and events on selected device; checkpoints saved
  to CPU tensors; reports samples/s.
- `training/ppo.py` — rollout snapshots and action/loss tensors converted safely from CUDA to
  host numpy; device derived from model parameters.
- `training/train.py` — `--device`, model to device, peak VRAM reporting.
- `transfer/train_transfer.py` — `--device`, `--batch`, full GPU dataset/model path, frozen
  conv stem, CPU-tensor checkpoint save, VRAM reporting.

CPU fallback verified after these changes: `training.train` 256 steps and
`transfer.train_transfer` 1 epoch both completed on `--device cpu`.

## Measured throughput (same 5,560-sample balanced progression dataset)

| Workload | Device | Batch | samples/s | Epoch / run time | Peak VRAM |
|---|---|---:|---:|---:|---:|
| Transfer BC (conv+KEYSTONE) | CPU | 256 | 7,234 | 0.8 s | n/a |
| Transfer BC | CPU | 2048 | 6,342 | 0.9 s | n/a |
| Transfer BC | GPU (cu130) | 256 | 330 | 16.8 s | — |
| Transfer BC | GPU | 2048 | 373 | 14.9 s | — |
| Transfer BC | GPU | 8192 | 1,389–3,873 | 4.0 / 1.4 s | 1,772 MB |
| Plain BC (KEYSTONE only) | CPU | 256 | 35,658 | 0.2 s | n/a |
| Plain BC | GPU | 256 | 4,146 | 1.3 s | — |
| PPO 256 steps | CPU | — | — | 14.6 s | n/a |
| PPO 256 steps | GPU | — | — | 33.1 s | 73.5 MB |

**Conclusion:** CUDA works, but for these very small networks and tiny batches the CPU is
faster (kernel-launch overhead dominates). Larger batches help the GPU but do not beat CPU
here. We therefore keep **CPU as the practical training device** and retain GPU support for
larger future models. No mixed precision was enabled (no stability or speed benefit expected
at this scale).

## Stability and compatibility checks

- GPU smoke: model parameters, batches and loss on CUDA; backward updates executed;
  all losses finite; no NaNs.
- Checkpoint save/load: GPU-trained `gpu_b8192.pt` loaded with the **system CPU-only torch**,
  forward pass finite, legal macro argmax produced — CPU inference compatibility confirmed.
- Deployment: `submission/agent.py` uses the device supplied by the scorer, sets one thread,
  and never touches CUDA-specific state. Official validator PASSED after integration.
- Resource coordination: only short GPU pilots were run; no other agent's process or
  environment was terminated.

## Remaining limitations

- Linux/Docker validation still unverified on this Windows host.
- GPU advantage may appear only with substantially larger encoders/batches; not pursued here
  to avoid destabilising the frozen champion.
- `pin_memory` was not used because the whole dataset is copied to device once at start.
