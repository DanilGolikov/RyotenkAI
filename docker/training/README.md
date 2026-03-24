# Training Docker Image

Stable GPU runtime for LLM fine-tuning. The image provides CUDA, PyTorch and all training dependencies; the RyotenkAI pipeline delivers training code into the run workspace at execution time.

## Image Contents

- **Base:** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- **Python:** 3.12 (Conda)
- **PyTorch:** included in base image (2.5.1+cu124)
- **Deps:** everything from `requirements.runtime.txt`
- **SSH:** `openssh-server` (for RunPod access)

## Build & Push

```bash
cd docker/training

# Build and push (auto-increments version)
./build_and_push.sh --username ryotenkai

# Build without pushing
./build_and_push.sh --username ryotenkai --no-push

# Specify exact version
./build_and_push.sh --username ryotenkai --version v2.6.0

# Minor/major bump
./build_and_push.sh --username ryotenkai --bump minor
```

The pipeline **pulls** the image from Docker Hub at runtime — no local builds required.

## Pipeline Config

### single_node

```yaml
providers:
  single_node:
    connect:
      ssh:
        alias: my-gpu-node
    training:
      workspace_path: /home/user/ryotenkai_training
      execution_mode: docker
      docker_image: "ryotenkai/ryotenkai-training-runtime:latest"
```

### RunPod

```yaml
providers:
  runpod:
    training:
      image_name: "ryotenkai/ryotenkai-training-runtime:latest"
```

## Execution Contract

| Provider | How it works |
|----------|-------------|
| **single_node** | Pipeline creates a run dir on the host, starts the container with `-v <run_dir>:/workspace`, sets `PYTHONPATH=/workspace`. Logs and markers are written to `/workspace`. |
| **RunPod** | Pipeline SSHs into the pod container and runs `python -m src.training.run_training` directly. |

## File Structure

```
docker/training/
├── Dockerfile.runtime       # Training runtime image
├── build_and_push.sh        # Build & push to Docker Hub
├── entrypoint.sh            # Container entrypoint (SSH + exec)
├── requirements.runtime.txt # Python dependencies
├── ../../scripts/runtime_check.py  # Dependency validation script (copied into image at build)
└── README.md
```

## Troubleshooting

### NVIDIA runtime not working

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu24.04 nvidia-smi
```

If it fails, install `nvidia-container-toolkit` on the host.

### Out of memory

- Reduce batch size / sequence length in training config
- Use gradient checkpointing (`gradient_checkpointing: true`)
- Use LoRA with lower rank
