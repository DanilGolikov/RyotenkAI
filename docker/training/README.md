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

# Build and push (auto-increments patch: v1.0.1 → v1.0.2)
./build_and_push.sh --username ryotenkai

# Build without pushing
./build_and_push.sh --username ryotenkai --no-push

# Specify exact version
./build_and_push.sh --username ryotenkai --version v1.2.0

# Minor/major bump
./build_and_push.sh --username ryotenkai --bump minor
```

The pipeline **pulls** the image from Docker Hub at runtime — no local
builds required. After a successful push, copy the new tag into
``src/runner/__about__.py:_DEFAULT_RUNTIME_IMAGE`` (the script prints
the exact line at the end of its run).

### Tag policy

Tags are clean semver (``v1.0.2``). The dependency profile (CUDA,
Python, Torch, vLLM base) is **not** in the tag — it lives in:

- **OCI labels** on the image (``docker inspect <image> | jq '.[0].Config.Labels'``)
- A ``/opt/ryotenkai/version.txt`` text file inside the container

```bash
# Inspect a built image's profile without pulling all the layers:
docker inspect ryotenkai/ryotenkai-training-runtime:v1.0.2 \
  | jq '.[0].Config.Labels'
# {
#   "org.opencontainers.image.title": "ryotenkai-training-runtime",
#   "org.opencontainers.image.version": "1.0.2",
#   "ryotenkai.cuda": "12.4",
#   "ryotenkai.python": "3.12",
#   "ryotenkai.torch": "2.5.1",
#   …
# }

# Or read the file directly:
docker run --rm ryotenkai/ryotenkai-training-runtime:v1.0.2 cat /opt/ryotenkai/version.txt
# ryotenkai-training-runtime v1.0.2
# cuda: 12.4
# python: 3.12
# torch: 2.5.1
```

Why: keeping the tag short means
``src/runner/__about__.py:_DEFAULT_RUNTIME_IMAGE`` doesn't have to
re-spell every dependency every time you bump CUDA. Bumping a base
dependency is a build-args change, not a tag rename.

### Bumping a base dependency (CUDA / Python / Torch)

1. Update the ``FROM`` line in ``Dockerfile.runtime`` to the new base.
2. Update ``DEFAULT_CUDA_VERSION`` / ``DEFAULT_PYTHON_VERSION`` /
   ``DEFAULT_TORCH_VERSION`` defaults in ``build_and_push.sh`` so the
   labels match the actual base.
3. Run ``./build_and_push.sh --bump minor`` (or ``major`` for breaking
   changes).
4. Update ``src/runner/__about__.py`` to the new tag the script prints.

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
├── runtime_check.py         # Dependency validation script (copied into image at build)
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
