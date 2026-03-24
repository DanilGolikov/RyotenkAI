# Inference Docker Image (vLLM)

vLLM-based inference runtime with LoRA merge dependencies and SSH access.

> In the future, additional inference images with different engines may be added alongside this one. The `-vllm` suffix distinguishes this image from other engine variants.

## Image Contents

- **Base:** `vllm/vllm-openai:v0.6.3.post1`
- **Merge deps:** `peft`, `accelerate`, `safetensors`, `huggingface-hub`
- **SSH server:** for control-plane access (RunPod + single_node)
- **merge_lora.py:** LoRA merge script baked into `/opt/helix/`

One image covers all vLLM inference scenarios:
- **single_node**: pipeline uses it as both `merge_image` and `serve_image`
- **RunPod Pods**: pipeline pulls it as the pod image

## Build & Push

```bash
cd docker/inference

# Build and push (auto-increments version)
./build_and_push.sh --username ryotenkai

# Build without pushing
./build_and_push.sh --username ryotenkai --no-push

# Specify exact version
./build_and_push.sh --username ryotenkai --version v1.2.0

# Minor/major bump
./build_and_push.sh --username ryotenkai --bump minor
```

## Pipeline Config

```yaml
inference:
  engine: vllm
  engines:
    vllm:
      merge_image: "ryotenkai/inference-vllm:v1.0"
      serve_image: "ryotenkai/inference-vllm:v1.0"

  # RunPod pods (if used):
  # providers:
  #   runpod:
  #     inference:
  #       pod:
  #         image_name: "ryotenkai/inference-vllm:v1.0"
```

## How It Works

The pipeline **pulls** the image from Docker Hub at deploy time — no local builds required.

1. `build_and_push.sh` builds the image and pushes it to Docker Hub (both versioned tag and `:latest`)
2. Pipeline SSHs to the inference node and runs `docker pull`
3. For merge jobs: runs merge_lora.py inside the container (ephemeral)
4. For serving: starts vLLM serve in a detached container

## File Structure

```
docker/inference/
├── Dockerfile          # vLLM inference image
├── build_and_push.sh   # Build & push to Docker Hub
├── entrypoint.sh       # Container entrypoint (SSH + exec)
├── scripts/
│   └── merge_lora.py   # LoRA adapter merge script
└── README.md
```

## Manual Build

```bash
docker build -f Dockerfile -t ryotenkai/inference-vllm:v1.0 docker/inference
```

## Troubleshooting

### NVIDIA runtime not found

```bash
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Out of memory during merge

- Use a smaller `max_model_len` in vLLM config
- Reduce `gpu_memory_utilization` (e.g., 0.80)
- Use quantization (`awq` or `gptq`)
