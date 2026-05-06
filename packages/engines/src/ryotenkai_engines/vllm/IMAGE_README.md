# Inference Docker Image (vLLM)

vLLM-based inference runtime with LoRA merge dependencies and SSH access.

> Additional engines (SGLang, MAX, …) may live alongside this one as
> sibling folders under `packages/engines/src/ryotenkai_engines/`. Each
> engine ships its own `Dockerfile` + `engine.toml`; the `-vllm` suffix
> on the image tag distinguishes them.

## Image Contents

- **Base:** `vllm/vllm-openai:v0.6.3.post1`
- **Merge deps:** `peft`, `accelerate`, `safetensors`, `huggingface-hub`
- **SSH server:** for control-plane access (RunPod + single_node)
- **merge_lora.py:** LoRA merge script baked into `/opt/helix/`

One image covers all vLLM inference scenarios:
- **single_node**: provider uses it both for adapter merging and vLLM serve
- **RunPod Pods**: provider pulls it as the pod image

## Build & Push

The unified script walks every engine folder and builds them with the
convention name `{username}/inference-{engine.id}:{engine.version}`,
where `id` and `version` come from each engine's `engine.toml`. Drop a
new engine folder ⇒ no script edits needed.

```bash
# Build and push every engine in packages/engines/src/ryotenkai_engines/
./packages/engines/scripts/build_and_push_engines.sh --username ryotenkai

# Build only — skip push
./packages/engines/scripts/build_and_push_engines.sh --username ryotenkai --no-push

# Restrict to a single engine
./packages/engines/scripts/build_and_push_engines.sh --username ryotenkai --engine vllm
```

To bump this engine's version, edit `engine.toml [engine].version` (our
integration-contract semver — the image tag follows it 1:1) then re-run
the script.

## Pipeline Config

```yaml
inference:
  enabled: true
  provider: single_node
  engine:
    kind: vllm
    max_model_len: 8192
    quantization: awq
```

The image to pull is resolved by `ryotenkai_engines.images.resolve_image`
(env override → provider override → manifest default → convention).

## How It Works

The pipeline **pulls** the image from the registry at deploy time — no
local builds required on the inference node.

1. `build_and_push_engines.sh` builds and pushes the image (versioned
   tag + `:latest`).
2. The provider SSHes / connects to the inference target and runs
   `docker pull`.
3. For merge jobs: runs `merge_lora.py` inside the container (ephemeral).
4. For serving: the engine returns a `LaunchSpec`; the provider formats
   it via `ryotenkai_providers.inference.launch.format_docker_run` and
   starts a detached container.

## File Structure

```
packages/engines/src/ryotenkai_engines/vllm/
├── Dockerfile          # vLLM inference image
├── engine.toml         # capabilities + version metadata
├── runtime.py          # IInferenceEngine implementation
├── config.py           # VLLMEngineConfig
├── entrypoint.sh       # Container entrypoint (SSH + exec)
├── merge_lora.py       # LoRA adapter merge script
└── IMAGE_README.md
```

## Manual Build

```bash
docker build \
  -f packages/engines/src/ryotenkai_engines/vllm/Dockerfile \
  -t ryotenkai/inference-vllm:1.0.0 \
  packages/engines/src/ryotenkai_engines/vllm
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
